"""Same-day industry-relative MLP. Complements trees that already use cat_6 as a group intercept."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.dataset import neutralize_groups_1d, ols_residual_1d, precompute_ind_cols
from src.models.gru_ts import pearson_ic_loss


class CSMLPNet(nn.Module):
    def __init__(
        self,
        d_feat: int,
        hidden: int,
        dropout: float,
        cat_cardinalities: list[int] | None = None,
        cat_embed_dim: int = 8,
    ):
        super().__init__()
        cards = [int(c) for c in (cat_cardinalities or []) if int(c) > 0]
        self.embeds = nn.ModuleList(nn.Embedding(c, cat_embed_dim) for c in cards) if cards else None
        in_dim = d_feat + (len(cards) * cat_embed_dim if cards else 0)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, cats: torch.Tensor | None = None) -> torch.Tensor:
        h = x
        if self.embeds is not None:
            if cats is None:
                raise ValueError("CSMLPNet was built with embeddings but cats is None")
            pieces = [h]
            for i, emb in enumerate(self.embeds):
                idx = cats[:, i].clamp(0, emb.num_embeddings - 1)
                pieces.append(emb(idx))
            h = torch.cat(pieces, dim=-1)
        return self.mlp(h).squeeze(-1)


class CSMLPModel:
    """Per-stock MLP on industry z-scores. Optional industry-residual labels. No history, no raw."""

    def __init__(self, cfg: dict[str, Any], seed: int = 42):
        self.cfg = dict(cfg)
        self.seed = int(seed)
        self.cols = [int(i) for i in (cfg.get("num_indices") or [])]
        if not self.cols:
            raise ValueError("CS MLP needs features.num_indices")
        self.industry_col = int(cfg.get("industry_col", 6))
        self.hidden_size = int(cfg.get("hidden_size", 64))
        self.dropout = float(cfg.get("dropout", 0.1))
        self.clip = cfg.get("clip", 5.0)
        self.max_train_stocks = cfg.get("max_train_stocks_per_day")
        self.min_train_stocks = int(cfg.get("min_train_stocks", 8))
        self.log_every = int(cfg.get("log_every_days", 400))
        self.cat_indices = [int(i) for i in (cfg.get("cat_indices") or [])]
        if self.industry_col in self.cat_indices:
            raise ValueError("do not feed cat_6 into the MLP; trees already use it as a group intercept")
        self.cat_embed_dim = int(cfg.get("cat_embed_dim", 8))
        self.loss_name = str(cfg.get("loss", "pearson_ic")).lower()
        self.residual_industry = bool(cfg.get("residual_industry", True))
        self.target_mode = str(cfg.get("target_mode") or ("industry" if self.residual_industry else "y1"))
        self.hard_repeat = int(cfg.get("hard_repeat", 1))
        self.hard_days: set[int] = set()
        self.aux_pred: np.ndarray | None = None
        self.freeze_hidden = bool(cfg.get("freeze_hidden", False))
        self.net: CSMLPNet | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ind_sel: np.ndarray | None = None
        self.cat_sel: np.ndarray | None = None
        self.industry: np.ndarray | None = None
        self.cat_cardinalities: list[int] = []

    def prepare_features(self, data: dict[str, Any]) -> None:
        print(
            f"precompute industry z-score cols={self.cols} "
            f"industry_col={self.industry_col} T={data['num_x'].shape[0]}"
        )
        self.ind_sel = precompute_ind_cols(
            data["num_x"],
            data["mask_x"],
            data["cat_x"],
            self.cols,
            self.industry_col,
        )
        if self.clip is not None:
            np.clip(self.ind_sel, -float(self.clip), float(self.clip), out=self.ind_sel)
        self.industry = np.asarray(data["cat_x"][..., self.industry_col])
        if self.cat_indices:
            self.cat_sel = np.asarray(data["cat_x"][..., self.cat_indices], dtype=np.int32)
            maxes = self.cat_sel.max(axis=(0, 1))
            self.cat_cardinalities = [int(m) + 1 for m in np.asarray(maxes).ravel()]
        else:
            self.cat_sel = None
            self.cat_cardinalities = []
        print(
            f"ind_sel={self.ind_sel.shape} ~{self.ind_sel.nbytes / 1e9:.2f}GB "
            f"device={self.device} cats={self.cat_indices or '-'} "
            f"target_mode={self.target_mode} hard_repeat={self.hard_repeat}"
        )

    def _build_net(self) -> CSMLPNet:
        return CSMLPNet(
            len(self.cols),
            self.hidden_size,
            self.dropout,
            cat_cardinalities=self.cat_cardinalities,
            cat_embed_dim=self.cat_embed_dim,
        ).to(self.device)

    def _cats_tensor(self, t: int, idx: np.ndarray) -> torch.Tensor | None:
        if self.cat_sel is None:
            return None
        cats = np.ascontiguousarray(self.cat_sel[int(t), idx])
        return torch.from_numpy(cats.astype(np.int64, copy=False)).to(self.device)

    def _x_tensor(self, t: int, idx: np.ndarray) -> torch.Tensor:
        if self.ind_sel is None:
            raise RuntimeError("call prepare_features first")
        x = np.ascontiguousarray(self.ind_sel[int(t), idx])
        return torch.from_numpy(x).to(self.device)

    def _loss(self, pred: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
        if self.loss_name in {"pearson_ic", "ic"}:
            return pearson_ic_loss(pred, yb)
        return nn.functional.mse_loss(pred, yb)

    def _day_target(self, data: dict[str, Any], t: int, idx: np.ndarray) -> np.ndarray:
        y = np.asarray(data["y1"][t], dtype=np.float32)
        mask = data["mask_y"][t]
        if self.target_mode == "tree_ols":
            if self.aux_pred is None:
                raise RuntimeError("target_mode=tree_ols needs aux_pred")
            y = ols_residual_1d(y, self.aux_pred[t], mask)
        elif self.target_mode == "industry":
            groups = self.industry[t] if self.industry is not None else data["cat_x"][t][:, self.industry_col]
            y = neutralize_groups_1d(y, groups, mask)
        return y[idx]

    def _apply_freeze_hidden(self) -> None:
        if self.net is None:
            raise RuntimeError("CS MLP net missing")
        for name, p in self.net.named_parameters():
            p.requires_grad = name.startswith("mlp.6.")
        trainable = [n for n, p in self.net.named_parameters() if p.requires_grad]
        frozen = [n for n, p in self.net.named_parameters() if not p.requires_grad]
        print(f"freeze_hidden=True trainable={trainable} frozen={len(frozen)} tensors")

    def fit(
        self,
        data: dict[str, Any],
        train_start: int,
        train_end: int,
        valid: dict[str, Any] | None = None,
        scorer=None,
    ) -> dict[str, Any]:
        from src.metrics import mean_rank_ic

        score_fn = scorer or mean_rank_ic
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if self.ind_sel is None:
            self.prepare_features(data)

        if self.net is None:
            self.net = self._build_net()
        if self.freeze_hidden:
            self._apply_freeze_hidden()
        lr = float(self.cfg.get("lr", 1e-3))
        decay = float(self.cfg.get("weight_decay", 1e-4))
        epochs = int(self.cfg.get("max_epochs", 8))
        patience = int(self.cfg.get("patience", 3))
        trainable = [p for p in self.net.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("no trainable CS MLP parameters")
        opt = torch.optim.Adam(trainable, lr=lr, weight_decay=decay)
        rng = np.random.default_rng(self.seed)
        best_ic = -1e9
        best_state: dict[str, torch.Tensor] | None = None
        stale = 0
        history: list[dict[str, Any]] = []
        n_params = sum(p.numel() for p in self.net.parameters())
        print(
            f"cs_mlp params={n_params} hidden={self.hidden_size} "
            f"loss={self.loss_name} target_mode={self.target_mode} "
            f"hard_days={len(self.hard_days)} hard_repeat={self.hard_repeat} "
            f"freeze_hidden={self.freeze_hidden}"
        )

        base_days = list(range(int(train_start), int(train_end)))
        days: list[int] = []
        for t in base_days:
            n = self.hard_repeat if t in self.hard_days else 1
            days.extend([t] * max(n, 1))
        for epoch in range(epochs):
            self.net.train()
            rng.shuffle(days)
            losses: list[float] = []
            for i, t in enumerate(days):
                mask_y = np.asarray(data["mask_y"][t], dtype=bool)
                idx = np.flatnonzero(mask_y)
                if idx.size < self.min_train_stocks:
                    continue
                if self.max_train_stocks and idx.size > int(self.max_train_stocks):
                    idx = np.sort(rng.choice(idx, size=int(self.max_train_stocks), replace=False))
                y = self._day_target(data, t, idx)
                xb = self._x_tensor(t, idx)
                yb = torch.from_numpy(np.ascontiguousarray(y)).to(self.device)
                pred = self.net(xb, self._cats_tensor(t, idx))
                loss = self._loss(pred, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                losses.append(float(loss.item()))
                if self.log_every and (i + 1) % self.log_every == 0:
                    print(
                        f"  epoch {epoch + 1} day {i + 1}/{len(days)} "
                        f"loss={float(np.mean(losses[-self.log_every:])):.5f}",
                        flush=True,
                    )
            valid_ic = None
            if valid is not None:
                pred_v = self.predict_panel(valid, data)
                valid_ic = float(score_fn(pred_v, valid["y1"], valid["mask_y"]))
                if np.isfinite(valid_ic) and valid_ic > best_ic:
                    best_ic = valid_ic
                    best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                    stale = 0
                else:
                    stale += 1
            mean_loss = float(np.mean(losses)) if losses else 0.0
            print(f"epoch {epoch + 1}/{epochs} loss={mean_loss:.5f} valid_RankIC={valid_ic}", flush=True)
            history.append({"epoch": epoch + 1, "loss": mean_loss, "valid_ic": valid_ic})
            if valid is not None and stale >= patience:
                print("early stop")
                break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return {"best_valid_ic": best_ic if best_ic > -1e8 else None, "history": history}

    def predict_panel(
        self,
        split_data: dict[str, Any],
        full_data: dict[str, Any] | None = None,
        fill_invalid: float = 0.0,
    ) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("CS MLP is not fitted")
        if self.ind_sel is None:
            if full_data is None:
                raise RuntimeError("call prepare_features first")
            self.prepare_features(full_data)
        self.net.eval()
        start = int(split_data.get("start", 0))
        n_days, n_stocks = split_data["mask_x"].shape
        out = np.full((n_days, n_stocks), float(fill_invalid), dtype=np.float32)
        with torch.no_grad():
            for local_t in range(n_days):
                mask = np.asarray(split_data["mask_x"][local_t], dtype=bool)
                idx = np.flatnonzero(mask)
                if idx.size == 0:
                    continue
                t = start + local_t
                pred = self.net(self._x_tensor(t, idx), self._cats_tensor(t, idx))
                out[local_t, idx] = pred.detach().cpu().numpy().astype(np.float32)
        return out

    def save(self, path: str | Path) -> None:
        if self.net is None:
            raise RuntimeError("CS MLP is not fitted")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path)
        meta = {
            "cfg": self.cfg,
            "seed": self.seed,
            "cols": self.cols,
            "industry_col": self.industry_col,
            "hidden_size": self.hidden_size,
            "dropout": self.dropout,
            "clip": self.clip,
            "cat_indices": self.cat_indices,
            "cat_embed_dim": self.cat_embed_dim,
            "cat_cardinalities": self.cat_cardinalities,
            "loss": self.loss_name,
            "residual_industry": self.residual_industry,
            "target_mode": self.target_mode,
            "hard_repeat": self.hard_repeat,
        }
        path.with_suffix(path.suffix + ".meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self, path: str | Path) -> None:
        path = Path(path)
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.cfg = meta.get("cfg", self.cfg)
            self.seed = int(meta.get("seed", self.seed))
            self.cols = [int(i) for i in meta.get("cols", self.cols)]
            self.industry_col = int(meta.get("industry_col", self.industry_col))
            self.hidden_size = int(meta.get("hidden_size", self.hidden_size))
            self.dropout = float(meta.get("dropout", self.dropout))
            self.clip = meta.get("clip", self.clip)
            self.cat_indices = [int(i) for i in meta.get("cat_indices", self.cat_indices)]
            self.cat_embed_dim = int(meta.get("cat_embed_dim", self.cat_embed_dim))
            self.cat_cardinalities = [int(c) for c in (meta.get("cat_cardinalities") or [])]
            self.loss_name = str(meta.get("loss", self.loss_name)).lower()
            self.residual_industry = bool(meta.get("residual_industry", self.residual_industry))
            self.target_mode = str(meta.get("target_mode", self.target_mode))
            self.hard_repeat = int(meta.get("hard_repeat", self.hard_repeat))
        self.net = self._build_net()
        try:
            state = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(path, map_location=self.device)
        self.net.load_state_dict(state)


CSMLPModel = CSMLPModel
