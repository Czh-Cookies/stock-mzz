"""Qlib-style per-stock GRU on a short window of cross-section z-scores."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.dataset import gather_windows, history_coverage_fraction, precompute_cs_cols, select_gru_train_idx


class GRUNet(nn.Module):
    def __init__(
        self,
        d_feat: int,
        hidden: int,
        layers: int,
        dropout: float,
        cat_cardinalities: list[int] | None = None,
        cat_embed_dim: int = 8,
        cs_hidden_norm: bool = False,
    ):
        super().__init__()
        self.cs_hidden_norm = bool(cs_hidden_norm)
        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        cards = [int(c) for c in (cat_cardinalities or []) if int(c) > 0]
        self.embeds = nn.ModuleList(nn.Embedding(c, cat_embed_dim) for c in cards) if cards else None
        in_fc = hidden + (len(cards) * cat_embed_dim if cards else 0)
        if self.cs_hidden_norm:
            in_fc *= 2
        self.fc = nn.Linear(in_fc, 1)

    def forward(self, x: torch.Tensor, cats: torch.Tensor | None = None) -> torch.Tensor:
        out, _ = self.rnn(x)
        h = out[:, -1, :]
        if self.embeds is not None:
            if cats is None:
                raise ValueError("GRUNet was built with category embeddings but cats is None")
            pieces = [h]
            for i, emb in enumerate(self.embeds):
                idx = cats[:, i].clamp(0, emb.num_embeddings - 1)
                pieces.append(emb(idx))
            h = torch.cat(pieces, dim=-1)
        if self.cs_hidden_norm and int(h.shape[0]) >= 8:
            mu = h.mean(dim=0, keepdim=True)
            sd = h.std(dim=0, keepdim=True).clamp_min(1e-6)
            h = torch.cat([h, (h - mu) / sd], dim=-1)
        elif self.cs_hidden_norm:
            h = torch.cat([h, torch.zeros_like(h)], dim=-1)
        return self.fc(h).squeeze(-1)


def pearson_ic_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """1 - Pearson correlation on the current-day batch. Ranking-friendly, no future days."""
    pred = pred - pred.mean()
    target = target - target.mean()
    denom = (pred.norm() * target.norm()).clamp_min(eps)
    return 1.0 - (pred * target).sum() / denom


class GRUModel:
    """One sequence per stock; last step is the prediction-day CS z-score (no future days)."""

    def __init__(self, cfg: dict[str, Any], seed: int = 42):
        self.cfg = dict(cfg)
        self.seed = int(seed)
        self.cols = [int(i) for i in (cfg.get("num_indices") or [])]
        if not self.cols:
            raise ValueError("GRU needs features.num_indices")
        self.length = int(cfg.get("length", 10))
        self.include_current = bool(cfg.get("include_current_day", True))
        self.hidden_size = int(cfg.get("hidden_size", 64))
        self.num_layers = int(cfg.get("num_layers", 1))
        self.dropout = float(cfg.get("dropout", 0.1))
        self.clip = cfg.get("clip", 5.0)
        self.max_train_stocks = cfg.get("max_train_stocks_per_day")
        self.min_train_stocks = int(cfg.get("min_train_stocks", 8))
        self.sample_mode = str(cfg.get("sample_mode") or "uniform").lower()
        self.industry_col = int(cfg.get("industry_col", 6))
        self.min_history_coverage = float(cfg.get("min_history_coverage") or 0.0)
        self.log_every = int(cfg.get("log_every_days", 400))
        self.cat_indices = [int(i) for i in (cfg.get("cat_indices") or [])]
        self.cat_embed_dim = int(cfg.get("cat_embed_dim", 8))
        self.loss_name = str(cfg.get("loss", "mse")).lower()
        # Default 0 keeps gru-no-today-recent-n2000-001 bit-identical.
        self.window_jitter = float(cfg.get("window_jitter", 0.0) or 0.0)
        self.freeze_backbone = bool(cfg.get("freeze_backbone", False))
        self.loss_weight = str(cfg.get("loss_weight") or "none").lower()
        self.drop_corr_threshold = float(cfg.get("drop_corr_threshold") or 0.0)
        self.append_mask_channel = bool(cfg.get("append_mask_channel", False))
        self.append_breadth_channel = bool(cfg.get("append_breadth_channel", False))
        self.target_space = str(cfg.get("target_space") or "raw").lower()
        self.min_day_n_x = cfg.get("min_day_n_x")
        self.rank_loss_weight = float(cfg.get("rank_loss_weight") or 0.0)
        self.cs_hidden_norm = bool(cfg.get("cs_hidden_norm", False))
        self.net: GRUNet | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cs_sel: np.ndarray | None = None
        self.cat_sel: np.ndarray | None = None
        self.cat_cardinalities: list[int] = []
        self.industry_panel: np.ndarray | None = None
        self.day_breadth: np.ndarray | None = None

    def prepare_features(self, data: dict[str, Any]) -> None:
        print(f"precompute CS z-score cols={self.cols} T={data['num_x'].shape[0]}")
        self.cs_sel = precompute_cs_cols(data["num_x"], data["mask_x"], self.cols)
        if self.clip is not None:
            np.clip(self.cs_sel, -float(self.clip), float(self.clip), out=self.cs_sel)
        print(f"cs_sel={self.cs_sel.shape} ~{self.cs_sel.nbytes / 1e9:.2f}GB device={self.device}")
        self.day_breadth = np.asarray(data["mask_x"], dtype=np.float32).mean(axis=1)
        if self.cat_indices:
            self.cat_sel = np.asarray(data["cat_x"][..., self.cat_indices], dtype=np.int32)
            maxes = self.cat_sel.max(axis=(0, 1))
            self.cat_cardinalities = [int(m) + 1 for m in np.asarray(maxes).ravel()]
            print(f"cat_indices={self.cat_indices} cardinalities={self.cat_cardinalities}")
        else:
            self.cat_sel = None
            self.cat_cardinalities = []
        if self.sample_mode in {"industry_balanced", "balanced"}:
            self.industry_panel = np.asarray(data["cat_x"][..., self.industry_col])
            print(f"industry_col={self.industry_col} for train sampling")
        else:
            self.industry_panel = None

    def _build_net(self) -> GRUNet:
        dropout = self.dropout if self.num_layers > 1 else 0.0
        extra = int(self.append_mask_channel) + int(self.append_breadth_channel)
        return GRUNet(
            len(self.cols) + extra,
            self.hidden_size,
            self.num_layers,
            dropout,
            cat_cardinalities=self.cat_cardinalities,
            cat_embed_dim=self.cat_embed_dim,
            cs_hidden_norm=self.cs_hidden_norm,
        ).to(self.device)

    def _cats_tensor(self, global_t: int, idx: np.ndarray) -> torch.Tensor | None:
        if self.cat_sel is None:
            return None
        cats = np.ascontiguousarray(self.cat_sel[int(global_t), idx])
        return torch.from_numpy(cats.astype(np.int64, copy=False)).to(self.device)

    def _forward(self, x: np.ndarray, global_t: int, idx: np.ndarray) -> torch.Tensor:
        if self.net is None:
            raise RuntimeError("GRU is not fitted")
        xb = torch.from_numpy(np.ascontiguousarray(x)).to(self.device)
        return self.net(xb, self._cats_tensor(global_t, idx))

    def _decorate_windows(
        self, x: np.ndarray, mask_x: np.ndarray, global_t: int, idx: np.ndarray
    ) -> np.ndarray:
        if not self.append_mask_channel and not self.append_breadth_channel:
            return x
        end = int(global_t) + (1 if self.include_current else 0)
        start = end - int(self.length)
        length = int(self.length)
        batch = int(idx.shape[0])
        t_len = int(mask_x.shape[0])
        sl0 = max(0, start)
        sl1 = min(t_len, end)
        extras: list[np.ndarray] = []
        if self.append_mask_channel:
            ch = np.zeros((batch, length), dtype=np.float32)
            if sl0 < sl1:
                ch[:, sl0 - start : sl0 - start + (sl1 - sl0)] = np.asarray(
                    mask_x[sl0:sl1][:, idx], dtype=np.float32
                ).T
            extras.append(ch[..., None])
        if self.append_breadth_channel:
            ch = np.zeros((batch, length), dtype=np.float32)
            if sl0 < sl1 and self.day_breadth is not None:
                b = np.asarray(self.day_breadth[sl0:sl1], dtype=np.float32)
                ch[:, sl0 - start : sl0 - start + b.size] = b[None, :]
            extras.append(ch[..., None])
        return np.concatenate([x, *extras], axis=-1)

    def _batch_target(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float32)
        if self.target_space not in {"cs_rank", "rank"}:
            return y
        from scipy.stats import rankdata

        r = rankdata(y, method="average").astype(np.float32)
        sd = float(r.std())
        if sd < 1e-6:
            return y
        return ((r - r.mean()) / sd).astype(np.float32)

    def _pairwise_rank_loss(self, pred: torch.Tensor, yb: torch.Tensor, n_pairs: int = 4096) -> torch.Tensor:
        n = int(pred.numel())
        if n < 8:
            return pred.new_zeros(())
        i = torch.randint(0, n, (n_pairs,), device=pred.device)
        j = torch.randint(0, n, (n_pairs,), device=pred.device)
        dy = yb[i] - yb[j]
        dp = pred[i] - pred[j]
        keep = dy.abs() > 1e-8
        if int(keep.sum()) < 8:
            return pred.new_zeros(())
        return nn.functional.softplus(-torch.sign(dy[keep]) * dp[keep]).mean()

    def _loss(self, pred: torch.Tensor, yb: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        if self.loss_name in {"pearson_ic", "ic"}:
            return pearson_ic_loss(pred, yb)
        if weights is None:
            mse = nn.functional.mse_loss(pred, yb)
        else:
            w = weights.clamp_min(1e-6)
            w = w / w.mean().clamp_min(1e-8)
            mse = (w * (pred - yb).pow(2)).mean()
        if self.rank_loss_weight > 0:
            return mse + float(self.rank_loss_weight) * self._pairwise_rank_loss(pred, yb)
        return mse

    def _apply_freeze(self) -> None:
        if self.net is None:
            raise RuntimeError("GRU net missing")
        for name, p in self.net.named_parameters():
            p.requires_grad = name.startswith("fc.")
        trainable = [n for n, p in self.net.named_parameters() if p.requires_grad]
        frozen = [n for n, p in self.net.named_parameters() if not p.requires_grad]
        print(f"freeze_backbone=True trainable={trainable} frozen={len(frozen)} tensors")

    def _drop_high_corr_train(self, data: dict[str, Any], train_start: int, train_end: int) -> None:
        thresh = float(self.drop_corr_threshold or 0.0)
        if thresh <= 0 or self.cs_sel is None:
            return
        t0, t1 = int(train_start), int(train_end)
        x = self.cs_sel[t0:t1]
        m = np.asarray(data["mask_x"][t0:t1], dtype=bool)
        n_col = int(x.shape[-1])
        cols_arr = []
        for i in range(n_col):
            v = np.asarray(x[..., i][m], dtype=np.float64)
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            if v.size > 250_000:
                v = v[:: max(1, v.size // 250_000)]
            cols_arr.append(v)
        n = min(a.size for a in cols_arr)
        mat = np.stack([a[:n] for a in cols_arr], axis=1)
        corr = np.corrcoef(mat, rowvar=False)
        keep: list[int] = []
        dropped: list[int] = []
        for i in range(n_col):
            if any(np.isfinite(corr[i, j]) and abs(float(corr[i, j])) >= thresh for j in keep):
                dropped.append(int(self.cols[i]))
            else:
                keep.append(i)
        if len(keep) < 8:
            print(f"drop_corr_threshold={thresh} aborted kept={len(keep)}")
            return
        if not dropped:
            print(f"drop_corr_threshold={thresh} dropped none kept={self.cols}")
            return
        self.cs_sel = np.ascontiguousarray(self.cs_sel[..., keep])
        self.cols = [self.cols[i] for i in keep]
        print(f"drop_corr_threshold={thresh} dropped={dropped} kept={self.cols}")

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
        if self.cs_sel is None:
            self.prepare_features(data)
        self._drop_high_corr_train(data, train_start, train_end)

        if self.net is None:
            self.net = self._build_net()
        if self.freeze_backbone:
            self._apply_freeze()
        lr = float(self.cfg.get("lr", 1e-3))
        decay = float(self.cfg.get("weight_decay", 1e-4))
        epochs = int(self.cfg.get("max_epochs", 8))
        patience = int(self.cfg.get("patience", 3))
        trainable = [p for p in self.net.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("no trainable GRU parameters")
        opt = torch.optim.Adam(trainable, lr=lr, weight_decay=decay)
        rng = np.random.default_rng(self.seed)
        best_ic = -1e9
        best_state: dict[str, torch.Tensor] | None = None
        stale = 0
        history: list[dict[str, Any]] = []
        n_params = sum(p.numel() for p in self.net.parameters())
        n_train = sum(p.numel() for p in trainable)
        print(
            f"gru params={n_params} trainable={n_train} hidden={self.hidden_size} L={self.length} "
            f"include_t={self.include_current} loss={self.loss_name} cats={self.cat_indices or '-'} "
            f"window_jitter={self.window_jitter} freeze_backbone={self.freeze_backbone} "
            f"sample_mode={self.sample_mode} min_history_coverage={self.min_history_coverage} "
            f"loss_weight={self.loss_weight} clip={self.clip} "
            f"mask_ch={self.append_mask_channel} breadth_ch={self.append_breadth_channel} "
            f"target_space={self.target_space} min_day_n_x={self.min_day_n_x} "
            f"cs_hidden_norm={self.cs_hidden_norm}"
        )

        days = list(range(int(train_start), int(train_end)))
        for epoch in range(epochs):
            self.net.train()
            rng.shuffle(days)
            losses: list[float] = []
            for i, t in enumerate(days):
                mask_y = np.asarray(data["mask_y"][t], dtype=bool)
                if self.min_day_n_x is not None:
                    n_listed = int(np.asarray(data["mask_x"][t], dtype=bool).sum())
                    if n_listed < int(self.min_day_n_x):
                        continue
                groups_t = None
                if self.industry_panel is not None:
                    groups_t = np.asarray(self.industry_panel[t])
                idx = select_gru_train_idx(
                    mask_y_t=mask_y,
                    mask_x=data["mask_x"],
                    t=int(t),
                    length=self.length,
                    rng=rng,
                    min_train_stocks=self.min_train_stocks,
                    max_stocks=self.max_train_stocks,
                    sample_mode=self.sample_mode,
                    groups_t=groups_t,
                    min_history_coverage=self.min_history_coverage,
                )
                if idx.size < self.min_train_stocks:
                    continue
                x = gather_windows(
                    self.cs_sel,
                    data["mask_x"],
                    t,
                    idx,
                    self.length,
                    include_current=self.include_current,
                )
                if self.window_jitter > 0:
                    drop = rng.random(x.shape[:2]) < self.window_jitter
                    drop[:, -1] = False
                    x = np.array(x, copy=True, dtype=np.float32)
                    x[drop] = 0.0
                x = self._decorate_windows(x, data["mask_x"], t, idx)
                y = self._batch_target(np.asarray(data["y1"][t][idx], dtype=np.float32))
                yb = torch.from_numpy(y).to(self.device)
                pred = self._forward(x, t, idx)
                w_t = None
                if self.loss_weight in {"history_coverage", "coverage"}:
                    cov = history_coverage_fraction(data["mask_x"], int(t), self.length, idx)
                    w_t = torch.from_numpy(np.asarray(cov, dtype=np.float32)).to(self.device)
                loss = self._loss(pred, yb, w_t)
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
            gc.collect()
            if valid is not None and stale >= patience:
                print("early stop")
                break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return {"best_valid_ic": best_ic if best_ic > -1e8 else None, "history": history}

    def _mask_panel(self, split_data: dict[str, Any], full_data: dict[str, Any] | None) -> np.ndarray:
        if full_data is not None and "mask_x" in full_data:
            return full_data["mask_x"]
        if "panel_mask_x" in split_data:
            return split_data["panel_mask_x"]
        raise RuntimeError("need full-panel mask_x for lookback across split boundaries")

    def predict_panel(
        self,
        split_data: dict[str, Any],
        full_data: dict[str, Any] | None = None,
        fill_invalid: float = 0.0,
    ) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("GRU is not fitted")
        if self.cs_sel is None:
            if full_data is None:
                raise RuntimeError("call prepare_features first")
            self.prepare_features(full_data)
        self.net.eval()
        mask_panel = self._mask_panel(split_data, full_data)
        start = int(split_data.get("start", 0))
        n_days, n_stocks = split_data["mask_x"].shape
        out = np.full((n_days, n_stocks), float(fill_invalid), dtype=np.float32)
        with torch.no_grad():
            for local_t in range(n_days):
                mask = np.asarray(split_data["mask_x"][local_t], dtype=bool)
                idx = np.flatnonzero(mask)
                if idx.size == 0:
                    continue
                global_t = start + local_t
                x = gather_windows(
                    self.cs_sel,
                    mask_panel,
                    global_t,
                    idx,
                    self.length,
                    include_current=self.include_current,
                )
                x = self._decorate_windows(x, mask_panel, global_t, idx)
                pred = self._forward(x, global_t, idx)
                out[local_t, idx] = pred.detach().cpu().numpy().astype(np.float32)
        return out

    def save(self, path: str | Path) -> None:
        if self.net is None:
            raise RuntimeError("GRU is not fitted")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path)
        meta = {
            "cfg": self.cfg,
            "seed": self.seed,
            "cols": self.cols,
            "length": self.length,
            "include_current": self.include_current,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "clip": self.clip,
            "cat_indices": self.cat_indices,
            "cat_embed_dim": self.cat_embed_dim,
            "cat_cardinalities": self.cat_cardinalities,
            "loss": self.loss_name,
            "window_jitter": self.window_jitter,
            "rank_loss_weight": self.rank_loss_weight,
            "cs_hidden_norm": self.cs_hidden_norm,
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
            self.length = int(meta.get("length", self.length))
            self.include_current = bool(meta.get("include_current", self.include_current))
            self.hidden_size = int(meta.get("hidden_size", self.hidden_size))
            self.num_layers = int(meta.get("num_layers", self.num_layers))
            self.dropout = float(meta.get("dropout", self.dropout))
            self.clip = meta.get("clip", self.clip)
            self.cat_indices = [int(i) for i in meta.get("cat_indices", self.cfg.get("cat_indices") or [])]
            self.cat_embed_dim = int(meta.get("cat_embed_dim", self.cfg.get("cat_embed_dim", 8)))
            self.cat_cardinalities = [int(c) for c in (meta.get("cat_cardinalities") or [])]
            self.loss_name = str(meta.get("loss", self.cfg.get("loss", "mse"))).lower()
            self.window_jitter = float(meta.get("window_jitter", self.cfg.get("window_jitter", 0.0)) or 0.0)
            self.cs_hidden_norm = bool(meta.get("cs_hidden_norm", self.cfg.get("cs_hidden_norm", False)))
        self.net = self._build_net()
        try:
            state = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(path, map_location=self.device)
        self.net.load_state_dict(state)


GRUModel = GRUModel
