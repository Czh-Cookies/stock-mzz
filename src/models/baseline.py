"""Cross-section LightGBM baseline (stage 2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.dataset import (
    build_sample_features,
    cat_feature_col_indices,
    feature_names,
    history_width,
    market_state_width,
    resolve_cat_indices,
)


class LightGBMBaseline:
    """Fit on train rows, predict a (T, S) score panel for a split."""

    def __init__(self, params: dict[str, Any], feature_cfg: dict[str, Any] | None = None, seed: int = 42):
        self.params = dict(params)
        self.feature_cfg = dict(feature_cfg or {})
        self.seed = int(seed)
        self.model = None
        self.booster = None
        self.cat_indices: list[int] = resolve_cat_indices(self.feature_cfg)
        self.cat_feature_indices: list[int] = []
        self.feature_name_list: list[str] = []

    def fit(self, x, y, group=None) -> None:
        x = np.asarray(x)
        y = np.asarray(y)
        n_num = self._infer_n_num(x.shape[1])
        self.cat_feature_indices = cat_feature_col_indices(n_num, self.cat_indices, self.feature_cfg)
        self.feature_name_list = feature_names(n_num, self.cat_indices, self.feature_cfg)

        params = dict(self.params)
        params.setdefault("random_state", self.seed)
        params.setdefault("n_jobs", -1)
        params.setdefault("verbosity", -1)
        objective = str(params.get("objective", "regression")).lower()
        fit_kw: dict[str, Any] = {}
        if self.cat_feature_indices:
            fit_kw["categorical_feature"] = self.cat_feature_indices
        if objective in {"lambdarank", "rank_xendcg"}:
            from lightgbm import LGBMRanker

            if group is None:
                raise ValueError("LambdaRank 需要按日 group")
            self.model = LGBMRanker(**params)
            self.model.fit(x, y, group=np.asarray(group, dtype=np.int32), **fit_kw)
        else:
            from lightgbm import LGBMRegressor

            self.model = LGBMRegressor(**params)
            self.model.fit(x, y, **fit_kw)
        self.booster = self.model.booster_

    def predict_panel(
        self,
        split_data: dict[str, Any],
        fill_invalid: float = 0.0,
        num_iteration: int | None = None,
    ) -> np.ndarray:
        iters = [num_iteration]
        return self.predict_panel_iters(split_data, iters, fill_invalid=fill_invalid)[num_iteration]

    def predict_panel_iters(
        self,
        split_data: dict[str, Any],
        iterations: list[int | None],
        fill_invalid: float = 0.0,
    ) -> dict[int | None, np.ndarray]:
        if self.booster is None and self.model is None:
            raise RuntimeError("model is not fitted")
        n_days, n_stocks, _ = split_data["num_x"].shape
        outs = {
            n: np.full((n_days, n_stocks), float(fill_invalid), dtype=np.float32)
            for n in iterations
        }
        for t in range(n_days):
            if n_days > 400 and t > 0 and t % 400 == 0:
                print(f"  predict day {t}/{n_days}", flush=True)
            mask = np.asarray(split_data["mask_x"][t], dtype=bool)
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                continue
            x = build_sample_features(
                split_data["num_x"][t],
                split_data["cat_x"][t],
                mask,
                idx,
                self.cat_indices,
                self.feature_cfg,
                global_t=int(split_data.get("start", 0)) + t,
                panel_num_x=split_data.get("panel_num_x"),
                panel_mask_x=split_data.get("panel_mask_x"),
            )
            for n in iterations:
                outs[n][t, idx] = self._predict_rows(x, num_iteration=n)
        return outs

    def feature_importance(self) -> list[tuple[str, float]]:
        if self.model is None and self.booster is None:
            raise RuntimeError("model is not fitted")
        gains = np.asarray(self.model.feature_importances_ if self.model is not None else self.booster.feature_importance(), dtype=np.float64)
        names = self.feature_name_list or [f"f{i}" for i in range(gains.size)]
        pairs = list(zip(names, gains.tolist()))
        pairs.sort(key=lambda item: item[1], reverse=True)
        return pairs

    def save(self, path: str | Path) -> None:
        if self.booster is None:
            raise RuntimeError("model is not fitted")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # LightGBM's C save_model does not accept non-ASCII Windows paths.
        path.write_text(self.booster.model_to_string(), encoding="utf-8")
        meta = {
            "params": self.params,
            "feature_cfg": self.feature_cfg,
            "cat_indices": self.cat_indices,
            "cat_feature_indices": self.cat_feature_indices,
            "feature_names": self.feature_name_list,
            "seed": self.seed,
        }
        path.with_suffix(path.suffix + ".meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self, path: str | Path) -> None:
        from lightgbm import Booster

        path = Path(path)
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.params = meta.get("params", self.params)
            self.feature_cfg = meta.get("feature_cfg", self.feature_cfg)
            self.cat_indices = list(meta.get("cat_indices", resolve_cat_indices(self.feature_cfg)))
            self.cat_feature_indices = list(meta.get("cat_feature_indices") or [])
            self.feature_name_list = list(meta.get("feature_names") or [])
            self.seed = int(meta.get("seed", self.seed))
        self.booster = Booster(model_str=path.read_text(encoding="utf-8"))
        self.model = None

    def _infer_n_num(self, n_feat: int) -> int:
        n_cat = len(self.cat_indices)
        n_blocks = numeric_block_count_safe(self.feature_cfg)
        hist_w = history_width(self.feature_cfg)
        mkt_w = market_state_width(self.feature_cfg)
        n_num, rem = divmod(n_feat - n_cat - hist_w - mkt_w, max(n_blocks, 1))
        if rem != 0 or n_num <= 0:
            raise ValueError(f"cannot infer numeric width from n_feat={n_feat} n_cat={n_cat} blocks={n_blocks}")
        return n_num

    def _predict_rows(self, x: np.ndarray, num_iteration: int | None = None) -> np.ndarray:
        if self.booster is not None:
            kw = {} if num_iteration is None else {"num_iteration": int(num_iteration)}
            return np.asarray(self.booster.predict(x, **kw), dtype=np.float32)
        if self.model is not None:
            if num_iteration is not None:
                return np.asarray(self.model.predict(x, num_iteration=int(num_iteration)), dtype=np.float32)
            return np.asarray(self.model.predict(x), dtype=np.float32)
        raise RuntimeError("model is not fitted")


def numeric_block_count_safe(feature_cfg: dict[str, Any]) -> int:
    from src.dataset import numeric_block_count

    return numeric_block_count(feature_cfg)


LightGBMBaseline = LightGBMBaseline
