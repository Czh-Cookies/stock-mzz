"""Blend time-axis and cross-section scores (stage 5).

Weights and gates are fit on valid only, then locked for test.
All transforms use the same day's scores / categories / coverage — no future days.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import rankdata


def coverage_gate_blend(
    temporal: np.ndarray,
    cross_section: np.ndarray,
    mask_x: np.ndarray,
    tau: float,
    weight_low: float,
    weight_high: float,
    space: str = "raw",
) -> np.ndarray:
    """Same-day coverage gate. High coverage uses weight_high on the temporal branch."""
    if space == "rank":
        temporal = panel_cs_rank(temporal, mask_x)
        cross_section = panel_cs_rank(cross_section, mask_x)
    elif space != "raw":
        raise ValueError(f"unknown coverage space {space!r}")
    n_x = np.asarray(mask_x).sum(axis=1)
    high = n_x >= float(tau)
    pred = np.empty_like(temporal, dtype=np.float32)
    pred[~high] = linear_blend(temporal[~high], cross_section[~high], weight_low)
    pred[high] = linear_blend(temporal[high], cross_section[high], weight_high)
    return pred


def linear_blend(temporal: np.ndarray, cross_section: np.ndarray, weight: float) -> np.ndarray:
    """weight 是时序支的权重，0 表示纯截面，1 表示纯时序。"""
    if temporal.shape != cross_section.shape:
        raise ValueError(f"shape mismatch: {temporal.shape} vs {cross_section.shape}")
    w = float(weight)
    return (w * temporal + (1.0 - w) * cross_section).astype(np.float32)


def stack_mlp_blend(
    mlp: np.ndarray,
    gated: np.ndarray,
    mask_x: np.ndarray,
    weight: float,
    space: str = "rank",
) -> np.ndarray:
    """Second-stage blend of industry MLP onto a gated GRU/tree panel. weight is MLP."""
    if space == "rank":
        return linear_blend(panel_cs_rank(mlp, mask_x), panel_cs_rank(gated, mask_x), weight)
    if space == "raw":
        return linear_blend(mlp, gated, weight)
    raise ValueError(f"unknown mlp space {space!r}")


def n2000_hard_gate_mlp(
    gru: np.ndarray,
    tree: np.ndarray,
    mlp: np.ndarray,
    mask_x: np.ndarray,
    tau: float,
    weight_low: float = 0.25,
    weight_high: float = 0.4,
    mlp_weight: float = 0.15,
    space: str = "raw",
    mlp_space: str = "rank",
) -> np.ndarray:
    """Frozen gru-no-today-recent-n2000-001 fusion recipe. Do not change defaults."""
    gated = coverage_gate_blend(gru, tree, mask_x, tau, weight_low, weight_high, space)
    return stack_mlp_blend(mlp, gated, mask_x, mlp_weight, mlp_space)


def panel_rank_mean(preds: list[np.ndarray], mask: np.ndarray) -> np.ndarray:
    """Equal-weight mean of same-day ranks. Aligns with RankIC; ignores raw scale."""
    if not preds:
        raise ValueError("panel_rank_mean needs at least one panel")
    ranks = [panel_cs_rank(p, mask) for p in preds]
    out = np.mean(np.stack(ranks, axis=0), axis=0)
    return np.asarray(out, dtype=np.float32)


def panel_cs_rank(pred: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Each day: rank among mask=True stocks. Invalid stays 0."""
    out = np.zeros_like(pred, dtype=np.float32)
    for t in range(pred.shape[0]):
        m = np.asarray(mask[t], dtype=bool)
        if int(m.sum()) < 2:
            continue
        out[t, m] = rankdata(pred[t, m], method="average").astype(np.float32)
    return out


def neutralize_industry(pred: np.ndarray, industry: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Subtract same-day industry mean of scores. Stock-axis residual."""
    out = np.array(pred, dtype=np.float32, copy=True)
    for t in range(pred.shape[0]):
        m = np.asarray(mask[t], dtype=bool)
        if int(m.sum()) < 2:
            continue
        scores = out[t, m].copy()
        groups = np.asarray(industry[t, m])
        for gid in np.unique(groups):
            idx = groups == gid
            if int(idx.sum()) < 2:
                continue
            scores[idx] -= float(scores[idx].mean())
        out[t, m] = scores
    return out


class FusionModel:
    """Search a small valid-only menu, then lock."""

    def __init__(self, cfg: dict[str, Any] | None = None):
        self.cfg = dict(cfg or {})
        self.locked: dict[str, Any] = {}

    def fit(
        self,
        temporal_valid: np.ndarray,
        cs_valid: np.ndarray,
        label: np.ndarray,
        mask_y: np.ndarray,
        mask_x: np.ndarray,
        industry: np.ndarray | None = None,
        scorer=None,
    ) -> dict[str, Any]:
        from src.metrics import mean_rank_ic

        score_fn = scorer or mean_rank_ic
        grid = [float(x) for x in (self.cfg.get("weight_grid") or [0.0, 0.25, 0.5, 0.75, 1.0])]
        results: list[dict[str, Any]] = []

        def consider(name: str, pred: np.ndarray, **meta: Any) -> None:
            ic = float(score_fn(pred, label, mask_y))
            row = {"name": name, "ic": ic, **meta}
            results.append(row)

        consider("cs_only", cs_valid, weight=0.0)
        consider("temporal_only", temporal_valid, weight=1.0)

        for w in grid:
            consider("raw_blend", linear_blend(temporal_valid, cs_valid, w), weight=w)

        t_rank = panel_cs_rank(temporal_valid, mask_x)
        c_rank = panel_cs_rank(cs_valid, mask_x)
        for w in grid:
            consider("rank_blend", linear_blend(t_rank, c_rank, w), weight=w, space="rank")

        if industry is not None:
            t_neu = neutralize_industry(t_rank, industry, mask_x)
            c_neu = neutralize_industry(c_rank, industry, mask_x)
            for w in grid:
                consider(
                    "rank_blend_industry",
                    linear_blend(t_neu, c_neu, w),
                    weight=w,
                    space="rank_industry",
                )

        n_x = np.asarray(mask_x).sum(axis=1)
        tau = float(np.median(n_x))
        high = n_x >= tau
        for w_low in (0.5, 0.75, 1.0):
            for w_high in (0.5, 0.75, 1.0):
                pred = np.empty_like(t_rank, dtype=np.float32)
                pred[~high] = linear_blend(t_rank[~high], c_rank[~high], w_low)
                pred[high] = linear_blend(t_rank[high], c_rank[high], w_high)
                consider(
                    "coverage_gate",
                    pred,
                    weight_low=w_low,
                    weight_high=w_high,
                    tau=tau,
                    space="rank",
                )

        results.sort(key=lambda r: r["ic"], reverse=True)
        self.locked = dict(results[0])
        self.locked["valid_leaderboard"] = results[:8]
        return self.locked

    def predict(
        self,
        temporal: np.ndarray,
        cs: np.ndarray,
        mask_x: np.ndarray,
        industry: np.ndarray | None = None,
        spec: dict[str, Any] | None = None,
    ) -> np.ndarray:
        spec = spec or self.locked
        if not spec:
            raise RuntimeError("fusion is not fitted")
        name = spec["name"]
        if name == "cs_only":
            return np.asarray(cs, dtype=np.float32)
        if name == "temporal_only":
            return np.asarray(temporal, dtype=np.float32)
        if name == "raw_blend":
            return linear_blend(temporal, cs, float(spec["weight"]))

        t_rank = panel_cs_rank(temporal, mask_x)
        c_rank = panel_cs_rank(cs, mask_x)
        if spec.get("space") == "rank_industry":
            if industry is None:
                raise ValueError("industry neutralization needs cat_x[..., 6]")
            t_rank = neutralize_industry(t_rank, industry, mask_x)
            c_rank = neutralize_industry(c_rank, industry, mask_x)
        if name in {"rank_blend", "rank_blend_industry"}:
            return linear_blend(t_rank, c_rank, float(spec["weight"]))
        if name == "coverage_gate":
            n_x = np.asarray(mask_x).sum(axis=1)
            tau = float(spec["tau"])
            high = n_x >= tau
            pred = np.empty_like(t_rank, dtype=np.float32)
            pred[~high] = linear_blend(t_rank[~high], c_rank[~high], float(spec["weight_low"]))
            pred[high] = linear_blend(t_rank[high], c_rank[high], float(spec["weight_high"]))
            return pred
        raise ValueError(f"unknown fusion {name!r}")


_EPS = 1e-8


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(z, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def coverage_soft_weights(n_x: np.ndarray, tau: float, temperature: float) -> np.ndarray:
    """Day-level high-coverage activation in (0, 1). temperature→0 approaches a step at tau."""
    scale = max(float(temperature), 1e-6)
    return _sigmoid((np.asarray(n_x, dtype=np.float64) - float(tau)) / scale)


def stock_history_coverage(
    panel_mask_x: np.ndarray,
    panel_origin: int,
    split_start: int,
    n_days: int,
    length: int,
) -> np.ndarray:
    """Past-only listing rate in [t-L, t). Causal; day t is excluded."""
    panel = np.asarray(panel_mask_x, dtype=bool)
    n_stocks = int(panel.shape[1])
    length = int(length)
    out = np.zeros((int(n_days), n_stocks), dtype=np.float32)
    origin = int(panel_origin)
    for local_t in range(int(n_days)):
        end = int(split_start) + local_t - origin
        start = max(0, end - length)
        if end <= start:
            continue
        out[local_t] = panel[start:end].mean(axis=0)
    return out


def panel_group_zscore(pred: np.ndarray, industry: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Within-day, within-industry z-score of scores. Groups with <2 valid stocks stay 0."""
    out = np.zeros_like(pred, dtype=np.float32)
    for t in range(pred.shape[0]):
        m = np.asarray(mask[t], dtype=bool) & np.isfinite(pred[t])
        if int(m.sum()) < 2:
            continue
        scores = np.asarray(pred[t], dtype=np.float64)
        groups = np.asarray(industry[t])
        for gid in np.unique(groups[m]):
            idx = m & (groups == gid)
            if int(idx.sum()) < 2:
                continue
            sl = scores[idx]
            sd = float(sl.std())
            if sd < _EPS:
                continue
            out[t, idx] = ((sl - sl.mean()) / sd).astype(np.float32)
    return out


def prepare_branch(
    pred: np.ndarray,
    mask_x: np.ndarray,
    space: str = "raw",
    industry: np.ndarray | None = None,
    group_z: bool = False,
) -> np.ndarray:
    """Same-day rank / industry z-score. No future days."""
    out = np.asarray(pred, dtype=np.float32)
    if space == "rank":
        out = panel_cs_rank(out, mask_x)
    elif space != "raw":
        raise ValueError(f"unknown space {space!r}")
    if group_z:
        if industry is None:
            raise ValueError("group_z needs industry")
        out = panel_group_zscore(out, industry, mask_x)
    return out


def coverage_stock_three_way(
    gru: np.ndarray,
    tree: np.ndarray,
    mlp: np.ndarray | None,
    mask_x: np.ndarray,
    *,
    tau: float,
    temperature: float,
    weight_low: float,
    weight_high: float,
    stock_cov: np.ndarray | None = None,
    stock_boost: float = 0.0,
    stock_center: float = 0.5,
    stock_temp: float = 0.15,
    w_mlp: float = 0.0,
    hard: bool = False,
) -> np.ndarray:
    """Blend GRU vs tree vs MLP.

    Day gate: high listing-count days get more GRU (soft logistic, or hard step if hard=True).
    Stock gate: names with denser past listings get extra GRU weight; sparse names keep the tree.
    """
    n_x = np.asarray(mask_x).sum(axis=1).astype(np.float64)
    if hard:
        alpha = (n_x >= float(tau)).astype(np.float64)
    else:
        alpha = coverage_soft_weights(n_x, tau, temperature)
    w = (float(weight_low) + alpha * (float(weight_high) - float(weight_low)))[:, None]
    if stock_cov is not None and float(stock_boost) != 0.0:
        a_s = _sigmoid((np.asarray(stock_cov, dtype=np.float64) - float(stock_center)) / max(float(stock_temp), 1e-6))
        w = w + float(stock_boost) * (2.0 * a_s - 1.0)
    w_m = float(w_mlp)
    w = np.clip(w, 0.0, max(0.0, 1.0 - w_m))
    w_t = 1.0 - w - w_m
    out = w * np.asarray(gru, dtype=np.float64) + w_t * np.asarray(tree, dtype=np.float64)
    if mlp is not None and w_m != 0.0:
        out = out + w_m * np.asarray(mlp, dtype=np.float64)
    return out.astype(np.float32)
