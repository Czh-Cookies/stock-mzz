"""Official RankIC (Jingge brief): Spearman on stocks S for each day, then mean over time T."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return np.nan
    ra = rankdata(a, method="average")
    rb = rankdata(b, method="average")
    if np.std(ra) == 0 or np.std(rb) == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def rank_ic_series(
    pred: np.ndarray,
    label: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """pred/label: (T, S). mask True means the point is valid."""
    if pred.shape != label.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs label {label.shape}")
    t_len, _ = pred.shape
    out = np.full(t_len, np.nan, dtype=np.float64)
    for t in range(t_len):
        valid = np.isfinite(pred[t]) & np.isfinite(label[t])
        if mask is not None:
            valid &= mask[t].astype(bool)
        out[t] = _spearman(pred[t, valid], label[t, valid])
    return out


def mean_rank_ic(
    pred: np.ndarray,
    label: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    series = rank_ic_series(pred, label, mask)
    return float(np.nanmean(series))


mean_rank_ic = mean_rank_ic
rank_ic_series = rank_ic_series
