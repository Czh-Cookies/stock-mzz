"""Panel slices and sample construction.

Causal contract:
- Cross-section / industry z-scores use only that day's mask_x stocks.
- History stats use days in [global_t - L, global_t) only. Day t and later never enter.
- Do not shuffle rows across time. Do not put y1 of other stocks into X.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np

from .io import read_zstd, split_ranges

SplitName = Literal["train", "valid", "test", "history"]

_EPS = 1e-8


def load_panel(path: str) -> dict[str, Any]:
    return read_zstd(path)


def drop_task2_label(data: dict[str, Any]) -> None:
    """Drop y2 so it cannot leak into features; also frees a bit of RAM."""
    data.pop("y2", None)


def split_bounds(data: dict[str, Any], split: SplitName) -> tuple[int, int]:
    t_len = int(data["num_x"].shape[0])
    ranges = split_ranges(data)
    if split == "history":
        return 0, int(data["train_start_idx"])
    start, end = ranges[split]
    return start, t_len if end is None else end


def slice_split(data: dict[str, Any], split: SplitName) -> dict[str, Any]:
    start, end = split_bounds(data, split)
    out = {
        "start": start,
        "end": end,
        "num_x": data["num_x"][start:end],
        "cat_x": data["cat_x"][start:end],
        "y1": data["y1"][start:end],
        "mask_x": data["mask_x"][start:end],
        "mask_y": data["mask_y"][start:end],
    }
    if "y2" in data:
        out["y2"] = data["y2"][start:end]
    # Views of the full panel so history can look back across split boundaries.
    out["panel_num_x"] = data["num_x"]
    out["panel_mask_x"] = data["mask_x"]
    return out


def keep_recent_days(split: dict[str, Any], n_days: int) -> dict[str, Any]:
    """Keep the last n_days of a split for fitting. History still looks back via panel_*."""
    keep = int(n_days)
    n = int(split["num_x"].shape[0])
    if keep <= 0 or keep >= n:
        return split
    skip = n - keep
    out = dict(split)
    for key in ("num_x", "cat_x", "y1", "mask_x", "mask_y", "y2", "industry"):
        if key in out and out[key] is not None:
            out[key] = out[key][skip:]
    out["start"] = int(split["start"]) + skip
    return out


def split_cache_dir(root: Path | None = None) -> Path:
    base = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    return base / "outputs" / "split_cache"


def industry_panel(split: dict[str, Any], industry_col: int = 6) -> np.ndarray:
    if split.get("industry") is not None:
        return np.asarray(split["industry"])
    return np.asarray(split["cat_x"][..., int(industry_col)])


def dump_split_cache(
    data: dict[str, Any],
    dest: Path | None = None,
    *,
    industry_col: int = 6,
    splits: tuple[str, ...] = ("train", "valid", "test"),
) -> list[Path]:
    """Save labels/masks/industry only. Fusion scripts can skip the 8.5GB panel."""
    dest = split_cache_dir() if dest is None else Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in splits:
        sl = slice_split(data, name)
        path = dest / f"{name}.npz"
        np.savez(
            path,
            y1=np.array(sl["y1"], dtype=np.float32, copy=True),
            mask_x=np.array(sl["mask_x"], dtype=bool, copy=True),
            mask_y=np.array(sl["mask_y"], dtype=bool, copy=True),
            industry=np.array(sl["cat_x"][..., int(industry_col)], dtype=np.int16, copy=True),
            start=np.int32(sl["start"]),
            end=np.int32(sl["end"]),
            industry_col=np.int32(industry_col),
        )
        written.append(path)
        print(f"  wrote {path} y1={sl['y1'].shape} {path.stat().st_size / 1e6:.1f}MB")
    return written


def load_split_cache(split: SplitName, dest: Path | None = None) -> dict[str, Any]:
    dest = split_cache_dir() if dest is None else Path(dest)
    path = dest / f"{split}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as z:
        return {
            "start": int(z["start"]),
            "end": int(z["end"]),
            "y1": np.array(z["y1"]),
            "mask_x": np.array(z["mask_x"]),
            "mask_y": np.array(z["mask_y"]),
            "industry": np.array(z["industry"]),
            "industry_col": int(z["industry_col"]),
        }


def load_eval_splits(
    *,
    cache_dir: Path | None = None,
    data_path: str | Path | None = None,
    industry_col: int = 6,
    splits: tuple[str, ...] = ("valid", "test"),
    dump_if_missing: bool = True,
) -> tuple[dict[str, dict[str, Any]], str]:
    """Load valid/test labels from cache. Falls back to the full panel once and dumps."""
    import gc

    cache_dir = split_cache_dir() if cache_dir is None else Path(cache_dir)
    if all((cache_dir / f"{name}.npz").is_file() for name in splits):
        return {name: load_split_cache(name, cache_dir) for name in splits}, "cache"
    if data_path is None:
        missing = [name for name in splits if not (cache_dir / f"{name}.npz").is_file()]
        raise FileNotFoundError(
            f"split cache missing {missing} under {cache_dir}; run scripts/dump_split_cache.py"
        )
    data = load_panel(str(data_path))
    drop_task2_label(data)
    if dump_if_missing:
        dump_split_cache(data, cache_dir, industry_col=industry_col)
    del data
    gc.collect()
    return {name: load_split_cache(name, cache_dir) for name in splits}, "panel+dump"


def iter_days(split_data: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield one trading day at a time to avoid holding extra copies."""
    n_days = split_data["num_x"].shape[0]
    for t in range(n_days):
        item = {
            "t": t,
            "num_x": split_data["num_x"][t],
            "cat_x": split_data["cat_x"][t],
            "y1": split_data["y1"][t],
            "mask_x": split_data["mask_x"][t],
            "mask_y": split_data["mask_y"][t],
        }
        if "y2" in split_data:
            item["y2"] = split_data["y2"][t]
        yield item


def resolve_cat_indices(feature_cfg: dict[str, Any]) -> list[int]:
    indices = list(feature_cfg.get("cat_indices") or [])
    if feature_cfg.get("use_stock_id_cat") and 5 not in indices:
        indices.append(5)
    return sorted(set(int(i) for i in indices))


def history_spec(feature_cfg: dict[str, Any]) -> dict[str, Any] | None:
    hist = feature_cfg.get("history") or {}
    if not hist.get("enabled"):
        return None
    return hist


def history_width(feature_cfg: dict[str, Any]) -> int:
    hist = history_spec(feature_cfg)
    if hist is None:
        return 0
    n_idx = len(hist.get("num_indices") or [])
    n_stats = len(hist.get("stats") or ["mean", "std", "last"])
    short = int(hist.get("short_length") or 0)
    if short > 0:
        n_stats += len(hist.get("short_stats") or ["last"])
    return n_idx * n_stats


def market_state_spec(feature_cfg: dict[str, Any]) -> dict[str, Any] | None:
    mkt = feature_cfg.get("market_state") or {}
    if not mkt.get("enabled"):
        return None
    return mkt


def market_state_width(feature_cfg: dict[str, Any]) -> int:
    """Same-day market aggregates. Constant within a day; no future, no mask_y / y1."""
    mkt = market_state_spec(feature_cfg)
    if mkt is None:
        return 0
    n = 2 if mkt.get("include_coverage", True) else 0
    n += len(mkt.get("num_indices") or []) * len(mkt.get("stats") or ["mean", "std"])
    return n


def _day_market_vector(num_t: np.ndarray, mask_x_t: np.ndarray, mkt: dict[str, Any]) -> np.ndarray:
    mask = np.asarray(mask_x_t, dtype=bool)
    n_valid = int(mask.sum())
    n_stocks = int(mask.size)
    vec: list[float] = []
    if mkt.get("include_coverage", True):
        vec.append(n_valid / max(n_stocks, 1))
        vec.append(float(np.log1p(n_valid)))
    cols = np.asarray(mkt.get("num_indices") or [], dtype=np.int64)
    stats = list(mkt.get("stats") or ["mean", "std"])
    if cols.size:
        if n_valid >= 2:
            xm = np.asarray(num_t[mask][:, cols], dtype=np.float32)
            by_name = {"mean": xm.mean(axis=0), "std": xm.std(axis=0)}
            for stat in stats:
                block = by_name.get(stat)
                if block is None:
                    raise ValueError(f"unknown market_state stat {stat!r}; use mean/std")
                vec.extend(np.asarray(block, dtype=np.float64).ravel().tolist())
        else:
            vec.extend([0.0] * int(cols.size * len(stats)))
    return np.asarray(vec, dtype=np.float32)


def numeric_block_count(feature_cfg: dict[str, Any]) -> int:
    n = 0
    if feature_cfg.get("include_raw", True):
        n += 1
    if feature_cfg.get("cs_zscore", True):
        n += 1
    if feature_cfg.get("industry_zscore", True):
        n += 1
    return n


def feature_names(n_num: int, cat_indices: list[int], feature_cfg: dict[str, Any]) -> list[str]:
    names: list[str] = []
    if feature_cfg.get("include_raw", True):
        names.extend(f"raw_{i}" for i in range(n_num))
    if feature_cfg.get("cs_zscore", True):
        names.extend(f"cs_{i}" for i in range(n_num))
    if feature_cfg.get("industry_zscore", True):
        names.extend(f"ind_{i}" for i in range(n_num))
    names.extend(f"cat_{i}" for i in cat_indices)
    hist = history_spec(feature_cfg)
    if hist is not None:
        stats = list(hist.get("stats") or ["mean", "std", "last"])
        for stat in stats:
            for col in hist.get("num_indices") or []:
                names.append(f"hist_{stat}_{int(col)}")
        short = int(hist.get("short_length") or 0)
        if short > 0:
            for stat in hist.get("short_stats") or ["last"]:
                for col in hist.get("num_indices") or []:
                    names.append(f"hist{short}_{stat}_{int(col)}")
    mkt = market_state_spec(feature_cfg)
    if mkt is not None:
        if mkt.get("include_coverage", True):
            names.extend(["mkt_coverage", "mkt_log_n"])
        for stat in mkt.get("stats") or ["mean", "std"]:
            for col in mkt.get("num_indices") or []:
                names.append(f"mkt_{stat}_{int(col)}")
    return names


def cat_feature_col_indices(n_num: int, cat_indices: list[int], feature_cfg: dict[str, Any]) -> list[int]:
    offset = n_num * numeric_block_count(feature_cfg)
    return list(range(offset, offset + len(cat_indices)))


def _zscore_masked(num_t: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Column z-score on one day, statistics from mask=True stocks only."""
    out = np.zeros_like(num_t, dtype=np.float32)
    if int(mask.sum()) < 2:
        return out
    xm = num_t[mask]
    mu = xm.mean(axis=0)
    sd = xm.std(axis=0)
    sd = np.where(sd < _EPS, 1.0, sd)
    out[mask] = ((xm - mu) / sd).astype(np.float32, copy=False)
    return out


def precompute_cs_cols(num_x: np.ndarray, mask_x: np.ndarray, cols: list[int]) -> np.ndarray:
    """(T, S, F) daily cross-section z-score of selected columns. Causal per day."""
    cols_i = np.asarray(cols, dtype=np.int64)
    t_len, n_stocks, _ = num_x.shape
    out = np.zeros((t_len, n_stocks, cols_i.size), dtype=np.float32)
    for t in range(t_len):
        out[t] = _zscore_masked(np.asarray(num_x[t][:, cols_i], dtype=np.float32), mask_x[t])
    return out


def gather_windows(
    cs_sel: np.ndarray,
    mask_x: np.ndarray,
    global_t: int,
    stock_idx: np.ndarray,
    length: int,
    include_current: bool = True,
    **kwargs: Any,
) -> np.ndarray:
    """Right-aligned windows ending at t (or t-1). Shape (B, L, F). Missing days are 0."""
    if "include_current" in kwargs:
        include_current = bool(kwargs.pop("include_current"))
    if kwargs:
        raise TypeError(f"unexpected gather_windows kwargs {sorted(kwargs)}")
    end = int(global_t) + (1 if include_current else 0)
    start = end - int(length)
    stock_idx = np.asarray(stock_idx, dtype=np.intp)
    length = int(length)
    n_feat = int(cs_sel.shape[-1])
    t_len = int(cs_sel.shape[0])
    out = np.zeros((stock_idx.size, length, n_feat), dtype=np.float32)
    sl0 = max(0, start)
    sl1 = min(t_len, end)
    if sl0 >= sl1:
        return out
    out0 = sl0 - start
    out1 = out0 + (sl1 - sl0)
    chunk = np.ascontiguousarray(cs_sel[sl0:sl1][:, stock_idx])
    alive = np.asarray(mask_x[sl0:sl1], dtype=bool)[:, stock_idx]
    chunk = np.transpose(chunk, (1, 0, 2))
    chunk[~alive.T] = 0.0
    out[:, out0:out1, :] = chunk
    return out


def _industry_zscore(num_t: np.ndarray, mask: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Within-industry z-score on one day. Groups with <2 valid stocks stay 0."""
    out = np.zeros_like(num_t, dtype=np.float32)
    valid = np.flatnonzero(mask)
    if valid.size < 2:
        return out
    g = groups[valid]
    for gid in np.unique(g):
        idx = valid[g == gid]
        if idx.size < 2:
            continue
        xm = num_t[idx]
        mu = xm.mean(axis=0)
        sd = xm.std(axis=0)
        sd = np.where(sd < _EPS, 1.0, sd)
        out[idx] = ((xm - mu) / sd).astype(np.float32, copy=False)
    return out


def precompute_ind_cols(
    num_x: np.ndarray,
    mask_x: np.ndarray,
    cat_x: np.ndarray,
    cols: list[int],
    industry_col: int = 6,
) -> np.ndarray:
    """(T, S, F) within-industry z-score of selected columns. Causal per day."""
    cols_i = np.asarray(cols, dtype=np.int64)
    t_len, n_stocks, _ = num_x.shape
    out = np.zeros((t_len, n_stocks, cols_i.size), dtype=np.float32)
    for t in range(t_len):
        out[t] = _industry_zscore(
            np.asarray(num_x[t][:, cols_i], dtype=np.float32),
            mask_x[t],
            cat_x[t][:, int(industry_col)],
        )
    return out


def neutralize_groups_1d(
    values: np.ndarray,
    groups: np.ndarray,
    mask: np.ndarray,
    min_size: int = 2,
) -> np.ndarray:
    """Subtract same-day group mean. Used for industry-residual labels, not features."""
    out = np.asarray(values, dtype=np.float32).copy()
    valid = np.flatnonzero(np.asarray(mask, dtype=bool) & np.isfinite(out))
    if valid.size == 0:
        return out
    g = np.asarray(groups)[valid]
    for gid in np.unique(g):
        idx = valid[g == gid]
        if idx.size < int(min_size):
            continue
        out[idx] -= float(out[idx].mean())
    return out


def ols_residual_1d(y: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Within-day residual of y after OLS on pred. Ranking error left by a frozen model."""
    out = np.asarray(y, dtype=np.float32).copy()
    m = np.asarray(mask, dtype=bool) & np.isfinite(out) & np.isfinite(pred)
    idx = np.flatnonzero(m)
    if idx.size < 3:
        return out
    x = np.asarray(pred, dtype=np.float64)[idx]
    z = np.asarray(y, dtype=np.float64)[idx]
    xc = x - x.mean()
    zc = z - z.mean()
    var = float(np.dot(xc, xc))
    if var < 1e-12:
        out[idx] = zc.astype(np.float32)
        return out
    b = float(np.dot(xc, zc) / var)
    out[idx] = (zc - b * xc).astype(np.float32)
    return out


def build_day_features(
    num_t: np.ndarray,
    cat_t: np.ndarray,
    mask_x_t: np.ndarray,
    stock_idx: np.ndarray,
    cat_indices: list[int],
    feature_cfg: dict[str, Any],
) -> np.ndarray:
    """Build rows for selected stocks. Z-score stats always use the full day's mask_x."""
    mask = np.asarray(mask_x_t, dtype=bool)
    blocks: list[np.ndarray] = []
    if feature_cfg.get("include_raw", True):
        blocks.append(np.asarray(num_t[stock_idx], dtype=np.float32))
    if feature_cfg.get("cs_zscore", True):
        cs = _zscore_masked(num_t, mask)
        blocks.append(cs[stock_idx])
    if feature_cfg.get("industry_zscore", True):
        industry_col = int(feature_cfg.get("industry_col", 6))
        ind = _industry_zscore(num_t, mask, cat_t[:, industry_col])
        blocks.append(ind[stock_idx])
    if cat_indices:
        cats = np.asarray(cat_t[stock_idx][:, cat_indices], dtype=np.float32)
        blocks.append(cats)
    if not blocks:
        raise ValueError("no feature blocks enabled")
    return np.concatenate(blocks, axis=1)


def build_hist_features(
    panel_num_x: np.ndarray,
    panel_mask_x: np.ndarray,
    global_t: int,
    stock_idx: np.ndarray,
    feature_cfg: dict[str, Any],
) -> np.ndarray:
    """Past-only stats on selected numeric columns. Windows are [t-L, t)."""
    hist = history_spec(feature_cfg)
    n_sel = int(np.asarray(stock_idx).shape[0])
    width = history_width(feature_cfg)
    if hist is None or n_sel == 0 or width == 0:
        return np.zeros((n_sel, width), dtype=np.float32)
    cols = np.asarray(hist.get("num_indices") or [], dtype=np.int64)
    source = str(hist.get("source", "cs_zscore"))
    halflife = float(hist.get("ewm_halflife", 3))
    stats = list(hist.get("stats") or ["mean", "std", "last"])
    parts = [
        _hist_window_stats(
            panel_num_x,
            panel_mask_x,
            global_t,
            int(hist.get("length", 10)),
            stock_idx,
            cols,
            source,
            stats,
            halflife,
        )
    ]
    short = int(hist.get("short_length") or 0)
    if short > 0:
        short_stats = list(hist.get("short_stats") or ["last"])
        parts.append(
            _hist_window_stats(
                panel_num_x,
                panel_mask_x,
                global_t,
                short,
                stock_idx,
                cols,
                source,
                short_stats,
                halflife,
            )
        )
    return np.concatenate(parts, axis=1)


def build_market_state_features(
    num_t: np.ndarray,
    mask_x_t: np.ndarray,
    stock_idx: np.ndarray,
    feature_cfg: dict[str, Any],
    *,
    global_t: int | None = None,
    panel_num_x: np.ndarray | None = None,
    panel_mask_x: np.ndarray | None = None,
) -> np.ndarray:
    """Broadcast same-day market stats to each selected stock. Uses mask_x only."""
    n_sel = int(np.asarray(stock_idx).shape[0])
    width = market_state_width(feature_cfg)
    if width == 0 or n_sel == 0:
        return np.zeros((n_sel, width), dtype=np.float32)
    mkt = market_state_spec(feature_cfg)
    assert mkt is not None
    today = _day_market_vector(num_t, mask_x_t, mkt)
    if today.size != width:
        raise ValueError(f"market_state width {width} vs built {today.size}")
    rel_l = int(mkt.get("relative_length") or 0)
    if rel_l > 0:
        if panel_num_x is None or panel_mask_x is None or global_t is None:
            raise ValueError("relative market_state needs panel_num_x, panel_mask_x, global_t")
        start = max(0, int(global_t) - rel_l)
        end = int(global_t)
        if end > start:
            past = np.stack(
                [_day_market_vector(panel_num_x[i], panel_mask_x[i], mkt) for i in range(start, end)],
                axis=0,
            )
            mu = past.mean(axis=0)
            sd = past.std(axis=0)
            sd = np.where(sd < _EPS, 1.0, sd)
            today = ((today - mu) / sd).astype(np.float32, copy=False)
        else:
            today = np.zeros((width,), dtype=np.float32)
    return np.broadcast_to(today, (n_sel, width)).copy()


def _hist_window_stats(
    panel_num_x: np.ndarray,
    panel_mask_x: np.ndarray,
    global_t: int,
    length: int,
    stock_idx: np.ndarray,
    cols: np.ndarray,
    source: str,
    stats: list[str],
    ewm_halflife: float,
) -> np.ndarray:
    n_sel = int(np.asarray(stock_idx).shape[0])
    width = int(cols.size * len(stats))
    start = max(0, int(global_t) - int(length))
    end = int(global_t)
    if end <= start or width == 0:
        return np.zeros((n_sel, width), dtype=np.float32)
    win = _history_window(panel_num_x, panel_mask_x, start, end, stock_idx, cols, source)
    wmask = np.asarray(panel_mask_x[start:end][:, stock_idx], dtype=bool)
    win_m = np.where(wmask[:, :, None], win, np.nan)
    count = wmask.sum(axis=0).astype(np.float32)
    last = None
    with np.errstate(all="ignore"):
        mu = np.nansum(win_m, axis=0) / np.maximum(count[:, None], 1.0)
        mu = np.where(count[:, None] > 0, mu, 0.0).astype(np.float32)
        var = np.nansum((win_m - mu[None, :, :]) ** 2, axis=0) / np.maximum(count[:, None], 1.0)
        sd = np.sqrt(np.maximum(var, 0.0)).astype(np.float32)
        sd = np.where(count[:, None] >= 2, sd, 0.0)
        if "last" in stats or "delta" in stats:
            last_idx = np.where(wmask, np.arange(wmask.shape[0], dtype=np.int32)[:, None], -1).max(axis=0)
            last = np.zeros((n_sel, cols.size), dtype=np.float32)
            has = last_idx >= 0
            sel = np.flatnonzero(has)
            if sel.size:
                last[sel] = win[last_idx[sel], sel, :]
        ewm = None
        if "ewm" in stats:
            age = np.arange(win.shape[0] - 1, -1, -1, dtype=np.float32)
            decay = np.power(0.5, age / max(float(ewm_halflife), 1e-3)).astype(np.float32)
            w = decay[:, None, None]
            valid = wmask[:, :, None]
            num = np.where(valid, win * w, 0.0).sum(axis=0)
            den = np.where(valid, w, 0.0).sum(axis=0)
            ewm = np.where(den > 0, num / np.maximum(den, 1e-8), 0.0).astype(np.float32)
        by_name = {
            "mean": mu,
            "std": sd,
            "last": last,
            "delta": None if last is None else (last - mu).astype(np.float32),
            "ewm": ewm,
        }
        blocks: list[np.ndarray] = []
        for stat in stats:
            block = by_name.get(stat)
            if block is None:
                raise ValueError(f"unknown history stat {stat!r}; use mean/std/last/delta/ewm")
            blocks.append(block)
    return np.concatenate(blocks, axis=1)


def _history_window(
    panel_num_x: np.ndarray,
    panel_mask_x: np.ndarray,
    start: int,
    end: int,
    stock_idx: np.ndarray,
    cols: np.ndarray,
    source: str,
) -> np.ndarray:
    """(L, n_sel, n_cols) history tensor. cs_zscore uses each past day's full-market ranks."""
    if source == "raw":
        return np.asarray(panel_num_x[start:end][:, stock_idx][:, :, cols], dtype=np.float32)
    if source != "cs_zscore":
        raise ValueError(f"history.source must be raw or cs_zscore, got {source!r}")
    length = end - start
    n_sel = int(np.asarray(stock_idx).shape[0])
    win = np.empty((length, n_sel, cols.size), dtype=np.float32)
    for i in range(length):
        z = _zscore_masked(
            np.asarray(panel_num_x[start + i][:, cols], dtype=np.float32),
            np.asarray(panel_mask_x[start + i], dtype=bool),
        )
        win[i] = z[stock_idx]
    return win


def build_sample_features(
    num_t: np.ndarray,
    cat_t: np.ndarray,
    mask_x_t: np.ndarray,
    stock_idx: np.ndarray,
    cat_indices: list[int],
    feature_cfg: dict[str, Any],
    *,
    global_t: int,
    panel_num_x: np.ndarray | None = None,
    panel_mask_x: np.ndarray | None = None,
) -> np.ndarray:
    day = build_day_features(num_t, cat_t, mask_x_t, stock_idx, cat_indices, feature_cfg)
    blocks: list[np.ndarray] = [day]
    if history_spec(feature_cfg) is not None:
        if panel_num_x is None or panel_mask_x is None:
            raise ValueError("history features need panel_num_x and panel_mask_x")
        blocks.append(build_hist_features(panel_num_x, panel_mask_x, global_t, stock_idx, feature_cfg))
    mkt = build_market_state_features(
        num_t,
        mask_x_t,
        stock_idx,
        feature_cfg,
        global_t=global_t,
        panel_num_x=panel_num_x,
        panel_mask_x=panel_mask_x,
    )
    if mkt.shape[1] > 0:
        blocks.append(mkt)
    if len(blocks) == 1:
        return day
    return np.concatenate(blocks, axis=1)


def _stocks_for_day(
    mask_x: np.ndarray,
    mask_y: np.ndarray,
    require_label: bool,
    max_stocks: int | None,
    rng: np.random.Generator,
    groups: np.ndarray | None = None,
) -> np.ndarray:
    if require_label:
        idx = np.flatnonzero(np.asarray(mask_y, dtype=bool))
    else:
        idx = np.flatnonzero(np.asarray(mask_x, dtype=bool))
    if max_stocks is not None and idx.size > int(max_stocks):
        if groups is not None:
            idx = _stratified_choice(idx, np.asarray(groups)[idx], int(max_stocks), rng)
        else:
            idx = np.sort(rng.choice(idx, size=int(max_stocks), replace=False))
    return idx


def _stratified_choice(
    idx: np.ndarray,
    groups: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample k stocks, keeping industry mix so full-market days are not one-group dominated."""
    idx = np.asarray(idx)
    groups = np.asarray(groups)
    if idx.size <= k:
        return np.sort(idx)
    uniq, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    n_g = int(uniq.size)
    if n_g >= k:
        pick = rng.choice(n_g, size=k, replace=False)
        out = np.empty(k, dtype=idx.dtype)
        for i, gi in enumerate(pick):
            members = idx[inv == gi]
            out[i] = members[int(rng.integers(0, members.size))]
        return np.sort(out)
    alloc = np.floor(k * counts / counts.sum()).astype(np.int64)
    alloc = np.maximum(alloc, 1)
    extra = int(alloc.sum() - k)
    order = np.argsort(-alloc)
    i = 0
    while extra > 0 and i < n_g * 50:
        gi = int(order[i % n_g])
        if alloc[gi] > 1:
            alloc[gi] -= 1
            extra -= 1
        i += 1
    remain = counts - alloc
    extra = int(k - alloc.sum())
    for gi in np.argsort(-remain):
        if extra <= 0:
            break
        take = min(extra, int(max(remain[gi], 0)))
        alloc[gi] += take
        extra -= take
    parts = []
    for gi, n_take in enumerate(alloc):
        members = idx[inv == gi]
        n_take = min(int(n_take), int(members.size))
        if n_take <= 0:
            continue
        if n_take == members.size:
            parts.append(members)
        else:
            parts.append(rng.choice(members, n_take, replace=False))
    return np.sort(np.concatenate(parts)) if parts else np.sort(idx[:k])


def history_coverage_fraction(
    mask_x: np.ndarray,
    t: int,
    length: int,
    idx: np.ndarray,
) -> np.ndarray:
    """Share of days in [t-length, t) with mask_x. Causal; day t is not used."""
    start = max(0, int(t) - int(length))
    end = int(t)
    idx = np.asarray(idx)
    if end <= start or idx.size == 0:
        return np.ones(idx.shape[0], dtype=np.float64)
    win = np.asarray(mask_x[start:end], dtype=bool)[:, idx]
    return win.mean(axis=0)


def select_gru_train_idx(
    *,
    mask_y_t: np.ndarray,
    mask_x: np.ndarray,
    t: int,
    length: int,
    rng: np.random.Generator,
    min_train_stocks: int = 8,
    max_stocks: int | None = None,
    sample_mode: str = "uniform",
    groups_t: np.ndarray | None = None,
    min_history_coverage: float = 0.0,
) -> np.ndarray:
    """Train-day stock index. Defaults match the old uniform cap (n2000-safe)."""
    idx = np.flatnonzero(np.asarray(mask_y_t, dtype=bool))
    if idx.size < int(min_train_stocks):
        return idx
    cov_cut = float(min_history_coverage or 0.0)
    if cov_cut > 0:
        cov = history_coverage_fraction(mask_x, t, length, idx)
        kept = idx[cov >= cov_cut]
        if kept.size >= int(min_train_stocks):
            idx = kept
    mode = str(sample_mode or "uniform").lower()
    cap = int(max_stocks) if max_stocks else None
    if cap is not None and idx.size > cap:
        if mode in {"industry_balanced", "balanced"} and groups_t is not None:
            idx = _stratified_choice(idx, np.asarray(groups_t)[idx], cap, rng)
        else:
            idx = np.sort(rng.choice(idx, size=cap, replace=False))
    return np.sort(np.asarray(idx))


def group_sizes(coords: np.ndarray) -> np.ndarray:
    """Consecutive per-day row counts. Rows must stay in time order."""
    t = np.asarray(coords[:, 0])
    if t.size == 0:
        return np.zeros((0,), dtype=np.int32)
    breaks = np.flatnonzero(t[1:] != t[:-1]) + 1
    idx = np.concatenate(([0], breaks, [int(t.size)]))
    return np.diff(idx).astype(np.int32)


def within_day_relevance(y: np.ndarray, coords: np.ndarray, n_grades: int = 5) -> np.ndarray:
    """Bucket y1 into 0..n_grades-1 within each day for LambdaRank.

    LightGBM lambdarank only accepts small integer labels (default max 31).
    Using 800 unique ranks therefore fails; 5 grades also regularizes.
    """
    from scipy.stats import rankdata

    y = np.asarray(y)
    t = np.asarray(coords[:, 0])
    out = np.empty(y.shape[0], dtype=np.float32)
    grades = int(n_grades)
    start = 0
    n_all = int(y.size)
    while start < n_all:
        end = start + 1
        while end < n_all and t[end] == t[start]:
            end += 1
        n = end - start
        ranks = rankdata(y[start:end], method="ordinal")
        out[start:end] = np.minimum((ranks - 1) * grades // n, grades - 1)
        start = end
    return out


def flatten_masked_rows(
    split_data: dict[str, Any],
    cat_indices: list[int] | None = None,
    require_label: bool = True,
    feature_cfg: dict[str, Any] | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten a split into LightGBM-style rows.

    Training should pass require_label=True so only mask_y rows are used.
    Cross-section stats still use that day's full mask_x universe.

    Returns
    -------
    x : (N, F) float32
    y : (N,) float32
    coords : (N, 2) int32 with columns [local_t, stock]
    """
    feature_cfg = dict(feature_cfg or {})
    if cat_indices is None:
        cat_indices = resolve_cat_indices(feature_cfg)
    else:
        cat_indices = list(cat_indices)
        feature_cfg.setdefault("cat_indices", cat_indices)

    n_days, n_stocks, n_num = split_data["num_x"].shape
    max_stocks = feature_cfg.get("max_train_stocks_per_day") if require_label else None
    rng = np.random.default_rng(seed)
    stratify = bool(feature_cfg.get("stratify_industry", False))
    industry_col = int(feature_cfg.get("industry_col", 6))

    day_indices: list[np.ndarray] = []
    n_rows = 0
    for t in range(n_days):
        groups = split_data["cat_x"][t][:, industry_col] if stratify else None
        idx = _stocks_for_day(
            split_data["mask_x"][t],
            split_data["mask_y"][t],
            require_label,
            max_stocks,
            rng,
            groups=groups,
        )
        day_indices.append(idx)
        n_rows += int(idx.size)

    n_feat = (
        n_num * numeric_block_count(feature_cfg)
        + len(cat_indices)
        + history_width(feature_cfg)
        + market_state_width(feature_cfg)
    )
    x = np.empty((n_rows, n_feat), dtype=np.float32)
    y = np.empty((n_rows,), dtype=np.float32)
    coords = np.empty((n_rows, 2), dtype=np.int32)

    cursor = 0
    for t, idx in enumerate(day_indices):
        if idx.size == 0:
            continue
        sl = slice(cursor, cursor + idx.size)
        x[sl] = build_sample_features(
            split_data["num_x"][t],
            split_data["cat_x"][t],
            split_data["mask_x"][t],
            idx,
            cat_indices,
            feature_cfg,
            global_t=int(split_data.get("start", 0)) + t,
            panel_num_x=split_data.get("panel_num_x"),
            panel_mask_x=split_data.get("panel_mask_x"),
        )
        y[sl] = np.asarray(split_data["y1"][t][idx], dtype=np.float32)
        coords[sl, 0] = t
        coords[sl, 1] = idx
        cursor += idx.size

    return x, y, coords


drop_task2_label = drop_task2_label
dump_split_cache = dump_split_cache
flatten_masked_rows = flatten_masked_rows
within_day_relevance = within_day_relevance
