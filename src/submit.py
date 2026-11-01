from __future__ import annotations

from pathlib import Path

import numpy as np

# Official test panel for this data drop. Tt = T - test_start_idx.
EXPECTED_TEST_SHAPE = (442, 5282)


def sanitize_submission(pred: np.ndarray, shape: tuple[int, int] = EXPECTED_TEST_SHAPE) -> np.ndarray:
    """float32 (Tt, S); non-finite -> 0. Matches the RankIC upload page."""
    arr = np.asarray(pred, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"submission must be 2-D (Tt, S), got {arr.shape}")
    if arr.shape != shape:
        raise ValueError(f"submission shape {arr.shape} != expected {shape}")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(arr, dtype=np.float32)


def check_submission_array(pred: np.ndarray, shape: tuple[int, int] = EXPECTED_TEST_SHAPE) -> np.ndarray:
    """Reject files the platform will bounce: rank, dtype, shape, NaN/Inf, pickle."""
    arr = np.asarray(pred)
    if type(arr) is not np.ndarray and not isinstance(arr, np.ndarray):
        raise ValueError(f"submission must be np.ndarray, got {type(pred)!r}")
    if arr.ndim != 2:
        raise ValueError(f"submission must be 2-D (Tt, S), got {arr.shape}")
    if arr.shape != shape:
        raise ValueError(f"submission shape {arr.shape} != expected {shape}")
    if arr.dtype != np.float32:
        raise ValueError(f"submission dtype {arr.dtype} != float32")
    bad = int((~np.isfinite(arr)).sum())
    if bad:
        raise ValueError(f"submission has {bad} NaN/Inf values; platform rejects these")
    return arr


def save_submission(pred: np.ndarray, path: str | Path) -> Path:
    """Save test predictions so `np.load(path)` returns the array directly."""
    arr = sanitize_submission(pred)
    check_submission_array(arr)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    loaded = np.load(path, allow_pickle=False)
    if loaded.shape != arr.shape or loaded.dtype != np.float32:
        raise RuntimeError("saved file is not reloadable as float32 (Tt, S)")
    if not np.isfinite(loaded).all():
        raise RuntimeError("saved file still contains NaN/Inf")
    return path


save_submission = save_submission
sanitize_submission = sanitize_submission
check_submission_array = check_submission_array
EXPECTED_TEST_SHAPE = EXPECTED_TEST_SHAPE
