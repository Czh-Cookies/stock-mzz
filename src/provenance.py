"""Content identities for split caches. No pickle or inferred legacy trust."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


class CacheValidationError(ValueError):
    pass


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def array_sha256(value: np.ndarray) -> str:
    a = np.ascontiguousarray(value)
    if a.dtype.hasobject:
        raise CacheValidationError("object arrays cannot establish cache identity")
    h = hashlib.sha256(canonical([a.shape, a.dtype.str]).encode())
    h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def panel_contract(data: dict, industry_col: int = 6) -> dict:
    source = data.get("_source_sha256")
    if not isinstance(source, str) or len(source) != 64:
        raise CacheValidationError("source SHA256 missing; load through load_panel before caching")
    t, s, f = data["num_x"].shape
    c = data["cat_x"].shape[2]
    if not 0 <= industry_col < c:
        raise CacheValidationError("industry column out of range")
    def axis(key, size):
        values = data.get(key, list(range(size)))
        if len(values) != size:
            raise CacheValidationError(f"{key} length mismatch")
        return [str(v) for v in values]
    bounds = [int(data[k]) for k in ("train_start_idx", "valid_start_idx", "test_start_idx")] + [t]
    if not 0 <= bounds[0] < bounds[1] < bounds[2] <= t:
        raise CacheValidationError("invalid split boundaries")
    return {
        "schema": 1, "source_sha256": source, "shape": [t, s, f, c],
        "bounds": bounds,
        "num_features": axis("num_feature_ids", f),
        "cat_features": axis("cat_feature_ids", c),
        # Without named IDs, positions are bound to the immutable source hash.
        "stock_order": axis("stock_ids", s),
        "preprocessing": {"industry_col": int(industry_col), "industry_dtype": "int32",
                          "label_dtype": "float32", "mask_dtype": "bool", "version": 1},
    }


def validate_contract(contract: dict, expected: dict | None, source_sha256: str | None,
                      industry_col: int) -> None:
    required = {"schema", "source_sha256", "shape", "bounds", "num_features",
                "cat_features", "stock_order", "preprocessing"}
    if not required <= contract.keys() or contract["schema"] != 1:
        raise CacheValidationError("unsupported or incomplete cache contract; rebuild cache")
    if expected is None and source_sha256 is None:
        raise CacheValidationError("cache requires an independently supplied source/contract")
    if expected is not None and canonical(contract) != canonical(expected):
        raise CacheValidationError("cache contract mismatch: source/split/features/order/preprocessing")
    if source_sha256 is not None and contract["source_sha256"] != source_sha256:
        raise CacheValidationError("cache source mismatch")
    if contract["preprocessing"] != {"industry_col": int(industry_col), "industry_dtype": "int32",
                                     "label_dtype": "float32", "mask_dtype": "bool", "version": 1}:
        raise CacheValidationError("cache preprocessing mismatch")
    t, s, f, c = contract["shape"]
    if (len(contract["stock_order"]) != s or len(contract["num_features"]) != f
            or len(contract["cat_features"]) != c or contract["bounds"][-1] != t):
        raise CacheValidationError("cache axis schema mismatch")
