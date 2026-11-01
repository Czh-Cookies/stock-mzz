"""Validate a Jingge task-1 npy before upload.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\check_submission.py --pred submissions/task1_fusion_n2000_softgate.npy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.submit import EXPECTED_TEST_SHAPE, check_submission_array, save_submission  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Jingge submission npy")
    parser.add_argument("--pred", required=True)
    parser.add_argument("--shape", default=None, help="Tt,S  e.g. 442,5282")
    parser.add_argument(
        "--fix",
        default=None,
        help="if set, write a sanitized float32 npy (NaN/Inf -> 0) to this path",
    )
    args = parser.parse_args()

    path = Path(args.pred)
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    pred = np.load(path, allow_pickle=False)
    shape = EXPECTED_TEST_SHAPE
    if args.shape:
        a, b = args.shape.split(",")
        shape = (int(a), int(b))
    if args.fix:
        out = save_submission(pred, args.fix)
        pred = np.load(out, allow_pickle=False)
        print(f"sanitized -> {out}")
    arr = check_submission_array(pred, shape)
    print(
        f"ok path={path} shape={arr.shape} dtype={arr.dtype} "
        f"finite={int(np.isfinite(arr).sum())}/{arr.size} "
        f"min={float(np.nanmin(arr)):.6g} max={float(np.nanmax(arr)):.6g}"
    )


if __name__ == "__main__":
    main()
