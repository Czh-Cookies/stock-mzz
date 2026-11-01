"""Check the y1 test-set npy before uploading to satc.jingle.ai.

Default file is today's candidate (GRU n2000 + cols 69/73, conservative gate).
Does not overwrite submissions/task1_y1_upload.npy.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\submit_test.py
    .\\.venv\\Scripts\\python.exe scripts\\submit_test.py --pred submissions\\task1_y1_submit_today.npy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.submit import EXPECTED_TEST_SHAPE, check_submission_array  # noqa: E402

DEFAULT_PRED = ROOT / "submissions" / "task1_y1_submit_today.npy"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Jingge task-1 test submission npy")
    parser.add_argument("--pred", default=str(DEFAULT_PRED))
    args = parser.parse_args()

    path = Path(args.pred)
    if not path.is_file():
        raise SystemExit(f"missing {path}")

    # Same load the platform uses: no allow_pickle.
    pred = np.load(path, allow_pickle=False)
    arr = check_submission_array(pred, EXPECTED_TEST_SHAPE)
    print(f"ok {path.resolve()}")
    print(f"shape={arr.shape} dtype={arr.dtype}")
    print(f"finite={int(np.isfinite(arr).sum())}/{arr.size}")
    print(f"min={float(arr.min()):.6g} max={float(arr.max()):.6g}")
    print("upload this npy on 课题一 · 预测 y1")
    print("does not overwrite submissions/task1_y1_upload.npy")


if __name__ == "__main__":
    main()
