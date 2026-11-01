"""Validate and copy a test prediction into submission format.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\export_submission.py --pred outputs/baseline_test.npy --out submissions/task1.npy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.submit import EXPECTED_TEST_SHAPE, save_submission  # noqa: E402

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--out", default="submissions/task1_y1.npy")
    parser.add_argument("--expect-shape", default="442,5282")
    args = parser.parse_args()

    pred = np.load(args.pred, allow_pickle=False)
    expect = tuple(int(x) for x in args.expect_shape.split(","))
    if pred.shape != expect:
        raise SystemExit(f"shape {pred.shape} != expected {expect}")
    path = save_submission(pred, args.out)
    loaded = np.load(path, allow_pickle=False)
    print(f"wrote {path} shape={loaded.shape} dtype={loaded.dtype}")
    print("ready for RankIC page: 课题一 · 预测 y1")
    if loaded.shape != EXPECTED_TEST_SHAPE:
        print(f"current dataset expects {EXPECTED_TEST_SHAPE}; re-check test_start_idx if this differs")


if __name__ == "__main__":
    main()
