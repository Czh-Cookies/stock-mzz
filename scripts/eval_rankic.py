"""Compute mean RankIC of a prediction panel against y1.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\eval_rankic.py --pred outputs/baseline_valid.npy --split valid
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_data_path  # noqa: E402
from src.dataset import load_eval_splits, load_split_cache, split_cache_dir  # noqa: E402
from src.metrics import mean_rank_ic, rank_ic_series  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="npy of shape (T_split, S)")
    parser.add_argument("--split", choices=["train", "valid"], default="valid")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--dump-series", default=None, help="optional csv path for per-day RankIC")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cache = split_cache_dir(ROOT)
    try:
        split = load_split_cache(args.split, cache, data_path=resolve_data_path(cfg, args.data))
        print(f"labels from cache {cache / (args.split + '.npz')}")
    except FileNotFoundError:
        data_path = resolve_data_path(cfg, args.data)
        splits, src = load_eval_splits(
            cache_dir=cache,
            data_path=data_path,
            splits=(args.split,),
        )
        split = splits[args.split]
        print(f"labels from {src}")
    pred = np.load(args.pred)
    label = split["y1"]
    mask = split["mask_y"]
    if pred.shape != label.shape:
        raise SystemExit(f"pred {pred.shape} vs label {label.shape}")

    series = rank_ic_series(pred, label, mask)
    score = mean_rank_ic(pred, label, mask)
    print(f"split={args.split} shape={pred.shape} mean RankIC={score:.6f}")
    print(f"daily RankIC: min={np.nanmin(series):.6f} median={np.nanmedian(series):.6f} max={np.nanmax(series):.6f}")
    if args.dump_series:
        out = Path(args.dump_series)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(out, series, delimiter=",")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
