"""Dump valid/test/train labels and masks so fusion scripts skip data.z.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\dump_split_cache.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_data_path  # noqa: E402
from src.dataset import (  # noqa: E402
    drop_task2_label,
    dump_split_cache,
    load_panel,
    load_eval_splits,
    split_cache_dir,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache split labels/masks without num_x")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--industry-col", type=int, default=6)
    parser.add_argument("--dest", default=None, help="new cache directory; preserve historical artifacts")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_path = resolve_data_path(cfg, args.data)
    dest = Path(args.dest) if args.dest else split_cache_dir(ROOT)
    names = ("train", "valid", "test")
    if not args.force and all((dest / f"{name}.npz").is_file() for name in names):
        load_eval_splits(cache_dir=dest, data_path=data_path, splits=names,
                         industry_col=args.industry_col, dump_if_missing=False)
        print(f"verified cache at {dest}")
        for name in names:
            path = dest / f"{name}.npz"
            print(f"  {path.name} {path.stat().st_size / 1e6:.1f}MB")
        return

    if any((dest / f"{name}.npz").exists() for name in names):
        raise SystemExit("preserve old caches: choose a new --dest directory")
    t0 = time.perf_counter()
    data = load_panel(str(data_path))
    drop_task2_label(data)
    print(f"loaded panel in {time.perf_counter() - t0:.1f}s")
    dump_split_cache(data, dest, industry_col=args.industry_col, splits=names)
    print(f"wrote {dest} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
