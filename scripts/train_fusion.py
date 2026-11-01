"""Stage 5: fuse time-axis and cross-section scores on valid, lock, then test.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\train_fusion.py --config configs/fusion.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_data_path  # noqa: E402
from src.dataset import industry_panel, load_eval_splits, split_cache_dir  # noqa: E402
from src.metrics import mean_rank_ic  # noqa: E402
from src.models.fusion import FusionModel  # noqa: E402
from src.submit import save_submission  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse temporal and cross-section predictions")
    parser.add_argument("--config", default="configs/fusion.yaml")
    parser.add_argument("--data", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    inputs = cfg.get("inputs") or {}
    paths = cfg.get("paths") or {}
    fusion_cfg = cfg.get("fusion") or {}
    industry_col = int(inputs.get("industry_col", 6))

    t0 = time.perf_counter()
    data_path = None
    cache_dir = split_cache_dir(ROOT)
    if not (cache_dir / "valid.npz").is_file() or not (cache_dir / "test.npz").is_file():
        data_path = resolve_data_path(cfg, args.data)
    splits, src = load_eval_splits(
        cache_dir=cache_dir,
        data_path=data_path,
        industry_col=industry_col,
        splits=("valid", "test"),
    )
    valid = splits["valid"]
    test = splits["test"]
    print(f"loaded {src} in {time.perf_counter() - t0:.1f}s")

    temporal_valid = np.load(ROOT / inputs["temporal_valid"])
    cs_valid = np.load(ROOT / inputs["cross_section_valid"])
    temporal_test = np.load(ROOT / inputs["temporal_test"])
    cs_test = np.load(ROOT / inputs["cross_section_test"])

    model = FusionModel(fusion_cfg)
    locked = model.fit(
        temporal_valid,
        cs_valid,
        valid["y1"],
        valid["mask_y"],
        valid["mask_x"],
        industry=industry_panel(valid, industry_col),
    )
    print("valid leaderboard:")
    for row in locked.get("valid_leaderboard") or []:
        extra = {k: v for k, v in row.items() if k not in {"valid_leaderboard", "name", "ic"}}
        print(f"  {row['ic']:.6f}  {row['name']}  {extra}")
    print(f"LOCKED {locked['name']} ic={locked['ic']:.6f}")

    valid_pred = model.predict(
        temporal_valid,
        cs_valid,
        valid["mask_x"],
        industry=industry_panel(valid, industry_col),
    )
    test_pred = model.predict(
        temporal_test,
        cs_test,
        test["mask_x"],
        industry=industry_panel(test, industry_col),
    )
    confirm = mean_rank_ic(valid_pred, valid["y1"], valid["mask_y"])
    print(f"confirm valid RankIC={confirm:.6f}")

    valid_path = ROOT / paths.get("valid_pred", "outputs/fusion_valid.npy")
    test_path = ROOT / paths.get("test_pred", "outputs/fusion_test.npy")
    sub_path = ROOT / paths.get("submission", "submissions/task1_fusion.npy")
    valid_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(valid_path, valid_pred)
    np.save(test_path, test_pred)
    save_submission(test_pred, sub_path)
    meta = {k: v for k, v in locked.items() if k != "valid_leaderboard"}
    meta["leaderboard"] = locked.get("valid_leaderboard")
    lock_path = ROOT / paths.get("lock", "outputs/fusion_lock.json")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"wrote {sub_path} shape={test_pred.shape}")
    print(f"wrote {lock_path}")
    print("VALID_RANKIC", f"{confirm:.6f}")


if __name__ == "__main__":
    main()
