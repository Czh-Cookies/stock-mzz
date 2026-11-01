"""Locked hist vs baseline 0.7, then conservative gate. No weight search."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_eval_splits, split_cache_dir  # noqa: E402
from src.metrics import mean_rank_ic  # noqa: E402
from src.models.fusion import linear_blend, n2000_hard_gate_mlp  # noqa: E402
from src.submit import save_submission  # noqa: E402

TREE_W = 0.7
BASE = 0.11840


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hist-valid", required=True)
    parser.add_argument("--hist-test", required=True)
    parser.add_argument("--gru-valid", default="outputs/gru_n2000_add6973_e1_valid.npy")
    parser.add_argument("--gru-test", default="outputs/gru_n2000_add6973_e1_test.npy")
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    cache = split_cache_dir(ROOT)
    splits, _ = load_eval_splits(cache_dir=cache, splits=("valid", "test"))
    valid, test = splits["valid"], splits["test"]
    tau = float(np.quantile(np.asarray(valid["mask_x"]).sum(1).astype(np.float64), 0.6))
    hist_v = np.load(ROOT / args.hist_valid)
    hist_t = np.load(ROOT / args.hist_test)
    base_v = np.load(ROOT / "outputs/baseline_valid.npy")
    base_t = np.load(ROOT / "outputs/baseline_test.npy")
    tree_v = linear_blend(hist_v, base_v, TREE_W)
    tree_t = linear_blend(hist_t, base_t, TREE_W)
    hist_ic = float(mean_rank_ic(hist_v, valid["y1"], valid["mask_y"]))
    tree_ic = float(mean_rank_ic(tree_v, valid["y1"], valid["mask_y"]))
    print(f"hist valid {hist_ic:.6f}")
    print(f"tree raw_blend0.7 valid {tree_ic:.6f} (old tree 0.106185)")

    gru_v = np.load(ROOT / args.gru_valid)
    gru_t = np.load(ROOT / args.gru_test)
    mlp_v = np.load(ROOT / "outputs/cs_mlp_valid.npy")
    mlp_t = np.load(ROOT / "outputs/cs_mlp_test.npy")
    pred_v = n2000_hard_gate_mlp(
        gru_v, tree_v, mlp_v, valid["mask_x"], tau, 0.25, 0.4, 0.15, "raw", "rank"
    )
    fused = float(mean_rank_ic(pred_v, valid["y1"], valid["mask_y"]))
    print(f"FUSED_RANKIC {fused:.6f} vs BASE {BASE:.5f} delta={fused - BASE:+.6f}")
    pred_t = n2000_hard_gate_mlp(
        gru_t, tree_t, mlp_t, test["mask_x"], tau, 0.25, 0.4, 0.15, "raw", "rank"
    )
    tag = str(args.tag)
    np.save(ROOT / f"outputs/fusion_{tag}_valid.npy", pred_v)
    np.save(ROOT / f"outputs/fusion_{tag}_test.npy", pred_t)
    save_submission(pred_t, ROOT / f"submissions/task1_fusion_{tag}.npy")
    print(f"wrote submissions/task1_fusion_{tag}.npy")
    print("does not overwrite submissions/task1_y1_upload.npy")


if __name__ == "__main__":
    main()
