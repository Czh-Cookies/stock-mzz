"""Frozen conservative gate for a new GRU npy pair. No tau/weight search.

Does not overwrite task1_y1_upload.npy or champion add6973 files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_eval_splits, split_cache_dir  # noqa: E402
from src.metrics import mean_rank_ic  # noqa: E402
from src.models.fusion import n2000_hard_gate_mlp  # noqa: E402
from src.submit import save_submission  # noqa: E402

BASE = 0.11840
W_LOW, W_HIGH, TAU_Q, W_MLP = 0.25, 0.4, 0.6, 0.15


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gru-valid", required=True)
    parser.add_argument("--gru-test", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    splits, src = load_eval_splits(cache_dir=split_cache_dir(ROOT), splits=("valid", "test"))
    valid, test = splits["valid"], splits["test"]
    tau = float(np.quantile(np.asarray(valid["mask_x"]).sum(1).astype(np.float64), TAU_Q))
    gru_v = np.load(ROOT / args.gru_valid)
    gru_t = np.load(ROOT / args.gru_test)
    tree_v = np.load(ROOT / "outputs/fusion_valid.npy")
    tree_t = np.load(ROOT / "outputs/fusion_test.npy")
    mlp_v = np.load(ROOT / "outputs/cs_mlp_valid.npy")
    mlp_t = np.load(ROOT / "outputs/cs_mlp_test.npy")
    gru_ic = float(mean_rank_ic(gru_v, valid["y1"], valid["mask_y"]))
    pred_v = n2000_hard_gate_mlp(
        gru_v, tree_v, mlp_v, valid["mask_x"], tau, W_LOW, W_HIGH, W_MLP, "raw", "rank"
    )
    fused = float(mean_rank_ic(pred_v, valid["y1"], valid["mask_y"]))
    print(f"src={src} tau={tau:.1f} gru_valid={gru_ic:.6f}")
    print(f"FUSED_RANKIC {fused:.6f} vs BASE {BASE:.5f} delta={fused - BASE:+.6f}")
    keep = fused > BASE + 0.0005
    print("KEEP" if keep else "rollback")
    pred_t = n2000_hard_gate_mlp(
        gru_t, tree_t, mlp_t, test["mask_x"], tau, W_LOW, W_HIGH, W_MLP, "raw", "rank"
    )
    tag = str(args.tag)
    np.save(ROOT / f"outputs/fusion_{tag}_valid.npy", pred_v)
    np.save(ROOT / f"outputs/fusion_{tag}_test.npy", pred_t)
    out = ROOT / f"submissions/task1_fusion_{tag}.npy"
    save_submission(pred_t, out)
    print(f"wrote {out}")
    print("does not overwrite submissions/task1_y1_upload.npy")
    if not keep:
        print("do not upload; stay on previous official 0.125679 file")


if __name__ == "__main__":
    main()
