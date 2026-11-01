"""Conservative coverage gate only. No weight search.

Usage:
    python scripts/eval_conservative_gate.py --gru-valid ... --out-prefix freeze_gate_head_e2
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

W_LOW = 0.25
W_HIGH = 0.4
TAU_Q = 0.6
W_MLP = 0.15


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gru-valid", required=True)
    parser.add_argument("--gru-test", required=True)
    parser.add_argument("--tree-valid", default="outputs/fusion_valid.npy")
    parser.add_argument("--tree-test", default="outputs/fusion_test.npy")
    parser.add_argument("--mlp-valid", default="outputs/cs_mlp_valid.npy")
    parser.add_argument("--mlp-test", default="outputs/cs_mlp_test.npy")
    parser.add_argument("--tag", default="freeze_gate_head_e2")
    args = parser.parse_args()

    cache = split_cache_dir(ROOT)
    splits, src = load_eval_splits(cache_dir=cache, splits=("valid", "test"))
    valid, test = splits["valid"], splits["test"]
    n_x = np.asarray(valid["mask_x"]).sum(axis=1).astype(np.float64)
    tau = float(np.quantile(n_x, TAU_Q))
    print(
        f"conservative gate w_low={W_LOW} w_high={W_HIGH} tau_q={TAU_Q} tau={tau:.1f} "
        f"mlp={W_MLP} src={src}"
    )

    gru_v = np.load(ROOT / args.gru_valid)
    gru_t = np.load(ROOT / args.gru_test)
    tree_v = np.load(ROOT / args.tree_valid)
    tree_t = np.load(ROOT / args.tree_test)
    mlp_v = np.load(ROOT / args.mlp_valid)
    mlp_t = np.load(ROOT / args.mlp_test)

    pred_v = n2000_hard_gate_mlp(
        gru_v, tree_v, mlp_v, valid["mask_x"], tau, W_LOW, W_HIGH, W_MLP, "raw", "rank"
    )
    fused_ic = float(mean_rank_ic(pred_v, valid["y1"], valid["mask_y"]))
    print(f"FUSED_RANKIC {fused_ic:.6f}")

    pred_t = n2000_hard_gate_mlp(
        gru_t, tree_t, mlp_t, test["mask_x"], tau, W_LOW, W_HIGH, W_MLP, "raw", "rank"
    )
    tag = str(args.tag)
    valid_path = ROOT / f"outputs/fusion_{tag}_valid.npy"
    test_path = ROOT / f"outputs/fusion_{tag}_test.npy"
    sub_path = ROOT / f"submissions/task1_fusion_{tag}.npy"
    np.save(valid_path, pred_v)
    np.save(test_path, pred_t)
    save_submission(pred_t, sub_path)
    print(f"wrote {sub_path}")
    print("does not overwrite submissions/task1_y1_upload.npy")


if __name__ == "__main__":
    main()
