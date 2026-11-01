"""Small-step fusion variants on locked add6973 GRU. Valid only. No weight grid."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_eval_splits, split_cache_dir  # noqa: E402
from src.metrics import mean_rank_ic  # noqa: E402
from src.models.fusion import (  # noqa: E402
    coverage_gate_blend,
    n2000_hard_gate_mlp,
    neutralize_industry,
    stack_mlp_blend,
)
from src.submit import save_submission  # noqa: E402

BASE = 0.11840
W_LOW, W_HIGH, W_MLP = 0.25, 0.4, 0.15


def main() -> None:
    cache = split_cache_dir(ROOT)
    splits, src = load_eval_splits(cache_dir=cache, splits=("valid", "test"))
    valid, test = splits["valid"], splits["test"]
    tau = float(np.quantile(np.asarray(valid["mask_x"]).sum(1).astype(np.float64), 0.6))
    gru_v = np.load(ROOT / "outputs/gru_n2000_add6973_e1_valid.npy")
    gru_t = np.load(ROOT / "outputs/gru_n2000_add6973_e1_test.npy")
    tree_v = np.load(ROOT / "outputs/fusion_valid.npy")
    tree_t = np.load(ROOT / "outputs/fusion_test.npy")
    mlp_v = np.load(ROOT / "outputs/cs_mlp_valid.npy")
    mlp_t = np.load(ROOT / "outputs/cs_mlp_test.npy")
    ind_v = valid["industry"]
    ind_t = test["industry"]
    mask_v = valid["mask_x"]
    mask_t = test["mask_x"]

    def ic(pred):
        return float(mean_rank_ic(pred, valid["y1"], valid["mask_y"]))

    rows = []

    def consider(name, pred_v, pred_t):
        v = ic(pred_v)
        delta = v - BASE
        keep = "KEEP" if v > BASE + 1e-5 else "ROLLBACK"
        print(f"{v:.6f}  {delta:+.6f}  {keep}  {name}")
        rows.append((v, name, pred_v, pred_t, keep))

    cons_v = n2000_hard_gate_mlp(gru_v, tree_v, mlp_v, mask_v, tau, W_LOW, W_HIGH, W_MLP, "raw", "rank")
    cons_t = n2000_hard_gate_mlp(gru_t, tree_t, mlp_t, mask_t, tau, W_LOW, W_HIGH, W_MLP, "raw", "rank")
    consider("baseline_raw_gate", cons_v, cons_t)

    consider(
        "gate_rank_space",
        n2000_hard_gate_mlp(gru_v, tree_v, mlp_v, mask_v, tau, W_LOW, W_HIGH, W_MLP, "rank", "rank"),
        n2000_hard_gate_mlp(gru_t, tree_t, mlp_t, mask_t, tau, W_LOW, W_HIGH, W_MLP, "rank", "rank"),
    )
    consider(
        "mlp_raw_second",
        n2000_hard_gate_mlp(gru_v, tree_v, mlp_v, mask_v, tau, W_LOW, W_HIGH, W_MLP, "raw", "raw"),
        n2000_hard_gate_mlp(gru_t, tree_t, mlp_t, mask_t, tau, W_LOW, W_HIGH, W_MLP, "raw", "raw"),
    )

    gru_nv = neutralize_industry(gru_v, ind_v, mask_v)
    gru_nt = neutralize_industry(gru_t, ind_t, mask_t)
    consider(
        "neu_gru_then_gate",
        n2000_hard_gate_mlp(gru_nv, tree_v, mlp_v, mask_v, tau, W_LOW, W_HIGH, W_MLP, "raw", "rank"),
        n2000_hard_gate_mlp(gru_nt, tree_t, mlp_t, mask_t, tau, W_LOW, W_HIGH, W_MLP, "raw", "rank"),
    )

    tree_nv = neutralize_industry(tree_v, ind_v, mask_v)
    tree_nt = neutralize_industry(tree_t, ind_t, mask_t)
    consider(
        "neu_gru_tree_then_gate",
        n2000_hard_gate_mlp(gru_nv, tree_nv, mlp_v, mask_v, tau, W_LOW, W_HIGH, W_MLP, "raw", "rank"),
        n2000_hard_gate_mlp(gru_nt, tree_nt, mlp_t, mask_t, tau, W_LOW, W_HIGH, W_MLP, "raw", "rank"),
    )

    gated_v = coverage_gate_blend(gru_v, tree_v, mask_v, tau, W_LOW, W_HIGH, "raw")
    gated_t = coverage_gate_blend(gru_t, tree_t, mask_t, tau, W_LOW, W_HIGH, "raw")
    consider(
        "neu_gated_then_mlp",
        stack_mlp_blend(mlp_v, neutralize_industry(gated_v, ind_v, mask_v), mask_v, W_MLP, "rank"),
        stack_mlp_blend(mlp_t, neutralize_industry(gated_t, ind_t, mask_t), mask_t, W_MLP, "rank"),
    )

    best = max(rows, key=lambda r: r[0])
    print(f"BEST {best[1]} {best[0]:.6f}")
    if best[4] == "KEEP" and best[1] != "baseline_raw_gate":
        tag = "fuse_" + best[1]
        save_submission(best[3], ROOT / f"submissions/task1_fusion_{tag}.npy")
        np.save(ROOT / f"outputs/fusion_{tag}_valid.npy", best[2])
        print(f"wrote submissions/task1_fusion_{tag}.npy")
    else:
        print("no fusion lift; stay on 0.11840 baseline")


if __name__ == "__main__":
    main()
