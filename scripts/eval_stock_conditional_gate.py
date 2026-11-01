"""Fixed stock-coverage nest on the locked conservative day gate. No weight search.

Sparse names (history coverage < 0.5) always use the low GRU weight.
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
from src.models.fusion import coverage_gate_blend, linear_blend, stack_mlp_blend, stock_history_coverage  # noqa: E402
from src.submit import save_submission  # noqa: E402

W_LOW = 0.25
W_HIGH = 0.4
TAU_Q = 0.6
W_MLP = 0.15
STOCK_CUT = 0.5
LOOKBACK = 10


def _fuse(gru, tree, mlp, mask_x, tau, stock_cov):
    gated = coverage_gate_blend(gru, tree, mask_x, tau, W_LOW, W_HIGH, "raw")
    sparse = np.asarray(stock_cov, dtype=np.float64) < STOCK_CUT
    if np.any(sparse):
        forced = linear_blend(gru, tree, W_LOW)
        gated = np.where(sparse, forced, gated).astype(np.float32)
    return stack_mlp_blend(mlp, gated, mask_x, W_MLP, "rank")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gru-valid", default="outputs/gru_no_today_recent_n2000_valid.npy")
    parser.add_argument("--gru-test", default="outputs/gru_no_today_recent_n2000_test.npy")
    parser.add_argument("--tag", default="stock_cond_e1")
    args = parser.parse_args()

    cache = split_cache_dir(ROOT)
    splits, src = load_eval_splits(cache_dir=cache, splits=("valid", "test"))
    valid, test = splits["valid"], splits["test"]
    n_x = np.asarray(valid["mask_x"]).sum(axis=1).astype(np.float64)
    tau = float(np.quantile(n_x, TAU_Q))
    print(
        f"stock-cond gate w_low={W_LOW} w_high={W_HIGH} tau_q={TAU_Q} tau={tau:.1f} "
        f"stock_cut={STOCK_CUT} mlp={W_MLP} src={src}"
    )

    gru_v = np.load(ROOT / args.gru_valid)
    gru_t = np.load(ROOT / args.gru_test)
    tree_v = np.load(ROOT / "outputs/fusion_valid.npy")
    tree_t = np.load(ROOT / "outputs/fusion_test.npy")
    mlp_v = np.load(ROOT / "outputs/cs_mlp_valid.npy")
    mlp_t = np.load(ROOT / "outputs/cs_mlp_test.npy")

    panel = np.asarray(valid["panel_mask_x"] if "panel_mask_x" in valid else None)
    # split cache may not have panel_mask_x; reconstruct coverage from mask_x only is wrong.
    # load_eval_splits should provide start indices. Use cached mask_x of train+valid if needed.

    from src.dataset import load_split_cache

    train_c = load_split_cache("train", cache)
    # Concatenate train/valid/test mask_x along time for lookback.
    # test cache also exists.
    test_c = splits["test"]
    # For valid, panel origin 0 would need full history. split cache train starts at train_start.
    # stock_history_coverage(panel_mask_x, panel_origin, split_start, n_days, length)
    # If we only have split masks, lookback is truncated at split start — acceptable if train
    # cache is contiguous before valid.
    panel_mask = np.concatenate(
        [np.asarray(train_c["mask_x"], dtype=bool), np.asarray(valid["mask_x"], dtype=bool), np.asarray(test_c["mask_x"], dtype=bool)],
        axis=0,
    )
    valid_start_in_panel = int(np.asarray(train_c["mask_x"]).shape[0])
    test_start_in_panel = valid_start_in_panel + int(valid["mask_x"].shape[0])
    cov_v = stock_history_coverage(panel_mask, 0, valid_start_in_panel, valid["mask_x"].shape[0], LOOKBACK)
    cov_t = stock_history_coverage(panel_mask, 0, test_start_in_panel, test_c["mask_x"].shape[0], LOOKBACK)

    pred_v = _fuse(gru_v, tree_v, mlp_v, valid["mask_x"], tau, cov_v)
    fused_ic = float(mean_rank_ic(pred_v, valid["y1"], valid["mask_y"]))
    print(f"FUSED_RANKIC {fused_ic:.6f}")

    pred_t = _fuse(gru_t, tree_t, mlp_t, test_c["mask_x"], tau, cov_t)
    tag = str(args.tag)
    np.save(ROOT / f"outputs/fusion_{tag}_valid.npy", pred_v)
    np.save(ROOT / f"outputs/fusion_{tag}_test.npy", pred_t)
    sub = ROOT / f"submissions/task1_fusion_{tag}.npy"
    save_submission(pred_t, sub)
    print(f"wrote {sub}")
    print("does not overwrite submissions/task1_y1_upload.npy")


if __name__ == "__main__":
    main()
