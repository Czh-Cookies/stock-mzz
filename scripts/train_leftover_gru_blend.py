"""Independent leftover GRU branch, then fixed 0.2 mix into the champion GRU.

Does not edit champion configs, checkpoints, or train_gru.py.
Hidden size / layers stay at the n2000 GRU defaults (64, 1 layer). Mix weight is
hard-coded 0.2 and is not trained.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\train_leftover_gru_blend.py
    .\\.venv\\Scripts\\python.exe scripts\\train_leftover_gru_blend.py --skip-train
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_eval_splits, split_cache_dir  # noqa: E402
from src.metrics import mean_rank_ic  # noqa: E402
from src.models.fusion import n2000_hard_gate_mlp  # noqa: E402
from src.submit import save_submission  # noqa: E402

# Champion GRU (official test 0.121626; local fused valid 0.11840). Do not overwrite.
CHAMP_GRU_VALID = ROOT / "outputs/gru_n2000_add6973_e1_valid.npy"
CHAMP_GRU_TEST = ROOT / "outputs/gru_n2000_add6973_e1_test.npy"
LEFTOVER_CFG = ROOT / "configs/gru_n2000_leftover_e1.yaml"
LEFTOVER_VALID = ROOT / "outputs/gru_n2000_leftover_e1_valid.npy"
LEFTOVER_TEST = ROOT / "outputs/gru_n2000_leftover_e1_test.npy"
PYTHON = ROOT / ".venv/Scripts/python.exe"

# Fixed; not a learned parameter.
LEFTOVER_WEIGHT = 0.2
W_LOW = 0.25
W_HIGH = 0.4
TAU_Q = 0.6
W_MLP = 0.15
LOCAL_FUSED_BASE = 0.11840
TAG = "leftover_w02"


def _daily_spearman(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    vals: list[float] = []
    for t in range(a.shape[0]):
        m = np.asarray(mask[t], dtype=bool) & np.isfinite(a[t]) & np.isfinite(b[t])
        if int(m.sum()) < 8:
            continue
        r = spearmanr(a[t, m], b[t, m]).statistic
        if np.isfinite(r):
            vals.append(float(r))
    return float(np.mean(vals)) if vals else float("nan")


def _train_leftover() -> None:
    py = str(PYTHON if PYTHON.is_file() else sys.executable)
    cmd = [py, "-u", str(ROOT / "scripts/train_gru.py"), "--config", str(LEFTOVER_CFG)]
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    print("train leftover:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Leftover GRU + fixed 0.2 mix + conservative gate")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="reuse existing leftover npy; do not call train_gru.py",
    )
    args = parser.parse_args()

    if not CHAMP_GRU_VALID.is_file() or not CHAMP_GRU_TEST.is_file():
        raise SystemExit(f"missing champion GRU preds: {CHAMP_GRU_VALID} / {CHAMP_GRU_TEST}")

    if not args.skip_train:
        _train_leftover()
    if not LEFTOVER_VALID.is_file() or not LEFTOVER_TEST.is_file():
        raise SystemExit(f"missing leftover preds: {LEFTOVER_VALID} / {LEFTOVER_TEST}")

    champ_v = np.load(CHAMP_GRU_VALID)
    champ_t = np.load(CHAMP_GRU_TEST)
    side_v = np.load(LEFTOVER_VALID)
    side_t = np.load(LEFTOVER_TEST)
    mix_v = ((1.0 - LEFTOVER_WEIGHT) * champ_v + LEFTOVER_WEIGHT * side_v).astype(np.float32)
    mix_t = ((1.0 - LEFTOVER_WEIGHT) * champ_t + LEFTOVER_WEIGHT * side_t).astype(np.float32)

    mix_valid_path = ROOT / "outputs/gru_mix_leftover_w02_valid.npy"
    mix_test_path = ROOT / "outputs/gru_mix_leftover_w02_test.npy"
    np.save(mix_valid_path, mix_v)
    np.save(mix_test_path, mix_t)

    splits, src = load_eval_splits(cache_dir=split_cache_dir(ROOT), splits=("valid", "test"))
    valid, test = splits["valid"], splits["test"]
    tau = float(np.quantile(np.asarray(valid["mask_x"]).sum(axis=1).astype(np.float64), TAU_Q))
    y, my, mask = valid["y1"], valid["mask_y"], valid["mask_x"]
    tree_v = np.load(ROOT / "outputs/fusion_valid.npy")
    tree_t = np.load(ROOT / "outputs/fusion_test.npy")
    mlp_v = np.load(ROOT / "outputs/cs_mlp_valid.npy")
    mlp_t = np.load(ROOT / "outputs/cs_mlp_test.npy")

    corr = _daily_spearman(champ_v, side_v, mask)
    print(f"leftover_weight={LEFTOVER_WEIGHT} (fixed, not trained)")
    print(f"daily Spearman leftover vs champion GRU {corr:.4f} (L12 clone was 0.9997)")
    print(f"leftover GRU valid RankIC {mean_rank_ic(side_v, y, my):.6f}")
    print(f"champion GRU valid RankIC {mean_rank_ic(champ_v, y, my):.6f}")
    print(f"mix GRU valid RankIC {mean_rank_ic(mix_v, y, my):.6f}")

    pred_v = n2000_hard_gate_mlp(
        mix_v, tree_v, mlp_v, valid["mask_x"], tau, W_LOW, W_HIGH, W_MLP, "raw", "rank"
    )
    fused = float(mean_rank_ic(pred_v, y, my))
    print(
        f"conservative gate w_low={W_LOW} w_high={W_HIGH} tau_q={TAU_Q} tau={tau:.1f} "
        f"mlp={W_MLP} src={src}"
    )
    print(f"FUSED_RANKIC {fused:.6f} vs local champ {LOCAL_FUSED_BASE:.5f} delta={fused - LOCAL_FUSED_BASE:+.6f}")
    print("official champion test RankIC 0.121626 is unchanged on disk")

    pred_t = n2000_hard_gate_mlp(
        mix_t, tree_t, mlp_t, test["mask_x"], tau, W_LOW, W_HIGH, W_MLP, "raw", "rank"
    )
    np.save(ROOT / f"outputs/fusion_{TAG}_valid.npy", pred_v)
    np.save(ROOT / f"outputs/fusion_{TAG}_test.npy", pred_t)
    sub = ROOT / f"submissions/task1_fusion_{TAG}.npy"
    save_submission(pred_t, sub)
    print(f"wrote {mix_valid_path}")
    print(f"wrote {sub}")
    print("does not overwrite outputs/gru_n2000_add6973_e1_*.npy")
    print("does not overwrite submissions/task1_fusion_add6973_e1.npy")
    print("does not overwrite submissions/task1_y1_upload.npy")


if __name__ == "__main__":
    main()
