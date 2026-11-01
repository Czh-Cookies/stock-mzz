"""Climb path: rank-average complementary GRUs, then frozen conservative gate.

Does not search tau/weights on valid (that overfit 243 days: ensemble grid picked
full_mix=0, s43_mix=0). Does not overwrite task1_y1_upload.npy.

Preferred GRU set (skip missing files):
  1) gru_n2000_add6973  — columns 69/73, official-path champion branch
  2) gru_full_add6973   — same columns, full-universe sample
  3) leftover GRU       — unused trajectory columns
Fallback: n2000 / full without 69/73.

Tree: hist_lgbm_add6973 raw_blend 0.7 vs baseline if both exist, else fusion_valid.

Train first if npy missing:
  python scripts/train_gru.py --config configs/gru_n2000_add6973_e1.yaml
  python scripts/train_gru.py --config configs/gru_full_add6973_e1.yaml
  python scripts/train_baseline.py --config configs/hist_lgbm_add6973.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_eval_splits, split_cache_dir  # noqa: E402
from src.metrics import mean_rank_ic  # noqa: E402
from src.models.fusion import linear_blend, n2000_hard_gate_mlp, panel_rank_mean  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from src.submit import save_submission  # noqa: E402

W_LOW, W_HIGH, TAU_Q, W_MLP = 0.25, 0.4, 0.6, 0.15
TREE_W = 0.7

GRU_CANDIDATES = (
    ("add6973_n2000", "outputs/gru_n2000_add6973_e1_valid.npy", "outputs/gru_n2000_add6973_e1_test.npy"),
    ("add6973_full", "outputs/gru_full_add6973_e1_valid.npy", "outputs/gru_full_add6973_e1_test.npy"),
    ("leftover", "outputs/gru_n2000_leftover_e1_valid.npy", "outputs/gru_n2000_leftover_e1_test.npy"),
    ("n2000", "outputs/gru_no_today_recent_n2000_valid.npy", "outputs/gru_no_today_recent_n2000_test.npy"),
    ("full", "outputs/gru_no_today_recent_full_valid.npy", "outputs/gru_no_today_recent_full_test.npy"),
)


def _load_pair(valid_rel: str, test_rel: str) -> tuple[np.ndarray, np.ndarray] | None:
    vp, tp = ROOT / valid_rel, ROOT / test_rel
    if vp.is_file() and tp.is_file():
        return np.load(vp), np.load(tp)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank-average GRUs + conservative gate")
    parser.add_argument("--data", default=None)
    args = parser.parse_args()

    cache = split_cache_dir(ROOT)
    data_path = None
    if not (cache / "valid.npz").is_file():
        from src.config import load_config, resolve_data_path

        data_path = resolve_data_path(load_config("configs/default.yaml"), args.data)
    splits, src = load_eval_splits(cache_dir=cache, data_path=data_path, splits=("valid", "test"))
    valid, test = splits["valid"], splits["test"]
    tau = float(np.quantile(np.asarray(valid["mask_x"]).sum(1).astype(np.float64), TAU_Q))
    print(f"src={src} conservative tau_q={TAU_Q} tau={tau:.1f}")

    gru_v: list[np.ndarray] = []
    gru_t: list[np.ndarray] = []
    used: list[str] = []
    for name, vr, tr in GRU_CANDIDATES:
        pair = _load_pair(vr, tr)
        if pair is None:
            print(f"skip missing {name}")
            continue
        if name in {"n2000", "full"} and any(u.startswith("add6973") for u in used):
            # weaker same-recipe fallback; keep only if no 69/73 GRU exists
            continue
        gru_v.append(pair[0])
        gru_t.append(pair[1])
        used.append(name)
        ic_g = float(mean_rank_ic(pair[0], valid["y1"], valid["mask_y"]))
        print(f"use GRU {name} valid_RankIC={ic_g:.6f}")
    if not gru_v:
        raise SystemExit("no GRU valid/test npy found; train add6973 / n2000 first")

    mix_v = gru_v[0] if len(gru_v) == 1 else panel_rank_mean(gru_v, valid["mask_x"])
    mix_t = gru_t[0] if len(gru_t) == 1 else panel_rank_mean(gru_t, test["mask_x"])
    if len(gru_v) >= 2:
        mix_ic = float(mean_rank_ic(mix_v, valid["y1"], valid["mask_y"]))
        print(f"rank-mean GRU valid_RankIC={mix_ic:.6f}")
        for i in range(1, len(used)):
            rs = []
            for t in range(gru_v[0].shape[0]):
                m = np.asarray(valid["mask_x"][t], dtype=bool)
                if int(m.sum()) < 8:
                    continue
                r = spearmanr(gru_v[0][t, m], gru_v[i][t, m]).statistic
                if np.isfinite(r):
                    rs.append(float(r))
            print(f"daily Spearman {used[0]} vs {used[i]} {float(np.mean(rs)) if rs else float('nan'):.4f}")


    hist = _load_pair("outputs/hist_lgbm_add6973_valid.npy", "outputs/hist_lgbm_add6973_test.npy")
    base = _load_pair("outputs/baseline_valid.npy", "outputs/baseline_test.npy")
    fused = _load_pair("outputs/fusion_valid.npy", "outputs/fusion_test.npy")
    # hist_add6973 alone is 0.1017 < old hist 0.1042; blending it dropped fused ~0.0015.
    # Rank-mean GRU into raw gate drops to ~0.10 because ranks ~1..S mix with tree raw scores.
    if fused is not None:
        tree_v, tree_t = fused
        tree_note = "fusion_valid (locked 0.7 hist vs baseline)"
    elif hist is not None and base is not None:
        tree_v = linear_blend(hist[0], base[0], TREE_W)
        tree_t = linear_blend(hist[1], base[1], TREE_W)
        tree_note = "hist_add6973 raw_blend0.7 baseline (fallback)"
    else:
        raise SystemExit("need fusion_valid or hist_lgbm_add6973 + baseline")
    print(f"tree={tree_note}")

    mlp = _load_pair("outputs/cs_mlp_valid.npy", "outputs/cs_mlp_test.npy")
    if mlp is None:
        raise SystemExit("missing cs_mlp valid/test npy")
    mlp_v, mlp_t = mlp

    pred_v = n2000_hard_gate_mlp(
        mix_v, tree_v, mlp_v, valid["mask_x"], tau, W_LOW, W_HIGH, W_MLP, "raw", "rank"
    )
    ic = float(mean_rank_ic(pred_v, valid["y1"], valid["mask_y"]))
    print(f"CLIMB_VALID_RANKIC {ic:.6f}  grus={used}")

    if used == ["add6973_n2000"] or used == ["n2000"]:
        print("single GRU; rank ensemble had nothing extra to average")

    pred_t = n2000_hard_gate_mlp(
        mix_t, tree_t, mlp_t, test["mask_x"], tau, W_LOW, W_HIGH, W_MLP, "raw", "rank"
    )
    np.save(ROOT / "outputs/fusion_climb_rank_ensemble_valid.npy", pred_v)
    np.save(ROOT / "outputs/fusion_climb_rank_ensemble_test.npy", pred_t)
    out = ROOT / "submissions/task1_fusion_climb_rank_ensemble.npy"
    save_submission(pred_t, out)
    meta = {
        "valid_rankic": ic,
        "grus": used,
        "tree": tree_note,
        "gate": {"tau_q": TAU_Q, "tau": tau, "weight_low": W_LOW, "weight_high": W_HIGH, "mlp": W_MLP},
        "note": "no valid grid; conservative gate; does not overwrite task1_y1_upload.npy",
    }
    (ROOT / "outputs/fusion_climb_rank_ensemble_lock.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {out}")
    print("upload this npy after scripts/check_submission.py; keep previous 0.125679 file as backup")


if __name__ == "__main__":
    main()
