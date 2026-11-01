"""One-shot fixed mixes on locked conservative gate. No weight search."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_eval_splits, split_cache_dir  # noqa: E402
from src.metrics import mean_rank_ic  # noqa: E402
from src.models.fusion import n2000_hard_gate_mlp  # noqa: E402

BASE = 0.11840


def main() -> None:
    splits, _ = load_eval_splits(cache_dir=split_cache_dir(ROOT), splits=("valid",))
    v = splits["valid"]
    tau = float(np.quantile(np.asarray(v["mask_x"]).sum(1).astype(np.float64), 0.6))
    y, my, mask = v["y1"], v["mask_y"], v["mask_x"]
    tree = np.load(ROOT / "outputs/fusion_valid.npy")
    mlp = np.load(ROOT / "outputs/cs_mlp_valid.npy")
    champ = np.load(ROOT / "outputs/gru_n2000_add6973_e1_valid.npy")
    l12 = np.load(ROOT / "outputs/gru_n2000_L12_e1_valid.npy")
    mlp47 = np.load(ROOT / "outputs/cs_mlp_add47_e8_valid.npy")
    mlp3 = np.load(ROOT / "outputs/cs_mlp_add3_e8_valid.npy")

    def fused(gru, mlp_p):
        pred = n2000_hard_gate_mlp(gru, tree, mlp_p, mask, tau, 0.25, 0.4, 0.15, "raw", "rank")
        ic = float(mean_rank_ic(pred, y, my))
        return ic

    mix = (0.5 * champ + 0.5 * l12).astype(np.float32)
    rows = [
        ("champ", fused(champ, mlp)),
        ("gru_0.5_champ_L12", fused(mix, mlp)),
        ("champ+mlp47", fused(champ, mlp47)),
        ("L12+mlp47", fused(l12, mlp47)),
        ("champ+mlp3", fused(champ, mlp3)),
    ]
    for name, ic in rows:
        flag = "KEEP" if ic > BASE + 0.0005 else "rollback"
        print(f"{name:22s} {ic:.6f} {ic - BASE:+.6f} {flag}")


if __name__ == "__main__":
    main()
