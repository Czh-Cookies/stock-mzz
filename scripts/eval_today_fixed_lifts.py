"""Fixed-rule fusion lifts. No weight grid. Conservative 0.4/0.25/tau_q=0.6 stays unless noted."""

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
    linear_blend,
    n2000_hard_gate_mlp,
    neutralize_industry,
    stack_mlp_blend,
)
from src.submit import save_submission  # noqa: E402

W_LOW, W_HIGH, TAU_Q, W_MLP = 0.25, 0.4, 0.6, 0.15


def ic(pred, split):
    return float(mean_rank_ic(pred, split["y1"], split["mask_y"]))


def fuse(gru, tree, mlp, mask, tau, space="raw"):
    return n2000_hard_gate_mlp(gru, tree, mlp, mask, tau, W_LOW, W_HIGH, W_MLP, space, "rank")


def industry_size_gate(gru, tree, mlp, mask, industry, tau):
    gated = np.empty_like(gru, dtype=np.float32)
    for t in range(gru.shape[0]):
        m = np.asarray(mask[t], dtype=bool)
        g = np.asarray(industry[t])
        sizes = np.zeros_like(g, dtype=np.int32)
        for gid in np.unique(g[m]):
            sizes[g == gid] = int((m & (g == gid)).sum())
        med = float(np.median(sizes[m])) if int(m.sum()) else 0.0
        w = np.where(sizes >= med, W_HIGH, W_LOW).astype(np.float64)
        day = w * gru[t] + (1.0 - w) * tree[t]
        # still mix day-coverage on top using same tau
        n = float(m.sum())
        w_day = W_HIGH if n >= tau else W_LOW
        gated[t] = (0.5 * w_day + 0.5 * w) * gru[t] + (1.0 - (0.5 * w_day + 0.5 * w)) * tree[t]
    return stack_mlp_blend(mlp, gated, mask, W_MLP, "rank")


def main() -> None:
    cache = split_cache_dir(ROOT)
    splits, src = load_eval_splits(cache_dir=cache, splits=("valid", "test"))
    valid, test = splits["valid"], splits["test"]
    n_x = np.asarray(valid["mask_x"]).sum(axis=1).astype(np.float64)
    tau = float(np.quantile(n_x, TAU_Q))
    print(f"src={src} tau_q={TAU_Q} tau={tau:.1f}")

    gru_n = np.load(ROOT / "outputs/gru_no_today_recent_n2000_valid.npy")
    gru_f = np.load(ROOT / "outputs/gru_no_today_recent_full_valid.npy")
    tree = np.load(ROOT / "outputs/fusion_valid.npy")
    mlp = np.load(ROOT / "outputs/cs_mlp_valid.npy")
    gru_n_t = np.load(ROOT / "outputs/gru_no_today_recent_n2000_test.npy")
    gru_f_t = np.load(ROOT / "outputs/gru_no_today_recent_full_test.npy")
    tree_t = np.load(ROOT / "outputs/fusion_test.npy")
    mlp_t = np.load(ROOT / "outputs/cs_mlp_test.npy")
    hard_v = np.load(ROOT / "outputs/fusion_full_hardgate_valid.npy")
    hard_t = np.load(ROOT / "outputs/fusion_full_hardgate_test.npy")
    cons_v = fuse(gru_n, tree, mlp, valid["mask_x"], tau)
    cons_t = fuse(gru_n_t, tree_t, mlp_t, test["mask_x"], tau)

    rows = []
    def report(name, pred_v, pred_t=None):
        v = ic(pred_v, valid)
        rows.append((v, name, pred_v, pred_t))
        print(f"{v:.6f}  {name}")

    report("conservative_n2000", cons_v, cons_t)
    report("full_hardgate_cached", hard_v, hard_t)
    report("mix050_cons_hardgate", 0.5 * cons_v + 0.5 * hard_v, 0.5 * cons_t + 0.5 * hard_t)
    report("mix030_cons_hardgate", 0.7 * cons_v + 0.3 * hard_v, 0.7 * cons_t + 0.3 * hard_t)
    mix_gru = 0.5 * gru_n + 0.5 * gru_f
    mix_gru_t = 0.5 * gru_n_t + 0.5 * gru_f_t
    report("mix050_gru_then_cons", fuse(mix_gru, tree, mlp, valid["mask_x"], tau), fuse(mix_gru_t, tree_t, mlp_t, test["mask_x"], tau))
    report("full_gru_cons", fuse(gru_f, tree, mlp, valid["mask_x"], tau), fuse(gru_f_t, tree_t, mlp_t, test["mask_x"], tau))
    report("n2000_rank_space", fuse(gru_n, tree, mlp, valid["mask_x"], tau, "rank"), fuse(gru_n_t, tree_t, mlp_t, test["mask_x"], tau, "rank"))
    gru_neu = neutralize_industry(gru_n, valid["industry"], valid["mask_x"])
    gru_neu_t = neutralize_industry(gru_n_t, test["industry"], test["mask_x"])
    report("n2000_ind_neu_then_cons", fuse(gru_neu, tree, mlp, valid["mask_x"], tau), fuse(gru_neu_t, tree_t, mlp_t, test["mask_x"], tau))
    report(
        "industry_size_plus_day_gate",
        industry_size_gate(gru_n, tree, mlp, valid["mask_x"], valid["industry"], tau),
        industry_size_gate(gru_n_t, tree_t, mlp_t, test["mask_x"], test["industry"], tau),
    )

    # train-tau: use last 800 train days coverage quantile from train cache
    train = __import__("src.dataset", fromlist=["load_split_cache"]).load_split_cache("train", cache)
    n_train = np.asarray(train["mask_x"][-800:]).sum(axis=1).astype(np.float64)
    tau_tr = float(np.quantile(n_train, TAU_Q))
    print(f"tau_train800={tau_tr:.1f}")
    report(
        "conservative_tau_from_train",
        fuse(gru_n, tree, mlp, valid["mask_x"], tau_tr),
        fuse(gru_n_t, tree_t, mlp_t, test["mask_x"], tau_tr),
    )

    rows.sort(reverse=True)
    best_v, best_name, best_pv, best_pt = rows[0]
    base = rows[[r[1] for r in rows].index("conservative_n2000")][0]
    print(f"BEST {best_name} {best_v:.6f} vs cons {base:.6f} delta={best_v-base:+.6f}")
    if best_pt is None:
        return
    spike = best_v > base + 0.003
    if spike:
        print("SPIKE: treat as valid overfit, not writing as submit")
        return
    if best_name == "conservative_n2000" or best_v < base - 1e-6:
        print("no lift; not replacing submit candidate")
        return
    tag = "today_fixed_" + best_name
    np.save(ROOT / f"outputs/fusion_{tag}_valid.npy", best_pv)
    np.save(ROOT / f"outputs/fusion_{tag}_test.npy", best_pt)
    path = ROOT / f"submissions/task1_fusion_{tag}.npy"
    save_submission(best_pt, path)
    print(f"wrote {path}")
    print("does not overwrite submissions/task1_y1_upload.npy")


if __name__ == "__main__":
    main()
