"""Coverage-gated GRU vs tree fusion. No retrain.

Old FusionModel coverage_gate only searched GRU weight in {0.5, 0.75, 1.0} on rank
space. Global optimum was raw 0.25, so that grid could not beat 0.110. This script
searches 0.15/0.25/0.4 in raw and rank, and both directions (more GRU when crowded
or when sparse). Does not overwrite fusion_gru_blend / fusion_cs_mlp.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\eval_coverage_gru.py
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
from src.dataset import load_eval_splits, split_cache_dir  # noqa: E402
from src.metrics import mean_rank_ic, rank_ic_series  # noqa: E402
from src.models.fusion import FusionModel, coverage_gate_blend, linear_blend  # noqa: E402
from src.submit import save_submission  # noqa: E402

WEIGHTS = [0.15, 0.25, 0.4]
TAU_QS = [0.4, 0.5, 0.6]
SPACES = ("raw", "rank")
BASE_GRU_BLEND = 0.110069
BASE_CS_MLP = 0.111264


def _quartile_table(n_x: np.ndarray, series: dict[str, np.ndarray]) -> None:
    qs = np.nanpercentile(n_x, [0, 25, 50, 75, 100])
    print(
        f"coverage min={qs[0]:.0f} p25={qs[1]:.0f} median={qs[2]:.0f} "
        f"p75={qs[3]:.0f} max={qs[4]:.0f}"
    )
    edges = np.nanpercentile(n_x, [0, 25, 50, 75, 100])
    print("coverage quartile  n_days  " + "  ".join(f"{k:>10s}" for k in series))
    for i in range(4):
        lo, hi = edges[i], edges[i + 1]
        if i < 3:
            sel = (n_x >= lo) & (n_x < hi)
        else:
            sel = (n_x >= lo) & (n_x <= hi)
        bits = [f"{float(np.nanmean(v[sel])):.4f}" for v in series.values()]
        print(f"  Q{i + 1} [{lo:.0f},{hi:.0f}]  {int(sel.sum()):5d}  " + "  ".join(f"{b:>10s}" for b in bits))


def main() -> None:
    parser = argparse.ArgumentParser(description="Coverage-gated GRU blend")
    parser.add_argument("--data", default=None)
    args = parser.parse_args()

    cfg = load_config("configs/fusion_gru_blend.yaml")
    t0 = time.perf_counter()
    cache_dir = split_cache_dir(ROOT)
    data_path = None
    if not (cache_dir / "valid.npz").is_file() or not (cache_dir / "test.npz").is_file():
        data_path = resolve_data_path(cfg, args.data)
    splits, src = load_eval_splits(
        cache_dir=cache_dir,
        data_path=data_path,
        splits=("valid", "test"),
    )
    valid = splits["valid"]
    test = splits["test"]
    print(f"loaded {src} in {time.perf_counter() - t0:.1f}s")

    gru_v = np.load(ROOT / "outputs/gru_valid.npy")
    tree_v = np.load(ROOT / "outputs/fusion_valid.npy")
    gru_t = np.load(ROOT / "outputs/gru_test.npy")
    tree_t = np.load(ROOT / "outputs/fusion_test.npy")

    n_x = np.asarray(valid["mask_x"]).sum(axis=1).astype(np.float64)
    gru_ic = rank_ic_series(gru_v, valid["y1"], valid["mask_y"])
    tree_ic = rank_ic_series(tree_v, valid["y1"], valid["mask_y"])
    blend25 = linear_blend(gru_v, tree_v, 0.25)
    blend_ic = rank_ic_series(blend25, valid["y1"], valid["mask_y"])
    print(f"global raw_blend 0.25 RankIC={mean_rank_ic(blend25, valid['y1'], valid['mask_y']):.6f}")
    print(
        f"corr(coverage, RankIC) tree={float(np.corrcoef(n_x, tree_ic)[0, 1]):.3f} "
        f"gru={float(np.corrcoef(n_x, gru_ic)[0, 1]):.3f} "
        f"blend25={float(np.corrcoef(n_x, blend_ic)[0, 1]):.3f}"
    )
    _quartile_table(n_x, {"tree": tree_ic, "gru": gru_ic, "blend25": blend_ic})

    results: list[dict] = []
    for space in SPACES:
        for q in TAU_QS:
            tau = float(np.quantile(n_x, q))
            n_high = int((n_x >= tau).sum())
            for w_low in WEIGHTS:
                for w_high in WEIGHTS:
                    pred = coverage_gate_blend(gru_v, tree_v, valid["mask_x"], tau, w_low, w_high, space)
                    ic = mean_rank_ic(pred, valid["y1"], valid["mask_y"])
                    results.append(
                        {
                            "name": "coverage_gate",
                            "ic": float(ic),
                            "space": space,
                            "tau": tau,
                            "tau_q": q,
                            "n_high": n_high,
                            "weight_low": w_low,
                            "weight_high": w_high,
                            "more_gru_when": "high" if w_high > w_low else ("low" if w_high < w_low else "equal"),
                        }
                    )
    results.sort(key=lambda r: r["ic"], reverse=True)
    print("top coverage gates:")
    for row in results[:8]:
        print(
            f"  {row['ic']:.6f}  {row['space']} q={row['tau_q']} tau={row['tau']:.0f} "
            f"w_low={row['weight_low']} w_high={row['weight_high']} ({row['more_gru_when']} cov)"
        )
    equal = [r for r in results if r["weight_low"] == r["weight_high"] == 0.25 and r["space"] == "raw"]
    print(f"raw equal-0.25 (any tau) {[round(r['ic'], 6) for r in equal]}")

    locked = dict(results[0])
    print(f"LOCKED {locked['name']} ic={locked['ic']:.6f} { {k: locked[k] for k in locked if k != 'ic'} }")

    out_json = ROOT / "outputs/fusion_gru_cov_lock.json"
    out_json.write_text(
        json.dumps({"best": locked, "leaderboard": results[:12], "n_candidates": len(results)}, indent=2),
        encoding="utf-8",
    )

    if locked["ic"] <= BASE_GRU_BLEND + 1e-4:
        print(f"gate did not beat {BASE_GRU_BLEND:.6f}; not writing a new submission")
        print(f"total {time.perf_counter() - t0:.1f}s")
        print("VALID_RANKIC", f"{locked['ic']:.6f}")
        return

    valid_pred = coverage_gate_blend(
        gru_v, tree_v, valid["mask_x"], locked["tau"], locked["weight_low"], locked["weight_high"], locked["space"]
    )
    test_pred = coverage_gate_blend(
        gru_t, tree_t, test["mask_x"], locked["tau"], locked["weight_low"], locked["weight_high"], locked["space"]
    )
    np.save(ROOT / "outputs/fusion_gru_cov_valid.npy", valid_pred)
    np.save(ROOT / "outputs/fusion_gru_cov_test.npy", test_pred)
    save_submission(test_pred, ROOT / "submissions/task1_fusion_gru_cov.npy")
    print("wrote submissions/task1_fusion_gru_cov.npy (did not overwrite 0.110/0.111)")

    cs_v_path = ROOT / "outputs/cs_mlp_valid.npy"
    if cs_v_path.is_file():
        cs_v = np.load(cs_v_path)
        cs_t = np.load(ROOT / "outputs/cs_mlp_test.npy")
        blender = FusionModel({"weight_grid": [0.0, 0.15, 0.25, 0.4, 0.5, 0.7, 1.0]})
        stacked = blender.fit(cs_v, valid_pred, valid["y1"], valid["mask_y"], valid["mask_x"])
        print("stack cs_mlp on gated GRU:")
        for row in (stacked.get("valid_leaderboard") or [])[:5]:
            extra = {k: v for k, v in row.items() if k not in {"valid_leaderboard", "name", "ic"}}
            print(f"  {row['ic']:.6f}  {row['name']}  {extra}")
        print(f"STACK_LOCKED {stacked['name']} ic={stacked['ic']:.6f}")
        if float(stacked["ic"]) > BASE_CS_MLP + 1e-4:
            out_t = blender.predict(cs_t, test_pred, test["mask_x"])
            np.save(ROOT / "outputs/fusion_cs_mlp_cov_valid.npy", blender.predict(cs_v, valid_pred, valid["mask_x"]))
            np.save(ROOT / "outputs/fusion_cs_mlp_cov_test.npy", out_t)
            save_submission(out_t, ROOT / "submissions/task1_fusion_cs_mlp_cov.npy")
            print("wrote submissions/task1_fusion_cs_mlp_cov.npy (did not overwrite 0.111)")
        else:
            print(f"cs_mlp stack did not beat {BASE_CS_MLP:.6f}")

    print(f"total {time.perf_counter() - t0:.1f}s")
    print("VALID_RANKIC", f"{locked['ic']:.6f}")


if __name__ == "__main__":
    main()
