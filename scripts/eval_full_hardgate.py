"""Swap full-universe GRU into the frozen n2000 hard-gate stack.

Does not retrain trees / cs_mlp and does not overwrite n2000 artifacts.
Recomputes the 0.117177 recipe on n2000 GRU, then on the new GRU, then a
small valid-only gate grid.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\eval_full_hardgate.py --config configs/fusion_full_hardgate.yaml
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ablation_log import maybe_update_leader, render_block, upsert_experiment  # noqa: E402
from src.config import load_config, resolve_data_path  # noqa: E402
from src.dataset import load_eval_splits, split_cache_dir  # noqa: E402
from src.metrics import mean_rank_ic  # noqa: E402
from src.models.fusion import n2000_hard_gate_mlp  # noqa: E402
from src.submit import save_submission  # noqa: E402


def _load(path: Path) -> np.ndarray:
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    return np.load(path)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _gate(
    gru: np.ndarray,
    tree: np.ndarray,
    mlp: np.ndarray,
    mask: np.ndarray,
    tau: float,
    w_low: float,
    w_high: float,
    w_mlp: float,
    space: str,
    mlp_space: str,
) -> np.ndarray:
    return n2000_hard_gate_mlp(gru, tree, mlp, mask, tau, w_low, w_high, w_mlp, space, mlp_space)


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-universe GRU + frozen hard gate")
    parser.add_argument("--config", default="configs/fusion_full_hardgate.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--skip-log", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp_id = str(cfg.get("experiment_id") or "gru-no-today-recent-full-001")
    inputs = cfg.get("inputs") or {}
    paths = cfg.get("paths") or {}
    base = cfg.get("baseline") or {}
    search = cfg.get("search") or {}
    documented = float(base.get("documented_valid_rankic", 0.117177))
    space = str(base.get("space", "raw"))
    mlp_space = str(base.get("mlp_space", "rank"))
    w_low0 = float(base.get("weight_low", 0.25))
    w_high0 = float(base.get("weight_high", 0.4))
    w_mlp0 = float(base.get("mlp_weight", 0.15))
    tau_q0 = float(base.get("tau_q", 0.6))

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
    valid, test = splits["valid"], splits["test"]
    yv, mv, mx_v, mx_t = valid["y1"], valid["mask_y"], valid["mask_x"], test["mask_x"]
    n_x = np.asarray(mx_v).sum(axis=1).astype(np.float64)
    tau0 = float(np.quantile(n_x, tau_q0))

    gru_new_v = _load(ROOT / str(inputs["temporal_valid"]))
    gru_new_t = _load(ROOT / str(inputs["temporal_test"]))
    tree_v = _load(ROOT / str(inputs["cross_section_valid"]))
    tree_t = _load(ROOT / str(inputs["cross_section_test"]))
    mlp_v = _load(ROOT / str(inputs["mlp_valid"]))
    mlp_t = _load(ROOT / str(inputs["mlp_test"]))
    gru_ref_v = _load(ROOT / str(inputs["ref_temporal_valid"]))

    ref_pred = _gate(gru_ref_v, tree_v, mlp_v, mx_v, tau0, w_low0, w_high0, w_mlp0, space, mlp_space)
    ref_ic = float(mean_rank_ic(ref_pred, yv, mv))
    print(f"loaded {src}; n2000 frozen recipe RankIC={ref_ic:.6f} (documented {documented:.6f})")
    if abs(ref_ic - documented) > 0.005:
        print("warning: n2000 复算偏差大，检查树 / cs_mlp / n2000 GRU 是否被覆盖")

    frozen_new = _gate(gru_new_v, tree_v, mlp_v, mx_v, tau0, w_low0, w_high0, w_mlp0, space, mlp_space)
    frozen_ic = float(mean_rank_ic(frozen_new, yv, mv))
    print(f"full GRU + frozen gate RankIC={frozen_ic:.6f}  delta={frozen_ic - ref_ic:+.6f}")

    best = {
        "ic": frozen_ic,
        "tau_q": tau_q0,
        "tau": tau0,
        "weight_low": w_low0,
        "weight_high": w_high0,
        "mlp_weight": w_mlp0,
        "note": "frozen n2000 recipe on full GRU",
    }
    n_cand = 0
    for q, lo, hi, wm in itertools.product(
        [float(x) for x in (search.get("tau_qs") or [tau_q0])],
        [float(x) for x in (search.get("weight_lows") or [w_low0])],
        [float(x) for x in (search.get("weight_highs") or [w_high0])],
        [float(x) for x in (search.get("mlp_weights") or [w_mlp0])],
    ):
        if hi < lo:
            continue
        tau = float(np.quantile(n_x, q))
        pred = _gate(gru_new_v, tree_v, mlp_v, mx_v, tau, lo, hi, wm, space, mlp_space)
        ic = float(mean_rank_ic(pred, yv, mv))
        n_cand += 1
        if ic > best["ic"]:
            best = {
                "ic": ic,
                "tau_q": q,
                "tau": tau,
                "weight_low": lo,
                "weight_high": hi,
                "mlp_weight": wm,
                "note": "valid grid",
            }
    print(f"best valid RankIC={best['ic']:.6f} over {n_cand} candidates  spec={best}")

    tau = float(best["tau"])
    valid_pred = _gate(
        gru_new_v,
        tree_v,
        mlp_v,
        mx_v,
        tau,
        float(best["weight_low"]),
        float(best["weight_high"]),
        float(best["mlp_weight"]),
        space,
        mlp_space,
    )
    test_pred = _gate(
        gru_new_t,
        tree_t,
        mlp_t,
        mx_t,
        tau,
        float(best["weight_low"]),
        float(best["weight_high"]),
        float(best["mlp_weight"]),
        space,
        mlp_space,
    )
    confirm = float(mean_rank_ic(valid_pred, yv, mv))
    print(f"confirm valid RankIC={confirm:.6f}")

    valid_path = ROOT / paths.get("valid_pred", "outputs/fusion_full_hardgate_valid.npy")
    test_path = ROOT / paths.get("test_pred", "outputs/fusion_full_hardgate_test.npy")
    sub_path = ROOT / paths.get("submission", "submissions/task1_fusion_full_hardgate.npy")
    lock_path = ROOT / paths.get("lock", "outputs/fusion_full_hardgate_lock.json")
    valid_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(valid_path, valid_pred)
    np.save(test_path, test_pred)
    save_submission(test_pred, sub_path)
    print(f"wrote {sub_path}")

    elapsed = time.perf_counter() - t0
    payload = {
        "experiment_id": exp_id,
        "n2000_recomputed": ref_ic,
        "full_frozen_gate": frozen_ic,
        "best_valid_rankic": confirm,
        "delta_vs_n2000": float(confirm - ref_ic),
        "locked": _jsonable(best),
        "does_not_overwrite": [
            "configs/gru_no_today_recent_n2000.yaml",
            "submissions/task1_fusion_recent_gru_n2000_cov_mlp.npy",
            "submissions/task1_y1_upload.npy",
        ],
    }
    lock_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {lock_path}")

    if confirm > documented + 1e-4:
        conclusion = (
            f"**作为新 GRU 支保留。** valid {confirm:.6f}，对照冻结 n2000 门控 {ref_ic:.6f}。"
            "未覆盖 n2000 提交文件。"
        )
        next_step = "若高于上一主候选，用 submissions/task1_fusion_full_hardgate.npy 再交一次。"
    else:
        conclusion = (
            f"**未超过 n2000 硬门控。** valid {confirm:.6f} vs {ref_ic:.6f}。"
            "全市场抽样这一刀不够，下一刀看崩日而不是再加密网格。"
        )
        next_step = "保留 n2000 主提交；诊断高覆盖日 / 负 RankIC 日。"

    if not args.skip_log:
        block = render_block(
            exp_id,
            config="configs/gru_full_submit_e1.yaml + configs/fusion_full_hardgate.yaml",
            inputs="全市场 GRU（末 800 天、不含当天）+ 冻结 fusion 树 + 原 cs_mlp",
            history="窗口 = 10，[t-L, t)",
            model="同 n2000 GRU；每天训练股票 = 当日全部有标签票；覆盖度硬门控 + rank MLP",
            loss="MSE；验收 RankIC",
            valid_ic=confirm,
            baseline_ic=documented,
            recomputed_baseline_ic=ref_ic,
            delta=float(confirm - ref_ic),
            ablations=[
                {"name": "n2000_frozen_recipe", "ic": ref_ic, "note": "对照"},
                {"name": "full_frozen_recipe", "ic": frozen_ic, "note": "只换 GRU"},
                {"name": "full_valid_grid", "ic": confirm, "note": str(best.get("note"))},
            ],
            locked=_jsonable(best),
            elapsed_s=elapsed,
            conclusion=conclusion,
            next_step=next_step,
        )
        dest = upsert_experiment(exp_id, block)
        maybe_update_leader(confirm, exp_id)
        print(f"wrote {dest}")

    print(f"total {elapsed:.1f}s")
    print("REF_RANKIC", f"{ref_ic:.6f}")
    print("VALID_RANKIC", f"{confirm:.6f}")


if __name__ == "__main__":
    main()
