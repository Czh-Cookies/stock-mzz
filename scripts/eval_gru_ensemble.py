"""Mix n2000 / full / seed-43 GRU into the coverage gate. Valid-only search.

Does not overwrite task1_y1_upload.npy or the n2000 yaml.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\eval_gru_ensemble.py --config configs/fusion_gru_ensemble.yaml
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
from src.models.fusion import linear_blend, n2000_hard_gate_mlp  # noqa: E402
from src.submit import save_submission  # noqa: E402


def _load(path: Path) -> np.ndarray:
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    return np.load(path)


def _maybe(path: Path) -> np.ndarray | None:
    return np.load(path) if path.is_file() else None


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _mix(a: np.ndarray, b: np.ndarray | None, w: float) -> np.ndarray:
    if b is None or w <= 0:
        return a
    if w >= 1:
        return b
    return linear_blend(b, a, w)


def _gate(gru, tree, mlp, mask, tau, lo, hi, wm, space, mlp_space):
    return n2000_hard_gate_mlp(gru, tree, mlp, mask, tau, lo, hi, wm, space, mlp_space)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fusion_gru_ensemble.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--skip-log", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp_id = str(cfg.get("experiment_id") or "gru-ensemble-full-s43-001")
    inputs = cfg.get("inputs") or {}
    paths = cfg.get("paths") or {}
    base = cfg.get("baseline") or {}
    search = cfg.get("search") or {}
    documented = float(base.get("documented_valid_rankic", 0.121023))
    n2000_doc = float(base.get("n2000_valid_rankic", 0.117177))
    space = str(base.get("space", "raw"))
    mlp_space = str(base.get("mlp_space", "rank"))

    t0 = time.perf_counter()
    cache_dir = split_cache_dir(ROOT)
    data_path = None
    if not (cache_dir / "valid.npz").is_file():
        data_path = resolve_data_path(cfg, args.data)
    splits, src = load_eval_splits(cache_dir=cache_dir, data_path=data_path, splits=("valid", "test"))
    valid, test = splits["valid"], splits["test"]
    yv, mv, mx_v, mx_t = valid["y1"], valid["mask_y"], valid["mask_x"], test["mask_x"]
    n_x = np.asarray(mx_v).sum(axis=1).astype(np.float64)

    n2000_v = _load(ROOT / str(inputs["n2000_valid"]))
    n2000_t = _load(ROOT / str(inputs["n2000_test"]))
    full_v = _load(ROOT / str(inputs["full_valid"]))
    full_t = _load(ROOT / str(inputs["full_test"]))
    s43_v = _maybe(ROOT / str(inputs.get("s43_valid") or ""))
    s43_t = _maybe(ROOT / str(inputs.get("s43_test") or ""))
    tree_v = _load(ROOT / str(inputs["cross_section_valid"]))
    tree_t = _load(ROOT / str(inputs["cross_section_test"]))
    mlp_v = _load(ROOT / str(inputs["mlp_valid"]))
    mlp_t = _load(ROOT / str(inputs["mlp_test"]))

    tau_ref = float(np.quantile(n_x, 0.6))
    ref = _gate(n2000_v, tree_v, mlp_v, mx_v, tau_ref, 0.25, 0.4, 0.15, space, mlp_space)
    ref_ic = float(mean_rank_ic(ref, yv, mv))
    tau_win = float(np.quantile(n_x, 0.7))
    prev = _gate(full_v, tree_v, mlp_v, mx_v, tau_win, 0.2, 0.6, 0.1, space, mlp_space)
    prev_ic = float(mean_rank_ic(prev, yv, mv))
    print(f"loaded {src}; n2000 recipe {ref_ic:.6f}; full-grid recipe {prev_ic:.6f}")

    best: dict[str, Any] = {"ic": -1e9}
    n_cand = 0
    s43_mixes = [0.0] if s43_v is None else [float(x) for x in (search.get("s43_mix") or [0.0])]
    for fm, sm, q, lo, hi, wm in itertools.product(
        [float(x) for x in (search.get("full_mix") or [1.0])],
        s43_mixes,
        [float(x) for x in (search.get("tau_qs") or [0.7])],
        [float(x) for x in (search.get("weight_lows") or [0.2])],
        [float(x) for x in (search.get("weight_highs") or [0.6])],
        [float(x) for x in (search.get("mlp_weights") or [0.1])],
    ):
        if hi < lo:
            continue
        gru_v = _mix(n2000_v, full_v, fm)
        gru_v = _mix(gru_v, s43_v, sm)
        tau = float(np.quantile(n_x, q))
        pred = _gate(gru_v, tree_v, mlp_v, mx_v, tau, lo, hi, wm, space, mlp_space)
        ic = float(mean_rank_ic(pred, yv, mv))
        n_cand += 1
        if ic > best["ic"]:
            best = {
                "ic": ic,
                "full_mix": fm,
                "s43_mix": sm,
                "tau_q": q,
                "tau": tau,
                "weight_low": lo,
                "weight_high": hi,
                "mlp_weight": wm,
            }
    print(f"best valid RankIC={best['ic']:.6f} over {n_cand} candidates  spec={best}")

    gru_v = _mix(n2000_v, full_v, float(best["full_mix"]))
    gru_v = _mix(gru_v, s43_v, float(best["s43_mix"]))
    gru_t = _mix(n2000_t, full_t, float(best["full_mix"]))
    gru_t = _mix(gru_t, s43_t, float(best["s43_mix"]))
    valid_pred = _gate(
        gru_v, tree_v, mlp_v, mx_v, float(best["tau"]),
        float(best["weight_low"]), float(best["weight_high"]), float(best["mlp_weight"]),
        space, mlp_space,
    )
    test_pred = _gate(
        gru_t, tree_t, mlp_t, mx_t, float(best["tau"]),
        float(best["weight_low"]), float(best["weight_high"]), float(best["mlp_weight"]),
        space, mlp_space,
    )
    confirm = float(mean_rank_ic(valid_pred, yv, mv))
    print(f"confirm valid RankIC={confirm:.6f}")

    valid_path = ROOT / paths.get("valid_pred", "outputs/fusion_gru_ensemble_valid.npy")
    test_path = ROOT / paths.get("test_pred", "outputs/fusion_gru_ensemble_test.npy")
    sub_path = ROOT / paths.get("submission", "submissions/task1_fusion_gru_ensemble.npy")
    lock_path = ROOT / paths.get("lock", "outputs/fusion_gru_ensemble_lock.json")
    valid_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(valid_path, valid_pred)
    np.save(test_path, test_pred)
    save_submission(test_pred, sub_path)
    print(f"wrote {sub_path}")

    elapsed = time.perf_counter() - t0
    lock_path.write_text(
        json.dumps(
            {
                "experiment_id": exp_id,
                "n2000_recipe": ref_ic,
                "full_grid_recipe": prev_ic,
                "best_valid_rankic": confirm,
                "locked": _jsonable(best),
                "does_not_overwrite": [
                    "submissions/task1_y1_upload.npy",
                    "submissions/task1_fusion_full_hardgate.npy",
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if confirm > documented + 1e-4:
        conclusion = f"**作为新候选。** valid {confirm:.6f}，对照上一刀 {documented:.6f} / n2000 门控 {ref_ic:.6f}。"
        next_step = "用 submissions/task1_fusion_gru_ensemble.npy 再交一次；不要覆盖 0.120075 那份备份。"
    else:
        conclusion = f"**未超过上一刀。** valid {confirm:.6f} vs {documented:.6f}。种子/浅融合这一刀不够。"
        next_step = "主提交仍用 full hardgate 或 0.120075 那份；下一刀看崩日。"

    if not args.skip_log:
        block = render_block(
            exp_id,
            config="configs/gru_full_seed43.yaml + configs/fusion_gru_ensemble.yaml",
            inputs="n2000 GRU + 全市场 seed42 + 全市场 seed43 + 冻结树 + 原 cs_mlp",
            history="窗口 = 10，[t-L, t)",
            model="GRU 浅融合后再覆盖度硬门控；valid 锁 mix 与门控",
            loss="不重训树/MLP；验收 RankIC",
            valid_ic=confirm,
            baseline_ic=documented,
            recomputed_baseline_ic=ref_ic,
            delta=float(confirm - documented),
            ablations=[
                {"name": "n2000_frozen", "ic": ref_ic, "note": str(n2000_doc)},
                {"name": "full_grid", "ic": prev_ic, "note": "上一刀"},
                {"name": "ensemble_grid", "ic": confirm, "note": "本刀"},
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
    print("VALID_RANKIC", f"{confirm:.6f}")


if __name__ == "__main__":
    main()
