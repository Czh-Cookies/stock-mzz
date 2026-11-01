"""Soft coverage + stock-history gate on frozen n2000 GRU / tree / cs_mlp panels.

Does not retrain the baseline and does not overwrite n2000 checkpoints or the 0.117 submission.
Recomputes the documented hard-gate recipe first, then searches the new gate.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\eval_softgate_lift.py --config configs/fusion_n2000_softgate.yaml
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
from src.dataset import industry_panel, load_eval_splits, load_split_cache, split_cache_dir  # noqa: E402
from src.metrics import mean_rank_ic  # noqa: E402
from src.models.fusion import (  # noqa: E402
    coverage_stock_three_way,
    linear_blend,
    n2000_hard_gate_mlp,
    prepare_branch,
    stack_mlp_blend,
    stock_history_coverage,
)
from src.submit import save_submission  # noqa: E402

REQUIRED_VALID = (
    ("temporal_valid", "n2000 GRU valid"),
    ("cross_section_valid", "tree fusion valid"),
    ("mlp_valid", "cs_mlp valid"),
)
REQUIRED_TEST = (
    ("temporal_test", "n2000 GRU test"),
    ("cross_section_test", "tree fusion test"),
    ("mlp_test", "cs_mlp test"),
)


def _load_npy(path: Path) -> np.ndarray:
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
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _concat_mask(cache: Path) -> tuple[np.ndarray, int] | tuple[None, None]:
    parts: list[np.ndarray] = []
    origin = None
    for name in ("train", "valid", "test"):
        if not (cache / f"{name}.npz").is_file():
            return None, None
        sl = load_split_cache(name, cache)
        if origin is None:
            origin = int(sl["start"])
        parts.append(np.asarray(sl["mask_x"], dtype=bool))
    return np.concatenate(parts, axis=0), int(origin or 0)


def _apply_spec(
    gru: np.ndarray,
    tree: np.ndarray,
    mlp: np.ndarray,
    mask_x: np.ndarray,
    industry: np.ndarray,
    stock_cov: np.ndarray | None,
    spec: dict[str, Any],
) -> np.ndarray:
    space = str(spec.get("space", "raw"))
    group_z = bool(spec.get("group_z", False))
    g = prepare_branch(gru, mask_x, space, industry, group_z)
    t = prepare_branch(tree, mask_x, space, industry, group_z)
    gated = coverage_stock_three_way(
        g,
        t,
        None,
        mask_x,
        tau=float(spec["tau"]),
        temperature=float(spec.get("temperature", 1.0)),
        weight_low=float(spec["weight_low"]),
        weight_high=float(spec["weight_high"]),
        stock_cov=stock_cov,
        stock_boost=float(spec.get("stock_boost", 0.0)),
        stock_center=float(spec.get("stock_center", 0.5)),
        stock_temp=float(spec.get("stock_temp", 0.15)),
        w_mlp=0.0,
        hard=bool(spec.get("hard", False)),
    )
    w_m = float(spec.get("w_mlp", 0.0))
    if w_m == 0.0:
        return gated
    return stack_mlp_blend(mlp, gated, mask_x, w_m, str(spec.get("mlp_space", "rank")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Soft coverage gate lift over n2000 baseline")
    parser.add_argument("--config", default="configs/fusion_n2000_softgate.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--skip-log", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp_id = str(cfg.get("experiment_id") or "fusion-n2000-softgate-001")
    inputs = cfg.get("inputs") or {}
    paths = cfg.get("paths") or {}
    base = cfg.get("baseline") or {}
    gate = cfg.get("gate") or {}
    industry_col = int(inputs.get("industry_col", 6))
    documented = float(base.get("documented_valid_rankic", 0.117177))
    target = float(base.get("target_valid_rankic", 0.12))

    t0 = time.perf_counter()
    missing = [
        f"{label} ({inputs.get(key)})"
        for key, label in REQUIRED_VALID + REQUIRED_TEST
        if not (ROOT / str(inputs.get(key, ""))).is_file()
    ]
    if missing:
        raise SystemExit(
            "缺少预测文件，先按基准栈训完且不要覆盖 n2000 产物：\n  - "
            + "\n  - ".join(missing)
            + "\nGRU: .\\.venv\\Scripts\\python.exe scripts\\train_gru.py --config configs/gru_no_today_recent_n2000.yaml"
            + "\n树融合: configs/baseline.yaml → hist_lgbm.yaml → fusion.yaml"
            + "\nMLP: .\\.venv\\Scripts\\python.exe scripts\\train_cs_mlp.py --config configs/cs_mlp.yaml"
        )

    cache_dir = split_cache_dir(ROOT)
    data_path = None
    need = ("valid", "test")
    if not all((cache_dir / f"{name}.npz").is_file() for name in need):
        data_path = resolve_data_path(cfg, args.data)
    splits, src = load_eval_splits(
        cache_dir=cache_dir,
        data_path=data_path,
        industry_col=industry_col,
        splits=need,
    )
    valid = splits["valid"]
    test = splits["test"]
    print(f"loaded {src} labels/masks")

    gru_v = _load_npy(ROOT / inputs["temporal_valid"])
    tree_v = _load_npy(ROOT / inputs["cross_section_valid"])
    mlp_v = _load_npy(ROOT / inputs["mlp_valid"])
    gru_t = _load_npy(ROOT / inputs["temporal_test"])
    tree_t = _load_npy(ROOT / inputs["cross_section_test"])
    mlp_t = _load_npy(ROOT / inputs["mlp_test"])
    aux_v = _maybe(ROOT / str(inputs.get("aux_temporal_valid") or ""))
    aux_t = _maybe(ROOT / str(inputs.get("aux_temporal_test") or ""))

    yv, mv = valid["y1"], valid["mask_y"]
    mx_v, mx_t = valid["mask_x"], test["mask_x"]
    ind_v = industry_panel(valid, industry_col)
    ind_t = industry_panel(test, industry_col)
    n_x = np.asarray(mx_v).sum(axis=1).astype(np.float64)

    lookback = int(gate.get("lookback", 10))
    panel_mask, origin = _concat_mask(cache_dir)
    if panel_mask is not None and origin is not None:
        cov_v = stock_history_coverage(panel_mask, origin, int(valid["start"]), mx_v.shape[0], lookback)
        cov_t = stock_history_coverage(panel_mask, origin, int(test["start"]), mx_t.shape[0], lookback)
        stock_center = float(np.nanmedian(cov_v))
    else:
        cov_v = None
        cov_t = None
        stock_center = 0.5
        print("split_cache 缺 train.npz，个股历史覆盖加权关闭")

    tau_base = float(np.quantile(n_x, float(base.get("tau_q", 0.6))))
    base_pred = n2000_hard_gate_mlp(
        gru_v,
        tree_v,
        mlp_v,
        mx_v,
        tau_base,
        float(base.get("weight_low", 0.25)),
        float(base.get("weight_high", 0.4)),
        float(base.get("mlp_weight", 0.15)),
        str(base.get("space", "raw")),
        str(base.get("mlp_space", "rank")),
    )
    base_ic = float(mean_rank_ic(base_pred, yv, mv))
    print(f"recomputed baseline RankIC={base_ic:.6f} (documented {documented:.6f})")
    if abs(base_ic - documented) > 0.005:
        print("warning: 复算与文档 0.117177 偏差较大，检查是否用了 n2000 GRU / fusion / 原 cs_mlp")

    ablations: list[dict[str, Any]] = [
        {"name": "A0_hard_day_gate_mlp", "ic": base_ic, "note": "复算原硬门控+rank MLP 0.15"}
    ]

    spaces = [str(s) for s in (gate.get("spaces") or ["raw"])]
    tau_qs = [float(x) for x in (gate.get("tau_qs") or [0.6])]
    temps = [float(x) for x in (gate.get("temperatures") or [200.0])]
    pairs = [tuple(float(x) for x in p) for p in (gate.get("weight_pairs") or [[0.25, 0.4]])]
    boosts = [float(x) for x in (gate.get("stock_boosts") or [0.0])]
    stock_temps = [float(x) for x in (gate.get("stock_temps") or [0.15])]
    w_mlps = [float(x) for x in (gate.get("mlp_weights") or [0.15])]
    group_flags = [bool(x) for x in (gate.get("group_z") or [False])]
    aux_mixes = [float(x) for x in (gate.get("aux_mix") or [0.0])]

    def score_spec(name: str, spec: dict[str, Any], gru_use: np.ndarray) -> dict[str, Any]:
        pred = _apply_spec(gru_use, tree_v, mlp_v, mx_v, ind_v, cov_v, spec)
        ic = float(mean_rank_ic(pred, yv, mv))
        return {"name": name, "ic": ic, **spec}

    a1_best: dict[str, Any] = {"ic": -1e9}
    for space, q, temp, pair in itertools.product(spaces, tau_qs, temps, pairs):
        spec = {
            "space": space,
            "group_z": False,
            "tau": float(np.quantile(n_x, q)),
            "tau_q": q,
            "temperature": temp,
            "weight_low": pair[0],
            "weight_high": pair[1],
            "stock_boost": 0.0,
            "w_mlp": float(base.get("mlp_weight", 0.15)),
            "mlp_space": str(base.get("mlp_space", "rank")),
            "hard": False,
        }
        row = score_spec("A1", spec, gru_v)
        if row["ic"] > a1_best["ic"]:
            a1_best = row
    ablations.append(
        {
            "name": "A1_soft_day_gate",
            "ic": a1_best["ic"],
            "note": f"tau_q={a1_best.get('tau_q')} T={a1_best.get('temperature')}",
        }
    )

    a2_best: dict[str, Any] = {"ic": -1e9}
    for boost, st in itertools.product(boosts, stock_temps):
        spec = {
            "space": str(base.get("space", "raw")),
            "group_z": False,
            "tau": tau_base,
            "tau_q": float(base.get("tau_q", 0.6)),
            "temperature": 1.0,
            "weight_low": float(base.get("weight_low", 0.25)),
            "weight_high": float(base.get("weight_high", 0.4)),
            "stock_boost": boost,
            "stock_center": stock_center,
            "stock_temp": st,
            "w_mlp": float(base.get("mlp_weight", 0.15)),
            "mlp_space": str(base.get("mlp_space", "rank")),
            "hard": True,
        }
        row = score_spec("A2", spec, gru_v)
        if row["ic"] > a2_best["ic"]:
            a2_best = row
    ablations.append(
        {
            "name": "A2_hard_day_plus_stock",
            "ic": a2_best["ic"],
            "note": f"boost={a2_best.get('stock_boost')}",
        }
    )

    best: dict[str, Any] = {"ic": -1e9}
    n_cand = 0
    use_boosts = boosts if cov_v is not None else [0.0]
    for space, group_z, q, temp, pair, boost, st, w_m in itertools.product(
        spaces, group_flags, tau_qs, temps, pairs, use_boosts, stock_temps, w_mlps
    ):
        spec = {
            "space": space,
            "group_z": group_z,
            "tau": float(np.quantile(n_x, q)),
            "tau_q": q,
            "temperature": temp,
            "weight_low": pair[0],
            "weight_high": pair[1],
            "stock_boost": boost if cov_v is not None else 0.0,
            "stock_center": stock_center,
            "stock_temp": st,
            "w_mlp": w_m,
            "mlp_space": str(base.get("mlp_space", "rank")),
            "hard": False,
            "aux_mix": 0.0,
        }
        row = score_spec("search", spec, gru_v)
        n_cand += 1
        if row["ic"] > best["ic"]:
            best = row
    ablations.append({"name": "A3_soft_stock_mlp_grid", "ic": best["ic"], "note": f"{n_cand} candidates"})

    if aux_v is not None:
        a_aux: dict[str, Any] = {"ic": -1e9}
        lock_body = {k: v for k, v in best.items() if k not in {"name", "ic"}}
        for mix in aux_mixes:
            if mix <= 0:
                continue
            mixed = linear_blend(aux_v, gru_v, mix)
            row = score_spec("A4_aux", dict(lock_body), mixed)
            row["aux_mix"] = mix
            if row["ic"] > a_aux["ic"]:
                a_aux = row
        if a_aux["ic"] > -1e8:
            ablations.append(
                {
                    "name": "A4_aux_winjitter_mix",
                    "ic": a_aux["ic"],
                    "note": f"mix={a_aux.get('aux_mix')}",
                }
            )
            if a_aux["ic"] > best["ic"]:
                best = a_aux
    else:
        ablations.append({"name": "A4_aux_winjitter_mix", "ic": None, "note": "无 winjitter 预测，跳过"})

    print("ablations:")
    for row in ablations:
        ic = row.get("ic")
        ic_s = f"{ic:.6f}" if isinstance(ic, float) else "skip"
        print(f"  {str(row['name']):28s} {ic_s}  {row.get('note', '')}")

    lock_spec = {k: v for k, v in best.items() if k not in {"name", "ic"}}
    print(f"LOCKED ic={best['ic']:.6f} spec={lock_spec}")

    gru_v_use, gru_t_use = gru_v, gru_t
    mix = float(lock_spec.get("aux_mix", 0.0) or 0.0)
    if mix > 0 and aux_v is not None and aux_t is not None:
        gru_v_use = linear_blend(aux_v, gru_v, mix)
        gru_t_use = linear_blend(aux_t, gru_t, mix)

    valid_pred = _apply_spec(gru_v_use, tree_v, mlp_v, mx_v, ind_v, cov_v, lock_spec)
    test_pred = _apply_spec(gru_t_use, tree_t, mlp_t, mx_t, ind_t, cov_t, lock_spec)
    confirm = float(mean_rank_ic(valid_pred, yv, mv))
    print(f"confirm valid RankIC={confirm:.6f}")

    valid_path = ROOT / paths.get("valid_pred", "outputs/fusion_n2000_softgate_valid.npy")
    test_path = ROOT / paths.get("test_pred", "outputs/fusion_n2000_softgate_test.npy")
    sub_path = ROOT / paths.get("submission", "submissions/task1_fusion_n2000_softgate.npy")
    lock_path = ROOT / paths.get("lock", "outputs/fusion_n2000_softgate_lock.json")
    ab_path = ROOT / paths.get("ablation", "outputs/fusion_n2000_softgate_ablation.json")
    valid_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(valid_path, valid_pred)
    np.save(test_path, test_pred)
    save_submission(test_pred, sub_path)

    payload = {
        "experiment_id": exp_id,
        "documented_baseline": documented,
        "recomputed_baseline": base_ic,
        "new_valid_rankic": confirm,
        "delta": float(confirm - documented),
        "hit_target": bool(confirm >= target),
        "locked": _jsonable(lock_spec),
        "ablations": _jsonable(ablations),
        "n_candidates": n_cand,
        "paths": {
            "valid_pred": str(valid_path.relative_to(ROOT)),
            "test_pred": str(test_path.relative_to(ROOT)),
            "submission": str(sub_path.relative_to(ROOT)),
        },
        "does_not_overwrite": [
            "configs/gru_no_today_recent_n2000.yaml",
            "checkpoints/gru_no_today_recent_n2000.pt",
            "submissions/task1_fusion_recent_gru_n2000_cov_mlp.npy",
        ],
    }
    lock_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    ab_path.write_text(json.dumps(payload["ablations"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {sub_path}")
    print(f"wrote {lock_path}")

    elapsed = time.perf_counter() - t0
    if confirm >= target:
        conclusion = f"**保留为新候选。** valid {confirm:.6f} ≥ {target}。未覆盖 n2000 基线产物。"
        next_step = "测试集只用于最终提交，不要回灌调参。"
    elif confirm > documented + 1e-4:
        conclusion = (
            f"**作为融合规则保留，门槛未到。** valid {confirm:.6f}，对照文档 {documented:.6f}，"
            f"距 {target} 约 {target - confirm:.4f}。"
        )
        next_step = "可再训 configs/gru_n2000_winjitter.yaml 后重跑本脚本做 A4。"
    else:
        conclusion = f"**未超过文档基准。** valid {confirm:.6f} vs {documented:.6f}。硬门控复算 {base_ic:.6f} 仍可用。"
        next_step = "检查预测文件是否为 n2000 GRU；不要改 gru_no_today_recent_n2000.yaml。"

    if not args.skip_log:
        block = render_block(
            exp_id,
            config="configs/fusion_n2000_softgate.yaml + scripts/eval_softgate_lift.py",
            inputs="冻结 n2000 GRU + fusion 树 + 原 cs_mlp；门控用当天 mask_x 与 [t-L,t) 个股上市率",
            history=f"窗口 = {lookback}（仅用于个股历史覆盖，不重训）",
            model="软覆盖度 logistic + 个股历史覆盖加权 + 可选行业分数 z-score；valid 上锁定",
            loss="不训练；验收 RankIC",
            valid_ic=confirm,
            baseline_ic=documented,
            recomputed_baseline_ic=base_ic,
            delta=float(confirm - documented),
            ablations=ablations,
            locked=_jsonable(lock_spec),
            elapsed_s=elapsed,
            conclusion=conclusion,
            next_step=next_step,
        )
        dest = upsert_experiment(exp_id, block)
        maybe_update_leader(confirm, exp_id)
        print(f"wrote {dest}")

    print(f"total {elapsed:.1f}s")
    print("BASELINE_RANKIC", f"{base_ic:.6f}")
    print("VALID_RANKIC", f"{confirm:.6f}")


if __name__ == "__main__":
    main()
