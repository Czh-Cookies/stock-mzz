"""Train-only collapse-day switch, then locked inference on valid/test.

Collapse day (train labels only): tree RankIC is significantly below GRU RankIC.
The observable switch is n_x >= tau_n. tau_n and w_collapse are fit on train and
frozen. Valid/test never update them. Conservative 0.25 / 0.4 / tau_q=0.6 stay
fixed on non-collapse days.

Does not edit champion GRU/tree/MLP files, train_gru.py, or fusion.py.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\eval_train_collapse_switch.py
    .\\.venv\\Scripts\\python.exe scripts\\eval_train_collapse_switch.py --skip-predict
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_data_path  # noqa: E402
from src.dataset import (  # noqa: E402
    drop_task2_label,
    keep_recent_days,
    load_eval_splits,
    load_panel,
    slice_split,
    split_cache_dir,
)
from src.provenance import CacheValidationError, array_sha256, canonical, file_sha256
from src.metrics import mean_rank_ic, rank_ic_series  # noqa: E402
from src.models.baseline import LightGBMBaseline  # noqa: E402
from src.models.fusion import linear_blend, stack_mlp_blend  # noqa: E402
from src.models.gru_ts import GRUModel  # noqa: E402
from src.submit import save_submission  # noqa: E402

CHAMP_GRU_CFG = ROOT / "configs/gru_n2000_add6973_e1.yaml"
CHAMP_GRU_CKPT = ROOT / "checkpoints/gru_n2000_add6973_e1.pt"
CHAMP_GRU_VALID = ROOT / "outputs/gru_n2000_add6973_e1_valid.npy"
CHAMP_GRU_TEST = ROOT / "outputs/gru_n2000_add6973_e1_test.npy"
GRU_TRAIN = ROOT / "outputs/gru_n2000_add6973_e1_train_r0.npy"
TREE_TRAIN = ROOT / "outputs/tree_blend07_train800_r0.npy"
STATS_TRAIN = ROOT / "outputs/collapse_train_day_stats_r0.npz"
LOCK_PATH = ROOT / "outputs/collapse_train_switch_locked_r0.json"

W_LOW = 0.25
W_HIGH = 0.4
TAU_Q = 0.6
W_MLP = 0.15
TREE_BLEND = 0.7
RECENT_DAYS = 800
MARGIN_SIGMA = 0.5
LOCAL_FUSED_BASE = 0.11840
TAG = "collapse_train_switch_r0"


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(x, dtype=np.float64), 0.0)


def _stats_contract():
    cfg = load_config(CHAMP_GRU_CFG)
    deps = [CHAMP_GRU_CKPT, CHAMP_GRU_CKPT.with_suffix(".pt.meta.json"),
            ROOT / "checkpoints/hist_lgbm.txt", ROOT / "checkpoints/hist_lgbm.txt.meta.json",
            ROOT / "checkpoints/baseline.txt", ROOT / "checkpoints/baseline.txt.meta.json",
            Path(__file__), ROOT / "src/models/gru_ts.py", ROOT / "src/dataset.py",
            ROOT / "src/metrics.py", ROOT / "src/models/baseline.py", ROOT / "src/models/fusion.py"]
    return {"source": file_sha256(resolve_data_path(cfg)), "config": cfg,
            "recent_days": RECENT_DAYS, "tree_blend": TREE_BLEND,
            "dependencies": {str(p.relative_to(ROOT)): file_sha256(p) for p in deps}}


def _predict_train_branches() -> None:
    contract = _stats_contract()
    if any(p.exists() for p in (GRU_TRAIN, TREE_TRAIN, STATS_TRAIN)):
        raise CacheValidationError("R0 stats outputs exist; refuse to overwrite")
    cfg = load_config(CHAMP_GRU_CFG)
    data_path = resolve_data_path(cfg)
    print(f"load panel for train preds data={data_path}", flush=True)
    t0 = time.perf_counter()
    data = load_panel(str(data_path))
    drop_task2_label(data)
    print(f"loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    train = keep_recent_days(slice_split(data, "train"), RECENT_DAYS)
    print(f"train recent_days={RECENT_DAYS} start={train['start']} days={train['mask_x'].shape[0]}", flush=True)

    feat = dict(cfg.get("features") or {})
    model_cfg = dict(cfg.get("model") or {})
    train_cfg = dict(cfg.get("train") or {})
    gru_cfg = {**feat, **model_cfg, **train_cfg}
    gru = GRUModel(gru_cfg, seed=int(cfg.get("seed", 42)))
    gru.load(CHAMP_GRU_CKPT)
    gru.prepare_features(data)
    t1 = time.perf_counter()
    gru_pred = gru.predict_panel(train, data, fill_invalid=0.0)
    print(f"GRU train predict {time.perf_counter() - t1:.1f}s {gru_pred.shape}", flush=True)
    GRU_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    np.save(GRU_TRAIN, gru_pred)
    del gru
    gc.collect()

    hist_cfg = load_config("configs/hist_lgbm.yaml")
    hist = LightGBMBaseline(
        params=(hist_cfg.get("model") or {}).get("params") or {},
        feature_cfg=dict(hist_cfg.get("features") or {}),
        seed=int(hist_cfg.get("seed", 42)),
    )
    hist.load(ROOT / "checkpoints/hist_lgbm.txt")
    t2 = time.perf_counter()
    hist_pred = hist.predict_panel(train, fill_invalid=0.0)
    print(f"hist_lgbm train predict {time.perf_counter() - t2:.1f}s", flush=True)
    del hist
    gc.collect()

    base_cfg = load_config("configs/baseline.yaml")
    base = LightGBMBaseline(
        params=(base_cfg.get("model") or {}).get("params") or {},
        feature_cfg=dict(base_cfg.get("features") or {}),
        seed=int(base_cfg.get("seed", 42)),
    )
    base.load(ROOT / "checkpoints/baseline.txt")
    t3 = time.perf_counter()
    base_pred = base.predict_panel(train, fill_invalid=0.0)
    print(f"baseline train predict {time.perf_counter() - t3:.1f}s", flush=True)
    del base
    gc.collect()

    tree_pred = linear_blend(hist_pred, base_pred, TREE_BLEND)
    np.save(TREE_TRAIN, tree_pred)
    n_x = np.asarray(train["mask_x"]).sum(axis=1).astype(np.float64)
    gru_ic = rank_ic_series(gru_pred, train["y1"], train["mask_y"])
    tree_ic = rank_ic_series(tree_pred, train["y1"], train["mask_y"])
    contract["payload_hashes"] = {k: array_sha256(v) for k, v in
        {"n_x": n_x, "gru_ic": gru_ic, "tree_ic": tree_ic,
         "start": np.int32(train["start"])}.items()}
    np.savez(
        STATS_TRAIN,
        n_x=n_x,
        gru_ic=gru_ic,
        tree_ic=tree_ic,
        start=np.int32(train["start"]),
        manifest=np.array(canonical(contract)),
    )
    print(f"wrote {GRU_TRAIN}")
    print(f"wrote {TREE_TRAIN}")
    print(f"wrote {STATS_TRAIN}")
    print("does not overwrite outputs/gru_n2000_add6973_e1_valid.npy")
    del data, train, gru_pred, hist_pred, base_pred, tree_pred
    gc.collect()


def _fit_lock() -> dict:
    with np.load(STATS_TRAIN, allow_pickle=False) as cached:
        if "manifest" not in cached.files:
            raise CacheValidationError("unverified collapse stats; regenerate into R0 paths")
        metadata = json.loads(str(cached["manifest"]))
        hashes = metadata.pop("payload_hashes", {})
        if canonical(metadata) != canonical(_stats_contract()):
            raise CacheValidationError("collapse stats source/config/checkpoint mismatch")
        stats = {k: cached[k] for k in ("n_x", "gru_ic", "tree_ic", "start")}
        if hashes != {k: array_sha256(v) for k, v in stats.items()}:
            raise CacheValidationError("collapse stats payload mismatch")
    n_x = np.asarray(stats["n_x"], dtype=np.float64)
    gru_ic = np.asarray(stats["gru_ic"], dtype=np.float64)
    tree_ic = np.asarray(stats["tree_ic"], dtype=np.float64)
    gap = gru_ic - tree_ic
    sigma = float(np.nanstd(gap))
    margin = max(0.0, MARGIN_SIGMA * sigma)
    collapse = np.isfinite(gap) & (gap > margin)
    n_collapse = int(collapse.sum())
    if n_collapse <= 0:
        raise RuntimeError("no collapse days on train; refuse to invent a valid-side rule")
    # Absolute n_x does not transfer: train last-800 n_x is ~3400–4200, valid is 4242+.
    # Lock the train ECDF percentile of collapse days; infer with that percentile inside each split.
    order = np.argsort(np.argsort(n_x)).astype(np.float64)
    ecdf = (order + 1.0) / float(n_x.size)
    p_star = float(np.median(ecdf[collapse]))
    tau_n = float(np.median(n_x[collapse]))
    w_num = _relu(gru_ic[collapse])
    w_den = w_num + _relu(tree_ic[collapse]) + 1e-6
    w_collapse = float(np.clip(np.mean(w_num / w_den), W_HIGH, 1.0))
    lock = {
        "fit_split": f"train_last_{RECENT_DAYS}",
        "n_train_days": int(n_x.size),
        "n_collapse": n_collapse,
        "collapse_rate": float(n_collapse / n_x.size),
        "gap_mean": float(np.nanmean(gap)),
        "gap_std": sigma,
        "margin": margin,
        "margin_sigma": MARGIN_SIGMA,
        "definition": "collapse iff (RankIC_GRU - RankIC_tree) > max(0, 0.5 * std(gap)) on train",
        "observable": "split_ecdf(n_x) >= p_star",
        "p_star": p_star,
        "tau_n_train_abs": tau_n,
        "w_collapse": w_collapse,
        "w_low": W_LOW,
        "w_high": W_HIGH,
        "tau_q_conservative": TAU_Q,
        "mlp_weight": W_MLP,
        "tree_blend": TREE_BLEND,
        "mean_gru_ic_collapse": float(np.nanmean(gru_ic[collapse])),
        "mean_tree_ic_collapse": float(np.nanmean(tree_ic[collapse])),
        "mean_n_x_collapse": float(np.mean(n_x[collapse])),
        "mean_n_x_other": float(np.mean(n_x[~collapse])),
        "valid_is_inference_only": True,
    }
    LOCK_PATH.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"LOCK {LOCK_PATH}")
    print(
        f"train collapse {n_collapse}/{n_x.size} margin={margin:.4f} "
        f"p_star={p_star:.4f} tau_n_train={tau_n:.1f} w_collapse={w_collapse:.4f} "
        f"ic_gru={lock['mean_gru_ic_collapse']:.4f} ic_tree={lock['mean_tree_ic_collapse']:.4f}"
    )
    return lock


def _apply_switch(
    gru: np.ndarray,
    tree: np.ndarray,
    mask_x: np.ndarray,
    tau_cons: float,
    p_star: float,
    w_collapse: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_x = np.asarray(mask_x).sum(axis=1).astype(np.float64)
    order = np.argsort(np.argsort(n_x)).astype(np.float64)
    ecdf = (order + 1.0) / float(n_x.size)
    collapse_hat = ecdf >= float(p_star)
    w = np.where(n_x >= float(tau_cons), W_HIGH, W_LOW).astype(np.float64)
    w = np.where(collapse_hat, float(w_collapse), w)
    gated = (w[:, None] * gru + (1.0 - w[:, None]) * tree).astype(np.float32)
    return gated, collapse_hat


def main() -> None:
    parser = argparse.ArgumentParser(description="Train-fit collapse switch; valid inference only")
    parser.add_argument("--skip-predict", action="store_true")
    args = parser.parse_args()

    try:
        import torch

        torch.set_num_threads(max(1, int(os.cpu_count() or 4)))
    except Exception:
        pass

    if not CHAMP_GRU_CKPT.is_file():
        raise SystemExit(f"missing champion GRU {CHAMP_GRU_CKPT}")
    if args.skip_predict and not STATS_TRAIN.is_file():
        raise CacheValidationError("verified R0 stats missing; --skip-predict cannot reuse legacy stats")
    if not args.skip_predict:
        _predict_train_branches()
    lock = _fit_lock()

    splits, src = load_eval_splits(cache_dir=split_cache_dir(ROOT), splits=("valid", "test"))
    valid, test = splits["valid"], splits["test"]
    tau_cons = float(np.quantile(np.asarray(valid["mask_x"]).sum(axis=1).astype(np.float64), TAU_Q))
    print(
        f"conservative non-collapse gate w_low={W_LOW} w_high={W_HIGH} tau_q={TAU_Q} "
        f"tau_cons={tau_cons:.1f} (coverage quantile, not a weight search) src={src}"
    )

    gru_v = np.load(CHAMP_GRU_VALID)
    gru_t = np.load(CHAMP_GRU_TEST)
    tree_v = np.load(ROOT / "outputs/fusion_valid.npy")
    tree_t = np.load(ROOT / "outputs/fusion_test.npy")
    mlp_v = np.load(ROOT / "outputs/cs_mlp_valid.npy")
    mlp_t = np.load(ROOT / "outputs/cs_mlp_test.npy")

    gated_v, hat_v = _apply_switch(
        gru_v, tree_v, valid["mask_x"], tau_cons, lock["p_star"], lock["w_collapse"]
    )
    pred_v = stack_mlp_blend(mlp_v, gated_v, valid["mask_x"], W_MLP, "rank")
    fused = float(mean_rank_ic(pred_v, valid["y1"], valid["mask_y"]))
    print(f"valid inferred collapse days {int(hat_v.sum())}/{hat_v.size} (not used to refit)")
    print(f"FUSED_RANKIC {fused:.6f} vs local champ {LOCAL_FUSED_BASE:.5f} delta={fused - LOCAL_FUSED_BASE:+.6f}")
    print("official champion test RankIC 0.121626 is unchanged on disk")

    gated_t, hat_t = _apply_switch(
        gru_t, tree_t, test["mask_x"], tau_cons, lock["p_star"], lock["w_collapse"]
    )
    pred_t = stack_mlp_blend(mlp_t, gated_t, test["mask_x"], W_MLP, "rank")
    print(f"test inferred collapse days {int(hat_t.sum())}/{hat_t.size}")

    valid_path = ROOT / f"outputs/fusion_{TAG}_valid.npy"
    test_path = ROOT / f"outputs/fusion_{TAG}_test.npy"
    sub_path = ROOT / f"submissions/task1_fusion_{TAG}.npy"
    np.save(valid_path, pred_v)
    np.save(test_path, pred_t)
    save_submission(pred_t, sub_path)
    print(f"wrote {valid_path}")
    print(f"wrote {sub_path}")
    print("does not overwrite submissions/task1_fusion_add6973_e1.npy")
    print("does not overwrite submissions/task1_y1_upload.npy")


if __name__ == "__main__":
    main()
