"""Industry-residual cross-section MLP. Does not overwrite fusion_gru_blend artifacts.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\train_cs_mlp.py --config configs/cs_mlp.yaml
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_data_path  # noqa: E402
from src.dataset import drop_task2_label, load_panel, slice_split, split_bounds  # noqa: E402
from src.metrics import mean_rank_ic  # noqa: E402
from src.models.cs_mlp import CSMLPModel  # noqa: E402
from src.models.fusion import FusionModel  # noqa: E402
from src.submit import save_submission  # noqa: E402


def _drop_heavy_panels(*bags: dict) -> None:
    for bag in bags:
        for key in ("num_x", "cat_x", "panel_num_x"):
            bag.pop(key, None)
    gc.collect()


def _pairwise_pred_ic(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    vals = []
    for t in range(a.shape[0]):
        m = np.asarray(mask[t], dtype=bool) & np.isfinite(a[t]) & np.isfinite(b[t])
        if int(m.sum()) < 8:
            continue
        r = spearmanr(a[t, m], b[t, m]).statistic
        if np.isfinite(r):
            vals.append(float(r))
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train industry-residual cross-section MLP")
    parser.add_argument("--config", default="configs/cs_mlp.yaml")
    parser.add_argument("--data", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    feat = dict(cfg.get("features") or {})
    model_cfg = dict(cfg.get("model") or {})
    train_cfg = dict(cfg.get("train") or {})
    paths = cfg.get("paths") or {}
    fill_invalid = float(train_cfg.get("fill_invalid", 0.0))
    mlp_cfg = {**feat, **model_cfg, **train_cfg}

    try:
        import torch

        n_threads = int(train_cfg.get("num_threads") or os.cpu_count() or 4)
        torch.set_num_threads(max(1, n_threads))
    except Exception:
        pass

    print(f"config={args.config}")
    print(f"seed={seed}")
    print(
        f"CSMLP cols={len(feat.get('num_indices') or [])} hidden={model_cfg.get('hidden_size')} "
        f"loss={model_cfg.get('loss', 'pearson_ic')} residual={feat.get('residual_industry', True)} "
        f"cats={feat.get('cat_indices') or '-'} max_train_stocks={feat.get('max_train_stocks_per_day')}"
    )

    data_path = resolve_data_path(cfg, args.data)
    print(f"data={data_path}")

    t0 = time.perf_counter()
    data = load_panel(str(data_path))
    drop_task2_label(data)
    print(f"loaded in {time.perf_counter() - t0:.1f}s  num_x={tuple(data['num_x'].shape)}")

    train_start, train_end = split_bounds(data, "train")
    recent_days = train_cfg.get("recent_days") or feat.get("recent_days")
    if recent_days:
        train_start = max(int(train_start), int(train_end) - int(recent_days))
        print(f"train recent_days={int(recent_days)} start={train_start}")
    valid = slice_split(data, "valid")
    test = slice_split(data, "test")
    print(f"days train={train_end - train_start} valid={valid['mask_x'].shape[0]} test={test['mask_x'].shape[0]}")

    model = CSMLPModel(mlp_cfg, seed=seed)
    t1 = time.perf_counter()
    model.prepare_features(data)
    init_ckpt = paths.get("init_checkpoint")
    if init_ckpt:
        init_path = ROOT / str(init_ckpt)
        if not init_path.is_file():
            raise SystemExit(f"missing init_checkpoint {init_path}")
        model.load(init_path)
        model.cfg = {**model.cfg, **mlp_cfg}
        model.freeze_hidden = bool(mlp_cfg.get("freeze_hidden", False))
        model.max_train_stocks = mlp_cfg.get("max_train_stocks_per_day")
        print(f"loaded init_checkpoint={init_path}")
    _drop_heavy_panels(data, valid, test)
    print(f"ind cache {time.perf_counter() - t1:.1f}s; dropped num_x/cat_x")

    t2 = time.perf_counter()
    fit_info = model.fit(data, train_start, train_end, valid=valid)
    print(f"fit {time.perf_counter() - t2:.1f}s best_valid_ic={fit_info.get('best_valid_ic')}")

    t3 = time.perf_counter()
    valid_pred = model.predict_panel(valid, data, fill_invalid=fill_invalid)
    valid_ic = mean_rank_ic(valid_pred, valid["y1"], valid["mask_y"])
    print(f"valid mean RankIC={valid_ic:.6f}  predict {time.perf_counter() - t3:.1f}s")

    valid_pred_path = ROOT / paths.get("valid_pred", "outputs/cs_mlp_valid.npy")
    valid_pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(valid_pred_path, valid_pred)

    peers = {
        "baseline": ROOT / "outputs/baseline_valid.npy",
        "hist_lgbm": ROOT / "outputs/hist_lgbm_valid.npy",
        "fusion": ROOT / "outputs/fusion_valid.npy",
        "gru": ROOT / "outputs/gru_valid.npy",
        "fusion_gru_blend": ROOT / "outputs/fusion_gru_blend_valid.npy",
    }
    print("mean daily Spearman vs peers (lower = more complementary):")
    for name, path in peers.items():
        if not path.is_file():
            continue
        other = np.load(path)
        ic = _pairwise_pred_ic(valid_pred, other, valid["mask_x"])
        print(f"  vs {name:18s}  {ic:.4f}")

    t4 = time.perf_counter()
    test_pred = model.predict_panel(test, data, fill_invalid=fill_invalid)
    test_pred_path = ROOT / paths.get("test_pred", "outputs/cs_mlp_test.npy")
    np.save(test_pred_path, test_pred)
    submission = ROOT / paths.get("submission", "submissions/task1_cs_mlp.npy")
    save_submission(test_pred, submission)
    print(f"test pred {test_pred.shape} {test_pred.dtype}  predict+save {time.perf_counter() - t4:.1f}s")

    checkpoint = ROOT / paths.get("checkpoint", "checkpoints/cs_mlp.pt")
    model.save(checkpoint)
    print(f"wrote {checkpoint}")
    print(f"wrote {valid_pred_path}")
    print(f"wrote {submission}")
    print(f"does not overwrite submissions/task1_fusion_gru_blend.npy")

    if bool(train_cfg.get("skip_blend")):
        print("skip_blend=True; not searching fusion weights")
        print(f"total {time.perf_counter() - t0:.1f}s")
        print("VALID_RANKIC", f"{valid_ic:.6f}")
        return

    blend_path = ROOT / "outputs/fusion_gru_blend_valid.npy"
    blend_test_path = ROOT / "outputs/fusion_gru_blend_test.npy"
    if blend_path.is_file() and blend_test_path.is_file():
        fusion_cfg = dict(cfg.get("blend") or {})
        fusion_cfg.setdefault("weight_grid", [0.0, 0.15, 0.25, 0.4, 0.5, 0.7, 1.0])
        blender = FusionModel(fusion_cfg)
        locked = blender.fit(
            valid_pred,
            np.load(blend_path),
            valid["y1"],
            valid["mask_y"],
            valid["mask_x"],
            industry=model.industry[valid["start"] : valid["end"]] if model.industry is not None else None,
        )
        print("blend with fusion_gru_blend:")
        for row in (locked.get("valid_leaderboard") or [])[:6]:
            extra = {k: v for k, v in row.items() if k not in {"valid_leaderboard", "name", "ic"}}
            print(f"  {row['ic']:.6f}  {row['name']}  {extra}")
        print(f"BLEND_LOCKED {locked['name']} ic={locked['ic']:.6f}")
        if float(locked["ic"]) > 0.110069 + 1e-4:
            out_valid = blender.predict(
                valid_pred,
                np.load(blend_path),
                valid["mask_x"],
                industry=model.industry[valid["start"] : valid["end"]] if model.industry is not None else None,
            )
            out_test = blender.predict(
                test_pred,
                np.load(blend_test_path),
                test["mask_x"],
                industry=model.industry[test["start"] : test["end"]] if model.industry is not None else None,
            )
            default_run = str(paths.get("submission", "submissions/task1_cs_mlp.npy")).endswith("task1_cs_mlp.npy")
            if default_run:
                np.save(ROOT / "outputs/fusion_cs_mlp_valid.npy", out_valid)
                np.save(ROOT / "outputs/fusion_cs_mlp_test.npy", out_test)
                save_submission(out_test, ROOT / "submissions/task1_fusion_cs_mlp.npy")
                print("wrote submissions/task1_fusion_cs_mlp.npy (still not overwriting fusion_gru_blend)")
            else:
                print(f"blend {locked['ic']:.6f}; not overwriting task1_fusion_cs_mlp.npy")
        else:
            print("blend did not beat 0.110069; not writing a new main submission")

    print(f"total {time.perf_counter() - t0:.1f}s")
    print("VALID_RANKIC", f"{valid_ic:.6f}")


if __name__ == "__main__":
    main()
