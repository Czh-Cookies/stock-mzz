"""Qlib-style GRU on CS z-score windows. Does not overwrite hist_lgbm / fusion artifacts.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\train_gru.py --config configs/gru.yaml
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_data_path  # noqa: E402
from src.dataset import drop_task2_label, load_panel, slice_split, split_bounds  # noqa: E402
from src.metrics import mean_rank_ic  # noqa: E402
from src.models.gru_ts import GRUModel  # noqa: E402
from src.submit import save_submission  # noqa: E402


def _drop_heavy_panels(*bags: dict) -> None:
    for bag in bags:
        for key in ("num_x", "cat_x", "panel_num_x"):
            bag.pop(key, None)
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Qlib-style GRU temporal model")
    parser.add_argument("--config", default="configs/gru.yaml")
    parser.add_argument("--data", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    feat = dict(cfg.get("features") or {})
    model_cfg = dict(cfg.get("model") or {})
    train_cfg = dict(cfg.get("train") or {})
    paths = cfg.get("paths") or {}
    fill_invalid = float(train_cfg.get("fill_invalid", 0.0))
    gru_cfg = {**feat, **model_cfg, **train_cfg}

    try:
        import torch

        n_threads = int(train_cfg.get("num_threads") or os.cpu_count() or 4)
        torch.set_num_threads(max(1, n_threads))
    except Exception:
        pass

    print(f"config={args.config}")
    print(f"seed={seed}")
    print(
        f"GRU L={feat.get('length')} include_t={feat.get('include_current_day', True)} "
        f"cols={len(feat.get('num_indices') or [])} hidden={model_cfg.get('hidden_size')} "
        f"max_train_stocks={feat.get('max_train_stocks_per_day')} "
        f"sample_mode={feat.get('sample_mode') or 'uniform'} "
        f"min_history_coverage={feat.get('min_history_coverage') or 0} "
        f"freeze_backbone={model_cfg.get('freeze_backbone', False)} "
        f"loss={model_cfg.get('loss', 'mse')} loss_weight={model_cfg.get('loss_weight') or feat.get('loss_weight') or 'none'} "
        f"clip={feat.get('clip')} drop_corr={feat.get('drop_corr_threshold') or 0} cats={feat.get('cat_indices') or '-'}"
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

    model = GRUModel(gru_cfg, seed=seed)
    t1 = time.perf_counter()
    init_ckpt = paths.get("init_checkpoint")
    if init_ckpt:
        requested_features = model.feature_spec()
        init_path = ROOT / str(init_ckpt)
        if not init_path.is_file():
            raise SystemExit(f"missing init_checkpoint {init_path}")
        model.load(init_path, training_overrides={k: v for k, v in gru_cfg.items()
                   if k in {"lr", "weight_decay", "max_epochs", "patience", "freeze_backbone",
                            "max_train_stocks_per_day", "log_every_days"}})
        if model.feature_spec() != requested_features:
            raise ValueError("training config conflicts with checkpoint features; align config explicitly")
        print(f"loaded init_checkpoint={init_path} features={model.feature_spec()}")
    model.prepare_features(data)
    _drop_heavy_panels(data, valid, test)
    print(f"cs cache {time.perf_counter() - t1:.1f}s; dropped num_x/cat_x")

    t2 = time.perf_counter()
    fit_info = model.fit(data, train_start, train_end, valid=valid)
    print(f"fit {time.perf_counter() - t2:.1f}s best_valid_ic={fit_info.get('best_valid_ic')}")

    t3 = time.perf_counter()
    valid_pred = model.predict_panel(valid, data, fill_invalid=fill_invalid)
    valid_ic = mean_rank_ic(valid_pred, valid["y1"], valid["mask_y"])
    print(f"valid mean RankIC={valid_ic:.6f}  predict {time.perf_counter() - t3:.1f}s")

    valid_pred_path = ROOT / paths.get("valid_pred", "outputs/gru_valid.npy")
    valid_pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(valid_pred_path, valid_pred)

    t4 = time.perf_counter()
    test_pred = model.predict_panel(test, data, fill_invalid=fill_invalid)
    test_pred_path = ROOT / paths.get("test_pred", "outputs/gru_test.npy")
    test_pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(test_pred_path, test_pred)
    submission = ROOT / paths.get("submission", "submissions/task1_gru.npy")
    save_submission(test_pred, submission)
    print(f"test pred {test_pred.shape} {test_pred.dtype}  predict+save {time.perf_counter() - t4:.1f}s")

    checkpoint = ROOT / paths.get("checkpoint", "checkpoints/gru.pt")
    model.save(checkpoint)
    print(f"wrote {checkpoint}")
    print(f"wrote {valid_pred_path}")
    print(f"wrote {submission}")
    print("does not overwrite submissions/task1_fusion.npy")
    print(f"total {time.perf_counter() - t0:.1f}s")
    print("VALID_RANKIC", f"{valid_ic:.6f}")


if __name__ == "__main__":
    main()
