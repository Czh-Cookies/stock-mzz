"""Purged time-out protocol and immutable artifact contracts; no training entrypoint.

Only a verified TRAINING PREFIX (rows before official valid_start) may be passed
to cache preparation. This module never decodes data.z or accesses test labels.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import platform
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.metrics import mean_rank_ic, rank_ic_series
from src.provenance import canonical, file_sha256, array_sha256


class ProtocolError(ValueError):
    pass


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def load_protocol(path: str | Path) -> dict:
    p = json.loads(Path(path).read_text(encoding="utf-8"))
    bounds = p["official_bounds"]
    if len(bounds) != 4 or bounds != sorted(set(bounds)):
        raise ProtocolError("invalid official boundaries")
    gap = p["purge_steps"]
    if type(gap) is not int or gap < 1:
        raise ProtocolError("purge_steps must be a positive integer")
    previous = bounds[0]
    for name, (start, end) in p["folds"].items():
        if not bounds[0] < start-gap < start < end <= bounds[1] or start < previous:
            raise ProtocolError(f"invalid/overlapping fold {name}")
        previous = end
    for key in ("horizon_steps", "availability_lag_steps"):
        value = p["label"][key]
        if value is not None and (type(value) is not int or value < 0):
            raise ProtocolError(f"invalid {key}")
    return p


def label_lag(p: dict, *, exploratory: bool = False) -> int:
    h, lag = p["label"]["horizon_steps"], p["label"]["availability_lag_steps"]
    if h is None or lag is None:
        if not exploratory:
            raise ProtocolError("label horizon/availability unknown; only explicitly accepted exploratory runs allowed")
        return p["purge_steps"]
    required = h + lag
    if p["purge_steps"] < required:
        raise ProtocolError("purge shorter than label horizon plus availability lag")
    return required


def fold_spec(p: dict, name: str, recent_days: int | None = None) -> dict:
    if name not in p["folds"]:
        raise ProtocolError("unknown fold")
    a, b = p["folds"][name]
    end = a - p["purge_steps"]
    start = p["official_bounds"][0]
    if recent_days is not None:
        if type(recent_days) is not int or recent_days <= 0:
            raise ProtocolError("invalid recent_days")
        start = max(start, end-recent_days)
    return {"fold": name, "train": [start, end], "purge": [end, a], "evaluate": [a, b]}


def prefix_contract(data: dict, source_sha256: str, p: dict) -> dict:
    """Bind a separately prepared prefix. No test/official-valid rows are accepted."""
    cutoff = p["official_bounds"][1]
    if "y2" in data:
        raise ProtocolError("y2 must not enter the OOF prefix")
    if len(source_sha256) != 64 or any(c not in "0123456789abcdef" for c in source_sha256):
        raise ProtocolError("invalid source SHA256")
    shapes = {k: list(np.shape(data[k])) for k in ("num_x", "cat_x", "mask_x", "y1", "mask_y")}
    if any(shape[0] != cutoff for shape in shapes.values()):
        raise ProtocolError("prefix must end at official valid_start; full/test arrays forbidden")
    t, stocks, features = shapes["num_x"]
    if (len(shapes['cat_x']) != 3 or shapes['cat_x'][:2] != [t, stocks]
            or any(shapes[k] != [t, stocks] for k in ('mask_x', 'mask_y', 'y1'))):
        raise ProtocolError("prefix shape mismatch")
    axes = {}
    for key, size in (("num_feature_ids", features), ("cat_feature_ids", shapes['cat_x'][2]),
                      ("stock_ids", stocks)):
        vals = [str(v) for v in data.get(key, range(size))]
        if len(vals) != size or len(set(vals)) != size:
            raise ProtocolError(f"invalid ordered axis {key}")
        axes[key] = vals
    # Hash row by row to avoid allocating a second multi-GB contiguous panel.
    hashes = {}
    for key in shapes:
        h = hashlib.sha256()
        for row in data[key]:
            h.update(array_sha256(row).encode())
        hashes[key] = h.hexdigest()
    return {"schema": "oof-prefix-v1", "source_sha256": source_sha256,
            "bounds": p["official_bounds"], "shapes": shapes, "axes": axes, "payloads": hashes}


def run_contract(p: dict, fold: str, data_contract: dict, model: str, cfg: dict,
                 code_hashes: dict, *, seed: int = 42, recent_days=None,
                 exploratory: bool = False) -> dict:
    label_lag(p, exploratory=exploratory)
    if data_contract.get('schema') != 'oof-prefix-v1' or data_contract['bounds'] != p['official_bounds']:
        raise ProtocolError("verified prefix contract required")
    if not code_hashes or any(len(v) != 64 for v in code_hashes.values()):
        raise ProtocolError("working-tree code hashes required; Git HEAD alone is insufficient")
    if (cfg.get('paths') or {}).get('init_checkpoint'):
        raise ProtocolError("old checkpoint initialization is forbidden for baseline OOF")
    versions = {}
    for package in ("numpy", "scipy", "torch", "lightgbm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"runtime": {"python": platform.python_version(), "packages": versions},
            "schema": "oof-run-v1", "protocol_hash": digest(p), "protocol": copy.deepcopy(p),
            **fold_spec(p, fold, recent_days), "data": copy.deepcopy(data_contract),
            "model": model, "config": copy.deepcopy(cfg), "config_hash": digest(cfg),
            "code_hashes": dict(code_hashes), "seed": seed, "exploratory": exploratory,
            "score_space": p['score_space'], "initialization": "fresh", "outer_labels_for_fit": False}


def new_run(root: Path, run: dict) -> Path:
    """Never overwrite: even an identical rerun needs a new parent session directory."""
    root = Path(root).resolve()
    name = f"{run['fold']}-{digest(run)[:16]}-s{run['seed']}"
    dest = root / name
    dest.mkdir(parents=True, exist_ok=False)
    with (dest/'run.json').open('x', encoding='utf-8') as f:
        f.write(canonical(run))
    return dest


def _validate_run(run):
    p = run['protocol']
    label_lag(p, exploratory=run['exploratory'])
    if run['protocol_hash'] != digest(p) or run['config_hash'] != digest(run['config']):
        raise ProtocolError("run hash mismatch")
    expected = fold_spec(p, run['fold'])
    if (run['evaluate'] != expected['evaluate'] or run['purge'] != expected['purge']
            or not expected['train'][0] <= run['train'][0] < run['train'][1] == expected['train'][1]):
        raise ProtocolError("training overlaps evaluation or violates purge")
    if run.get('initialization') != 'fresh' or run.get('outer_labels_for_fit') is not False:
        raise ProtocolError("non-OOF training provenance")


def prepare_fold(data: dict, run: dict) -> tuple[dict, dict, dict]:
    """Return training inputs, label-free evaluation inputs, and evaluator-only labels.

    All label arrays seen by a trainer are capped before the purge, including
    callers that accidentally iterate beyond train_end. History features may use
    purge days; label values never enter features. Evaluation is day-causal.
    """
    _validate_run(run)
    actual = prefix_contract(data, run['data']['source_sha256'], run['protocol'])
    if actual != run['data']:
        raise ProtocolError("prefix source/axis/payload mismatch")
    start, stop = run['train']; a, b = run['evaluate']
    train = {k: data[k][:stop] for k in ('num_x', 'cat_x', 'mask_x', 'y1', 'mask_y')}
    # Labels are copied so their ndarray.base cannot expose outer-fold labels.
    train['y1'] = np.array(train['y1'], copy=True)
    train['mask_y'] = np.array(train['mask_y'], copy=True)
    train['y1'][:start] = np.nan
    train['mask_y'][:start] = False
    train.update(start=start, end=stop)
    evaluation = {k: data[k][:b] for k in ('num_x', 'cat_x', 'mask_x')}
    evaluation.update(start=a, end=b)
    labels = {'y1': np.asarray(data['y1'][a:b]), 'mask_y': np.asarray(data['mask_y'][a:b]),
              'mask_x': np.asarray(data['mask_x'][a:b])}
    return train, evaluation, labels


def write_label_cache(dest: Path, labels: dict, run: dict) -> Path:
    _validate_run(run)
    expected_shape = (run['evaluate'][1]-run['evaluate'][0], len(run['data']['axes']['stock_ids']))
    if set(labels) != {'y1', 'mask_y', 'mask_x'} or any(v.shape != expected_shape for v in labels.values()):
        raise ProtocolError("fold label shape/schema mismatch")
    meta = {'run_hash': digest(run), 'range': run['evaluate'],
            'hashes': {k: array_sha256(v) for k, v in labels.items()}}
    path = Path(dest)/'evaluation_labels.npz'
    with path.open('xb') as f:
        np.savez(f, **labels, manifest=np.array(canonical(meta)))
    return path


def load_label_cache(path: Path, run: dict) -> dict:
    _validate_run(run)
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {'y1', 'mask_y', 'mask_x', 'manifest'}:
            raise ProtocolError("unverified or malformed fold cache")
        meta = json.loads(str(z['manifest']))
        if meta['run_hash'] != digest(run) or meta['range'] != run['evaluate']:
            raise ProtocolError("wrong fold/config/source cache")
        out = {k: z[k] for k in ('y1', 'mask_y', 'mask_x')}
        if meta['hashes'] != {k: array_sha256(v) for k, v in out.items()}:
            raise ProtocolError("fold cache payload changed")
        shape = (run['evaluate'][1]-run['evaluate'][0], len(run['data']['axes']['stock_ids']))
        if any(v.shape != shape for v in out.values()):
            raise ProtocolError("fold cache shape mismatch")
        return out


def register_checkpoint(dest: Path, path: Path, run: dict, *, fitted_range: list,
                        selection_range: list | None = None) -> dict:
    _validate_run(run)
    path, dest = Path(path).resolve(), Path(dest).resolve()
    if path.parent != dest or fitted_range != run['train']:
        raise ProtocolError("checkpoint must be freshly trained inside this run and exact fit range")
    if selection_range is not None:
        raise ProtocolError("baseline registration requires a frozen training budget; inner-selection needs a new protocol")
    meta = {'run_hash': digest(run), 'checkpoint': path.name, 'sha256': file_sha256(path),
            'fitted_range': fitted_range, 'selection_range': selection_range}
    sidecar = path.with_suffix(path.suffix+'.meta.json')
    meta['model_metadata_sha256'] = file_sha256(sidecar) if sidecar.is_file() else None
    with (dest/'checkpoint.provenance.json').open('x', encoding='utf-8') as f:
        f.write(canonical(meta))
    return meta


def verify_checkpoint(dest: Path, run: dict) -> dict:
    _validate_run(run)
    dest = Path(dest)
    meta = json.loads((dest/'checkpoint.provenance.json').read_text(encoding='utf-8'))
    name = meta['checkpoint']
    if Path(name).name != name or meta['run_hash'] != digest(run) or meta['fitted_range'] != run['train']:
        raise ProtocolError("checkpoint provenance mismatch")
    cp = dest/name
    if file_sha256(cp) != meta['sha256']:
        raise ProtocolError("checkpoint bytes changed")
    sidecar = cp.with_suffix(cp.suffix+'.meta.json')
    current = file_sha256(sidecar) if sidecar.is_file() else None
    if current != meta['model_metadata_sha256']:
        raise ProtocolError("checkpoint metadata changed")
    return meta


def save_predictions(dest: Path, pred: np.ndarray, run: dict, *, score_space: str) -> Path:
    meta = verify_checkpoint(dest, run)
    shape = (run['evaluate'][1]-run['evaluate'][0], len(run['data']['axes']['stock_ids']))
    if pred.shape != shape or not np.isfinite(pred).all() or score_space not in ('raw', 'rank', 'zscore'):
        raise ProtocolError("prediction shape/finite/score-space mismatch")
    path = Path(dest)/'evaluation_predictions.npy'
    info = {'run_hash': digest(run), 'checkpoint_sha256': meta['sha256'],
            'range': run['evaluate'], 'score_space': score_space}
    if path.exists() or path.with_suffix('.json').exists():
        raise FileExistsError('prediction artifacts already exist')
    with path.open('xb') as f:
        np.save(f, pred, allow_pickle=False)
    info['sha256'] = file_sha256(path)
    with path.with_suffix('.json').open('x', encoding='utf-8') as f:
        f.write(canonical(info))
    return path


def evaluate_run(dest: Path) -> dict:
    dest = Path(dest)
    run = json.loads((dest/'run.json').read_text(encoding='utf-8'))
    cp = verify_checkpoint(dest, run)
    path = dest/'evaluation_predictions.npy'
    meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if (meta.get('score_space') not in ('raw', 'rank', 'zscore')
            or meta['run_hash'] != digest(run) or meta['checkpoint_sha256'] != cp['sha256']
            or meta['range'] != run['evaluate'] or meta['sha256'] != file_sha256(path)):
        raise ProtocolError("prediction provenance mismatch")
    labels = load_label_cache(dest/'evaluation_labels.npz', run)
    pred = np.load(path, allow_pickle=False)
    series = rank_ic_series(pred, labels['y1'], labels['mask_y'])
    if not np.isfinite(series).all():
        raise ProtocolError("undefined daily RankIC; refuse silent day exclusion")
    return {'fold': run['fold'], 'score': mean_rank_ic(pred, labels['y1'], labels['mask_y']),
            'days': len(series), 'negative_days': int((series < 0).sum()),
            'exploratory': run['exploratory'], 'run_hash': digest(run)}


def eligible_fusion_rows(p: dict, target_fold: str, prior_run: dict, *, exploratory=False) -> np.ndarray:
    """Availability convention: row t matures at t+horizon+lag, must be < fit_time."""
    _validate_run(prior_run)
    if prior_run['protocol_hash'] != digest(p):
        raise ProtocolError("fusion protocol mismatch")
    cutoff = p['folds'][target_fold][0]
    a, b = prior_run['evaluate']
    if b > cutoff:
        raise ProtocolError("same/future fold cannot train fusion")
    lag = label_lag(p, exploratory=exploratory)
    rows = np.arange(a, b)
    return rows[rows+lag < cutoff]
