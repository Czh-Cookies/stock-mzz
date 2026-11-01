"""Fresh, fixed-budget fold adapters. No outer labels or callbacks passed to fit."""
import gc
import json
from pathlib import Path
import numpy as np
from src.config import load_config
from src.oof import (ProtocolError, label_lag, fold_spec, run_contract, new_run,
                     register_checkpoint, save_predictions, write_label_cache, digest,
                     verify_checkpoint)
from src.provenance import file_sha256, canonical, array_sha256
from src.models.fusion import panel_cs_rank

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {'gru': 'configs/gru_no_today_recent_n2000.yaml', 'mlp': 'configs/cs_mlp.yaml',
           'baseline': 'configs/baseline.yaml', 'hist': 'configs/hist_lgbm.yaml'}
# Fixed before F1 scores: rank-space blend; no gate fitted on outer-fold masks/labels.
FIXED_WEIGHTS = {'baseline': .19125, 'hist': .44625, 'gru': .2125, 'mlp': .15}


def check_memory_budget(model, available_bytes=None):
    """Conservative startup floor, not a guarantee on peak working set."""
    if available_bytes is None:
        import ctypes
        import os
        if os.name != 'nt':
            raise ProtocolError('memory preflight currently supports Windows only')
        class Memory(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in
                ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]
        m = Memory(); m.length = ctypes.sizeof(m)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            raise ProtocolError('cannot measure available memory')
        available_bytes = m.available
    required = (6 if model in ('gru', 'mlp') else 10)*2**30
    if available_bytes < required:
        raise ProtocolError(f'available RAM below conservative {required/2**30:g} GiB startup floor')
    return required


def fixed_fusion(predictions, mask):
    if set(predictions) != set(FIXED_WEIGHTS):
        raise ProtocolError('fixed fusion requires all four declared branches')
    if any(v.shape != mask.shape or not np.isfinite(v).all() for v in predictions.values()):
        raise ProtocolError('fusion prediction shape/finite mismatch')
    return np.asarray(sum(FIXED_WEIGHTS[k]*panel_cs_rank(predictions[k], mask)
                          for k in FIXED_WEIGHTS), dtype=np.float32)


def fuse_runs(store, fold, run_dirs, dest):
    """Label-free, immutable fixed fusion with verified branch artifact chains."""
    if set(run_dirs) != set(FIXED_WEIGHTS):
        raise ProtocolError('all four branch run directories required')
    predictions, sources = {}, {}
    for model, directory in run_dirs.items():
        directory = Path(directory)
        run = json.loads((directory/'run.json').read_text(encoding='utf-8'))
        cp = verify_checkpoint(directory, run)
        if (run['model'] != model or run['fold'] != fold or run['data'] != store.contract
                or run['protocol_hash'] != digest(store.protocol) or run['seed'] != 42):
            raise ProtocolError('fusion branch model/fold/data/protocol/seed mismatch')
        path = directory/'evaluation_predictions.npy'
        meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        if (meta['run_hash'] != digest(run) or meta['checkpoint_sha256'] != cp['sha256']
                or meta['sha256'] != file_sha256(path) or meta['range'] != run['evaluate']
                or meta['score_space'] not in ('raw', 'rank', 'zscore')):
            raise ProtocolError('fusion prediction provenance mismatch')
        predictions[model] = np.load(path, allow_pickle=False)
        sources[model] = {'run_dir': str(directory.resolve()), 'run_hash': digest(run), **meta}
    _, evaluation = store.evaluation_inputs(fold)
    mask = evaluation['mask_x']
    pred = fixed_fusion(predictions, mask)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=False)
    path = dest/'evaluation_predictions.npy'
    with path.open('xb') as f:
        np.save(f, pred, allow_pickle=False)
    info = {'schema': 'fixed-fold-fusion-v1', 'fold': fold, 'weights': FIXED_WEIGHTS,
            'score_space': 'weighted_daily_ranks', 'data_hash': digest(store.contract),
            'protocol_hash': digest(store.protocol), 'mask_hash': array_sha256(mask),
            'sources': sources, 'sha256': file_sha256(path), 'fit_labels': False,
            'code_sha256': file_sha256(__file__)}
    with (dest/'fusion.json').open('x', encoding='utf-8') as f:
        f.write(canonical(info))
    return dest


def plan(protocol, fold, model):
    if model not in CONFIGS:
        raise ProtocolError('unknown model')
    cfg = load_config(CONFIGS[model], apply_local=False)
    recent = (cfg.get('train') or {}).get('recent_days')
    return {'model': model, 'config': cfg, **fold_spec(protocol, fold, recent),
            'recent_days': recent, 'selection': 'fixed max_epochs/n_estimators; no early stopping',
            'fusion_weights': FIXED_WEIGHTS, 'default_action': 'dry-run'}


def train_fold(store, protocol, fold, model_name, root, *, execute=False, accept_unknown=False,
               config=None):
    info = plan(protocol, fold, model_name)
    if not execute:
        return info
    label_lag(protocol, exploratory=accept_unknown)
    if store.protocol != protocol:
        raise ProtocolError('store/runner protocol mismatch')
    cfg = config if config is not None else info['config']  # synthetic tests use tiny explicit configs
    recent = (cfg.get('train') or {}).get('recent_days')
    code_files = list((ROOT/'src').rglob('*.py')) + [ROOT/'scripts/run_oof_fold.py']
    code = {str(p.relative_to(ROOT)): file_sha256(p) for p in sorted(code_files)}
    run = run_contract(protocol, fold, store.contract, model_name, cfg, code,
                       seed=int(cfg.get('seed', 42)), recent_days=recent, exploratory=accept_unknown)
    dest = new_run(Path(root), run)
    start, end = run['train']
    train = store.training_panel(fold, recent)
    feat = dict(cfg.get('features') or {})
    options = {**feat, **(cfg.get('model') or {}), **(cfg.get('train') or {})}
    seed = int(cfg.get('seed', 42))
    if model_name in ('gru', 'mlp'):
        from src.models.gru_ts import GRUModel
        from src.models.cs_mlp import CSMLPModel
        model = (GRUModel if model_name == 'gru' else CSMLPModel)(options, seed=seed)
        train.update(store.contract['axes'])
        model.prepare_features(train)
        model.fit(train, start, end, valid=None)
        cp = dest/'model.pt'
        model.save(cp)
        cards = list(model.cat_cardinalities)
        # Release precomputed train features before creating prediction features.
        if model_name == 'gru': model.cs_sel = None
        else: model.ind_sel = None
        del train; gc.collect()
        panel, evaluation = store.evaluation_inputs(fold)
        panel.update(store.contract['axes'])
        # Map unseen categories to existing fallback 0, using TRAIN cardinalities only.
        if model.cat_indices:
            cats = np.array(panel['cat_x'], copy=True)
            for col, card in zip(model.cat_indices, cards):
                values = cats[..., col]
                values[(values < 0) | (values >= card)] = 0
            panel['cat_x'] = cats
        model.prepare_features(panel)
        model.cat_cardinalities = cards
        pred = model.predict_panel(evaluation, panel)
    else:
        from src.models.baseline import LightGBMBaseline
        from src.dataset import flatten_masked_rows
        sl = {k: v[start:end] for k, v in train.items()}
        sl.update(start=start, end=end, panel_num_x=train['num_x'], panel_mask_x=train['mask_x'])
        x, y, _ = flatten_masked_rows(sl, feature_cfg=feat, seed=seed)
        model = LightGBMBaseline((cfg['model'].get('params') or {}), feat, seed)
        model.fit(x, y)
        cp = dest/'model.txt'; model.save(cp)
        del train, sl, x, y; gc.collect()
        _, evaluation = store.evaluation_inputs(fold)
        pred = model.predict_panel(evaluation)
    register_checkpoint(dest, cp, run, fitted_range=run['train'])
    save_predictions(dest, pred, run, score_space='raw')
    # Evaluation labels are obtained only AFTER fitting and prediction have finished.
    a, b = run['evaluate']
    labels = store.label_window(fold, 'evaluate', a, b)
    labels['mask_x'] = np.asarray(evaluation['mask_x'])
    write_label_cache(dest, labels, run)
    return {'run_dir': str(dest), 'model': model_name, 'fitted_range': run['train']}
