"""Immutable NPY prefix store. Conversion accepts arrays, NEVER an arbitrary pickle."""
import json
from pathlib import Path
import numpy as np
from src.oof import ProtocolError, prefix_contract, digest, fold_spec
from src.provenance import file_sha256, canonical

KEYS = ('num_x', 'cat_x', 'mask_x', 'y1', 'mask_y')


def export_prefix(data, source_sha256, protocol, dest, chunk_rows=8):
    if chunk_rows < 1:
        raise ProtocolError('chunk_rows must be positive')
    contract = prefix_contract(data, source_sha256, protocol)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=False)
    files = {}
    for key in KEYS:
        arr = data[key]
        if np.asarray(arr).dtype.hasobject:
            raise ProtocolError('object dtype forbidden')
        path = dest/f'{key}.npy'
        out = np.lib.format.open_memmap(path, mode='w+', dtype=arr.dtype, shape=arr.shape)
        for start in range(0, len(arr), chunk_rows):
            out[start:start+chunk_rows] = arr[start:start+chunk_rows]
        out.flush(); del out
        files[key] = {'sha256': file_sha256(path), 'dtype': arr.dtype.str, 'shape': list(arr.shape)}
    # Completion marker written last: incomplete conversion cannot be opened.
    with (dest/'manifest.json').open('x', encoding='utf-8') as f:
        f.write(canonical({'schema': 'prefix-store-v1', 'contract': contract,
                           'contract_hash': digest(contract), 'files': files}))
    return dest


class PrefixStore:
    def __init__(self, path, protocol, *, expected_source):
        self.path = Path(path)
        self.protocol = protocol
        self.manifest = json.loads((self.path/'manifest.json').read_text(encoding='utf-8'))
        c = self.manifest['contract']
        if (self.manifest['schema'] != 'prefix-store-v1' or c['source_sha256'] != expected_source
                or self.manifest.get('contract_hash') != digest(c)
                or c['bounds'] != protocol['official_bounds'] or set(self.manifest['files']) != set(KEYS)):
            raise ProtocolError('prefix source/protocol/schema mismatch')
        self.contract = c
        self._arrays = {}
        try:
            for key, spec in self.manifest['files'].items():
                p = self.path/f'{key}.npy'
                if file_sha256(p) != spec['sha256']:
                    raise ProtocolError(f'prefix file changed: {key}')
                arr = np.load(p, mmap_mode='r', allow_pickle=False)
                self._arrays[key] = arr
                if list(arr.shape) != spec['shape'] or arr.dtype.str != spec['dtype']:
                    raise ProtocolError('dtype/shape mismatch')
            check = {**self._arrays, **c['axes']}
            if prefix_contract(check, expected_source, protocol) != c:
                raise ProtocolError('prefix contract/payload mismatch')
        except Exception:
            self.close()
            raise

    def close(self):
        for array in self._arrays.values():
            array._mmap.close()
        self._arrays.clear()

    def label_window(self, fold, role, start, end, recent_days=None):
        spec = fold_spec(self.protocol, fold, recent_days)
        if role not in ('train', 'evaluate'):
            raise ProtocolError('invalid label role')
        lo, hi = spec[role]
        if not lo <= start < end <= hi:
            raise ProtocolError('label access outside authorized fold interval')
        return {k: np.array(self._arrays[k][start:end], copy=True) for k in ('y1', 'mask_y')}

    def features(self, end):
        if not 0 < end <= self.protocol['official_bounds'][1]:
            raise ProtocolError('feature range beyond training prefix')
        return {k: self._arrays[k][:end] for k in ('num_x', 'cat_x', 'mask_x')}

    def training_panel(self, fold, recent_days=None):
        spec = fold_spec(self.protocol, fold, recent_days)
        start, end = spec['train']
        panel = self.features(end)
        shape = panel['mask_x'].shape
        panel['y1'] = np.full(shape, np.nan, dtype=self._arrays['y1'].dtype)
        panel['mask_y'] = np.zeros(shape, bool)
        permitted = self.label_window(fold, 'train', start, end, recent_days)
        for key in ('y1', 'mask_y'):
            panel[key][start:end] = permitted[key]
        return panel

    def evaluation_inputs(self, fold):
        a, b = fold_spec(self.protocol, fold)['evaluate']
        panel = self.features(b)
        sl = {k: v[a:b] for k, v in panel.items()}
        sl.update(start=a, end=b, panel_num_x=panel['num_x'], panel_mask_x=panel['mask_x'])
        return panel, sl
