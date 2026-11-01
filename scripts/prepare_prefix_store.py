"""Copy an already safely exported training prefix into a verified mmap store.

Does not load pickle. Default dry-run does not even map input label files.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.oof import load_protocol
from src.prefix_store import export_prefix, KEYS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='directory of training-prefix NPY arrays, never data.z')
    p.add_argument('--output', required=True)
    p.add_argument('--source-sha256', required=True)
    p.add_argument('--axes', required=True, help='JSON with ordered stock_ids/num_feature_ids/cat_feature_ids')
    p.add_argument('--protocol', default=str(ROOT/'configs/oof_r11.json'))
    p.add_argument('--run', action='store_true')
    a = p.parse_args()
    if not a.run:
        print('dry-run: requires independently authorized training-only prefix; no pickle conversion'); return
    protocol = load_protocol(a.protocol)
    axes = json.loads(Path(a.axes).read_text(encoding='utf-8'))
    if set(axes) != {'stock_ids', 'num_feature_ids', 'cat_feature_ids'}:
        p.error('axes JSON must contain exactly the three ordered axes')
    data = {}
    try:
        for k in KEYS:
            data[k] = np.load(Path(a.input)/f'{k}.npy', mmap_mode='r', allow_pickle=False)
        print(export_prefix({**data, **axes}, a.source_sha256, protocol, a.output))
    finally:
        for arr in data.values():
            arr._mmap.close()


if __name__ == '__main__': main()
