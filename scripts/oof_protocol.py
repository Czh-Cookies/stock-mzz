"""R1.1 planning and provenance-checked scoring only. No training or data decoding."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.oof import load_protocol, fold_spec, digest, evaluate_run
from src.provenance import file_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['plan', 'evaluate'])
    parser.add_argument('--protocol', default=str(ROOT/'configs/oof_r11.json'))
    parser.add_argument('--source', help='optional source-file hash only; never deserialize')
    parser.add_argument('--run-dir')
    parser.add_argument('--out', help='optional NEW JSON file; existing file is rejected')
    args = parser.parse_args()
    if args.action == 'evaluate':
        if not args.run_dir:
            parser.error('--run-dir required')
        result = evaluate_run(Path(args.run_dir))
    else:
        p = load_protocol(args.protocol)
        result = {'protocol_hash': digest(p), 'protocol': p,
                  'folds': [dict(expanding=fold_spec(p, f), recent800=fold_spec(p, f, 800))
                            for f in p['folds']],
                  'source_sha256': file_sha256(args.source) if args.source else None,
                  'source_decoded': False, 'verified_prefix_ready': False,
                  'training_enabled': False,
                  'code_hashes': {str(path.relative_to(ROOT)): file_sha256(path)
                      for path in sorted((ROOT/'src').rglob('*.py'))},
                  'blockers': ['unknown label horizon/availability',
                               'verified training-only prefix not prepared',
                               'fold training runner not implemented in R1.1']}
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        with Path(args.out).open('x', encoding='utf-8') as f:
            f.write(rendered)
        print(f'wrote {args.out}')
    else:
        print(rendered)


if __name__ == '__main__':
    main()
