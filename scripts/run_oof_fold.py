"""Default dry-run; only prepared prefix stores are accepted, never data.z."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.oof import load_protocol
from src.fold_runner import plan, train_fold, CONFIGS
from src.prefix_store import PrefixStore


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fold', choices=['F1', 'F2', 'F3'], default='F1')
    p.add_argument('--model', choices=list(CONFIGS), default='gru')
    p.add_argument('--store'); p.add_argument('--source-sha256')
    p.add_argument('--protocol', default=str(ROOT/'configs/oof_r11.json'))
    p.add_argument('--output-root', default=str(ROOT/'outputs/oof_r12'))
    p.add_argument('--run', action='store_true')
    p.add_argument('--accept-unknown-horizon', action='store_true')
    args = p.parse_args(); protocol = load_protocol(args.protocol)
    if not args.run:
        print(json.dumps(plan(protocol, args.fold, args.model), ensure_ascii=False, indent=2)); return
    if not args.store or not args.source_sha256:
        p.error('--run requires --store and --source-sha256')
    if not args.accept_unknown_horizon and protocol['label']['horizon_steps'] is None:
        p.error('unknown label horizon: explicit --accept-unknown-horizon required')
    from src.fold_runner import check_memory_budget
    check_memory_budget(args.model)
    store = PrefixStore(args.store, protocol, expected_source=args.source_sha256)
    try:
        print(json.dumps(train_fold(store, protocol, args.fold, args.model, args.output_root,
            execute=True, accept_unknown=args.accept_unknown_horizon), ensure_ascii=False))
    finally:
        store.close()


if __name__ == '__main__': main()
