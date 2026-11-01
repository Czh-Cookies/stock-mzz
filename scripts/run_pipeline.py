"""Single entry for the Jingge track-1 pipeline.

Does not train anything unless you pass --run. Default is a dry-run that prints
commands, missing files, and the scoring axis of each stage.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\run_pipeline.py --list
    .\\.venv\\Scripts\\python.exe scripts\\run_pipeline.py --stage gru_n2000
    .\\.venv\\Scripts\\python.exe scripts\\run_pipeline.py --from cache --to check --run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import (  # noqa: E402
    STAGE_BY_ID,
    STAGES,
    existing_outputs,
    missing_inputs,
    scorecard_rows,
    stage_command,
)


def _slice_stages(from_id: str | None, to_id: str | None, only: str | None) -> list:
    if only:
        if only not in STAGE_BY_ID:
            raise SystemExit(f"unknown stage {only!r}; use --list")
        return [STAGE_BY_ID[only]]
    ids = [s.id for s in STAGES]
    start = ids.index(from_id) if from_id else 0
    end = ids.index(to_id) + 1 if to_id else len(ids)
    if from_id and from_id not in STAGE_BY_ID:
        raise SystemExit(f"unknown stage {from_id!r}")
    if to_id and to_id not in STAGE_BY_ID:
        raise SystemExit(f"unknown stage {to_id!r}")
    return list(STAGES[start:end])


def main() -> None:
    parser = argparse.ArgumentParser(description="Jingge y1 pipeline")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--stage", default=None, help="run / print a single stage id")
    parser.add_argument("--from", dest="from_id", default=None)
    parser.add_argument("--to", dest="to_id", default=None)
    parser.add_argument("--run", action="store_true", help="actually execute; default is dry-run")
    parser.add_argument("--scorecard", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    if args.list or args.scorecard:
        if args.list:
            print(f"{'id':12s}  {'axis':5s}  {'panel':5s}  title")
            for s in STAGES:
                panel = "YES" if s.loads_panel else "no"
                print(f"{s.id:12s}  {s.axis:5s}  {panel:5s}  {s.title}")
        if args.scorecard:
            print(json.dumps(scorecard_rows(), ensure_ascii=False, indent=2))
        return

    stages = _slice_stages(args.from_id, args.to_id, args.stage)
    py = args.python
    for s in stages:
        cmd = stage_command(s, python=py)
        miss = missing_inputs(s, ROOT)
        have = existing_outputs(s, ROOT)
        print("=" * 72)
        print(f"[{s.id}] {s.title}  axis={s.axis}  rubric={s.rubric}")
        print("cmd:", " ".join(cmd))
        print("note:", s.note)
        if s.loads_panel:
            print("memory: loads full panel — do not run another panel job in parallel")
        if miss:
            print("missing inputs:", ", ".join(miss))
        if have:
            print("already have:", ", ".join(have))
        if args.run:
            if miss:
                raise SystemExit(f"stage {s.id} blocked on missing inputs")
            proc = subprocess.run(cmd, cwd=ROOT)
            if proc.returncode != 0:
                raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
