"""Train-only RankIC of unused num cols on the same last-800 window as the champion GRU."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_data_path  # noqa: E402
from src.dataset import drop_task2_label, keep_recent_days, load_panel, slice_split  # noqa: E402
from src.metrics import mean_rank_ic  # noqa: E402

USED = {5, 6, 7, 8, 10, 11, 17, 18, 19, 20, 21, 22, 24, 25, 39, 40, 41, 57, 68, 69, 70, 73, 90}
FAILED_ADD = {74, 42, 66, 58}


def main() -> None:
    cfg = load_config("configs/gru_n2000_add6973_e1.yaml")
    data = load_panel(str(resolve_data_path(cfg)))
    drop_task2_label(data)
    train = keep_recent_days(slice_split(data, "train"), 800)
    num = train["num_x"]
    y = train["y1"]
    my = train["mask_y"]
    adj = set()
    for c in USED:
        adj.add(c - 1)
        adj.add(c + 1)
    print(f"scan train last800 RankIC n_days={num.shape[0]} n_cols={num.shape[-1]}", flush=True)
    rows: list[tuple[int, float, float]] = []
    for c in range(int(num.shape[-1])):
        ic = mean_rank_ic(num[..., c], y, my)
        rows.append((c, ic, abs(ic)))
        if (c + 1) % 20 == 0:
            print(f"  scanned {c + 1}/{num.shape[-1]}", flush=True)
    rows.sort(key=lambda x: -x[2])
    print("UNUSED |IC|>0.03 (last800 train):")
    for c, ic, a in rows:
        if c in USED or a < 0.03:
            continue
        tags = []
        if c in FAILED_ADD:
            tags.append("failed_add")
        if c in adj:
            tags.append("adjacent")
        print(f"  col={c:3d} ic={ic:+.6f} {' '.join(tags)}")
    print("TOP unused any IC:")
    n = 0
    for c, ic, a in rows:
        if c in USED:
            continue
        tags = []
        if c in FAILED_ADD:
            tags.append("failed_add")
        if c in adj:
            tags.append("adjacent")
        print(f"  col={c:3d} ic={ic:+.6f} {' '.join(tags)}")
        n += 1
        if n >= 20:
            break


if __name__ == "__main__":
    main()
