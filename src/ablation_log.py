"""Append or replace an experiment block in experiments.md. Never invent RankIC numbers."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments.md"
TEMPLATE_MARK = "模板："


def fmt_ic(value: float | None) -> str:
    if value is None or not isinstance(value, (int, float)):
        return "未算"
    return f"{float(value):.6f}"


def render_block(
    experiment_id: str,
    *,
    config: str,
    inputs: str,
    history: str,
    model: str,
    loss: str,
    valid_ic: float | None,
    baseline_ic: float | None,
    recomputed_baseline_ic: float | None,
    delta: float | None,
    ablations: list[dict[str, Any]],
    locked: dict[str, Any] | None,
    elapsed_s: float | None,
    conclusion: str,
    next_step: str,
    extra_lines: list[str] | None = None,
) -> str:
    lines = [
        f"## {experiment_id}",
        f"- 日期：{date.today().isoformat()}",
        f"- 代码/配置：`{config}`",
        f"- 输入特征：{inputs}",
        f"- 是否看历史：{history}",
        f"- 模型：{model}",
        f"- 损失：{loss}",
        f"- valid mean RankIC：{fmt_ic(valid_ic)}",
        f"- 对照基准 gru-no-today-recent-n2000-001 文档值：{fmt_ic(baseline_ic)}",
        f"- 对照基准本机复算：{fmt_ic(recomputed_baseline_ic)}",
        f"- Δ vs 文档基准：{fmt_ic(delta) if delta is not None else '未算'}",
        "- 消融：",
    ]
    if ablations:
        for row in ablations:
            name = row.get("name", "?")
            ic = fmt_ic(row.get("ic"))
            note = row.get("note") or ""
            extra = f"（{note}）" if note else ""
            lines.append(f"  - {name}: {ic}{extra}")
    else:
        lines.append("  - （无）")
    if locked:
        lock_bits = ", ".join(f"{k}={v}" for k, v in locked.items() if k != "ic")
        lines.append(f"- 锁定：{lock_bits}")
    lines.append("- train mean RankIC（如有）：未算")
    if elapsed_s is not None:
        lines.append(f"- 耗时 / 硬件：合计 {elapsed_s:.1f}s")
    else:
        lines.append("- 耗时 / 硬件：")
    lines.append(f"- 结论：{conclusion}")
    lines.append(f"- 下一步：{next_step}")
    if extra_lines:
        lines.extend(extra_lines)
    return "\n".join(lines) + "\n"


def upsert_experiment(experiment_id: str, body: str, path: Path | None = None) -> Path:
    dest = EXPERIMENTS if path is None else Path(path)
    text = dest.read_text(encoding="utf-8") if dest.is_file() else "# 实验记录\n\n"
    header = f"## {experiment_id}"
    start = text.find(header)
    if start >= 0:
        rest = text[start + len(header) :]
        nxt = rest.find("\n## ")
        end = start + len(header) + nxt if nxt >= 0 else len(text)
        tmpl = text.find(f"\n{TEMPLATE_MARK}", start)
        if tmpl >= 0 and tmpl < end:
            end = tmpl
        text = text[:start].rstrip() + "\n\n" + body.strip() + "\n\n" + text[end:].lstrip()
    else:
        mark = f"\n{TEMPLATE_MARK}"
        pos = text.find(mark)
        if pos >= 0:
            text = text[:pos].rstrip() + "\n\n" + body.strip() + "\n\n" + text[pos:].lstrip()
        else:
            text = text.rstrip() + "\n\n" + body.strip() + "\n"
    dest.write_text(text, encoding="utf-8")
    return dest


def maybe_update_leader(valid_ic: float, experiment_id: str, path: Path | None = None) -> None:
    """Only bump the top-of-file '当前最强' line when the new score is strictly better."""
    dest = EXPERIMENTS if path is None else Path(path)
    if not dest.is_file() or not np_isfinite(valid_ic):
        return
    text = dest.read_text(encoding="utf-8")
    prefix = "**当前最强：valid mean RankIC "
    i = text.find(prefix)
    if i < 0:
        return
    j = i + len(prefix)
    k = text.find("**", j)
    if k < 0:
        return
    try:
        old = float(text[j:k].split("（")[0])
    except ValueError:
        return
    if float(valid_ic) <= old + 1e-12:
        return
    new_head = (
        f"{prefix}{float(valid_ic):.6f}**（`{experiment_id}`）。"
        f"{'valid 已过 0.12。' if float(valid_ic) >= 0.12 else '门槛 0.12 未到。'}"
    )
    # replace the rest of that paragraph's first sentence only
    nl = text.find("\n", i)
    text = text[:i] + new_head + (text[nl:] if nl >= 0 else "")
    dest.write_text(text, encoding="utf-8")


def np_isfinite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


maybe_update_leader = maybe_update_leader
upsert_experiment = upsert_experiment
render_block = render_block
