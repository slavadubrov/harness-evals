"""Markdown tables for a run's summaries and comparisons."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from harness_evals.compare import Comparison

REQUIRED = [
    ("success", "Success"),
    ("pass^k", "pass^k"),
    ("violations", "Violations"),
    ("cost_per_task", "$ per task"),
    ("cost_per_success", "$ per success"),
    ("latency_p95", "p95 latency, s"),
    ("infra_errors", "Infra errors"),
    ("unknown_verdicts", "Undecided"),
]


def _fmt(key: str, v: Any, s: Mapping[str, Any]) -> str:
    if v is None:
        return "unknown"
    if key == "success":
        lo, hi = s["success_ci"]
        return f"{v:.2f} ({lo:.2f}–{hi:.2f})"
    if key == "pass^k":
        return f"{v:.2f} (k={s['trials']})"
    if key.startswith("cost"):
        out = f"${v:.5f}"
        return out + f" ({s['unknown_cost']} unknown)" if s.get("unknown_cost") else out
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def table(summaries: Mapping[str, Mapping[str, Any]], extra: list[str] | None = None) -> str:
    cols = REQUIRED + [(k, k) for k in extra or []]
    lines = [
        "| Harness | " + " | ".join(label for _, label in cols) + " |",
        "|---" * (len(cols) + 1) + "|",
    ]
    for name, s in summaries.items():
        cells = [_fmt(k, s.get(k), s) for k, _ in cols]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def comparison(c: Comparison) -> str:
    lo, hi = c.ci
    return "\n".join(
        [
            f"**{c.candidate}** against **{c.base}** on {c.tasks} tasks: "
            f"success {c.diff:+.3f} ({lo:+.3f} to {hi:+.3f}).",
            f"Better on {len(c.better)} tasks, worse on {len(c.worse)}.",
            f"Decision: **{c.decision}**. " + " ".join(c.reasons),
        ]
    )
