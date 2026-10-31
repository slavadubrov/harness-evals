"""Compare two harnesses on the same tasks and decide: adopt, reject or inconclusive."""

from __future__ import annotations

import operator
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from harness_evals.core import Grade
from harness_evals.metrics import bootstrap, summarize, task_rates
from harness_evals.usecase import UseCase

OPS = {">=": operator.ge, "<=": operator.le}


def gate_failures(summary: dict[str, Any], gates: dict[str, tuple[str, float]]) -> list[str]:
    """The absolute bars a harness misses. A metric that cannot be computed misses its
    bar, and so does a cost bar while any run's cost is unknown."""
    out = []
    for metric, (op, limit) in gates.items():
        v = summary.get(metric)
        if metric.startswith("cost") and summary.get("unknown_cost"):
            out.append(f"{metric}: cost unknown for {summary['unknown_cost']} runs")
        elif v is None:
            out.append(f"{metric}: not measured")
        elif not OPS[op](v, limit):
            out.append(f"{metric} = {v:.4g}, needs {op} {limit:g}")
    return out


@dataclass
class Comparison:
    base: str
    candidate: str
    tasks: int  # tasks both harnesses ran
    diff: float  # candidate success minus base success, averaged over shared tasks
    ci: tuple[float, float]  # paired interval of diff, resampling tasks
    better: list[str]  # tasks the candidate passed more often
    worse: list[str]
    summaries: dict[str, dict[str, Any]]
    decision: str = ""  # adopt, reject or inconclusive
    reasons: list[str] = field(default_factory=list)


def compare(
    usecase: UseCase,
    base: Sequence[Grade],
    candidate: Sequence[Grade],
    *,
    seed: int = 0,
) -> Comparison:
    """Pair the harnesses task by task: both ran the same tasks, so per-task differences
    cancel the variation that comes from task difficulty."""
    rb, rc = task_rates(base), task_rates(candidate)
    shared = sorted(rb.keys() & rc.keys())
    if not shared:
        raise ValueError("the harnesses share no tasks")
    d = [rc[t] - rb[t] for t in shared]
    names = (base[0].episode.harness or "base", candidate[0].episode.harness or "candidate")
    sb = summarize(base, usecase.metrics, seed)
    sc = summarize(candidate, usecase.metrics, seed)
    c = Comparison(
        base=names[0],
        candidate=names[1],
        tasks=len(shared),
        diff=statistics.fmean(d),
        ci=bootstrap(d, seed=seed),
        better=[t for t in shared if rc[t] > rb[t]],
        worse=[t for t in shared if rc[t] < rb[t]],
        summaries={names[0]: sb, names[1]: sc},
    )
    c.decision, c.reasons = decide(usecase, sb, sc, c.ci)
    return c


def decide(
    usecase: UseCase, sb: dict[str, Any], sc: dict[str, Any], ci: tuple[float, float]
) -> tuple[str, list[str]]:
    """The adoption rule. Without clear evidence for the candidate, keep the base."""
    if missed := gate_failures(sc, usecase.gates):
        return "reject", missed
    if sc["violations"] > sb["violations"]:
        return "reject", [f"more critical violations: {sc['violations']} vs {sb['violations']}"]
    if base_missed := gate_failures(sb, usecase.gates):
        return "adopt", ["the base misses the bar: " + "; ".join(base_missed)]
    lo, hi = ci
    m = usecase.margin
    if hi < -m:
        return "reject", [f"success is lower: {lo:+.3f} to {hi:+.3f}"]
    if lo > 0:
        return "adopt", [f"success is higher: {lo:+.3f} to {hi:+.3f}"]
    cb, cc = sb["cost_per_task"], sc["cost_per_task"]
    if m > 0 and lo > -m and cb is not None and cc is not None and cc < cb:
        return "adopt", [f"no worse than -{m:g} ({lo:+.3f} to {hi:+.3f}) and cheaper per task"]
    return "inconclusive", [
        f"success difference {lo:+.3f} to {hi:+.3f} includes no change; keep the base"
    ]
