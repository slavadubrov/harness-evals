"""Compare two harnesses on the same tasks and decide: adopt, reject or inconclusive."""

from __future__ import annotations

import operator
import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from math import isfinite
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
    for grades in (base, candidate):
        if len({g.episode.harness for g in grades}) != 1:
            raise ValueError("each comparison input must contain exactly one harness")
        trials = [(g.episode.task_id, g.episode.trial) for g in grades]
        if len(trials) != len(set(trials)):
            raise ValueError("comparison input contains duplicate task trials")
    if Counter(g.episode.task_id for g in base) != Counter(g.episode.task_id for g in candidate):
        raise ValueError("the harnesses must run the same tasks and number of trials per task")
    rb, rc = task_rates(base), task_rates(candidate)
    shared = sorted(rb)
    d = [rc[t] - rb[t] for t in shared]
    names = (base[0].episode.harness or "base", candidate[0].episode.harness or "candidate")
    if names[0] == names[1]:  # two versions of one harness
        names = (f"{names[0]} (base)", f"{names[1]} (candidate)")
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
    if not isfinite(usecase.margin) or usecase.margin < 0:
        raise ValueError("margin must be finite and nonnegative")
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
    if (
        lo >= -m
        and not (sb.get("unknown_cost") or sc.get("unknown_cost"))
        and cb is not None
        and cc is not None
        and cc < cb
    ):
        return "adopt", [f"no worse than -{m:g} ({lo:+.3f} to {hi:+.3f}) and cheaper per task"]
    return "inconclusive", [
        f"success difference {lo:+.3f} to {hi:+.3f} does not establish a benefit; keep the base"
    ]
