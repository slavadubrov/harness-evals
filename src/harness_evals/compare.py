"""Compare two harnesses on the same tasks and decide: adopt, reject or inconclusive."""

from __future__ import annotations

import operator
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isclose, isfinite
from typing import Any

from harness_evals.core import Grade
from harness_evals.metrics import (
    bootstrap,
    by_task,
    paired_interval,
    per_task,
    pooled,
    summarize,
    task_rates,
)
from harness_evals.usecase import UseCase, check_guardrails

OPS = {">=": operator.ge, "<=": operator.le}
Delta = tuple[float, tuple[float, float]]  # candidate minus base, and its 95% interval
# Compared in every comparison, beside success and the use case's metrics and guardrails.
COMPARED = ("cost_per_task", "latency_p50", "latency_p95", "calls_per_task", "tokens_per_task")
# Minimum sample policy for interval-based decisions; this does not guarantee coverage.
MIN_TASKS = 6


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
    # Candidate minus base for every compared metric, with a paired interval; None when a
    # metric is unknown for either harness.
    deltas: dict[str, Delta | None] = field(default_factory=dict)


def compare(
    usecase: UseCase,
    base: Sequence[Grade],
    candidate: Sequence[Grade],
    *,
    seed: int = 0,
) -> Comparison:
    """Pair the harnesses task by task: both ran the same tasks, so per-task differences
    cancel the variation that comes from task difficulty. Every compared metric gets a
    paired interval from the same resampled tasks."""
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
    tb, tc = by_task(base), by_task(candidate)
    deltas: dict[str, Delta | None] = {"success": (statistics.fmean(d), bootstrap(d, seed=seed))}
    for name in dict.fromkeys([*COMPARED, *usecase.metrics, *usecase.guardrails]):
        if name in deltas:
            continue
        if name == "pass^k":  # like success, an average of per-task values: pair the tasks
            vb, vc = (per_task(gs, name, sb["trials"]) for gs in (base, candidate))
            dk = [vc[t] - vb[t] for t in shared]
            deltas[name] = (statistics.fmean(dk), bootstrap(dk, seed=seed))
            continue
        unknown = name.startswith("cost") and (sb["unknown_cost"] or sc["unknown_cost"])
        if unknown or sb[name] is None or sc[name] is None:
            deltas[name] = None
            continue
        metric = pooled(name, usecase.metrics)
        ci = paired_interval([tb[t] for t in shared], [tc[t] for t in shared], metric, seed=seed)
        deltas[name] = None if ci is None else (sc[name] - sb[name], ci)
    diff, ci = deltas["success"]
    c = Comparison(
        base=names[0],
        candidate=names[1],
        tasks=len(shared),
        diff=diff,
        ci=ci,
        better=[t for t in shared if rc[t] > rb[t]],
        worse=[t for t in shared if rc[t] < rb[t]],
        summaries={names[0]: sb, names[1]: sc},
        deltas=deltas,
    )
    c.decision, c.reasons = decide(usecase, sb, sc, deltas)
    return c


def decide(
    usecase: UseCase,
    sb: dict[str, Any],
    sc: dict[str, Any],
    deltas: Mapping[str, Delta | None],
) -> tuple[str, list[str]]:
    """The adoption rule. Without clear evidence for the candidate, keep the base.

    The gates come first: a base that misses one is replaced by any candidate that clears
    them all. Between two harnesses that clear the gates, success and cost per task are
    what a candidate may win on, and ``guardrails`` limit how much other metrics may get
    worse. ``deltas`` holds candidate minus base with a paired interval for success, cost
    per task and every guardrail; compare() computes it."""
    if not isfinite(usecase.margin) or usecase.margin < 0:
        raise ValueError("margin must be finite and nonnegative")
    check_guardrails(usecase.guardrails, usecase.metrics)
    if missed := gate_failures(sc, usecase.gates):
        return "reject", missed
    if sc["violations"] > sb["violations"]:
        return "reject", [f"more critical violations: {sc['violations']} vs {sb['violations']}"]
    if base_missed := gate_failures(sb, usecase.gates):
        return "adopt", ["the base misses the bar: " + "; ".join(base_missed)]
    if sb.get("tasks", MIN_TASKS) < MIN_TASKS:
        return "inconclusive", [
            f"{sb['tasks']} tasks are too few for an interval to decide; keep the base and"
            f" add tasks (at least {MIN_TASKS})"
        ]
    _, (lo, hi) = deltas["success"]
    m = usecase.margin
    worse = [f"success is lower: {lo:+.3f} to {hi:+.3f}"] if hi < -m else []
    unsure = []
    for name, (op, limit) in usecase.guardrails.items():
        delta = deltas.get(name)
        if delta is None:
            unsure.append(f"{name} could not be compared")
            continue
        _, (dlo, dhi) = delta
        span = f"{name} changes by {_num(name, dlo)} to {_num(name, dhi)}"
        if not OPS[op](dhi if op == ">=" else dlo, limit):  # the whole interval is past it
            worse.append(f"{span}, past its limit {op} {limit:+g}")
        elif not OPS[op](dlo if op == ">=" else dhi, limit):
            unsure.append(f"{span}, which may pass its limit {op} {limit:+g}")
    if worse:
        return "reject", worse
    cost = deltas.get("cost_per_task")
    known = not (sb.get("unknown_cost") or sc.get("unknown_cost"))
    if lo > 0:
        win = f"success is higher: {lo:+.3f} to {hi:+.3f}"
    elif isclose(lo, hi, rel_tol=0.0, abs_tol=1e-15):
        return "inconclusive", [
            "a zero-width success interval does not establish the allowed loss; keep the base",
            *unsure,
        ]
    elif lo >= -m and known and cost is not None and cost[1][1] < 0:
        clo, chi = cost[1]
        win = (
            f"no worse than -{m:g} ({lo:+.3f} to {hi:+.3f}) and cheaper per task"
            f" ({_num('cost_per_task', clo)} to {_num('cost_per_task', chi)})"
        )
    else:
        return "inconclusive", [
            f"success difference {lo:+.3f} to {hi:+.3f} does not establish a benefit;"
            " keep the base",
            *unsure,
        ]
    if unsure:
        return "inconclusive", [f"{win}, but a guardrail may break; keep the base", *unsure]
    return "adopt", [win]


def _num(name: str, v: float) -> str:
    if name.startswith("cost"):
        return f"{v:+.5f}"
    return f"{v:+.2f}" if name.startswith("latency") else f"{v:+.3f}"
