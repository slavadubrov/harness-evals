"""Compare two harnesses on the same tasks and decide: adopt, reject or inconclusive."""

from __future__ import annotations

import operator
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import comb, isclose, isfinite
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


def sign_test(better: int, worse: int) -> float:
    """The two-sided p-value of a sign test over the tasks whose pass rate differs: how
    often chance alone splits them at least this unevenly. Tasks with no difference say
    nothing, so four better and none worse gives 0.125 however many tasks there are."""
    n, k = better + worse, min(better, worse)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def gate_failures(summary: dict[str, Any], gates: dict[str, tuple[str, float]]) -> list[str]:
    """The absolute bars a harness misses. A metric that cannot be computed misses its
    bar, and so does a cost bar while any run's cost is unknown."""
    return list(_missed(summary, gates).values())


def _missed(summary: dict[str, Any], gates: dict[str, tuple[str, float]]) -> dict[str, str]:
    """Each missed bar's metric, with the reason."""
    out = {}
    for metric, (op, limit) in gates.items():
        v = summary.get(metric)
        if metric.startswith("cost") and summary.get("unknown_cost"):
            out[metric] = f"{metric}: cost unknown for {summary['unknown_cost']} runs"
        elif v is None:
            out[metric] = f"{metric}: not measured"
        elif not OPS[op](v, limit):
            out[metric] = f"{metric} = {v:.4g}, needs {op} {limit:g}"
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
    level: float = 0.95  # the confidence level of every interval above


def compare(
    usecase: UseCase,
    base: Sequence[Grade],
    candidate: Sequence[Grade],
    *,
    seed: int = 0,
    step: bool = False,
    level: float = 0.95,
) -> Comparison:
    """Pair the harnesses task by task: both ran the same tasks, so per-task differences
    cancel the variation that comes from task difficulty. Every compared metric gets a
    paired interval from the same resampled tasks.

    ``step=True`` decides one step of an improvement loop instead of a release: a bar
    that the base misses too does not reject the candidate (see decide()). ``level`` is
    the confidence level of the paired intervals. For K proposals, a stricter level of
    ``1 - 0.05 / K`` shows which decisions still hold. With proposals chosen using earlier
    validation results, this does not guarantee a 5% campaign error rate."""
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")
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
    ci = bootstrap(d, seed=seed, level=level)
    deltas: dict[str, Delta | None] = {"success": (statistics.fmean(d), ci)}
    for name in dict.fromkeys([*COMPARED, *usecase.metrics, *usecase.guardrails]):
        if name in deltas:
            continue
        if name == "pass^k":  # like success, an average of per-task values: pair the tasks
            vb, vc = (per_task(gs, name, sb["trials"]) for gs in (base, candidate))
            dk = [vc[t] - vb[t] for t in shared]
            deltas[name] = (statistics.fmean(dk), bootstrap(dk, seed=seed, level=level))
            continue
        unknown = name.startswith("cost") and (sb["unknown_cost"] or sc["unknown_cost"])
        if unknown or sb[name] is None or sc[name] is None:
            deltas[name] = None
            continue
        metric = pooled(name, usecase.metrics)
        ci = paired_interval(
            [tb[t] for t in shared], [tc[t] for t in shared], metric, seed=seed, level=level
        )
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
        level=level,
    )
    c.decision, c.reasons = decide(
        usecase, sb, sc, deltas, step=step, changed=(len(c.better), len(c.worse)), level=level
    )
    return c


def decide(
    usecase: UseCase,
    sb: dict[str, Any],
    sc: dict[str, Any],
    deltas: Mapping[str, Delta | None],
    *,
    step: bool = False,
    changed: tuple[int, int] | None = None,
    level: float = 0.95,
) -> tuple[str, list[str]]:
    """The adoption rule. Without clear evidence for the candidate, keep the base.

    The gates come first: in a release, a base that misses one is replaced by any
    candidate that clears them all. Between two harnesses that clear the gates, success
    and cost per task are what a candidate may win on, and ``guardrails`` limit how much
    other metrics may get worse. ``deltas`` holds candidate minus base with a paired
    interval for success, cost per task and every guardrail; compare() computes it.

    ``step=True`` is the rule for one step of an improvement loop. A release asks whether
    the candidate may serve users, so a candidate below the bar is rejected. A step asks
    whether the candidate is a better base for the next change, so a bar that the base
    misses decides nothing either way: missing it too does not reject the candidate, and
    clearing it does not adopt the candidate. The intervals decide. A bar the base clears
    still rejects a candidate that misses it. A step is kept for higher success only, never for
    lower cost alone: two runs of one harness differ in tokens by sampling, and over a
    few dozen tasks that difference can look like a saving.

    ``changed`` is how many tasks the candidate passes more often and less often than the
    base. To adopt for higher success, more tasks must improve than worsen and the
    two-sided sign test must pass at ``level``. Without those counts the result for
    higher success is inconclusive."""
    if not isfinite(usecase.margin) or usecase.margin < 0:
        raise ValueError("margin must be finite and nonnegative")
    check_guardrails(usecase.guardrails, usecase.metrics)
    missed, base_missed = _missed(sc, usecase.gates), _missed(sb, usecase.gates)
    if new := [why for m, why in missed.items() if not (step and m in base_missed)]:
        return "reject", new
    if sc["violations"] > sb["violations"]:
        return "reject", [f"more critical violations: {sc['violations']} vs {sb['violations']}"]
    if base_missed and not missed and not step:
        return "adopt", ["the base misses the bar: " + "; ".join(base_missed.values())]
    # In a step, a bar the base misses decides nothing either way: the intervals decide.
    below = ["still below the bar: " + "; ".join(missed.values())] if missed else []
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
        if changed is None:
            return "inconclusive", [f"{win}, but the paired task counts are missing", *unsure]
        better, worse_count = changed
        p = sign_test(better, worse_count)
        if better <= worse_count or p > 1 - level:
            return "inconclusive", [
                f"success is higher on average ({lo:+.3f} to {hi:+.3f}), but the task counts"
                f" do not support the gain: {better} better, {worse_count} worse"
                f" (sign test p = {p:.3f}); keep the base",
                *unsure,
            ]
    elif isclose(lo, hi, rel_tol=0.0, abs_tol=1e-15):
        return "inconclusive", [
            "a zero-width success interval does not establish the allowed loss; keep the base",
            *unsure,
        ]
    elif not step and lo >= -m and known and cost is not None and cost[1][1] < 0:
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
    return "adopt", [win, *below]


def _num(name: str, v: float) -> str:
    if name.startswith("cost"):
        return f"{v:+.5f}"
    return f"{v:+.2f}" if name.startswith("latency") else f"{v:+.3f}"
