"""Compare two harnesses on the same tasks and decide: adopt, reject or inconclusive."""

from __future__ import annotations

import operator
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from math import comb, isclose, isfinite
from typing import Any

from harness_evals.core import Grade
from harness_evals.metrics import (
    LOWER_IS_BETTER,
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
ZERO = 1e-12  # a paired difference smaller than this counts as no change


def sign_test(better: int, worse: int) -> float:
    """The two-sided p-value of a sign test over the tasks whose pass rate differs: how
    often chance alone splits them at least this unevenly. Tasks with no difference say
    nothing, so four better and none worse gives 0.125 however many tasks there are.

    No decision uses this test; decide() uses wilcoxon_test(). It stays for comparison
    with the rule, as in the saved support campaign, which was decided with it."""
    n, k = better + worse, min(better, worse)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def wilcoxon_test(diffs: Sequence[float]) -> float:
    """The exact two-sided p-value of the Wilcoxon signed-rank test on paired differences.
    Zero differences are dropped. The rest are ranked by size, with average ranks for
    ties, and the statistic is the sum of the ranks of the positive differences. The
    p-value is how often random signs put that sum at least as far from its mean."""
    return _wilcoxon(diffs)[2]


def _wilcoxon(diffs: Sequence[float]) -> tuple[int, int, float]:
    """Twice the positive and the negative rank sums, and the two-sided p-value."""
    d = sorted((x for x in diffs if abs(x) >= ZERO), key=abs)
    if not d:
        return 0, 0, 1.0
    ranks = [0] * len(d)  # doubled average ranks, so ties stay integers
    i = 0
    while i < len(d):
        j = i
        while j + 1 < len(d) and abs(d[j + 1]) - abs(d[i]) < ZERO:
            j += 1
        ranks[i : j + 1] = [i + j + 2] * (j + 1 - i)
        i = j + 1
    total = sum(ranks)
    pos = sum(r for r, x in zip(ranks, d, strict=True) if x > 0)
    counts = [1] + [0] * total  # counts[w]: sign assignments whose positive sum is w
    for r in ranks:
        for w in range(total, r - 1, -1):
            counts[w] += counts[w - r]
    far = abs(2 * pos - total)
    extreme = sum(c for w, c in enumerate(counts) if abs(2 * w - total) >= far)
    return pos, total - pos, extreme / 2 ** len(d)


def run_permutation_test(
    base_runs: Sequence[Sequence[bool]], candidate_runs: Sequence[Sequence[bool]]
) -> float:
    """The exact two-sided p-value of a stratified permutation test over runs.

    ``base_runs[i]`` and ``candidate_runs[i]`` hold the pass results of task i's runs in
    each harness; the two may have different numbers of runs. If the change does nothing,
    the labels base and candidate can be swapped between the runs of one task. The
    statistic is the candidate's number of passed runs over all tasks. In one task it
    follows the hypergeometric distribution, and the tasks are independent. The p-value
    is the chance of a total at least as far from its mean as the observed one."""
    return _permutation(base_runs, candidate_runs)[1]


def _permutation(
    base_runs: Sequence[Sequence[bool]], candidate_runs: Sequence[Sequence[bool]]
) -> tuple[Fraction, float]:
    """The candidate's passed runs minus their mean under the null, and the p-value."""
    counts, total, obs, shift = [1], 1, 0, Fraction(0)
    for b, c in zip(base_runs, candidate_runs, strict=True):
        nb, nc = len(b), len(c)
        if not nb or not nc:
            raise ValueError("every task needs at least one run of each harness")
        k = sum(map(bool, b)) + sum(map(bool, c))
        shift += sum(map(bool, c)) - Fraction(k * nc, nb + nc)
        if k in (0, nb + nc):  # every relabelling gives the same count
            continue
        obs += sum(map(bool, c))
        task = [comb(k, x) * comb(nb + nc - k, nc - x) for x in range(nc + 1)]
        new = [0] * (len(counts) + nc)
        for i, a in enumerate(counts):
            for x, w in enumerate(task):
                new[i + x] += a * w
        counts, total = new, total * comb(nb + nc, nc)
    mean = obs - shift
    extreme = sum(n for x, n in enumerate(counts) if abs(x - mean) >= abs(shift))
    return shift, extreme / total


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
    objective: str = "success",
) -> Comparison:
    """Pair the harnesses task by task: both ran the same tasks, so per-task differences
    cancel the variation that comes from task difficulty. Every compared metric gets a
    paired interval from the same resampled tasks.

    ``step=True`` decides one step of an improvement loop instead of a release: a bar
    that the base misses too does not reject the candidate (see decide()). ``level`` is
    the confidence level of the paired intervals. For K proposals, a stricter level of
    ``1 - 0.05 / K`` shows which decisions still hold. With proposals chosen using earlier
    validation results, this does not guarantee a 5% campaign error rate.

    ``objective`` names the metric a step tries to improve; see decide()."""
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
        usecase,
        sb,
        sc,
        deltas,
        step=step,
        changed=(len(c.better), len(c.worse)),
        diffs=d,
        level=level,
        objective=objective,
        runs=tuple([[g.passed for g in ts[t]] for t in shared] for ts in (tb, tc)),
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
    diffs: Sequence[float] | None = None,
    level: float = 0.95,
    objective: str = "success",
    runs: tuple[Sequence[Sequence[bool]], Sequence[Sequence[bool]]] | None = None,
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
    clearing it does not adopt the candidate. A bar the base clears still rejects a
    candidate that misses it. With the default objective, a step is kept for higher
    success only, never for lower cost alone: two runs of one harness differ in tokens by
    sampling, and over a few dozen tasks that difference can look like a saving.

    In a release, ``diffs`` holds the per-task success differences, candidate minus base,
    and ``changed`` how many tasks the candidate passes more often and less often than the
    base. To adopt for higher success, the success interval must lie above zero, the ranks
    of the tasks that improve must sum to more than the ranks of the tasks that get worse,
    and the two-sided Wilcoxon signed-rank test must pass at ``level``. Without ``diffs``
    the result for higher success is inconclusive. Without ``changed`` the counts come
    from ``diffs``.

    In a step that tries to raise success, the runs decide and the success interval is
    only reported. ``runs`` holds the pass results of each task's runs, base first, then
    candidate, with the tasks in the same order. A task is broken when its own runs show a
    loss that run noise gives in at most ``1 - level`` of cases: the one-sided Fisher exact
    test on the task's passed and failed runs. Any broken task rejects the step. Otherwise
    the step is adopted only if the candidate passes more runs than the null mean and
    run_permutation_test() passes at ``level``. Without ``runs`` the result for higher
    success is inconclusive.

    ``objective`` other than success is a lower-is-better metric in ``deltas``, such as
    ``cost_per_task``, and applies to steps only. Gates, violations and guardrails apply
    as above. Success must hold: the step is rejected if the whole success interval is
    below ``-margin`` and inconclusive if its lower bound is. The step is adopted only if
    the whole interval of the objective is below zero. A guardrail on the objective itself
    rejects only when the whole interval is past its limit."""
    _check_objective(objective, deltas, step)
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
    runs_decide = step and objective == "success"  # the success interval is only reported
    worse = [f"success is lower: {lo:+.3f} to {hi:+.3f}"] if hi < -m and not runs_decide else []
    unsure = []
    for name, (op, limit) in usecase.guardrails.items():
        own = name == objective != "success"  # the objective decides its own direction
        delta = deltas.get(name)
        if delta is None:
            if not own:
                unsure.append(f"{name} could not be compared")
            continue
        _, (dlo, dhi) = delta
        span = f"{name} changes by {_num(name, dlo)} to {_num(name, dhi)}"
        if not OPS[op](dhi if op == ">=" else dlo, limit):  # the whole interval is past it
            worse.append(f"{span}, past its limit {op} {limit:+g}")
        elif not own and not OPS[op](dlo if op == ">=" else dhi, limit):
            unsure.append(f"{span}, which may pass its limit {op} {limit:+g}")
    if worse:
        return "reject", worse
    if objective != "success":
        return _objective_step(objective, deltas, (lo, hi), m, unsure, below)
    if step:
        return _success_step(deltas["success"], runs, level, unsure, below)
    cost = deltas.get("cost_per_task")
    known = not (sb.get("unknown_cost") or sc.get("unknown_cost"))
    if lo > 0:
        win = f"success is higher: {lo:+.3f} to {hi:+.3f}"
        if diffs is None:
            return "inconclusive", [f"{win}, but the paired task differences are missing", *unsure]
        better, worse_count = changed or (
            sum(x >= ZERO for x in diffs),
            sum(x <= -ZERO for x in diffs),
        )
        pos, neg, p = _wilcoxon(diffs)
        if pos <= neg or p > 1 - level:
            return "inconclusive", [
                f"success is higher on average ({lo:+.3f} to {hi:+.3f}), but the Wilcoxon"
                f" signed-rank test does not support the gain: {better} better,"
                f" {worse_count} worse, p = {p:.3f}; keep the base",
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


def _check_objective(objective: str, deltas: Mapping[str, Delta | None], step: bool) -> None:
    if objective == "success":
        return
    if not step:
        raise ValueError("objectives other than success apply to steps only: pass step=True")
    if objective not in LOWER_IS_BETTER or objective not in deltas:
        known = sorted(m for m in deltas if m in LOWER_IS_BETTER)
        raise ValueError(
            f"objective {objective!r}: use success or a lower-is-better metric with a paired"
            f" delta: {', '.join(known)}"
        )


def _objective_step(
    name: str,
    deltas: Mapping[str, Delta | None],
    success: tuple[float, float],
    m: float,
    unsure: list[str],
    below: list[str],
) -> tuple[str, list[str]]:
    """The end of decide() for a step that optimizes ``name`` while success holds."""
    lo, hi = success
    if lo < -m:
        return "inconclusive", [
            f"success may be lower by more than {m:g}: {lo:+.3f} to {hi:+.3f}; keep the base",
            *unsure,
        ]
    held = f"success holds: {lo:+.3f} to {hi:+.3f}, no lower than -{m:g}"
    delta = deltas[name]
    if delta is None:
        return "inconclusive", [f"{name} could not be compared; keep the base", held, *unsure]
    _, (olo, ohi) = delta
    span = f"{_num(name, olo)} to {_num(name, ohi)}"
    if ohi >= 0:
        return "inconclusive", [
            f"{name} changes by {span}, which does not establish a decrease; keep the base",
            held,
            *unsure,
        ]
    win = f"{name} is lower: {span}"
    if unsure:
        return "inconclusive", [f"{win}, but a guardrail may break; keep the base", held, *unsure]
    return "adopt", [win, held, *below]


def _success_step(
    success: Delta,
    runs: tuple[Sequence[Sequence[bool]], Sequence[Sequence[bool]]] | None,
    level: float,
    unsure: list[str],
    below: list[str],
) -> tuple[str, list[str]]:
    """The end of decide() for a step that tries to raise success."""
    d, (lo, hi) = success
    gain = f"success is higher by {d:+.3f} (interval {lo:+.3f} to {hi:+.3f})"
    if runs is None:
        return "inconclusive", [
            f"success changes by {d:+.3f}, but the run results of each task are missing;"
            " keep the base",
            *unsure,
        ]
    rates = [
        (Fraction(sum(map(bool, b)), len(b)), Fraction(sum(map(bool, c)), len(c)))
        for b, c in zip(*runs, strict=True)
    ]
    if broken := sum(_fisher_loss(b, c) <= 1 - level for b, c in zip(*runs, strict=True)):
        return "reject", [
            f"{broken} task{'s fail' if broken != 1 else ' fails'} runs"
            f" {'they' if broken != 1 else 'it'} passed with the base, by more than run noise"
            " explains; keep the base"
        ]
    shift, p = _permutation(*runs)
    better, worse = sum(rc > rb for rb, rc in rates), sum(rc < rb for rb, rc in rates)
    counts = f"{better} task{'s' * (better != 1)} better, {worse} worse, " + (
        f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"
    )
    if shift <= 0:
        return "inconclusive", [
            f"success difference {lo:+.3f} to {hi:+.3f} does not establish a benefit;"
            " keep the base",
            *unsure,
        ]
    if p > 1 - level:
        return "inconclusive", [
            f"{gain}, but the difference between runs does not support the gain: {counts};"
            " keep the base",
            *unsure,
        ]
    win = f"{gain}: {counts}"
    if unsure:
        return "inconclusive", [f"{win}, but a guardrail may break; keep the base", *unsure]
    return "adopt", [win, *below]


def _fisher_loss(base: Sequence[bool], candidate: Sequence[bool]) -> Fraction:
    """The one-sided Fisher exact p-value that one task's candidate runs pass fewer times
    than its base runs: the chance that the task's passed runs, dealt at random between
    the two harnesses, leave the candidate this few or fewer."""
    nb, nc = len(base), len(candidate)
    if not nb or not nc:
        raise ValueError("every task needs at least one run of each harness")
    c = sum(map(bool, candidate))
    k = sum(map(bool, base)) + c
    low = sum(comb(k, x) * comb(nb + nc - k, nc - x) for x in range(c + 1))
    return Fraction(low, comb(nb + nc, nc))


def _num(name: str, v: float) -> str:
    if name.startswith("cost"):
        return f"{v:+.5f}"
    return f"{v:+.2f}" if name.startswith("latency") else f"{v:+.3f}"
