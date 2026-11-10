"""The numbers every harness evaluation reports, with intervals computed over tasks.

Tasks are the unit of resampling: trials of one task are not independent evidence about
other tasks, so every interval resamples whole tasks with all their trials. Each metric
gets its own interval this way, so a design can be placed on two axes at once, such as
success against cost, with uncertainty on both.
"""

from __future__ import annotations

import random
import statistics
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from math import ceil, comb
from typing import Any

from harness_evals.core import Grade

Metric = Callable[[list[Grade]], float | None]
Tasks = Sequence[list[Grade]]  # each task's grades; a resample may draw a task twice

# Metrics where a smaller value is better; every other metric is better when larger.
LOWER_IS_BETTER = frozenset(
    {
        "violations",
        "cost_per_task",
        "cost_per_success",
        "latency_p50",
        "latency_p95",
        "calls_per_task",
        "tokens_per_task",
        "infra_errors",
        "unknown_verdicts",
        "unknown_cost",
    }
)


def by_task(grades: Sequence[Grade]) -> dict[str, list[Grade]]:
    out: dict[str, list[Grade]] = defaultdict(list)
    for g in grades:
        out[g.episode.task_id].append(g)
    return dict(out)


def by_harness(grades: Sequence[Grade]) -> dict[str, list[Grade]]:
    """The grades of each harness, in the order the harnesses first appear."""
    out: dict[str, list[Grade]] = defaultdict(list)
    for g in grades:
        out[g.episode.harness].append(g)
    return dict(out)


def task_rates(grades: Sequence[Grade]) -> dict[str, float]:
    """The share of trials each task passed."""
    return {t: sum(g.passed for g in gs) / len(gs) for t, gs in by_task(grades).items()}


def pass_hat_k(grades: Sequence[Grade], k: int) -> float:
    """pass^k: the chance that k independent trials of a task all pass, averaged over
    tasks. With n trials and c passes the unbiased estimate is C(c, k) / C(n, k)."""
    return statistics.fmean(_per_task(grades, k, lambda n, c: comb(c, k) / comb(n, k)))


def pass_at_k(grades: Sequence[Grade], k: int) -> float:
    """pass@k: the chance that at least one of k trials passes, averaged over tasks.
    Useful only when something reliable picks the passing trial, such as a test suite."""
    return statistics.fmean(_per_task(grades, k, lambda n, c: 1 - comb(n - c, k) / comb(n, k)))


def _per_task(grades: Sequence[Grade], k: int, f: Callable[[int, int], float]) -> list[float]:
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    out = []
    for t, gs in by_task(grades).items():
        n, c = len(gs), sum(g.passed for g in gs)
        if n < k:
            raise ValueError(f"task {t} has {n} trials, fewer than k={k}")
        out.append(f(n, c))
    return out


def bootstrap(
    values: Sequence[float], *, n: int = 2000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """Percentile interval of the mean, resampling the values with replacement."""
    if not values:
        raise ValueError("no values to resample")
    rng = random.Random(seed)
    k = len(values)
    means = [statistics.fmean(rng.choices(values, k=k)) for _ in range(n)]
    return _percentiles(means, level)


def percentile(xs: Sequence[float], p: float) -> float | None:
    """Nearest-rank percentile; None for an empty list."""
    if not xs:
        return None
    s = sorted(xs)
    return s[max(0, ceil(p / 100 * len(s)) - 1)]


def cost_per_task(grades: list[Grade]) -> float | None:
    """Mean cost of the runs whose cost is known; ``unknown_cost`` counts the rest."""
    known = [g.episode.cost_usd for g in grades if g.episode.cost_usd is not None]
    return statistics.fmean(known) if known else None


def cost_per_success(grades: list[Grade]) -> float | None:
    """Total cost divided by passing runs; unknown while any run's cost is unknown."""
    known = [g.episode.cost_usd for g in grades if g.episode.cost_usd is not None]
    passes = sum(g.passed for g in grades)
    return sum(known) / passes if passes and len(known) == len(grades) else None


def latency(p: float) -> Metric:
    """The p-th percentile of run latency; unknown while any run's latency is unknown."""

    def f(grades: list[Grade]) -> float | None:
        lat = [g.episode.latency_s for g in grades if g.episode.latency_s is not None]
        return percentile(lat, p) if len(lat) == len(grades) else None

    return f


def usage(field: str) -> Metric:
    """Mean model calls or tokens per run; unknown while any run does not report them.
    Unlike cost, they do not move with prices or provider cache hits."""

    def f(grades: list[Grade]) -> float | None:
        if field == "tokens":
            pairs = [(g.episode.input_tokens, g.episode.output_tokens) for g in grades]
            vals = [i + o for i, o in pairs if i is not None and o is not None]
        else:
            vals = [g.episode.model_calls for g in grades if g.episode.model_calls is not None]
        return statistics.fmean(vals) if vals and len(vals) == len(grades) else None

    return f


# Metrics of the pooled runs. A resample pools the runs of the tasks it drew.
POOLED: dict[str, Metric] = {
    "cost_per_task": cost_per_task,
    "cost_per_success": cost_per_success,
    "latency_p50": latency(50),
    "latency_p95": latency(95),
    "calls_per_task": usage("calls"),
    "tokens_per_task": usage("tokens"),
}


def pooled(name: str, custom: Mapping[str, Metric] | None = None) -> Metric:
    """The function behind a cost, latency or use-case metric, applied to pooled runs."""
    fn = POOLED.get(name) or (custom or {}).get(name)
    if fn is None:
        raise ValueError(f"no pooled metric named {name!r}")
    return fn


def per_task(grades: Sequence[Grade], name: str, k: int) -> dict[str, float]:
    """Success or pass^k of each task; the summary averages them over tasks."""
    if name == "success":
        return task_rates(grades)
    if name == "pass^k":
        values = _per_task(grades, k, lambda n, c: comb(c, k) / comb(n, k))
        return dict(zip(by_task(grades), values, strict=True))
    raise ValueError(f"no per-task metric named {name!r}")


def task_interval(
    tasks: Tasks,
    metric: Metric,
    *,
    n: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float] | None:
    """Percentile interval of ``metric`` on the pooled runs of ``n`` resamples of whole
    tasks. A metric that groups runs by task sees a task drawn twice as one task, so give
    use-case metrics a per-run definition, such as a share or a mean over runs."""
    values = [_safe(metric, _flat(tasks, idx)) for idx in _draws(len(tasks), n, seed)]
    return _percentiles(values, level)


def paired_interval(
    base: Tasks,
    candidate: Tasks,
    metric: Metric,
    *,
    n: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float] | None:
    """Interval of metric(candidate) minus metric(base). Both hold the same tasks in the
    same order, and each resample draws the same tasks for both harnesses."""
    diffs = []
    for idx in _draws(len(base), n, seed):
        b, c = _safe(metric, _flat(base, idx)), _safe(metric, _flat(candidate, idx))
        diffs.append(None if b is None or c is None else c - b)
    return _percentiles(diffs, level)


def _draws(m: int, n: int, seed: int) -> Iterable[list[int]]:
    # The same generator calls as bootstrap(): one seed draws the same tasks in both.
    rng = random.Random(seed)
    return (rng.choices(range(m), k=m) for _ in range(n))


def _flat(tasks: Tasks, idx: Sequence[int]) -> list[Grade]:
    return [g for i in idx for g in tasks[i]]


def _safe(metric: Metric, grades: list[Grade]) -> float | None:
    """A metric that cannot be computed on a resample, such as a share of no runs, is
    undefined there rather than an error."""
    try:
        return metric(grades)
    except (ArithmeticError, ValueError):
        return None


def _percentiles(values: Sequence[float | None], level: float) -> tuple[float, float] | None:
    """The middle ``level`` of the values. Undefined values (None) may fill at most one
    tail: if more resamples are undefined than that, the interval is unknown too."""
    known = sorted(v for v in values if v is not None)
    if not known or len(values) - len(known) > (1 - level) / 2 * len(values):
        return None
    n = len(known)
    return known[int((1 - level) / 2 * n)], known[min(n - 1, int((1 + level) / 2 * n))]


def summarize(
    grades: Sequence[Grade], metrics: dict[str, Metric] | None = None, seed: int = 0
) -> dict[str, Any]:
    """The required metrics, plus any use-case metrics passed in ``metrics``. Rates, costs,
    latencies, usage and use-case metrics come with a 95% interval over tasks: ``<name>_ci``."""
    if not grades:
        raise ValueError("no graded episodes")
    if len(names := {g.episode.harness for g in grades}) > 1:
        raise ValueError(
            f"summarize takes one harness, got {sorted(names)}; split them with by_harness()"
        )
    tasks = list(by_task(grades).values())
    rates = task_rates(grades)
    k = min(len(gs) for gs in tasks)
    eps = [g.episode for g in grades]
    runs = list(grades)
    s: dict[str, Any] = {
        "tasks": len(rates),
        "trials": k,
        "episodes": len(grades),
        "success": statistics.fmean(rates.values()),
        "success_ci": bootstrap(list(rates.values()), seed=seed),
        "pass^k": pass_hat_k(grades, k),
        "violations": sum(bool(g.violations) for g in grades),
        "violation_tasks": sorted({g.episode.task_id for g in grades if g.violations}),
        **{name: f(runs) for name, f in POOLED.items()},
        "unknown_cost": sum(e.cost_usd is None for e in eps),
        "infra_errors": sum(e.infra_error for e in eps),
        "unknown_verdicts": sum(bool(g.unknown) for g in grades),
        "models": sorted({m for e in eps for m in e.models}),
    }
    for name, fn in (metrics or {}).items():
        if name in s or name.endswith("_ci"):
            raise ValueError(f"custom metric {name!r} replaces a required metric or interval")
        s[name] = fn(runs)
    s["pass^k_ci"] = bootstrap(list(per_task(grades, "pass^k", k).values()), seed=seed)
    for name in [*POOLED, *(metrics or {})]:
        metric = pooled(name, metrics)
        s[f"{name}_ci"] = None if s[name] is None else task_interval(tasks, metric, seed=seed)
    return s


def pareto(
    summaries: Mapping[str, Mapping[str, Any]], x: str, y: str, *, lower: Iterable[str] = ()
) -> list[str]:
    """The harnesses that no other harness beats on both metrics: none is at least as good
    on ``x`` and on ``y`` and better on one of them. ``lower`` names custom metrics where
    smaller is better. The frontier uses point estimates; harnesses whose intervals overlap
    may swap places in another sample."""
    low = LOWER_IS_BETTER | ({lower} if isinstance(lower, str) else set(lower))

    def score(s: Mapping[str, Any], m: str) -> float | None:
        v = s.get(m)
        if not isinstance(v, int | float) or v != v:  # missing, not a number, or NaN
            return None
        return -v if m in low else v

    pts = {h: (score(s, x), score(s, y)) for h, s in summaries.items()}
    known = {h: p for h, p in pts.items() if None not in p}
    if summaries and not known:
        raise ValueError(f"no harness has both {x} and {y}")
    return [
        h
        for h, (a, b) in known.items()
        if not any(c >= a and d >= b and (c, d) != (a, b) for o, (c, d) in known.items() if o != h)
    ]


def failures(grades: Sequence[Grade]) -> dict[str, dict[str, int]]:
    """For each task, how many trials failed each check: where to read the traces first."""
    out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for g in grades:
        for name in g.failed + [f"unknown: {u}" for u in g.unknown]:
            out[g.episode.task_id][name] += 1
    return {t: dict(c) for t, c in out.items()}
