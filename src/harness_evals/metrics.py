"""The numbers every harness evaluation reports, with intervals computed over tasks.

Tasks are the unit of resampling: trials of one task are not independent evidence about
other tasks, so the intervals resample whole tasks with all their trials.
"""

from __future__ import annotations

import random
import statistics
from collections import defaultdict
from collections.abc import Callable, Sequence
from math import ceil, comb
from typing import Any

from harness_evals.core import Grade

Metric = Callable[[list[Grade]], float | None]


def by_task(grades: Sequence[Grade]) -> dict[str, list[Grade]]:
    out: dict[str, list[Grade]] = defaultdict(list)
    for g in grades:
        out[g.episode.task_id].append(g)
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
    means = sorted(statistics.fmean(rng.choices(values, k=k)) for _ in range(n))
    lo = means[int((1 - level) / 2 * n)]
    hi = means[min(n - 1, int((1 + level) / 2 * n))]
    return lo, hi


def percentile(xs: Sequence[float], p: float) -> float | None:
    """Nearest-rank percentile; None for an empty list."""
    if not xs:
        return None
    s = sorted(xs)
    return s[max(0, ceil(p / 100 * len(s)) - 1)]


def summarize(
    grades: Sequence[Grade], metrics: dict[str, Metric] | None = None, seed: int = 0
) -> dict[str, Any]:
    """The required metrics, plus any use-case metrics passed in ``metrics``."""
    if not grades:
        raise ValueError("no graded episodes")
    rates = task_rates(grades)
    k = min(len(gs) for gs in by_task(grades).values())
    eps = [g.episode for g in grades]
    known = [e.cost_usd for e in eps if e.cost_usd is not None]
    passes = sum(g.passed for g in grades)
    lat = [e.latency_s for e in eps if e.latency_s is not None]
    s: dict[str, Any] = {
        "tasks": len(rates),
        "trials": k,
        "episodes": len(grades),
        "success": statistics.fmean(rates.values()),
        "success_ci": bootstrap(list(rates.values()), seed=seed),
        "pass^k": pass_hat_k(grades, k),
        "violations": sum(bool(g.violations) for g in grades),
        "violation_tasks": sorted({g.episode.task_id for g in grades if g.violations}),
        "cost_per_task": statistics.fmean(known) if known else None,
        "cost_per_success": sum(known) / passes if len(known) == len(eps) and passes else None,
        "unknown_cost": len(eps) - len(known),
        "latency_p50": percentile(lat, 50) if len(lat) == len(eps) else None,
        "latency_p95": percentile(lat, 95) if len(lat) == len(eps) else None,
        "infra_errors": sum(e.infra_error for e in eps),
        "unknown_verdicts": sum(bool(g.unknown) for g in grades),
        "models": sorted({m for e in eps for m in e.models}),
    }
    for name, fn in (metrics or {}).items():
        if name in s:
            raise ValueError(f"custom metric {name!r} replaces a required metric")
        s[name] = fn(list(grades))
    return s


def failures(grades: Sequence[Grade]) -> dict[str, dict[str, int]]:
    """For each task, how many trials failed each check: where to read the traces first."""
    out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for g in grades:
        for name in g.failed + [f"unknown: {u}" for u in g.unknown]:
            out[g.episode.task_id][name] += 1
    return {t: dict(c) for t, c in out.items()}
