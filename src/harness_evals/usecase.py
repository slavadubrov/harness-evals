"""A use case: the tasks a harness is built for and the bar it must clear."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from typing import Any

from harness_evals.core import Episode, Grade, Task, grade
from harness_evals.metrics import POOLED, Metric


def check_guardrails(guardrails: Mapping[str, Any], metrics: Mapping[str, Metric]) -> None:
    """Each guardrail is (op, limit) on a metric that a comparison can pair."""
    comparable = {"success", "pass^k", *POOLED, *metrics}
    for metric, spec in guardrails.items():
        if metric not in comparable:
            raise ValueError(f"guardrail on {metric!r}: compare one of {sorted(comparable)}")
        ok = isinstance(spec, tuple) and len(spec) == 2 and spec[0] in (">=", "<=")
        if not ok or not isinstance(spec[1], int | float) or not isfinite(spec[1]):
            raise ValueError(f"guardrail on {metric!r} needs ('>=' or '<=', a finite limit)")


@dataclass
class UseCase:
    """``gates`` are absolute bars on a harness, such as ``{"violations": ("<=", 0)}``.
    ``metrics`` adds use-case metrics to the required ones. ``margin`` is how much success
    a cheaper harness may lose and still be adopted; 0 means it may lose none.
    ``guardrails`` bound how much other metrics may get worse in a comparison between two
    harnesses that clear the gates, as limits on candidate minus base:
    ``{"latency_p95": ("<=", 2.0)}`` lets p95 latency rise by at most 2 seconds."""

    name: str
    tasks: list[Task]
    gates: dict[str, tuple[str, float]] = field(default_factory=dict)
    metrics: dict[str, Metric] = field(default_factory=dict)
    margin: float = 0.0
    version: str = "1"
    guardrails: dict[str, tuple[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ids = [t.id for t in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate task ids in {self.name}")
        check_guardrails(self.guardrails, self.metrics)

    def split(self, name: str) -> list[Task]:
        return [t for t in self.tasks if t.split == name]

    def grade(self, episodes: Sequence[Episode]) -> list[Grade]:
        """Grade episodes against this use case. An unknown task id is an error, not a
        skip: an episode that cannot be graded must not disappear from the totals."""
        by_id = {t.id: t for t in self.tasks}
        return [grade(by_id[e.task_id], e) for e in episodes]
