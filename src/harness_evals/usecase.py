"""A use case: the tasks a harness is built for and the bar it must clear."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from harness_evals.core import Episode, Grade, Task, grade
from harness_evals.metrics import Metric


@dataclass
class UseCase:
    """``gates`` are absolute bars, such as ``{"violations": ("<=", 0)}``. ``metrics``
    adds use-case metrics to the required ones. ``margin`` is how much success a
    cheaper harness may lose and still be adopted; 0 means it may lose none."""

    name: str
    tasks: list[Task]
    gates: dict[str, tuple[str, float]] = field(default_factory=dict)
    metrics: dict[str, Metric] = field(default_factory=dict)
    margin: float = 0.0
    version: str = "1"

    def __post_init__(self) -> None:
        ids = [t.id for t in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate task ids in {self.name}")

    def split(self, name: str) -> list[Task]:
        return [t for t in self.tasks if t.split == name]

    def grade(self, episodes: Sequence[Episode]) -> list[Grade]:
        """Grade episodes against this use case. An unknown task id is an error, not a
        skip: an episode that cannot be graded must not disappear from the totals."""
        by_id = {t.id: t for t in self.tasks}
        return [grade(by_id[e.task_id], e) for e in episodes]
