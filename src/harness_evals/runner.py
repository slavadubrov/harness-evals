"""Run harnesses on a use case with everything except the harness held fixed.

An adapter connects one harness to the evaluator. It runs a list of tasks once and
returns one Episode per task, built from what the environment recorded. The runner
interleaves harnesses within each trial, so provider load and caches drift over all of
them alike, and it writes a manifest that says what was held fixed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harness_evals import __version__
from harness_evals.core import Episode, Task
from harness_evals.usecase import UseCase

Adapter = Callable[[list[Task], int], list[Episode]]  # (tasks, trial) -> one episode per task


def fingerprint(tasks: Iterable[Task]) -> str:
    """Hash of the task ids, inputs, splits and check names. Results compare only when
    their fingerprints match: a changed check makes a different evaluation."""
    h = hashlib.sha256()
    for t in sorted(tasks, key=lambda t: t.id):
        row = [t.id, t.split, t.input, [c.name for c in t.checks]]
        h.update(json.dumps(row, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def run(
    usecase: UseCase,
    harnesses: Mapping[str, Adapter],
    *,
    trials: int,
    frozen: Mapping[str, Any],
    split: str = "dev",
    out: Path | None = None,
) -> list[Episode]:
    """Run every harness ``trials`` times on the tasks of one split.

    ``frozen`` names what must not change between the harnesses you compare: at least
    the model, and usually its endpoint and sampling settings. Run the holdout split
    only for a final decision; each look at it makes it less of a holdout.
    """
    if "model" not in frozen:
        raise ValueError("frozen must name the model")
    tasks = usecase.split(split)
    if not tasks:
        raise ValueError(f"no tasks in split {split!r}")
    manifest = {
        "usecase": usecase.name,
        "usecase_version": usecase.version,
        "fingerprint": fingerprint(tasks),
        "split": split,
        "frozen": dict(frozen),
        "harnesses": list(harnesses),
        "trials": trials,
        "evaluator": f"harness-evals {__version__}",
        "started": _now(),
    }
    episodes: list[Episode] = []
    for trial in range(trials):
        for name, adapter in harnesses.items():
            got = {e.task_id: e for e in adapter(tasks, trial)}
            for t in tasks:  # a task with no episode is recorded as an error, not dropped
                e = got.get(t.id) or Episode(t.id, status="error", error="no episode returned")
                e.harness, e.trial = name, trial
                episodes.append(e)
    manifest["finished"] = _now()
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        save(episodes, out / "episodes.jsonl")
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return episodes


def comparable(a: Mapping[str, Any], b: Mapping[str, Any]) -> None:
    """Raise unless two manifests held the same things fixed."""
    for key in ("usecase", "usecase_version", "fingerprint", "split", "frozen"):
        if a.get(key) != b.get(key):
            raise ValueError(f"not comparable: {key} differs ({a.get(key)!r} vs {b.get(key)!r})")


def save(episodes: Iterable[Episode], path: Path) -> None:
    with open(path, "w") as f:
        for e in episodes:
            f.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")


def load(path: Path) -> list[Episode]:
    with open(path) as f:
        return [Episode.from_dict(json.loads(line)) for line in f if line.strip()]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
