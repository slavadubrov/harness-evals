"""Run harnesses on a use case with everything except the harness held fixed.

An adapter connects one harness to the evaluator. It runs a list of tasks once and
returns one Episode per task, built from what the environment recorded. The runner
interleaves harnesses within each trial, rotating the starting harness to balance
execution positions, and writes a manifest that says what was held fixed.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from harness_evals import __version__
from harness_evals.core import Episode, Task
from harness_evals.usecase import UseCase

Adapter = Callable[[list[Task], int], list[Episode]]  # (tasks, trial) -> one episode per task


def fingerprint(tasks: Iterable[Task]) -> str:
    """Hash of task ids, inputs, splits, check names and critical flags. Results compare
    only when their fingerprints match: a changed check makes a different evaluation."""
    h = hashlib.sha256()
    for t in sorted(tasks, key=lambda t: t.id):
        row = [t.id, t.split, t.input, [(c.name, c.critical) for c in t.checks]]
        h.update(json.dumps(row, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def run(
    usecase: UseCase,
    harnesses: Mapping[str, Adapter],
    *,
    trials: int,
    frozen: Mapping[str, Any],
    split: str = "dev",
    out: Path | str | None = None,
) -> list[Episode]:
    """Run every harness ``trials`` times on the tasks of one split.

    ``frozen`` names what must not change between the harnesses you compare: at least
    the model, and usually its endpoint and sampling settings. Run the holdout split
    only for a final decision; each look at it makes it less of a holdout.
    """
    if not frozen.get("model"):
        raise ValueError("frozen must name the model")
    if type(trials) is not int or trials < 1:
        raise ValueError("trials must be a positive integer")
    if not harnesses:
        raise ValueError("at least one harness is required")
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
    json.dumps(manifest)  # fail before the first trial, not after the last
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
    episodes: list[Episode] = []
    adapters = list(harnesses.items())
    warned: set[str] = set()
    for trial in range(trials):
        start = trial % len(adapters)
        for name, adapter in adapters[start:] + adapters[:start]:
            try:
                batch = list(adapter(deepcopy(tasks), trial))
            except Exception as exc:
                # A batch error loses partial evidence; catch per-task errors in adapters.
                error = f"adapter raised {exc!r}"[:300]
                if name not in warned:  # a bug in an adapter must not pass as a low score
                    warned.add(name)
                    warnings.warn(
                        f"harness {name!r}: {error}; every task of the batch counts as an"
                        " infrastructure error",
                        stacklevel=2,
                    )
                batch = [
                    Episode(t.id, status="error", infra_error=True, error=error) for t in tasks
                ]
            got = {e.task_id: e for e in batch}
            if len(got) != len(batch) or got.keys() - {t.id for t in tasks}:
                raise ValueError("adapter returned duplicate or unrequested episodes")
            for t in tasks:  # a task with no episode is recorded as an error, not dropped
                e = (
                    deepcopy(got[t.id])
                    if t.id in got
                    else Episode(
                        t.id, status="error", error="no episode returned", infra_error=True
                    )
                )
                e.harness, e.trial = name, trial
                if out is not None:  # an unsaveable episode stops the run now, not at the end
                    json.dumps(e.to_dict())
                episodes.append(e)
    manifest["finished"] = _now()
    if out is not None:
        manifest_text = json.dumps(manifest, indent=2) + "\n"
        save(episodes, out / "episodes.jsonl")
        (out / "manifest.json").write_text(manifest_text, encoding="utf-8")
    return episodes


def comparable(a: Mapping[str, Any], b: Mapping[str, Any]) -> None:
    """Raise unless two manifests held the same things fixed."""
    for key in ("usecase", "usecase_version", "fingerprint", "split", "frozen"):
        if not a.get(key) or not b.get(key):
            raise ValueError(f"not comparable: missing {key}")
        if a[key] != b[key]:
            raise ValueError(f"not comparable: {key} differs")
    if not isinstance(a["frozen"], Mapping) or not a["frozen"].get("model"):
        raise ValueError("not comparable: frozen must name the model")


def save(episodes: Iterable[Episode], path: Path) -> None:
    """Replace the saved episodes only after every record is written successfully."""
    path = Path(path)
    f = NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False)
    try:
        with f:
            for e in episodes:
                f.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")
        Path(f.name).replace(path)
    finally:
        Path(f.name).unlink(missing_ok=True)


def load(path: Path) -> list[Episode]:
    """The episodes that ``save`` or ``run(..., out=...)`` wrote."""
    with open(path, encoding="utf-8") as f:
        return [Episode.from_dict(json.loads(line)) for line in f if line.strip()]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
