"""harness-evals summarize|compare|frontier|failures --usecase module:USECASE episodes.jsonl ..."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from harness_evals.compare import compare
from harness_evals.core import Episode, Task
from harness_evals.metrics import summarize
from harness_evals.report import comparison, failure_map_svg, frontier_svg, table
from harness_evals.runner import comparable, fingerprint, load
from harness_evals.usecase import UseCase


def _usecase(ref: str) -> UseCase:
    module, _, attr = ref.partition(":")
    if os.getcwd() not in sys.path:  # a use case next to the runs, as a script would see it
        sys.path.insert(0, os.getcwd())
    try:
        uc = getattr(importlib.import_module(module), attr or "USECASE")
    except (ImportError, AttributeError) as exc:
        raise SystemExit(f"--usecase {ref}: {exc}") from None
    if not isinstance(uc, UseCase):
        raise SystemExit(f"{ref} is not a UseCase")
    return uc


def main(argv: list[str] | None = None) -> None:
    try:
        _main(argv)
    except (OSError, ValueError) as exc:  # a bad file or mismatched runs: one line, no trace
        raise SystemExit(f"harness-evals: {exc}") from None


def _check_coverage(episodes: list[Episode], tasks: list[Task], trials: int) -> None:
    """Each present harness must contain exactly one episode per expected task/trial."""
    if type(trials) is not int or trials < 1:
        raise ValueError("manifest trials must be a positive integer")
    if not episodes:
        raise ValueError("no episodes to check against the manifest")
    expected = {(t.id, i) for t in tasks for i in range(trials)}
    seen: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for e in episodes:
        if type(e.trial) is not int or not 0 <= e.trial < trials:
            raise ValueError(f"trial must be an integer from 0 to {trials - 1}: {e.trial!r}")
        pair = (e.task_id, e.trial)
        if pair not in expected:
            raise ValueError(f"task {e.task_id!r} is not in the manifest split")
        if pair in seen[e.harness]:
            raise ValueError(f"harness {e.harness!r} contains duplicate task trials")
        seen[e.harness].add(pair)
    for name, pairs in seen.items():
        if pairs != expected:
            raise ValueError(f"harness {name!r} is missing {len(expected - pairs)} task trials")


def _main(argv: list[str] | None) -> None:
    p = argparse.ArgumentParser(prog="harness-evals")
    p.add_argument("command", choices=["summarize", "compare", "frontier", "failures"])
    p.add_argument("episodes", nargs="+", type=Path, help="episodes.jsonl files")
    p.add_argument("--usecase", required=True, help="module:attribute of a UseCase")
    p.add_argument("--base", help="harness name to compare against")
    p.add_argument("--candidate", help="harness name to compare")
    p.add_argument("--x", default="cost_per_task", help="frontier: metric on the x axis")
    p.add_argument("--y", default="success", help="frontier: metric on the y axis")
    p.add_argument(
        "--lower",
        action="append",
        default=[],
        help="frontier: a custom metric where less is better",
    )
    p.add_argument("--by", choices=["checks", "runs"], default="checks", help="failures: columns")
    a = p.parse_args(argv)

    uc = _usecase(a.usecase)
    grades = defaultdict(list)
    reference = None
    for path in a.episodes:
        episodes = load(path)
        manifest_path = path.with_name("manifest.json")
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if not isinstance(manifest, dict):
                    raise ValueError("manifest must be a JSON object")
                comparable(
                    manifest,
                    {
                        "usecase": uc.name,
                        "usecase_version": uc.version,
                        "fingerprint": fingerprint(uc.split(manifest.get("split"))),
                        "split": manifest.get("split"),
                        "frozen": manifest.get("frozen"),
                    },
                )
                if reference is not None:
                    comparable(reference, manifest)
                _check_coverage(episodes, uc.split(manifest["split"]), manifest.get("trials"))
            except (OSError, ValueError) as exc:
                raise SystemExit(f"invalid manifest: {exc}") from None
            reference = manifest
        elif a.command == "compare":
            raise SystemExit("compare requires manifest.json next to each episodes file")
        new: dict[str, list] = defaultdict(list)
        for g in uc.grade(episodes):
            new[g.episode.harness].append(g)
        if again := sorted(set(new) & set(grades)):  # pooling two versions hides both
            raise SystemExit(
                f"harness {', '.join(again)} appears in more than one file; run each version"
                " under its own name, such as main and candidate"
            )
        grades.update(new)
    if a.command == "compare":
        if not (a.base in grades and a.candidate in grades):
            raise SystemExit(f"--base and --candidate must be among: {', '.join(grades)}")
        print(comparison(compare(uc, grades[a.base], grades[a.candidate])))
        return
    if a.command == "failures":  # an SVG on stdout: harness-evals failures ... > failures.svg
        print(failure_map_svg(uc, grades, by=a.by), end="")
        return
    sums = {h: summarize(gs, uc.metrics) for h, gs in grades.items()}
    if a.command == "frontier":  # an SVG on stdout: harness-evals frontier ... > plot.svg
        try:
            print(frontier_svg(sums, a.x, a.y, lower=a.lower), end="")
        except ValueError as exc:
            raise SystemExit(f"frontier: {exc}") from None
    else:
        print(table(sums, list(uc.metrics)))


if __name__ == "__main__":
    main()
