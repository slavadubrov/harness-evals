"""harness-evals summarize|compare|frontier|failures --usecase module:USECASE episodes.jsonl ..."""

from __future__ import annotations

import argparse
import importlib
import json
from collections import defaultdict
from pathlib import Path

from harness_evals.compare import compare
from harness_evals.metrics import summarize
from harness_evals.report import comparison, failure_map_svg, frontier_svg, table
from harness_evals.runner import comparable, fingerprint, load
from harness_evals.usecase import UseCase


def _usecase(ref: str) -> UseCase:
    module, _, attr = ref.partition(":")
    uc = getattr(importlib.import_module(module), attr or "USECASE")
    if not isinstance(uc, UseCase):
        raise SystemExit(f"{ref} is not a UseCase")
    return uc


def main(argv: list[str] | None = None) -> None:
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
            except (OSError, ValueError) as exc:
                raise SystemExit(f"invalid manifest: {exc}") from None
            reference = manifest
        elif a.command == "compare":
            raise SystemExit("compare requires manifest.json next to each episodes file")
        for g in uc.grade(load(path)):
            grades[g.episode.harness].append(g)
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
