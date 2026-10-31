"""harness-evals summarize|compare --usecase package.module:USECASE episodes.jsonl ..."""

from __future__ import annotations

import argparse
import importlib
from collections import defaultdict
from pathlib import Path

from harness_evals.compare import compare
from harness_evals.metrics import summarize
from harness_evals.report import comparison, table
from harness_evals.runner import load
from harness_evals.usecase import UseCase


def _usecase(ref: str) -> UseCase:
    module, _, attr = ref.partition(":")
    uc = getattr(importlib.import_module(module), attr or "USECASE")
    if not isinstance(uc, UseCase):
        raise SystemExit(f"{ref} is not a UseCase")
    return uc


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="harness-evals")
    p.add_argument("command", choices=["summarize", "compare"])
    p.add_argument("episodes", nargs="+", type=Path, help="episodes.jsonl files")
    p.add_argument("--usecase", required=True, help="module:attribute of a UseCase")
    p.add_argument("--base", help="harness name to compare against")
    p.add_argument("--candidate", help="harness name to compare")
    a = p.parse_args(argv)

    uc = _usecase(a.usecase)
    grades = defaultdict(list)
    for path in a.episodes:
        for g in uc.grade(load(path)):
            grades[g.episode.harness].append(g)
    if a.command == "summarize":
        print(table({h: summarize(gs, uc.metrics) for h, gs in grades.items()}, list(uc.metrics)))
        return
    if not (a.base in grades and a.candidate in grades):
        raise SystemExit(f"--base and --candidate must be among: {', '.join(grades)}")
    print(comparison(compare(uc, grades[a.base], grades[a.candidate])))


if __name__ == "__main__":
    main()
