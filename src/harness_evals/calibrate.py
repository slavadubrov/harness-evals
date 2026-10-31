"""Evaluate the evaluator: how often a grader agrees with labels a person wrote."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Iterable
from typing import Any


def agreement(pairs: Iterable[tuple[Hashable, Hashable]]) -> dict[str, Any]:
    """``pairs`` are (grader verdict, label) for the same runs. Labels must be written
    without seeing the grader's verdicts. For pass/fail verdicts, a false accept is a run
    the grader passed and the label failed: the error that lets a bad harness ship."""
    pairs = list(pairs)
    if not pairs:
        raise ValueError("no labelled pairs")
    n = len(pairs)
    agree = sum(g == lab for g, lab in pairs)
    confusion = Counter(pairs)
    graded, labelled = Counter(g for g, _ in pairs), Counter(lab for _, lab in pairs)
    chance = sum(graded[c] * labelled[c] for c in graded) / n**2
    out: dict[str, Any] = {
        "n": n,
        "agree": agree,
        "accuracy": agree / n,
        "kappa": (agree / n - chance) / (1 - chance) if chance < 1 else None,
        "confusion": {f"{g} as {lab}": c for (g, lab), c in sorted(confusion.items(), key=str)},
    }
    if set(graded) | set(labelled) <= {True, False}:
        out["false_accepts"] = confusion[(True, False)]
        out["false_rejects"] = confusion[(False, True)]
    return out
