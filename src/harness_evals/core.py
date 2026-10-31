"""Tasks, episodes, checks and grades: what every harness evaluation is built from.

A task states what a correct run must do. An episode records what one run did, as the
evaluator observed it. A check turns one obligation into a verdict: True, False, or None
when the evidence to decide is missing. A grade collects the verdicts of one episode.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

Verdict = bool | None  # None: the evidence needed to decide is missing


@dataclass(frozen=True)
class Action:
    """An effect the environment recorded, such as a refund it issued."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)

    def matches(self, name: str, args: dict[str, Any]) -> bool:
        return self.name == name and all(self.args.get(k) == v for k, v in args.items())


@dataclass
class Episode:
    """What the evaluator observed in one trial of one task by one harness."""

    task_id: str
    harness: str = ""
    trial: int = 0
    status: str = "completed"  # how the run ended: completed, waiting, error, ...
    reply: str | None = None  # what the user was told; None when nothing was sent
    actions: list[Action] = field(default_factory=list)  # from the environment, not the harness
    state_ok: Verdict = None  # the environment's check of its final state; None if not run
    cost_usd: float | None = None  # None means unknown, which is not the same as zero
    latency_s: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    model_calls: int | None = None
    models: list[str] = field(default_factory=list)  # model ids the provider reported
    error: str | None = None
    infra_error: bool = False  # the provider or the runner failed, not the harness
    judgements: dict[str, Verdict] = field(default_factory=dict)  # see judge()
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Episode:
        d = dict(d)
        d["actions"] = [Action(**a) for a in d.get("actions", [])]
        return cls(**d)


@dataclass(frozen=True)
class Check:
    """One obligation of a task. A failed ``critical`` check is a violation: a rule that
    must hold in every run, such as "never refund another customer's order"."""

    name: str
    fn: Callable[[Episode], Verdict]
    critical: bool = False

    def __call__(self, e: Episode) -> Verdict:
        return self.fn(e)


@dataclass(frozen=True)
class Task:
    """One case from the use case, with the checks a correct run must pass."""

    id: str
    input: Any
    checks: tuple[Check, ...]
    split: str = "dev"  # "dev" while you build; "holdout" only for the final decision
    tags: tuple[str, ...] = ()


@dataclass
class Grade:
    episode: Episode
    failed: list[str]
    unknown: list[str]  # checks that could not decide; the episode does not pass
    violations: list[str]  # failed critical checks

    @property
    def passed(self) -> bool:
        return not self.failed and not self.unknown


def grade(task: Task, e: Episode) -> Grade:
    """Run every check. Missing evidence fails closed: it never counts as a pass."""
    failed: list[str] = []
    unknown: list[str] = []
    violations: list[str] = []
    if e.error:
        failed.append("no run error")
    for c in task.checks:
        v = c(e)
        if v is None:
            unknown.append(c.name)
        elif not v:
            failed.append(c.name)
            if c.critical:
                violations.append(c.name)
    return Grade(e, failed, unknown, violations)


# ---------------------------------------------------------------------------------------
# Checks. Each name records its arguments, so changing a check changes the suite's
# fingerprint (see runner.fingerprint) and old results stop comparing with new ones.
# ---------------------------------------------------------------------------------------


def _call(name: str, args: dict[str, Any]) -> str:
    return name + "(" + ", ".join(f"{k}={v!r}" for k, v in sorted(args.items())) + ")"


def _text(e: Episode) -> str:
    return (e.reply or "").replace("’", "'")  # models often write a curly apostrophe


def state() -> Check:
    """The environment's own check of the final state passed."""
    return Check("state", lambda e: e.state_ok)


def did(name: str, **args: Any) -> Check:
    """The environment recorded at least one matching action."""
    return Check(
        f"did {_call(name, args)}", lambda e: any(a.matches(name, args) for a in e.actions)
    )


def never(name: str, **args: Any) -> Check:
    """No matching action happened, including one that was later undone."""
    return Check(
        f"never {_call(name, args)}",
        lambda e: not any(a.matches(name, args) for a in e.actions),
        critical=True,
    )


def at_most(n: int, name: str, **args: Any) -> Check:
    """At most ``n`` matching actions: a second refund for one request breaks it."""
    return Check(
        f"at most {n} {_call(name, args)}",
        lambda e: sum(a.matches(name, args) for a in e.actions) <= n,
        critical=True,
    )


def says(*patterns: str) -> Check:
    """The reply matches every regular expression, ignoring case."""
    rx = [re.compile(p, re.I) for p in patterns]
    return Check("says " + " & ".join(patterns), lambda e: all(r.search(_text(e)) for r in rx))


def never_says(*patterns: str, critical: bool = False) -> Check:
    """The reply matches none of the regular expressions, ignoring case."""
    rx = [re.compile(p, re.I) for p in patterns]
    return Check(
        "never says " + " | ".join(patterns),
        lambda e: not any(r.search(_text(e)) for r in rx),
        critical=critical,
    )


def ended(*statuses: str) -> Check:
    """The run ended in one of these statuses, for example "completed"."""
    return Check("ended " + "|".join(statuses), lambda e: e.status in statuses)


def judged(name: str) -> Check:
    """The verdict a judge stored under ``name``; see judge()."""
    return Check(f"judge {name}", lambda e: e.judgements.get(name))


def judge(
    tasks: Iterable[Task],
    episodes: Iterable[Episode],
    name: str,
    fn: Callable[[Task, Episode], Verdict],
) -> None:
    """Store ``fn``'s verdict on each episode under ``name``. A judge that raises or
    cannot decide leaves None, so an outage fails the episode instead of passing it."""
    by_id = {t.id: t for t in tasks}
    for e in episodes:
        try:
            e.judgements[name] = fn(by_id[e.task_id], e)
        except Exception:  # noqa: BLE001 - any judge failure is missing evidence
            e.judgements[name] = None
