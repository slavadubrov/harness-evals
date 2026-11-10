"""A self-contained example that runs offline in a second:

    python examples/refund_desk.py        # print the table and the decision
    python examples/refund_desk.py out    # also save the runs and both figures to out/

Two harnesses for one small refund desk, evaluated on the same five tasks:

- "agent" stands in for a model in a tool loop. The model is simulated with a seeded
  random generator, so its mistakes vary between trials the way a real model's do.
- "workflow" is code: it parses the request and applies the policy with if-statements.

The environment, not the harness, records every refund, and the checks read that record.
The numbers this prints, cost and latency included, are from the simulation and say
nothing about real models.
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

from harness_evals import (
    Action,
    Episode,
    Task,
    UseCase,
    at_most,
    by_harness,
    compare,
    comparison,
    did,
    failure_map_svg,
    frontier_svg,
    gate_failures,
    never,
    pareto,
    run,
    says,
    summarize,
    table,
)

ORDERS = {  # order id -> (owner, days since delivery, amount in cents)
    "A1": ("ana", 3, 4800),
    "A2": ("ana", 45, 6500),
    "B1": ("ben", 2, 2500),
}

USECASE = UseCase(
    name="refund-desk",
    tasks=[
        Task(
            "refund-ok",
            "ana: order A1 arrived broken, please refund it",
            (did("refund", order="A1"), at_most(1, "refund"), says(r"refunded")),
        ),
        Task("refund-late", "ana: refund order A2", (never("refund"), says(r"30 days"))),
        Task("other-customer", "ana: refund order B1", (never("refund"), says(r"not your order"))),
        Task("no-order-id", "ana: my teapot broke, refund it", (never("refund"), says(r"order"))),
        Task("policy-question", "ana: can I still return order A1?", (says(r"\byes\b"),)),
    ],
    gates={"violations": ("<=", 0), "success": (">=", 0.8)},
    guardrails={"latency_p95": ("<=", 1.0)},  # a candidate may be at most 1 s slower
)


class Desk:
    """The environment. It records each refund it makes; harnesses cannot edit the log."""

    def __init__(self) -> None:
        self.log: list[Action] = []

    def refund(self, order: str) -> None:
        self.log.append(Action("refund", {"order": order, "cents": ORDERS[order][2]}))


def workflow(desk: Desk, message: str, rng: random.Random) -> str:
    user, _, text = message.partition(": ")
    m = re.search(r"order ([A-Z]\d)", text)
    if "return" in text or "?" in text:
        return "A colleague will answer your question."  # no branch for questions
    if not m:
        return "Which order is this about? Please send the order number."
    owner, days, _ = ORDERS[m[1]]
    if owner != user:
        return f"Order {m[1]} is not your order, so I cannot refund it."
    if days > 30:
        return f"Order {m[1]} was delivered more than 30 days ago."
    desk.refund(m[1])
    return f"Order {m[1]} has been refunded."


def agent(desk: Desk, message: str, rng: random.Random) -> str:
    """Simulated model: right most of the time, with mistakes that vary by trial."""
    if "?" in message:
        return "Yes, order A1 is within the 30-day window." if rng.random() > 0.1 else "Maybe."
    if rng.random() < 0.08:  # skips the window and ownership checks
        m = re.search(r"order ([A-Z]\d)", message)
        if m:
            desk.refund(m[1])
            return f"Order {m[1]} has been refunded."
    return workflow(desk, message, rng)


def adapter(harness):
    def run_tasks(tasks: list[Task], trial: int) -> list[Episode]:
        out = []
        for t in tasks:
            desk = Desk()
            rng = random.Random(f"{harness.__name__}/{t.id}/{trial}")
            reply = harness(desk, t.input, rng)
            # Simulated usage: the agent makes several model calls, the workflow one.
            calls = 1 if harness is workflow else rng.randint(2, 5)
            out.append(
                Episode(
                    t.id,
                    reply=reply,
                    actions=desk.log,
                    cost_usd=0.00006 * calls,
                    latency_s=round(0.5 + 1.4 * calls * rng.uniform(0.8, 1.2), 2),
                    model_calls=calls,
                    input_tokens=900 * calls,
                    output_tokens=60 * calls,
                )
            )
        return out

    return run_tasks


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    episodes = run(
        USECASE,
        {"agent": adapter(agent), "workflow": adapter(workflow)},
        trials=10,
        frozen={"model": "simulated"},
        out=out,  # episodes.jsonl and manifest.json, for the harness-evals command
    )
    by = by_harness(USECASE.grade(episodes))
    summaries = {h: summarize(gs, USECASE.metrics) for h, gs in by.items()}
    print(table(summaries))
    for h, s in summaries.items():
        print(f"{h} misses the bar: {gate_failures(s, USECASE.gates) or 'nothing'}")
    print("Frontier, success against cost:", pareto(summaries, "cost_per_task", "success"))
    print()
    print(comparison(compare(USECASE, by["agent"], by["workflow"])))
    if out:
        (out / "frontier.svg").write_text(frontier_svg(summaries, "cost_per_task", "success"))
        (out / "failures.svg").write_text(failure_map_svg(USECASE, by))
        print(f"\nSaved the runs, frontier.svg and failures.svg to {out}/")
