# harness-evals

A small Python library for evaluating an agent harness or a workflow with the model held
fixed. You describe your use case as tasks with checks, connect each harness through a
short adapter, and get repeated-run metrics with intervals and a decision to adopt,
reject or keep investigating a change.

It is the companion code for the article on evaluating agent harnesses in the
[Recursive Harness Engineering](https://slavadubrov.com/blog/) series. The
[agent-harness-lab](https://github.com/slavadubrov/agent-harness-lab-public) installs it
to evaluate five designs of one support assistant, from a plain agent to a workflow with
no agent.

- No dependencies outside the standard library. Python 3.11 or later.
- It does not call models or run your harness. Your adapter does that; the library
  grades what the adapter observed.

## Install

```bash
uv add "harness-evals @ git+https://github.com/slavadubrov/harness-evals@v0.1.0"
# or
pip install "git+https://github.com/slavadubrov/harness-evals@v0.1.0"
```

## The idea

A model benchmark keeps the harness fixed and changes the model. A harness evaluation
does the opposite: the model, its endpoint and settings, the tasks and the checks stay
fixed, and only the harness changes. Then a difference in the results belongs to the
harness.

| Part | What it is | Where |
| --- | --- | --- |
| `UseCase` | The tasks a harness is built for, the bar it must clear (`gates`) and any use-case metrics | `usecase.py` |
| `Task` | One case: its input and the checks a correct run must pass | `core.py` |
| `Check` | One obligation. Returns True, False, or None when evidence is missing; `critical` marks a rule that must never break | `core.py` |
| `Episode` | What the evaluator observed in one run: status, reply, actions the environment recorded, cost, latency, models | `core.py` |
| adapter | Your function `(tasks, trial) -> list[Episode]` that runs one harness | your code |
| `run` | Runs every harness `trials` times, interleaved, and writes a manifest of what was held fixed | `runner.py` |
| `summarize`, `compare` | The metrics below, and a paired comparison with an adoption decision | `metrics.py`, `compare.py` |
| `agreement` | How often a grader agrees with labels a person wrote | `calibrate.py` |

## A task and an adapter

```python
from harness_evals import Episode, Task, UseCase, at_most, did, never, run, says

USECASE = UseCase(
    name="refund-desk",
    tasks=[
        Task("refund-ok", "ana: order A1 arrived broken, please refund it",
             (did("refund", order="A1"), at_most(1, "refund"), says(r"refunded"))),
        Task("refund-late", "ana: refund order A2",
             (never("refund"), says(r"30 days"))),
    ],
    gates={"violations": ("<=", 0), "pass^k": (">=", 0.9)},
)

def my_agent(tasks, trial):
    episodes = []
    for task in tasks:
        env = Environment()                        # records what really happened
        reply = run_my_agent(task.input, env)      # your harness
        episodes.append(Episode(task.id, reply=reply, actions=env.log,
                                cost_usd=env.cost, latency_s=env.seconds))
    return episodes

episodes = run(USECASE, {"agent": my_agent, "workflow": my_workflow},
               trials=10, frozen={"model": "provider/model-id", "temperature": 0})
```

`did`, `never`, `at_most`, `says`, `never_says`, `ended`, `state` and `judged` cover the
common obligations. A check is any function from an `Episode` to `True`, `False` or
`None`, so write your own for anything else.

[`examples/refund_desk.py`](examples/refund_desk.py) runs offline in a second. Its
"agent" is a simulated model with seeded mistakes, so its numbers describe the
simulation only:

```text
| Harness | Success | pass^k | Violations | $ per task | $ per success | p95 latency, s | Infra errors | Undecided |
|---|---|---|---|---|---|---|---|---|
| agent | 0.90 (0.78–1.00) | 0.60 (k=10) | 2 | $0.00000 | $0.00000 | 0.10 | 0 | 0 |
| workflow | 0.80 (0.40–1.00) | 0.80 (k=10) | 0 | $0.00000 | $0.00000 | 0.10 | 0 | 0 |

**workflow** against **agent** on 5 tasks: success -0.100 (-0.420 to +0.120).
Decision: **adopt**. the base misses the bar: violations = 2, needs <= 0
```

## Metrics

`summarize` always reports these:

| Metric | What it answers |
| --- | --- |
| `success` and `success_ci` | The share of runs that pass every check, averaged over tasks, with a 95% bootstrap interval that resamples tasks |
| `pass^k` | The chance that k runs of the same task all pass: C(c, k) / C(n, k) per task, averaged |
| `violations` | Runs that failed a critical check. Reported apart from success, because an average hides them |
| `cost_per_task`, `cost_per_success` | Dollars per run and per passing run. `unknown_cost` counts runs whose cost is unknown; it is never counted as zero |
| `latency_p50`, `latency_p95` | Nearest-rank percentiles |
| `infra_errors`, `unknown_verdicts` | Runs the provider or runner broke, and runs a check could not decide. Both count as failures and are reported |
| `models` | The model ids the provider reported, to catch a silent model change |

Add the metrics that matter for your use case through `UseCase.metrics`, for example
tool calls per task, how often a reply names an id, or a judge's score.

## The adoption rule

`compare(usecase, base, candidate)` pairs the two harnesses task by task and decides:

1. **reject** if the candidate misses a gate or has more critical violations;
2. **adopt** if the base misses a gate the candidate clears;
3. **reject** if the interval of the success difference lies below `-margin`;
4. **adopt** if it lies above 0, or if it lies above `-margin` and the candidate is cheaper;
5. otherwise **inconclusive**: keep the base, and add tasks or runs.

A cost gate fails while any run's cost is unknown, and a gate on a metric that could not
be computed fails too.

## What it guards against

- **A changed model or check.** `run` refuses to start without `frozen["model"]`, and
  `comparable(manifest_a, manifest_b)` raises when the frozen setup, the split or the
  task fingerprint differ. Check names include their arguments, so editing a check
  changes the fingerprint.
- **Grading the final state only.** Tasks carry checks on actions and replies; a
  harness that does nothing should fail every task.
- **Missing evidence.** A check that returns None fails the run, and a judge that raises
  stores None.
- **Tuning on the tasks you report.** Tasks have a `split`; `run` uses `"dev"` unless you
  ask for another.
- **An unmeasured judge.** `agreement(pairs)` reports accuracy, Cohen's kappa, false
  accepts and false rejects against labels.

## Limits

- The intervals treat your tasks as a sample of the requests the harness will see. With
  a few dozen tasks they are wide; that is information, not a defect.
- When every run of every task passes, the interval collapses to a point. Zero failures
  in n runs still allows a failure rate up to about 3/n at 95% confidence.
- The library trusts the episodes it is given. Build them from records the harness
  cannot edit: the environment's database, a service log, the provider's usage report.

## Development

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

MIT licensed.
