"""Evaluate an agent harness or workflow with the model held fixed."""

__version__ = "0.9.0"

from harness_evals.calibrate import agreement  # noqa: E402
from harness_evals.compare import (  # noqa: E402
    Comparison,
    compare,
    decide,
    gate_failures,
    run_permutation_test,
    sign_test,
    wilcoxon_test,
)
from harness_evals.core import (  # noqa: E402
    Action,
    Check,
    Episode,
    Grade,
    Task,
    at_most,
    did,
    ended,
    grade,
    judge,
    judged,
    never,
    never_says,
    says,
    state,
)
from harness_evals.metrics import (  # noqa: E402
    by_harness,
    failures,
    pareto,
    pass_at_k,
    pass_hat_k,
    summarize,
)
from harness_evals.report import (  # noqa: E402
    comparison,
    digest,
    failure_map_svg,
    frontier_svg,
    table,
)
from harness_evals.runner import comparable, fingerprint, load, run, save  # noqa: E402
from harness_evals.usecase import UseCase  # noqa: E402

__all__ = [
    "Action",
    "Check",
    "Comparison",
    "Episode",
    "Grade",
    "Task",
    "UseCase",
    "agreement",
    "at_most",
    "by_harness",
    "comparable",
    "compare",
    "comparison",
    "decide",
    "did",
    "digest",
    "ended",
    "failure_map_svg",
    "failures",
    "fingerprint",
    "frontier_svg",
    "gate_failures",
    "grade",
    "judge",
    "judged",
    "load",
    "never",
    "never_says",
    "pareto",
    "pass_at_k",
    "pass_hat_k",
    "run",
    "run_permutation_test",
    "save",
    "says",
    "sign_test",
    "state",
    "summarize",
    "table",
    "wilcoxon_test",
]
