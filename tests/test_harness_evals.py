import json
import subprocess
import sys
from math import isclose
from pathlib import Path

import pytest

from harness_evals import (
    Action,
    Episode,
    Task,
    UseCase,
    agreement,
    at_most,
    comparable,
    compare,
    did,
    fingerprint,
    gate_failures,
    grade,
    judge,
    judged,
    load,
    never,
    never_says,
    pass_at_k,
    pass_hat_k,
    run,
    says,
    state,
    summarize,
)

REFUND = Action("refund", {"order": "A1", "cents": 4800})


def task(*checks, id="t"):
    return Task(id, "input", tuple(checks))


def grades_for(pattern: dict[str, list[bool]], harness="h", cost=0.001):
    """pattern: task id -> pass/fail per trial, encoded through state_ok."""
    uc = UseCase("u", [task(state(), id=t) for t in pattern])
    eps = [
        Episode(t, harness, i, state_ok=ok, cost_usd=cost, latency_s=1.0)
        for t, oks in pattern.items()
        for i, ok in enumerate(oks)
    ]
    return uc, uc.grade(eps)


def test_pass_hat_k_and_pass_at_k_use_the_unbiased_estimators():
    _, gs = grades_for({"a": [True, True, False, False]})
    assert isclose(pass_hat_k(gs, 2), 1 / 6)  # C(2,2)/C(4,2)
    assert isclose(pass_at_k(gs, 2), 5 / 6)  # 1 - C(2,2)/C(4,2)
    assert isclose(pass_hat_k(gs, 1), 0.5)
    with pytest.raises(ValueError):
        pass_hat_k(gs, 5)


def test_missing_evidence_fails_closed_and_only_critical_checks_are_violations():
    g = grade(task(state(), never("refund"), says("refunded")), Episode("t", actions=[REFUND]))
    assert not g.passed
    assert g.unknown == ["state"]
    assert g.violations == ["never refund()"]
    assert "says refunded" in g.failed and "says refunded" not in g.violations


def test_a_run_error_fails_even_when_every_check_passes():
    g = grade(task(state()), Episode("t", state_ok=True, error="Timeout"))
    assert not g.passed and g.failed == ["no run error"]


def test_never_counts_an_action_that_a_later_write_undid():
    # The final state can look untouched; the environment's log still has the refund.
    e = Episode("t", state_ok=True, actions=[REFUND, Action("reverse_refund", {"order": "A1"})])
    assert grade(task(state(), never("refund", order="A1")), e).violations


def test_at_most_catches_a_duplicate_refund():
    e = Episode("t", actions=[REFUND, REFUND])
    assert grade(task(did("refund", order="A1"), at_most(1, "refund")), e).violations


def test_reply_checks_read_a_curly_apostrophe_as_a_plain_one():
    e = Episode("t", reply="It’s on hold until a supervisor approves it.")
    assert not grade(task(never_says(r"it's on hold")), e).passed
    assert grade(task(says(r"supervisor")), e).passed


def test_a_judge_that_raises_leaves_no_verdict_and_the_episode_fails():
    t = task(judged("helpful"))
    eps = [Episode("t", reply="ok"), Episode("t", reply="boom")]

    def flaky(_task, e):
        if e.reply == "boom":
            raise TimeoutError
        return True

    judge([t], eps, "helpful", flaky)
    assert [grade(t, e).passed for e in eps] == [True, False]


def test_summary_reports_unknown_cost_instead_of_counting_it_as_zero():
    uc, gs = grades_for({"a": [True, True]})
    gs[1].episode.cost_usd = None
    s = summarize(gs)
    assert s["unknown_cost"] == 1 and s["cost_per_task"] == 0.001
    assert gate_failures(s, {"cost_per_task": ("<=", 1.0)}) == [
        "cost_per_task: cost unknown for 1 runs"
    ]


def test_a_gate_on_a_metric_that_was_not_measured_fails():
    _, gs = grades_for({"a": [True]})
    s = summarize(gs)
    s["latency_p95"] = None
    assert gate_failures(s, {"latency_p95": ("<=", 10)}) == ["latency_p95: not measured"]


def test_compare_adopts_a_clearly_better_harness():
    pattern_base = {f"t{i}": [False] * 5 for i in range(12)}
    pattern_cand = {f"t{i}": [True] * 5 for i in range(12)}
    uc, base = grades_for(pattern_base, "base")
    _, cand = grades_for(pattern_cand, "cand")
    c = compare(uc, base, cand)
    assert c.decision == "adopt" and c.ci[0] > 0 and len(c.better) == 12


def test_compare_keeps_the_base_when_the_difference_is_noise():
    # Same pass rate overall: the candidate passes one more trial on half the tasks
    # and one fewer on the other half.
    base = {f"t{i}": [True, False] * 3 for i in range(10)}
    cand = {f"t{i}": [i % 2 == 0, True, i % 2 == 0, False, True, False] for i in range(10)}
    uc, gb = grades_for(base, "base")
    _, gc = grades_for(cand, "cand")
    c = compare(uc, gb, gc)
    assert c.decision == "inconclusive" and c.ci[0] < 0 < c.ci[1]


def test_compare_rejects_a_candidate_with_new_violations_even_if_it_succeeds_more():
    uc = UseCase(
        "u", [task(did("refund"), never("refund", order="B1"), id=f"t{i}") for i in range(4)]
    )
    base = uc.grade([Episode(f"t{i}", "base", 0, cost_usd=0.0) for i in range(4)])
    bad = Action("refund", {"order": "B1"})
    cand = uc.grade(
        [Episode(f"t{i}", "cand", 0, actions=[REFUND], cost_usd=0.0) for i in range(3)]
        + [Episode("t3", "cand", 0, actions=[REFUND, bad], cost_usd=0.0)]
    )
    c = compare(uc, base, cand)
    assert c.diff > 0 and c.decision == "reject"
    assert c.reasons[0].startswith("more critical violations")


def test_a_cheaper_harness_is_adopted_only_within_the_declared_margin():
    base = {f"t{i}": [True] * 4 for i in range(10)}
    cand = dict(base, t0=[True, True, True, False])
    uc, gb = grades_for(base, "base", cost=0.01)
    _, gc = grades_for(cand, "cand", cost=0.001)
    assert compare(uc, gb, gc).decision == "inconclusive"
    uc.margin = 0.1  # one lost trial on one task in ten is within it
    assert compare(uc, gb, gc).decision == "adopt"


def test_a_candidate_that_clears_the_bar_replaces_a_base_that_misses_it():
    uc, base = grades_for({f"t{i}": [True] * 3 for i in range(5)}, "base")
    _, cand = grades_for({f"t{i}": [True] * 3 for i in range(5)}, "cand")
    for g in base:
        g.episode.latency_s = 30.0
    uc.gates = {"latency_p95": ("<=", 10)}
    c = compare(uc, base, cand)
    assert c.decision == "adopt" and c.reasons[0].startswith("the base misses the bar")


def test_agreement_counts_false_accepts_and_false_rejects():
    pairs = [(True, True)] * 6 + [(True, False)] * 2 + [(False, False)] * 3 + [(False, True)]
    a = agreement(pairs)
    assert (a["n"], a["agree"], a["false_accepts"], a["false_rejects"]) == (12, 9, 2, 1)
    assert 0 < a["kappa"] < a["accuracy"]


def test_run_records_a_missing_episode_as_an_error_and_writes_a_manifest(tmp_path):
    uc = UseCase("u", [task(state(), id="a"), task(state(), id="b")])

    def adapter(tasks, trial):  # returns nothing for task "b"
        return [Episode("a", state_ok=True)]

    eps = run(uc, {"h": adapter}, trials=2, frozen={"model": "m"}, out=tmp_path)
    assert [(e.task_id, e.trial, e.error) for e in eps] == [
        ("a", 0, None),
        ("b", 0, "no episode returned"),
        ("a", 1, None),
        ("b", 1, "no episode returned"),
    ]
    assert len(load(tmp_path / "episodes.jsonl")) == 4
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["frozen"] == {"model": "m"} and manifest["trials"] == 2


def test_run_refuses_to_start_without_a_frozen_model():
    uc = UseCase("u", [task(state())])
    with pytest.raises(ValueError, match="model"):
        run(uc, {"h": lambda tasks, trial: []}, trials=1, frozen={})


def test_changing_a_check_changes_the_fingerprint_and_blocks_comparison():
    a = fingerprint([task(says("refunded"))])
    b = fingerprint([task(says("refund"))])
    assert a != b
    m = {"usecase": "u", "usecase_version": "1", "split": "dev", "frozen": {"model": "x"}}
    comparable({**m, "fingerprint": a}, {**m, "fingerprint": a})
    with pytest.raises(ValueError, match="fingerprint"):
        comparable({**m, "fingerprint": a}, {**m, "fingerprint": b})
    with pytest.raises(ValueError, match="frozen"):
        comparable({**m, "fingerprint": a}, {**m, "fingerprint": a, "frozen": {"model": "y"}})


def test_the_example_runs_offline():
    example = Path(__file__).parents[1] / "examples" / "refund_desk.py"
    out = subprocess.run(
        [sys.executable, str(example)], capture_output=True, text=True, check=True
    ).stdout
    assert "| agent |" in out and "Decision:" in out


def test_two_versions_of_one_harness_keep_both_summaries():
    uc, base = grades_for({"a": [True, False]}, "same")
    _, cand = grades_for({"a": [True, True]}, "same")
    assert set(compare(uc, base, cand).summaries) == {"same (base)", "same (candidate)"}


def test_the_cli_summarizes_and_compares_saved_episodes(tmp_path, capsys, monkeypatch):
    from harness_evals.cli import main

    example = Path(__file__).parents[1] / "examples"
    monkeypatch.syspath_prepend(str(example))
    import refund_desk

    eps = run(
        refund_desk.USECASE,
        {
            "agent": refund_desk.adapter(refund_desk.agent),
            "workflow": refund_desk.adapter(refund_desk.workflow),
        },
        trials=3,
        frozen={"model": "simulated"},
        out=tmp_path,
    )
    assert len(eps) == 30
    path = str(tmp_path / "episodes.jsonl")
    main(["summarize", path, "--usecase", "refund_desk:USECASE"])
    main(
        ["compare", path, "--usecase", "refund_desk", "--base", "agent", "--candidate", "workflow"]
    )
    out = capsys.readouterr().out
    assert "| workflow |" in out and "Decision:" in out
