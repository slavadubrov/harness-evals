import json
import subprocess
import sys
from math import isclose
from pathlib import Path

import pytest

from harness_evals import (
    Action,
    Check,
    Episode,
    Task,
    UseCase,
    agreement,
    at_most,
    comparable,
    compare,
    did,
    ended,
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
    save,
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
    assert says("refunded").name in g.failed and says("refunded").name not in g.violations


def test_a_run_error_fails_even_when_every_check_passes():
    g = grade(task(state()), Episode("t", state_ok=True, error="Timeout"))
    assert not g.passed and g.failed == ["no run error"]


@pytest.mark.parametrize("error", [{"infra_error": True}, {"status": "error"}, {"error": ""}])
def test_error_metadata_always_fails_the_episode(error):
    g = grade(task(state()), Episode("t", state_ok=True, **error))
    assert not g.passed


def test_action_checks_distinguish_missing_arguments_from_explicit_nulls():
    e = Episode("t", actions=[Action("refund")])
    assert not grade(task(did("refund", order=None)), e).passed
    assert grade(task(never("refund", order=None)), e).passed
    e.actions[0] = Action("refund", {"order": None})
    assert grade(task(did("refund", order=None)), e).passed


def test_grading_requires_the_episode_to_belong_to_the_task():
    with pytest.raises(ValueError, match="task"):
        grade(task(state()), Episode("other", state_ok=True))


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
    assert s["cost_per_success"] is None
    assert gate_failures(s, {"cost_per_task": ("<=", 1.0)}) == [
        "cost_per_task: cost unknown for 1 runs"
    ]


def test_a_gate_on_a_metric_that_was_not_measured_fails():
    _, gs = grades_for({"a": [True]})
    s = summarize(gs)
    s["latency_p95"] = None
    assert gate_failures(s, {"latency_p95": ("<=", 10)}) == ["latency_p95: not measured"]


def test_latency_gates_require_observations_for_every_run():
    _, gs = grades_for({"a": [True, True]})
    gs[1].episode.latency_s = None
    s = summarize(gs)
    assert s["latency_p50"] is None and s["latency_p95"] is None
    assert gate_failures(s, {"latency_p95": ("<=", 10)})


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


def test_missing_cost_cannot_establish_that_a_candidate_is_cheaper():
    uc, base = grades_for({"a": [True, True]}, "base", cost=0.01)
    _, cand = grades_for({"a": [True, True]}, "cand", cost=0.001)
    uc.margin = 0.1
    cand[1].episode.cost_usd = None
    assert compare(uc, base, cand).decision == "inconclusive"
    cand[1].episode.cost_usd = 0.001
    base[1].episode.cost_usd = None
    assert compare(uc, base, cand).decision == "inconclusive"


def test_equal_success_and_known_lower_cost_need_no_loss_margin():
    uc, base = grades_for({"a": [True, True]}, "base", cost=0.01)
    _, cand = grades_for({"a": [True, True]}, "cand", cost=0.001)
    assert compare(uc, base, cand).decision == "adopt"


@pytest.mark.parametrize("margin", [-0.1, float("nan"), float("inf")])
def test_comparison_requires_a_finite_nonnegative_margin(margin):
    uc, base = grades_for({"a": [True]}, "base")
    _, cand = grades_for({"a": [True]}, "cand")
    uc.margin = margin
    with pytest.raises(ValueError, match="margin"):
        compare(uc, base, cand)


def test_comparison_rejects_unpaired_tasks_and_mixed_harnesses():
    uc, base = grades_for({"a": [True], "b": [False]}, "base")
    _, cand = grades_for({"a": [True]}, "cand")
    with pytest.raises(ValueError, match="same tasks"):
        compare(uc, base, cand)
    with pytest.raises(ValueError, match="one harness"):
        compare(uc, base + cand, base)


def test_comparison_requires_unique_and_equally_repeated_trials():
    uc, base = grades_for({"a": [True, False]}, "base")
    _, cand = grades_for({"a": [True]}, "cand")
    with pytest.raises(ValueError, match="trials"):
        compare(uc, base, cand)
    with pytest.raises(ValueError, match="duplicate"):
        compare(uc, base, cand + cand)


def test_custom_metrics_cannot_replace_required_metrics():
    _, gs = grades_for({"a": [True]})
    with pytest.raises(ValueError, match="required metric"):
        summarize(gs, {"violations": lambda grades: 0})


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


@pytest.mark.parametrize("trials", [0, -1])
def test_run_requires_positive_trials(trials):
    uc = UseCase("u", [task(state())])
    with pytest.raises(ValueError, match="trials"):
        run(uc, {"h": lambda tasks, trial: []}, trials=trials, frozen={"model": "m"})


def test_run_rejects_empty_harnesses_and_an_empty_model():
    uc = UseCase("u", [task(state())])
    with pytest.raises(ValueError, match="harness"):
        run(uc, {}, trials=1, frozen={"model": "m"})
    with pytest.raises(ValueError, match="model"):
        run(uc, {"h": lambda tasks, trial: []}, trials=1, frozen={"model": None})


def test_run_preserves_observations_when_an_adapter_reuses_an_episode():
    uc = UseCase("u", [Task("t", {"values": ["fixed"]}, (state(),))])
    reused = Episode("t", state_ok=True)

    def adapter(tasks, trial):
        assert tasks[0].input == {"values": ["fixed"]}
        tasks[0].input["values"].clear()
        reused.actions.append(Action("refund"))
        tasks.clear()
        return [reused]

    eps = run(uc, {"a": adapter, "b": adapter}, trials=2, frozen={"model": "m"})
    assert [(e.harness, e.trial, len(e.actions)) for e in eps] == [
        ("a", 0, 1),
        ("b", 0, 2),
        ("b", 1, 3),
        ("a", 1, 4),
    ]
    assert reused.harness == ""


def test_run_rotates_each_harness_through_every_execution_position():
    uc = UseCase("u", [task(state())])
    order = []

    def adapter(name):
        def run_tasks(tasks, trial):
            order.append((trial, name))
            return [Episode("t", state_ok=True)]

        return run_tasks

    run(uc, {name: adapter(name) for name in "abc"}, trials=3, frozen={"model": "m"})
    assert order == [
        (trial, name) for trial, names in enumerate(("abc", "bca", "cab")) for name in names
    ]


@pytest.mark.parametrize("ids", [["t", "t"], ["t", "unexpected"]])
def test_run_rejects_duplicate_or_unrequested_episodes(ids):
    uc = UseCase("u", [task(state())])
    with pytest.raises(ValueError, match="episode"):
        run(
            uc,
            {"h": lambda tasks, trial: [Episode(t) for t in ids]},
            trials=1,
            frozen={"model": "m"},
        )


def test_adapter_outages_are_recorded_without_losing_other_trials(tmp_path):
    uc = UseCase("u", [task(state())])

    def adapter(tasks, trial):
        if trial == 1:
            raise TimeoutError("request failed")
        return [Episode("t", state_ok=True)]

    eps = run(uc, {"h": adapter}, trials=3, frozen={"model": "m"}, out=tmp_path)
    assert len(load(tmp_path / "episodes.jsonl")) == 3
    assert [g.passed for g in uc.grade(eps)] == [True, False, True]
    assert eps[1].infra_error and eps[1].status == "error"
    assert eps[1].error == "adapter raised TimeoutError"


@pytest.mark.parametrize("estimate", [pass_hat_k, pass_at_k])
def test_pass_estimators_require_at_least_one_trial(estimate):
    _, gs = grades_for({"a": [True, False]})
    with pytest.raises(ValueError, match="k"):
        estimate(gs, 0)


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


def test_criticality_and_pattern_boundaries_are_part_of_the_fingerprint():
    check = Check("custom", lambda e: True)
    critical = Check("custom", check.fn, critical=True)
    assert fingerprint([task(check)]) != fingerprint([task(critical)])
    assert fingerprint([task(says("a", "b"))]) != fingerprint([task(says("a & b"))])
    assert fingerprint([task(never_says("a", "b"))]) != fingerprint([task(never_says("a | b"))])
    assert fingerprint([task(ended("a", "b"))]) != fingerprint([task(ended("a|b"))])


def test_comparable_rejects_incomplete_manifests():
    manifest = {
        "usecase": "u",
        "usecase_version": "1",
        "fingerprint": "abc",
        "split": "dev",
        "frozen": {"model": "m"},
    }
    for key in manifest:
        incomplete = {k: v for k, v in manifest.items() if k != key}
        with pytest.raises(ValueError, match=key):
            comparable(incomplete, incomplete)
    with pytest.raises(ValueError, match="model"):
        comparable({**manifest, "frozen": {"model": None}}, {**manifest, "frozen": {"model": None}})


def test_failed_serialization_does_not_overwrite_saved_episodes(tmp_path):
    path = tmp_path / "episodes.jsonl"
    save([Episode("original", reply="返金")], path)
    original = path.read_bytes()
    with pytest.raises(TypeError):
        save([Episode("replacement"), Episode("invalid", extra={"value": object()})], path)
    assert path.read_bytes() == original
    assert load(path)[0].reply == "返金"
    assert list(tmp_path.iterdir()) == [path]


def test_invalid_manifest_does_not_overwrite_a_previous_run(tmp_path):
    uc = UseCase("u", [task(state())])
    harnesses = {"h": lambda tasks, trial: [Episode("t", state_ok=True)]}
    run(uc, harnesses, trials=1, frozen={"model": "m"}, out=tmp_path)
    original = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(TypeError):
        run(uc, harnesses, trials=2, frozen={"model": "m", "setting": object()}, out=tmp_path)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == original


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


@pytest.mark.parametrize("change", ["model", "checks", "version", "missing", "none"])
def test_cli_checks_manifests_before_comparing(tmp_path, monkeypatch, change):
    from types import SimpleNamespace

    from harness_evals.cli import main

    uc = UseCase("u", [task(state())])
    monkeypatch.setitem(sys.modules, "example_usecase", SimpleNamespace(USECASE=uc))
    paths = []
    for name in ("base", "candidate"):
        out = tmp_path / name
        frozen = {"model": "other" if change == "model" and name == "candidate" else "m"}
        run(
            uc,
            {name: lambda tasks, trial: [Episode("t", state_ok=True)]},
            trials=1,
            frozen=frozen,
            out=out,
        )
        paths.append(str(out / "episodes.jsonl"))
    if change == "checks":
        uc.tasks = [task(says("changed"))]
    elif change == "version":
        uc.version = "2"
    elif change == "missing":
        (tmp_path / "candidate" / "manifest.json").unlink()
    args = [
        "compare",
        *paths,
        "--usecase",
        "example_usecase",
        "--base",
        "base",
        "--candidate",
        "candidate",
    ]
    if change == "none":
        main(args)
    else:
        with pytest.raises(SystemExit, match="manifest|not comparable"):
            main(args)
