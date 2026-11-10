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
    by_harness,
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
    pareto,
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


@pytest.mark.parametrize("outcomes", [[True] * 10, [False] * 10, [True, False] * 5])
@pytest.mark.parametrize("margin", [0.0, 0.02])
def test_equal_task_success_cannot_establish_noninferiority(outcomes, margin):
    pattern = {f"t{i}": outcomes for i in range(16)}
    uc, base = grades_for(pattern, "base", cost=0.01)
    _, cand = grades_for(pattern, "cand", cost=0.001)
    uc.margin = margin
    c = compare(uc, base, cand)
    assert c.ci == (0.0, 0.0)  # keep the descriptive bootstrap result
    assert c.decision == "inconclusive" and "zero-width" in c.reasons[0]


@pytest.mark.parametrize("margin, decision", [(0.2, "inconclusive"), (0.05, "reject")])
def test_constant_success_loss_does_not_support_cost_only_adoption(margin, decision):
    uc, base = grades_for({f"t{i}": [True] * 10 for i in range(8)}, "base", cost=0.01)
    _, cand = grades_for({f"t{i}": [True] * 9 + [False] for i in range(8)}, "cand", cost=0.001)
    uc.margin = margin
    c = compare(uc, base, cand)
    assert c.ci[0] == c.ci[1] == pytest.approx(-0.1)
    assert c.decision == decision


def test_roundoff_cannot_make_a_constant_success_loss_support_cost_only_adoption():
    counts = [2, 3, 5, 10, 20, 40, 60, 99]
    uc, base = grades_for(
        {f"t{i}": [True] * n + [False] * (100 - n) for i, n in enumerate(counts)},
        "base",
        cost=0.01,
    )
    _, cand = grades_for(
        {f"t{i}": [True] * (n - 1) + [False] * (101 - n) for i, n in enumerate(counts)},
        "cand",
        cost=0.001,
    )
    uc.margin = 0.02
    c = compare(uc, base, cand)
    # Each task loses exactly 1/100; subtraction leaves a tiny numerical interval.
    assert 0 < c.ci[1] - c.ci[0] < 1e-15
    assert c.decision == "inconclusive" and "zero-width" in c.reasons[0]


def test_fewer_than_six_tasks_cannot_decide_on_an_interval():
    # One task, 5 of 10 against 6 of 10: the bootstrap interval is a point above zero.
    uc, base = grades_for({"a": [True] * 5 + [False] * 5}, "base")
    _, cand = grades_for({"a": [True] * 6 + [False] * 4}, "cand")
    c = compare(uc, base, cand)
    assert c.decision == "inconclusive" and "too few" in c.reasons[0]
    # The gates still decide: a base with a violation is replaced whatever the task count.
    uc.gates = {"success": (">=", 0.6)}
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

    with pytest.warns(UserWarning, match="TimeoutError"):
        eps = run(uc, {"h": adapter}, trials=3, frozen={"model": "m"}, out=tmp_path)
    assert len(load(tmp_path / "episodes.jsonl")) == 3
    assert [g.passed for g in uc.grade(eps)] == [True, False, True]
    assert eps[1].infra_error and eps[1].status == "error"
    assert eps[1].error == "adapter raised TimeoutError('request failed')"


def test_an_adapter_bug_is_warned_about_once_and_keeps_its_message():
    uc = UseCase("u", [task(state())])

    def adapter(tasks, trial):
        return undefined_name  # noqa: F821 - the bug under test

    with pytest.warns(UserWarning, match="NameError") as caught:
        eps = run(uc, {"h": adapter}, trials=3, frozen={"model": "m"})
    assert len(caught) == 1
    assert "undefined_name" in eps[0].error


def test_run_accepts_a_string_path_and_rejects_unsaveable_episodes_early(tmp_path):
    uc = UseCase("u", [task(state())])
    calls = []

    def adapter(tasks, trial):
        calls.append(trial)
        return [Episode("t", state_ok=True, extra={"at": object()})]

    with pytest.raises(TypeError):
        run(uc, {"h": adapter}, trials=5, frozen={"model": "m"}, out=str(tmp_path / "x"))
    assert calls == [0]  # stopped at the first batch, not after the last

    ok = {"h": lambda tasks, trial: [Episode("t", state_ok=True)]}
    run(uc, ok, trials=1, frozen={"model": "m"}, out=str(tmp_path / "y"))
    assert len(load(tmp_path / "y" / "episodes.jsonl")) == 1


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
    main(["frontier", path, "--usecase", "refund_desk", "--x", "latency_p95"])
    assert capsys.readouterr().out.startswith("<svg")


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


# -- several metrics, their intervals and the frontier ----------------------------------


def test_cost_and_latency_intervals_resample_whole_tasks():
    uc, gs = grades_for({"a": [True, True], "b": [True, True]})
    for g in gs:  # task a costs 1 per run and takes 1 s; task b costs 3 and takes 3 s
        g.episode.cost_usd = g.episode.latency_s = 1.0 if g.episode.task_id == "a" else 3.0
    s = summarize(gs)
    assert s["cost_per_task"] == 2.0 and s["latency_p50"] == 1.0
    # A resample of [a, a] costs 1 and one of [b, b] costs 3: both happen in 2,000 draws.
    assert s["cost_per_task_ci"] == (1.0, 3.0) and s["latency_p95_ci"] == (1.0, 3.0)
    assert s["pass^k_ci"] == s["success_ci"] == (1.0, 1.0)


def test_calls_and_tokens_per_task_are_unknown_until_every_run_reports_them():
    _, gs = grades_for({"a": [True, True], "b": [True, True]})
    for i, g in enumerate(gs):
        g.episode.model_calls, g.episode.input_tokens, g.episode.output_tokens = i, 100, 10 * i
    s = summarize(gs)
    assert s["calls_per_task"] == 1.5 and s["tokens_per_task"] == 115.0
    assert s["calls_per_task_ci"][0] <= 1.5 <= s["calls_per_task_ci"][1]
    gs[0].episode.output_tokens = None
    s = summarize(gs)
    assert s["tokens_per_task"] is None and s["tokens_per_task_ci"] is None


def test_use_case_metrics_get_an_interval_and_unknown_ones_get_none():
    _, gs = grades_for({"a": [True, False], "b": [True, True]})
    s = summarize(gs, {"replies": lambda grades: 1.0, "never": lambda grades: None})
    assert s["replies_ci"] == (1.0, 1.0) and s["never_ci"] is None
    with pytest.raises(ValueError, match="required metric"):
        summarize(gs, {"latency_p50_ci": lambda grades: 0.0})


def test_the_frontier_keeps_every_harness_that_no_other_beats_on_both_metrics():
    sums = {
        "agent": {"success": 0.9, "cost_per_task": 0.3},
        "workflow": {"success": 0.8, "cost_per_task": 0.1},
        "router": {"success": 0.7, "cost_per_task": 0.2},  # the workflow beats it on both
        "twin": {"success": 0.9, "cost_per_task": 0.3},  # a tie beats nobody
        "unpriced": {"success": 1.0, "cost_per_task": None},
    }
    assert pareto(sums, "cost_per_task", "success") == ["agent", "workflow", "twin"]
    calls = {"a": {"success": 0.9, "calls": 4.0}, "b": {"success": 0.9, "calls": 1.0}}
    assert pareto(calls, "calls", "success") == ["a"]
    assert pareto(calls, "calls", "success", lower=["calls"]) == ["b"]


def test_comparisons_report_paired_intervals_for_cost_latency_and_use_case_metrics():
    uc, base = grades_for({f"t{i}": [True] * 3 for i in range(6)}, "base", cost=0.01)
    _, cand = grades_for({f"t{i}": [True] * 3 for i in range(6)}, "cand", cost=0.002)
    uc.metrics = {"calls": lambda grades: 2.0}
    c = compare(uc, base, cand)
    assert set(c.deltas) == {"success", "calls", "cost_per_task", "latency_p50", "latency_p95"} | {
        "calls_per_task",
        "tokens_per_task",
    }
    assert c.deltas["tokens_per_task"] is None  # these episodes report no token counts
    diff, (lo, hi) = c.deltas["cost_per_task"]
    assert diff == pytest.approx(-0.008) and lo == pytest.approx(-0.008) == hi
    assert c.deltas["latency_p95"] == (0.0, (0.0, 0.0))


def test_cheaper_means_the_whole_cost_interval_lies_below_zero():
    uc, base = grades_for({f"t{i}": [True, False] for i in range(8)}, "base")
    _, cand = grades_for({f"t{i}": [True, i == 0] for i in range(8)}, "cand")
    # Success has a nonzero interval starting at zero; cost must establish the benefit.
    c = compare(uc, base, cand)
    assert c.ci[0] == 0 < c.ci[1]
    for i, g in enumerate(cand):  # cheaper on average, dearer on half the tasks
        g.episode.cost_usd = 0.0001 if i % 4 < 2 else 0.0017
    for g in base:
        g.episode.cost_usd = 0.001
    assert compare(uc, base, cand).decision == "inconclusive"
    for g in cand:
        g.episode.cost_usd = 0.0005
    assert compare(uc, base, cand).decision == "adopt"


def test_a_guardrail_rejects_a_cheaper_candidate_that_is_clearly_slower():
    uc, base = grades_for({f"t{i}": [True] * 3 for i in range(6)}, "base", cost=0.01)
    _, cand = grades_for({f"t{i}": [True] * 3 for i in range(6)}, "cand", cost=0.001)
    for g in cand:
        g.episode.latency_s = 6.0  # the base takes 1 s
    assert compare(uc, base, cand).decision == "inconclusive"
    uc.guardrails = {"latency_p95": ("<=", 2.0)}
    c = compare(uc, base, cand)
    assert c.decision == "reject" and c.reasons[0].startswith("latency_p95 changes by +5.00")


def test_a_guardrail_that_the_interval_may_cross_keeps_the_base():
    uc, base = grades_for({f"t{i}": [True] * 2 for i in range(10)}, "base", cost=0.01)
    _, cand = grades_for({f"t{i}": [True] * 2 for i in range(10)}, "cand", cost=0.001)
    for i, g in enumerate(cand):  # slower on some tasks only: the interval spans the limit
        g.episode.latency_s = 4.0 if i < 6 else 1.0
    uc.guardrails = {"latency_p50": ("<=", 1.0)}
    c = compare(uc, base, cand)
    assert c.decision == "inconclusive" and "may pass its limit" in c.reasons[-1]


def test_a_guardrail_on_a_metric_that_could_not_be_compared_keeps_the_base():
    uc, base = grades_for({f"t{i}": [True, True] for i in range(6)}, "base", cost=0.01)
    _, cand = grades_for({f"t{i}": [True, True] for i in range(6)}, "cand", cost=0.001)
    cand[0].episode.latency_s = None
    uc.guardrails = {"latency_p95": ("<=", 2.0)}
    c = compare(uc, base, cand)
    assert c.decision == "inconclusive" and c.reasons[-1] == "latency_p95 could not be compared"


@pytest.mark.parametrize(
    "guardrail",
    [{"unheard_of": ("<=", 1.0)}, {"latency_p95": ("<", 1.0)}, {"success": (">=", float("-inf"))}],
)
def test_guardrails_need_a_comparable_metric_an_operator_and_a_finite_limit(guardrail):
    with pytest.raises(ValueError, match="guardrail"):
        UseCase("u", [task(state())], guardrails=guardrail)


def test_the_frontier_plot_draws_every_harness_its_intervals_and_the_frontier():
    import xml.etree.ElementTree as ET

    from harness_evals.report import frontier_svg

    uc, a = grades_for({f"t{i}": [True, i > 0] for i in range(5)}, "agent", cost=0.0003)
    _, w = grades_for({f"t{i}": [i > 1] * 2 for i in range(5)}, "workflow", cost=0.00005)
    sums = {"agent": summarize(a), "workflow": summarize(w)}
    svg = ET.fromstring(frontier_svg(sums, "cost_per_task", "success", title="Success <> cost"))
    ns = {"s": "http://www.w3.org/2000/svg"}
    assert svg.find("s:title", ns).text == "Success <> cost"
    assert len(svg.findall("s:circle", ns)) == 2
    assert len(svg.find("s:polyline", ns).get("points").split()) == 2  # both on the frontier
    labels = [t.text for t in svg.findall("s:text", ns)]
    assert {"agent", "workflow", "$ per task", "Success"} <= set(labels)  # axes carry units
    with pytest.raises(ValueError, match="no harness"):
        frontier_svg(sums, "cost_per_task", "unmeasured")


def test_the_frontier_plot_keeps_labels_inside_names_what_it_left_out_and_can_zoom():
    import re
    import xml.etree.ElementTree as ET

    from harness_evals.report import frontier_svg

    sums = {
        "a-harness-with-a-long-name": {"cost_per_task": 0.0052, "success": 1.0},
        "close": {"cost_per_task": 0.0050, "success": 0.96},
        "no-cost": {"cost_per_task": None, "success": 0.9},
        "not-a-number": {"cost_per_task": float("nan"), "success": 0.9},
    }
    text = frontier_svg(sums, "cost_per_task", "success")
    ET.fromstring(text)  # well-formed, with no nan coordinates
    assert "nan" not in text.replace("not-a-number", "")
    assert "Left out, a metric is missing or incomplete: no-cost, not-a-number" in text
    x = float(re.search(r'x="([\d.]+)" y="[\d.]+" fill="#\w+" [^>]*>a-harness', text).group(1))
    assert x + 7.2 * len("a-harness-with-a-long-name") <= 640  # flipped to the left side
    ticks = re.findall(r'text-anchor="middle" class="mut" fill="#666">([\d.]+)<', text)
    assert ticks[0] == "0.000" and float(ticks[-1]) == 0.006  # no half-empty axis
    places = re.findall(r'class="halo \w+" x="(-?[\d.]+)" y="(-?[\d.]+)"', text)
    assert len(places) == 2 and all(0 < float(a) < 640 and 0 < float(b) < 452 for a, b in places)
    zoomed = frontier_svg(sums, "cost_per_task", "success", zero=False)
    ticks = re.findall(r'text-anchor="middle" class="mut" fill="#666">([\d.]+)<', zoomed)
    assert float(ticks[0]) > 0.004  # the axis starts just below the data


def test_the_table_shows_each_interval_beside_its_value():
    from harness_evals.report import table

    _, gs = grades_for({"a": [True, True], "b": [True, False]})
    row = table({"h": summarize(gs)}).splitlines()[-1]
    assert "| $0.00100 (0.00100–0.00100) |" in row and "| 1.00 (1.00–1.00) |" in row


def test_a_metric_undefined_or_failing_on_some_resamples_does_not_break_the_summary():
    _, gs = grades_for({"a": [True, True], "b": [False, False], "c": [False, False]})

    def share_of_refunds(grades):  # raises when the resample holds no task "a"
        hits = [g.passed for g in grades if g.episode.task_id == "a"]
        return sum(hits) / len(hits)

    s = summarize(gs, {"refunds named": share_of_refunds})
    assert s["refunds named"] == 1.0
    # Task "a" is missing from about 30% of resamples: too many to ignore, so no interval.
    assert s["refunds named_ci"] is None


def test_an_interval_is_unknown_when_more_than_one_tail_of_resamples_is_undefined():
    tasks = {f"t{i}": [i == 0] * 2 for i in range(10)}  # only one task ever passes
    _, gs = grades_for(tasks)
    s = summarize(gs)
    assert s["cost_per_success"] == pytest.approx(0.01)
    assert s["cost_per_success_ci"] is None  # most resamples have no passing run


def test_a_guardrail_that_may_break_is_named_when_it_blocks_a_gain():
    uc, base = grades_for({f"t{i}": [False] * 2 for i in range(10)}, "base")
    _, cand = grades_for({f"t{i}": [True] * 2 for i in range(10)}, "cand")
    for i, g in enumerate(cand):
        g.episode.latency_s = 4.0 if i < 6 else 1.0
    uc.guardrails = {"latency_p50": ("<=", 1.0)}
    c = compare(uc, base, cand)
    assert c.decision == "inconclusive"
    assert c.reasons[0].startswith("success is higher") and "guardrail may break" in c.reasons[0]


def test_a_base_that_misses_a_gate_is_replaced_before_guardrails_apply():
    uc, base = grades_for({f"t{i}": [True] * 2 for i in range(5)}, "base")
    _, cand = grades_for({f"t{i}": [True] * 2 for i in range(5)}, "cand")
    for g in base:
        g.episode.cost_usd = 1.0
    for g in cand:
        g.episode.latency_s = 9.0
    uc.gates, uc.guardrails = {"cost_per_task": ("<=", 0.5)}, {"latency_p95": ("<=", 1.0)}
    assert compare(uc, base, cand).decision == "adopt"


def test_decide_never_calls_a_candidate_cheaper_while_a_cost_is_unknown():
    from harness_evals import decide

    uc, base = grades_for({"a": [True, True]}, "base", cost=0.01)
    _, cand = grades_for({"a": [True, True]}, "cand", cost=0.001)
    sb, sc = summarize(base), summarize(cand)
    sc["unknown_cost"] = 1
    deltas = {"success": (0.0, (0.0, 0.0)), "cost_per_task": (-0.009, (-0.009, -0.009))}
    assert decide(uc, sb, sc, deltas)[0] == "inconclusive"


@pytest.mark.parametrize("name", ["foo_ci", "latency_p95"])
def test_custom_metric_names_cannot_shadow_required_metrics_or_intervals(name):
    _, gs = grades_for({"a": [True]})
    with pytest.raises(ValueError, match="replaces"):
        summarize(gs, {name: lambda grades: 1.0})


def test_malformed_guardrails_get_a_clear_error():
    for bad in ({"latency_p95": "<= 2"}, {"latency_p95": ("<=",)}, {"latency_p95": ("<=", "2")}):
        with pytest.raises(ValueError, match="guardrail"):
            UseCase("u", [task(state())], guardrails=bad)


def test_the_frontier_command_takes_lower_is_better_metrics_and_reports_unknown_ones(
    tmp_path, capsys, monkeypatch
):
    from types import SimpleNamespace

    from harness_evals.cli import main

    uc = UseCase("u", [task(state())], metrics={"tool calls": lambda gs: gs[0].episode.extra["n"]})
    monkeypatch.setitem(sys.modules, "frontier_usecase", SimpleNamespace(USECASE=uc))
    path = tmp_path / "episodes.jsonl"
    save(
        [
            Episode("t", "busy", 0, state_ok=True, extra={"n": 9.0}),
            Episode("t", "lean", 0, state_ok=True, extra={"n": 1.0}),
        ],
        path,
    )
    args = ["frontier", str(path), "--usecase", "frontier_usecase", "--x", "tool calls"]
    main([*args, "--lower", "tool calls"])
    svg = capsys.readouterr().out
    assert svg.count(">lean</text>") == 1 and 'class="halo hot"' in svg  # lean is on the frontier
    assert 'class="halo off"' in svg and ">busy</text>" in svg
    with pytest.raises(SystemExit, match="frontier"):
        main([*args[:-1], "unmeasured"])


def test_integer_latencies_from_json_keep_their_interval_in_the_table():
    from harness_evals.report import table

    _, gs = grades_for({"a": [True, True]})
    for g in gs:
        g.episode.latency_s = 2  # an int, as json.loads returns for 2
    row = table({"h": summarize(gs)}).splitlines()[-1]
    assert "| 2.00 (2.00–2.00) |" in row


# -- where each harness fails ------------------------------------------------------------


def _map_case():
    uc = UseCase(
        "u",
        [
            task(did("refund", order="A1"), says("refunded"), id="refund"),
            task(never("refund"), says("30 days"), id="late"),
        ],
    )
    eps = [
        Episode("refund", "agent", 0, actions=[REFUND], reply="refunded"),
        Episode("refund", "agent", 1, actions=[REFUND], reply="sorry"),  # says fails
        Episode("late", "agent", 0, actions=[REFUND], reply="30 days"),  # never fails
        Episode("late", "agent", 1, actions=[REFUND], reply="30 days"),
    ]
    return uc, {"agent": uc.grade(eps)}


def test_the_failure_map_gives_each_task_a_row_and_each_kind_of_check_a_column():
    import xml.etree.ElementTree as ET

    from harness_evals.report import check_kind, failure_map_svg

    assert check_kind("did refund(order='A1')") == "did refund"
    uc, grades = _map_case()
    svg = ET.fromstring(failure_map_svg(uc, grades))
    ns = {"s": "http://www.w3.org/2000/svg"}
    tips = {t.text for t in svg.iter("{http://www.w3.org/2000/svg}title")}
    assert "agent · refund · says: 1 of 2 runs failed" in tips
    assert "agent · late · never refund: 2 of 2 runs failed" in tips
    assert "agent · refund · did refund: 0 of 2 runs failed" in tips
    labels = {t.text for t in svg.findall(".//s:text", ns)}
    assert {"refund", "late", "did refund", "says", "never refund"} <= labels
    # "did refund" is not a check of "late", nor "never refund" of "refund": dots, not cells
    assert len(svg.findall("s:circle", ns)) == 2


def test_the_failure_map_by_run_has_one_cell_per_task_and_run():
    from harness_evals.report import failure_map_svg

    uc, grades = _map_case()
    svg = failure_map_svg(uc, grades, by="runs")
    assert svg.count("runs failed</title>") == 4
    assert "agent · refund · 2: 1 of 1 runs failed" in svg
    with pytest.raises(ValueError, match="by"):
        failure_map_svg(uc, grades, by="tasks")
    with pytest.raises(ValueError, match="no graded runs"):
        failure_map_svg(UseCase("v", [task(state(), id="other")]), grades)


def test_the_failures_command_draws_the_map_from_saved_runs(tmp_path, capsys, monkeypatch):
    from types import SimpleNamespace

    from harness_evals.cli import main

    uc, grades = _map_case()
    monkeypatch.setitem(sys.modules, "map_usecase", SimpleNamespace(USECASE=uc))
    path = tmp_path / "episodes.jsonl"
    save([g.episode for g in grades["agent"]], path)
    main(["failures", str(path), "--usecase", "map_usecase", "--by", "runs"])
    assert capsys.readouterr().out.startswith("<svg")


def test_critical_is_a_flag_on_action_checks_not_an_argument_filter():
    refunded = Episode("t", actions=[Action("refund", {"order": "A1"})])
    c = never("refund", critical=True)
    assert c.name == "never refund()" and c.critical and c(refunded) is False
    assert never("refund", critical=False).critical is False
    d = did("refund", critical=True)
    assert d.critical and d(refunded) is True
    assert at_most(0, "refund", critical=False).critical is False


def test_a_check_that_raises_is_undecided_not_a_crash():
    uc = UseCase("u", [task(Check("short reply", lambda e: len(e.reply) < 80))])
    (g,) = uc.grade([Episode("t", reply=None)])  # as after an adapter error
    assert not g.passed and g.unknown == ["short reply"]


def test_a_use_case_rejects_bad_gates_margins_and_bare_functions():
    with pytest.raises(ValueError, match="gate on 'sucess'"):
        UseCase("u", [task(state())], gates={"sucess": (">=", 0.8)})
    with pytest.raises(ValueError, match="gate on 'violations' needs"):
        UseCase("u", [task(state())], gates={"violations": ("==", 0)})
    with pytest.raises(ValueError, match="margin"):
        UseCase("u", [task(state())], margin=-1)
    with pytest.raises(ValueError, match="wrap"):
        UseCase("u", [task(lambda e: True)])


def test_summarize_refuses_grades_of_several_harnesses():
    _, a = grades_for({"t": [True, False]}, "a")
    _, b = grades_for({"t": [True, True]}, "b")
    with pytest.raises(ValueError, match="one harness"):
        summarize(a + b)
    assert {h: summarize(gs)["success"] for h, gs in by_harness(a + b).items()} == {
        "a": 0.5,
        "b": 1.0,
    }


def test_pareto_rejects_a_misspelt_metric_and_accepts_one_lower_name():
    sums = {"a": {"steps": 3, "success": 0.9}, "b": {"steps": 1, "success": 0.8}}
    with pytest.raises(ValueError, match="no harness has both"):
        pareto(sums, "cost", "success")
    assert pareto(sums, "steps", "success", lower="steps") == ["a", "b"]
    sums["c"] = {"steps": float("nan"), "success": 1.0}
    assert "c" not in pareto(sums, "steps", "success", lower=["steps"])


def test_the_command_refuses_one_harness_name_in_two_files(tmp_path):
    from types import SimpleNamespace

    from harness_evals.cli import main

    uc = UseCase("u", [task(state())])
    sys.modules["two_files_usecase"] = SimpleNamespace(USECASE=uc)
    try:
        for folder in ("main", "pr"):
            (tmp_path / folder).mkdir()
            save([Episode("t", "agent", 0, state_ok=True)], tmp_path / folder / "episodes.jsonl")
        files = [str(tmp_path / f / "episodes.jsonl") for f in ("main", "pr")]
        with pytest.raises(SystemExit, match="more than one file"):
            main(["summarize", *files, "--usecase", "two_files_usecase"])
        with pytest.raises(SystemExit, match="--usecase no_such_module"):
            main(["summarize", files[0], "--usecase", "no_such_module"])
        with pytest.raises(SystemExit, match="harness-evals:"):
            main(["summarize", str(tmp_path / "missing.jsonl"), "--usecase", "two_files_usecase"])
    finally:
        del sys.modules["two_files_usecase"]


def test_a_comparison_names_the_models_when_they_differ_even_after_a_rename():
    from harness_evals import comparison

    uc, base = grades_for({f"t{i}": [True, True] for i in range(6)}, "base")
    _, cand = grades_for({f"t{i}": [True, True] for i in range(6)}, "cand")
    for g in base:
        g.episode.models = ["small-0613"]
    for g in cand:
        g.episode.models = ["large-1106"]
    c = compare(uc, base, cand)
    c.base, c.candidate = "the current design", "the candidate"
    text = comparison(c)
    assert "Models differ: small-0613 against large-1106." in text
    assert "**the candidate** against **the current design**" in text


@pytest.mark.parametrize("verdict", ["false", "unknown", 1, 0, [], {}])
def test_invalid_check_verdicts_are_unknown(verdict):
    g = grade(task(Check("custom", lambda e: verdict, critical=True)), Episode("t"))
    assert not g.passed and g.unknown == ["custom"]
    assert g.failed == g.violations == []


def test_loaded_string_judge_and_state_verdicts_never_pass(tmp_path):
    path = tmp_path / "episodes.jsonl"
    path.write_text('{"task_id": "t", "state_ok": "true", "judgements": {"ok": "false"}}\n')
    (g,) = UseCase("u", [task(state(), judged("ok"))]).grade(load(path))
    assert not g.passed and g.unknown == ["state", "judge ok"]


@pytest.mark.parametrize(
    "change",
    [
        "missing_task",
        "missing_trial",
        "duplicate",
        "extra_task",
        "other_split",
        "empty",
        "negative_trial",
        "high_trial",
        "float_trial",
        "bool_trial",
        "string_trial",
        "missing_count",
        "zero_count",
        "float_count",
        "bool_count",
    ],
)
def test_cli_requires_exact_manifest_task_and_trial_coverage(tmp_path, monkeypatch, capsys, change):
    from types import SimpleNamespace

    from harness_evals.cli import main

    uc = UseCase(
        "u",
        [task(state(), id=f"t{i}") for i in range(7)]
        + [Task("holdout", "input", (state(),), split="holdout")],
    )
    monkeypatch.setitem(sys.modules, "coverage_usecase", SimpleNamespace(USECASE=uc))

    def adapter(tasks, trial):
        return [Episode(t.id, state_ok=True, cost_usd=0.01) for t in tasks]

    episodes = run(
        uc, {"base": adapter, "candidate": adapter}, trials=2, frozen={"model": "m"}, out=tmp_path
    )
    path = tmp_path / "episodes.jsonl"
    if change == "missing_task":
        episodes = [e for e in episodes if e.task_id != "t6"]
    elif change == "missing_trial":
        episodes = [e for e in episodes if not (e.task_id == "t6" and e.trial == 1)]
    elif change == "duplicate":
        episodes.append(episodes[0])
    elif change in ("extra_task", "other_split"):
        for e in episodes:
            if e.task_id == "t0":
                e.task_id = "unexpected" if change == "extra_task" else "holdout"
    elif change == "empty":
        episodes = []
    elif change.endswith("_trial"):
        episodes[0].trial = {
            "negative_trial": -1,
            "high_trial": 2,
            "float_trial": 0.0,
            "bool_trial": False,
            "string_trial": "0",
        }[change]
    else:
        manifest_path = tmp_path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if change == "missing_count":
            del manifest["trials"]
        else:
            manifest["trials"] = {"zero_count": 0, "float_count": 2.0, "bool_count": True}[change]
        manifest_path.write_text(json.dumps(manifest))
    save(episodes, path)
    with pytest.raises(SystemExit, match="coverage|trial|task|episodes"):
        main(
            [
                "compare",
                str(path),
                "--usecase",
                "coverage_usecase",
                "--base",
                "base",
                "--candidate",
                "candidate",
            ]
        )
    assert "Decision:" not in capsys.readouterr().out


def test_cli_accepts_complete_present_harnesses_from_one_manifest(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    from harness_evals.cli import main

    uc = UseCase("u", [task(state())])
    monkeypatch.setitem(sys.modules, "subset_usecase", SimpleNamespace(USECASE=uc))

    def adapter(tasks, trial):
        return [Episode("t", state_ok=True)]

    episodes = run(
        uc, {"base": adapter, "candidate": adapter}, trials=2, frozen={"model": "m"}, out=tmp_path
    )
    paths = []
    for name in ("base", "candidate"):
        path = tmp_path / f"{name}.jsonl"
        save([e for e in episodes if e.harness == name], path)
        paths.append(str(path))
    main(
        [
            "compare",
            *paths,
            "--usecase",
            "subset_usecase",
            "--base",
            "base",
            "--candidate",
            "candidate",
        ]
    )
    assert "Decision:" in capsys.readouterr().out


@pytest.mark.parametrize("x,y", [("cost_per_task", "success"), ("success", "cost_per_task")])
def test_cost_frontiers_omit_partial_costs_and_name_the_omission(x, y):
    import xml.etree.ElementTree as ET

    from harness_evals import frontier_svg

    _, complete = grades_for({"a": [True], "b": [True]}, "complete", cost=1.0)
    _, partial = grades_for({"a": [True], "b": [True]}, "partial", cost=0.0)
    partial[1].episode.cost_usd = None
    sums = {"complete": summarize(complete), "partial": summarize(partial)}
    assert sums["partial"]["cost_per_task"] == 0.0
    assert pareto(sums, x, y) == ["complete"]
    svg = frontier_svg(sums, x, y)
    root = ET.fromstring(svg)
    assert len(root.findall("{http://www.w3.org/2000/svg}circle")) == 1
    assert "Left out, a metric is missing or incomplete: partial" in svg
    assert pareto(sums, "latency_p50", "success") == ["complete", "partial"]
    with pytest.raises(ValueError, match="no harness has both"):
        pareto({"partial": sums["partial"]}, x, y)
    with pytest.raises(ValueError, match="no harness has both"):
        frontier_svg({"partial": sums["partial"]}, x, y)
