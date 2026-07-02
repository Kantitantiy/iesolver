"""
Summary aggregation tests — store'a sentetik satırlar yazıp özet fonksiyonlarını doğrula.
"""

from __future__ import annotations

import pytest

from ie_eval.analysis.report import format_comparison, format_summary
from ie_eval.analysis.stats import bootstrap_diff_ci, mcnemar_test
from ie_eval.analysis.summary import (
    compare_configs,
    per_problem_correctness,
    summarize_by_config,
)
from ie_eval.metrics import ProblemMetrics
from ie_eval.runner import RunRecord
from ie_eval.store import ResultStore
from ie_eval.validator import FeasibilityCheck


# =============================================================================
# Sentetik veri fabrikası
# =============================================================================
def _mk_rec(
    *, problem_id, config_id, run_idx,
    correct, execution=True, feasible=True, feas_checked=True,
    cost=0.001, tokens_in=100, tokens_out=50, llm_calls=3,
    elapsed=1.0, retries=0, error_class=None,
) -> RunRecord:
    metrics = ProblemMetrics(
        problem_id=problem_id,
        execution_rate=execution,
        numerical_match=correct,
        feasibility=FeasibilityCheck(feasible=feasible, violations=[], checked=feas_checked),
        elapsed_s=elapsed,
        total_tokens_in=tokens_in,
        total_tokens_out=tokens_out,
        total_cost_usd=cost,
        total_llm_calls=llm_calls,
        node_count=2,
        retry_count=retries,
        error_class=error_class,
        per_node={
            "n": {"latency_ms": elapsed * 1000, "tokens_in": tokens_in,
                  "tokens_out": tokens_out, "cost_usd": cost,
                  "llm_calls": llm_calls, "invocations": 1,
                  "error_class": error_class},
        },
    )
    return RunRecord(
        problem_id=problem_id, config_id=config_id, run_idx=run_idx,
        success=execution and correct, elapsed_s=elapsed, state={},
        error=None, metrics=metrics,
    )


def _seed_store(tmp_path, records):
    store = ResultStore(tmp_path / "analysis.sqlite")
    for r in records:
        store.persist(r)
    return store


# =============================================================================
# summarize_by_config
# =============================================================================
def test_summarize_empty_store(tmp_path):
    store = ResultStore(tmp_path / "x.sqlite")
    s = summarize_by_config(store, "any")
    assert s.n_problems == 0
    assert s.accuracy_mean == 0.0
    assert s.error_class_counts == {}


def test_summarize_single_config_3_runs_perfect(tmp_path):
    recs = []
    for pid in ("p1", "p2"):
        for run_idx in range(3):
            recs.append(_mk_rec(problem_id=pid, config_id="A", run_idx=run_idx, correct=True))
    store = _seed_store(tmp_path, recs)

    s = summarize_by_config(store, "A")
    assert s.n_problems == 2
    assert s.n_runs == 6
    assert s.n_runs_per_problem == 2  # her run_idx için 2 problem
    assert s.accuracy_mean == 1.0
    assert s.accuracy_std == 0.0
    assert s.accuracy_per_run == [1.0, 1.0, 1.0]
    assert s.execution_rate == 1.0
    assert s.feasibility_rate == 1.0
    assert s.total_cost_usd == pytest.approx(6 * 0.001)


def test_summarize_mixed_correctness_across_runs(tmp_path):
    """3 problem, 3 run. run 0: 3/3 doğru, run 1: 2/3, run 2: 1/3."""
    recs = [
        _mk_rec(problem_id="p1", config_id="A", run_idx=0, correct=True),
        _mk_rec(problem_id="p2", config_id="A", run_idx=0, correct=True),
        _mk_rec(problem_id="p3", config_id="A", run_idx=0, correct=True),

        _mk_rec(problem_id="p1", config_id="A", run_idx=1, correct=True),
        _mk_rec(problem_id="p2", config_id="A", run_idx=1, correct=True),
        _mk_rec(problem_id="p3", config_id="A", run_idx=1, correct=False),

        _mk_rec(problem_id="p1", config_id="A", run_idx=2, correct=True),
        _mk_rec(problem_id="p2", config_id="A", run_idx=2, correct=False),
        _mk_rec(problem_id="p3", config_id="A", run_idx=2, correct=False),
    ]
    store = _seed_store(tmp_path, recs)

    s = summarize_by_config(store, "A")
    assert s.accuracy_per_run == [1.0, pytest.approx(2/3), pytest.approx(1/3)]
    assert s.accuracy_mean == pytest.approx((1.0 + 2/3 + 1/3) / 3)
    assert s.accuracy_std > 0.0


def test_summarize_error_class_counts(tmp_path):
    recs = [
        _mk_rec(problem_id="p1", config_id="A", run_idx=0, correct=False,
                execution=False, error_class="SandboxTimeout"),
        _mk_rec(problem_id="p2", config_id="A", run_idx=0, correct=False,
                execution=False, error_class="SandboxTimeout"),
        _mk_rec(problem_id="p3", config_id="A", run_idx=0, correct=False,
                execution=False, error_class="SandboxFailure"),
    ]
    store = _seed_store(tmp_path, recs)
    s = summarize_by_config(store, "A")
    assert s.error_class_counts == {"SandboxTimeout": 2, "SandboxFailure": 1}


def test_summarize_ignores_other_configs(tmp_path):
    recs = [
        _mk_rec(problem_id="p1", config_id="A", run_idx=0, correct=True),
        _mk_rec(problem_id="p1", config_id="B", run_idx=0, correct=False),
    ]
    store = _seed_store(tmp_path, recs)
    s_a = summarize_by_config(store, "A")
    s_b = summarize_by_config(store, "B")
    assert s_a.accuracy_mean == 1.0
    assert s_b.accuracy_mean == 0.0


def test_summarize_feasibility_rate_denominator(tmp_path):
    """feasibility_checked=False sayılmamalı — denominator sadece kontrol edilenler."""
    recs = [
        _mk_rec(problem_id="p1", config_id="A", run_idx=0, correct=True,
                feasible=True, feas_checked=True),
        _mk_rec(problem_id="p2", config_id="A", run_idx=0, correct=True,
                feasible=False, feas_checked=True),
        _mk_rec(problem_id="p3", config_id="A", run_idx=0, correct=True,
                feasible=True, feas_checked=False),   # kontrol yok → sayılmaz
    ]
    store = _seed_store(tmp_path, recs)
    s = summarize_by_config(store, "A")
    assert s.feasibility_checked == 2       # p1 + p2
    assert s.feasibility_rate == 0.5         # p1 doğru / 2 kontrol


# =============================================================================
# per_problem_correctness — 4 politika
# =============================================================================
@pytest.fixture
def mixed_store(tmp_path):
    """p1: 3/3 doğru; p2: 2/3 doğru; p3: 1/3 doğru; p4: 0/3."""
    recs = []
    for run_idx in range(3):
        recs.append(_mk_rec(problem_id="p1", config_id="A", run_idx=run_idx, correct=True))
    for run_idx, c in enumerate([True, True, False]):
        recs.append(_mk_rec(problem_id="p2", config_id="A", run_idx=run_idx, correct=c))
    for run_idx, c in enumerate([True, False, False]):
        recs.append(_mk_rec(problem_id="p3", config_id="A", run_idx=run_idx, correct=c))
    for run_idx in range(3):
        recs.append(_mk_rec(problem_id="p4", config_id="A", run_idx=run_idx, correct=False))
    return _seed_store(tmp_path, recs)


def test_per_problem_majority(mixed_store):
    out = per_problem_correctness(mixed_store, "A", policy="majority")
    # p1: 3/3 → True; p2: 2/3 → True; p3: 1/3 → False; p4: 0/3 → False
    assert out == {"p1": True, "p2": True, "p3": False, "p4": False}


def test_per_problem_all(mixed_store):
    out = per_problem_correctness(mixed_store, "A", policy="all")
    assert out == {"p1": True, "p2": False, "p3": False, "p4": False}


def test_per_problem_any(mixed_store):
    out = per_problem_correctness(mixed_store, "A", policy="any")
    assert out == {"p1": True, "p2": True, "p3": True, "p4": False}


def test_per_problem_first(mixed_store):
    """run_idx=0'daki durum belirleyici."""
    out = per_problem_correctness(mixed_store, "A", policy="first")
    # run 0: p1=T, p2=T, p3=T, p4=F
    assert out == {"p1": True, "p2": True, "p3": True, "p4": False}


# =============================================================================
# compare_configs — 2×2 contingency
# =============================================================================
def test_compare_configs_full_2x2(tmp_path):
    """
    Problem | A | B
    --------+---+---
    p1      | T | T   (both correct)
    p2      | T | F   (only A)
    p3      | F | T   (only B)
    p4      | F | F   (both wrong)
    """
    recs = []
    for pid, ca, cb in [("p1", True, True), ("p2", True, False),
                         ("p3", False, True), ("p4", False, False)]:
        recs.append(_mk_rec(problem_id=pid, config_id="A", run_idx=0, correct=ca))
        recs.append(_mk_rec(problem_id=pid, config_id="B", run_idx=0, correct=cb))
    store = _seed_store(tmp_path, recs)

    comp = compare_configs(store, "A", "B", policy="first")
    assert comp.n_common_problems == 4
    assert comp.both_correct == 1
    assert comp.only_a_correct == 1
    assert comp.only_b_correct == 1
    assert comp.both_wrong == 1
    assert comp.accuracy_a == 0.5
    assert comp.accuracy_b == 0.5
    assert comp.accuracy_diff == 0.0


def test_compare_configs_only_uses_common_problems(tmp_path):
    """A: p1, p2, p3; B: p2, p3, p4 → common: p2, p3."""
    recs = [
        _mk_rec(problem_id="p1", config_id="A", run_idx=0, correct=True),
        _mk_rec(problem_id="p2", config_id="A", run_idx=0, correct=True),
        _mk_rec(problem_id="p3", config_id="A", run_idx=0, correct=False),
        _mk_rec(problem_id="p2", config_id="B", run_idx=0, correct=False),
        _mk_rec(problem_id="p3", config_id="B", run_idx=0, correct=True),
        _mk_rec(problem_id="p4", config_id="B", run_idx=0, correct=True),
    ]
    store = _seed_store(tmp_path, recs)
    comp = compare_configs(store, "A", "B", policy="first")
    assert comp.n_common_problems == 2   # p2 + p3
    assert comp.only_a_correct == 1      # p2: A doğru, B yanlış
    assert comp.only_b_correct == 1      # p3: A yanlış, B doğru


# =============================================================================
# Rapor formatları — smoke (kırılmasın; string içeriği spot kontrol)
# =============================================================================
def test_format_summary_contains_key_fields(tmp_path):
    recs = [
        _mk_rec(problem_id="p1", config_id="pipeline", run_idx=0, correct=True),
    ]
    store = _seed_store(tmp_path, recs)
    text = format_summary(summarize_by_config(store, "pipeline"))
    assert "pipeline" in text
    assert "pass@1" in text
    assert "execution_rate" in text
    assert "cost" in text


def test_format_comparison_with_stats(tmp_path):
    recs = []
    for pid in [f"p{i}" for i in range(30)]:
        # Pipeline hepsini doğru, single_shot yarısını
        recs.append(_mk_rec(problem_id=pid, config_id="pipeline", run_idx=0, correct=True))
        recs.append(_mk_rec(problem_id=pid, config_id="single_shot", run_idx=0,
                            correct=(int(pid[1:]) % 2 == 0)))
    store = _seed_store(tmp_path, recs)

    comp = compare_configs(store, "pipeline", "single_shot", policy="first")
    mc = mcnemar_test(comp.only_a_correct, comp.only_b_correct)
    a_marks = list(per_problem_correctness(store, "pipeline", policy="first").values())
    b_marks = list(per_problem_correctness(store, "single_shot", policy="first").values())
    ci = bootstrap_diff_ci(a_marks, b_marks, n_iterations=500, seed=42)

    text = format_comparison(comp, mc, ci)
    assert "pipeline" in text
    assert "single_shot" in text
    assert "McNemar" in text
    assert "Bootstrap" in text
    assert "χ²" in text or "exact" in text
