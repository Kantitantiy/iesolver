"""Deterministic validator tests (DESIGN_REVIEW §3.2)."""

from __future__ import annotations

import math

from ie_eval.problem import GroundTruth
from ie_eval.validator import (
    check_feasibility,
    extract_numbers,
    numerical_match,
)


# =============================================================================
# extract_numbers
# =============================================================================
def test_extract_numbers_basic():
    assert extract_numbers("Q* = 707.1") == [707.1]


def test_extract_numbers_multiple_and_scientific():
    got = extract_numbers("cost = 3.14e2, budget = -50")
    assert 314.0 in got
    assert -50.0 in got


def test_extract_numbers_thousand_separator():
    """10,000 → 10000; 3,14 → 3.14 (European decimal)."""
    got = extract_numbers("D = 10,000 units; approx pi = 3,14")
    assert 10000.0 in got
    assert 3.14 in got


def test_extract_numbers_from_empty():
    assert extract_numbers("") == []


# =============================================================================
# numerical_match
# =============================================================================
def test_numerical_match_exact():
    assert numerical_match(707.1, "Optimal Q = 707.1")


def test_numerical_match_within_tolerance():
    assert numerical_match(707.1, "Q = 707.5", tolerance_rel=1e-2)


def test_numerical_match_outside_tolerance():
    assert not numerical_match(707.1, "Q = 800", tolerance_rel=1e-2)


def test_numerical_match_empty_text():
    assert not numerical_match(707.1, "")


def test_numerical_match_small_expected_value():
    """max(|expected|, 1) denomu küçük değerlerde toleransı sabitler."""
    assert numerical_match(0.001, "value = 0.0011", tolerance_rel=1e-3)


# =============================================================================
# check_feasibility
# =============================================================================
def _eoq_feasibility(sol):
    return [] if sol.get("Q", -1) >= 0 else ["Q negative"]


def test_check_feasibility_ok():
    gt = GroundTruth(feasibility_fn=_eoq_feasibility)
    result = check_feasibility({"Q": 707.1}, gt)
    assert result.feasible
    assert result.violations == []
    assert result.checked


def test_check_feasibility_violation():
    gt = GroundTruth(feasibility_fn=_eoq_feasibility)
    result = check_feasibility({"Q": -1.0}, gt)
    assert not result.feasible
    assert "Q negative" in result.violations
    assert result.checked


def test_check_feasibility_no_function():
    """GroundTruth'ta feasibility_fn yoksa: feasible=True, checked=False."""
    gt = GroundTruth()
    result = check_feasibility({}, gt)
    assert result.feasible
    assert not result.checked


# =============================================================================
# IE-Case entegrasyon smoke — feasibility fonksiyonları LP kısıtlarını doğru sınıyor mu?
# =============================================================================
def test_ie_case_transport_feasibility_optimal_solution():
    from ie_eval.datasets.ie_case import ie_case_dataset
    problems = list(ie_case_dataset.load())
    transport = next(p for p in problems if p.id == "transport-2x3")
    result = check_feasibility(transport.ground_truth.solution, transport.ground_truth)
    assert result.feasible, f"Optimal solution violates constraints: {result.violations}"


def test_ie_case_transport_feasibility_bad_solution():
    from ie_eval.datasets.ie_case import ie_case_dataset
    problems = list(ie_case_dataset.load())
    transport = next(p for p in problems if p.id == "transport-2x3")
    bad = {k: 0.0 for k in transport.ground_truth.solution}
    result = check_feasibility(bad, transport.ground_truth)
    assert not result.feasible
    # Demand ihlalleri raporlanmalı
    assert any("demand not met" in v for v in result.violations)


def test_ie_case_eoq_analytical_matches_ground_truth():
    from ie_eval.datasets.ie_case import ie_case_dataset
    problems = list(ie_case_dataset.load())
    eoq = next(p for p in problems if p.id == "eoq-basic")
    expected = math.sqrt(2 * 10_000 * 50 / 2)
    assert abs(eoq.ground_truth.objective_value - expected) < 1e-9