"""
Ablation davranış testleri (EVALUATION_PLAN §5, A1–A4).

LLM ÇAĞRISI YAPILMAZ — tüm solve çağrıları sahte state döndürür.
Her test, ablation flag'inin beklenen etkisini doğrular.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ie_eval.ablations import (
    make_a1_solve,
    make_a2_solve,
    make_a3_correctness_fn,
    make_a4_solve,
    make_a5_solve,
    make_a6_solve,
)
from ie_eval.problem import GroundTruth, Problem
from ie_eval.runner import run_one


# =============================================================================
# Yardımcı
# =============================================================================
def _problem(gt_value: float | None = 42.0) -> Problem:
    return Problem(
        id="ablation-test",
        prompt="THE ABLATION PROMPT",
        ground_truth=GroundTruth(objective_value=gt_value, tolerance_rel=1e-2),
    )


# =============================================================================
# Fabrika fonksiyonları — __name__ doğrulaması
# =============================================================================
def test_a1_solve_fn_name():
    assert make_a1_solve().__name__ == "a1_no_refiner_solve"


def test_a2_solve_fn_name():
    assert make_a2_solve().__name__ == "a2_no_retry_solve"


def test_a4_solve_fn_name():
    assert make_a4_solve().__name__ == "a4_fast_only_solve"


def test_a6_solve_fn_name():
    assert make_a6_solve().__name__ == "a6_self_consistency_solve"


# =============================================================================
# A3 — make_a3_correctness_fn: LLM is_valid sinyalini kullanır
# =============================================================================
def test_a3_correctness_fn_is_valid_true():
    fn = make_a3_correctness_fn()
    assert fn({"is_valid": True, "execution_result": "wrong 999"}, _problem()) is True


def test_a3_correctness_fn_is_valid_false():
    fn = make_a3_correctness_fn()
    # Sayısal olarak doğru ama LLM validator False dedi
    assert fn({"is_valid": False, "execution_result": "42.0"}, _problem()) is False


def test_a3_correctness_fn_none_state_returns_false():
    assert make_a3_correctness_fn()(None, _problem()) is False


def test_a3_correctness_fn_missing_is_valid_returns_false():
    assert make_a3_correctness_fn()({}, _problem()) is False


# =============================================================================
# A3 — run_one entegrasyonu: correctness_fn override çalışıyor mu?
# =============================================================================
def test_a3_run_one_overrides_numerical_match_to_false():
    """correctness_fn False döndürdüğünde, 42.0 içeren execution_result'a
    rağmen metrics.numerical_match=False olmalı."""
    problem = _problem(gt_value=42.0)
    state = {
        "execution_result": "The answer is 42.0",
        "is_valid": False,
        "retry_count": 0,
        "metrics": {},
    }
    fake_solve = lambda prompt, data_path=None, **kw: state  # noqa: E731
    rec = run_one(problem, solve_fn=fake_solve, correctness_fn=make_a3_correctness_fn())

    assert rec.success
    assert rec.metrics is not None
    # Deterministik olarak True olurdu ama A3 is_valid=False döndürdü
    assert not rec.metrics.numerical_match


def test_a3_run_one_override_to_true_when_llm_says_valid():
    """is_valid=True, numerik olarak yanlış → A3 ile numerical_match=True."""
    problem = _problem(gt_value=42.0)
    state = {
        "execution_result": "completely wrong result: 999",
        "is_valid": True,  # LLM validator True dedi
        "retry_count": 0,
        "metrics": {},
    }
    fake_solve = lambda prompt, data_path=None, **kw: state  # noqa: E731
    rec = run_one(problem, solve_fn=fake_solve, correctness_fn=make_a3_correctness_fn())

    assert rec.metrics is not None
    assert rec.metrics.numerical_match  # LLM True dedi, deterministic yolu atladık


def test_a3_run_one_default_uses_deterministic():
    """correctness_fn=None (default) → deterministik numerical_match aktif."""
    problem = _problem(gt_value=42.0)
    state = {
        "execution_result": "The optimum is 42.0",
        "is_valid": False,
        "retry_count": 0,
        "metrics": {},
    }
    fake_solve = lambda prompt, data_path=None, **kw: state  # noqa: E731
    rec = run_one(problem, solve_fn=fake_solve, correctness_fn=None)
    assert rec.metrics is not None
    # Deterministik: "42.0" eşleşiyor → True (is_valid=False'a bakılmaz)
    assert rec.metrics.numerical_match


# =============================================================================
# iesolver state.py — ablation flag'leri empty_state'e geçer
# =============================================================================
def test_empty_state_ablation_defaults():
    from iesolver.state import empty_state

    s = empty_state("test prompt")
    assert s.get("enable_refiner") is True
    assert s.get("enable_validator_retry") is True
    assert s.get("fast_only") is False


def test_empty_state_ablation_flags_propagate():
    from iesolver.state import empty_state

    s = empty_state(
        "test prompt",
        enable_refiner=False,
        enable_validator_retry=False,
        fast_only=True,
    )
    assert s["enable_refiner"] is False
    assert s["enable_validator_retry"] is False
    assert s["fast_only"] is True


# =============================================================================
# graph.py — A1: _route_after_requirement
# =============================================================================
def test_a1_requirement_complete_no_refiner_goes_to_route():
    from iesolver.graph import _route_after_requirement

    assert _route_after_requirement({"is_complete": True, "enable_refiner": False}) == "route"


def test_a1_requirement_complete_with_refiner_goes_to_refine():
    from iesolver.graph import _route_after_requirement

    assert _route_after_requirement({"is_complete": True, "enable_refiner": True}) == "refine"


def test_a1_requirement_incomplete_goes_to_clarify_regardless_of_flag():
    from iesolver.graph import _route_after_requirement

    assert _route_after_requirement({"is_complete": False, "enable_refiner": False}) == "clarify"
    assert _route_after_requirement({"is_complete": False, "enable_refiner": True}) == "clarify"


# =============================================================================
# graph.py — A1: _route_after_clarify
# =============================================================================
def test_a1_clarify_auto_mode_no_refiner_goes_to_route():
    from iesolver.graph import _route_after_clarify

    assert _route_after_clarify({"auto_mode": True, "enable_refiner": False}) == "route"


def test_a1_clarify_auto_mode_with_refiner_goes_to_refine():
    from iesolver.graph import _route_after_clarify

    assert _route_after_clarify({"auto_mode": True, "enable_refiner": True}) == "refine"


def test_a1_clarify_interactive_always_loops_back():
    from iesolver.graph import _route_after_clarify

    # Interactive mode: refiner flag fark etmez, requirement'a döner
    assert _route_after_clarify({"auto_mode": False, "enable_refiner": False}) == "requirement"
    assert _route_after_clarify({"auto_mode": False, "enable_refiner": True}) == "requirement"


# =============================================================================
# graph.py — A2: _route_after_validate
# =============================================================================
def test_a2_no_retry_invalid_goes_to_report():
    from iesolver.graph import _route_after_validate

    state = {"is_valid": False, "retry_count": 0, "enable_validator_retry": False}
    assert _route_after_validate(state) == "report"


def test_a2_retry_enabled_invalid_goes_to_code_branch():
    from iesolver.graph import _route_after_validate

    state = {"is_valid": False, "retry_count": 1, "enable_validator_retry": True}
    assert _route_after_validate(state) == "code_branch"


def test_a2_retry_exhausted_goes_to_report():
    from iesolver.graph import _route_after_validate

    # MAX_RETRIES = 3; retry_count >= 3 → limit aşıldı
    state = {"is_valid": False, "retry_count": 3, "enable_validator_retry": True}
    assert _route_after_validate(state) == "report"


def test_a2_valid_goes_to_sensitivity_regardless_of_flag():
    from iesolver.graph import _route_after_validate

    for retry_flag in (True, False):
        state = {"is_valid": True, "retry_count": 0, "enable_validator_retry": retry_flag}
        assert _route_after_validate(state) == "sensitivity"


# =============================================================================
# lm.py — call_with_configured_lm: A4 fast_only yönlendirmesi
# =============================================================================
def test_call_with_configured_lm_fast_only_delegates_to_fast(monkeypatch):
    import iesolver.lm as lm_mod

    calls: list[str] = []
    monkeypatch.setattr(lm_mod, "call_with_fast_lm", lambda m, **kw: calls.append("fast") or "fast")
    monkeypatch.setattr(lm_mod, "call_with_reasoning_lm", lambda m, **kw: calls.append("rsn") or "rsn")

    result = lm_mod.call_with_configured_lm(object(), fast_only=True)
    assert result == "fast"
    assert calls == ["fast"]


def test_call_with_configured_lm_default_delegates_to_reasoning(monkeypatch):
    import iesolver.lm as lm_mod

    calls: list[str] = []
    monkeypatch.setattr(lm_mod, "call_with_fast_lm", lambda m, **kw: calls.append("fast") or "fast")
    monkeypatch.setattr(lm_mod, "call_with_reasoning_lm", lambda m, **kw: calls.append("rsn") or "rsn")

    result = lm_mod.call_with_configured_lm(object(), fast_only=False)
    assert result == "rsn"
    assert calls == ["rsn"]


# =============================================================================
# A5 — compiled_path bulunamazsa FileNotFoundError
# =============================================================================
def test_a5_raises_file_not_found_for_missing_path(tmp_path):
    missing = tmp_path / "compiled_nonexistent.json"
    with pytest.raises(FileNotFoundError, match="compiled signatures not found"):
        make_a5_solve(missing)


def test_a5_returns_callable_for_existing_path(tmp_path):
    """Dosya varsa FileNotFoundError fırlatılmaz; solve_fn döner."""
    compiled = tmp_path / "compiled.json"
    compiled.write_text('{"dummy": true}')   # içerik geçersiz ama dosya var
    fn = make_a5_solve(compiled)
    assert callable(fn)
    assert fn.__name__ == "a5_optimized_solve"
