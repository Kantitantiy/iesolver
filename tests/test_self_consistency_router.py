"""
A6 ablation testleri — StrategyRouter self-consistency (CLAUDE.md Düzeltme #5).

LLM ÇAĞRISI YAPILMAZ — call_with_fast_lm monkeypatch'lenir; dspy.majority
gerçek dspy.Prediction nesneleri üzerinde çalıştırılır.
"""

from __future__ import annotations

import dspy

import iesolver.nodes.route as route_mod
from iesolver.nodes.route import route_node


def _base_state(**overrides):
    state = {
        "essential_prompt": "Minimize transportation cost",
        "problem_type": "Mathematical Optimization",
        "strict_constraints": "supply >= demand",
    }
    state.update(overrides)
    return state


def test_route_node_single_sample_default(monkeypatch):
    """self_consistency_router=False (varsayılan) → tek çağrı, config verilmez, vote_summary yazılmaz."""
    captured_kwargs: dict = {}

    def fake_call(module, **kwargs):
        captured_kwargs.update(kwargs)
        return dspy.Prediction(
            execution_path="CODE", reasoning_framework="ReAct", rationale="single-shot rationale"
        )

    monkeypatch.setattr(route_mod, "call_with_fast_lm", fake_call)

    result = route_node(_base_state(self_consistency_router=False))

    assert result["execution_path"] == "CODE"
    assert result["rationale"] == "single-shot rationale"
    assert "router_vote_summary" not in result
    assert "config" not in captured_kwargs


def test_route_node_defaults_to_single_sample_when_flag_absent(monkeypatch):
    """State'te self_consistency_router hiç yoksa (eski checkpoint'ler) tek örnekleme çalışmalı."""
    def fake_call(module, **kwargs):
        assert "config" not in kwargs
        return dspy.Prediction(execution_path="NO_CODE", reasoning_framework="LeastToMost", rationale="r")

    monkeypatch.setattr(route_mod, "call_with_fast_lm", fake_call)

    result = route_node(_base_state())
    assert result["execution_path"] == "NO_CODE"


def test_route_node_self_consistency_majority_vote(monkeypatch):
    """self_consistency_router=True → n=3 örnekleme + çoğunluk oyu + vote_summary."""
    raw = dspy.Prediction.from_completions(
        [
            {"execution_path": "CODE", "reasoning_framework": "ReAct", "rationale": "r0"},
            {"execution_path": "CODE", "reasoning_framework": "ReAct", "rationale": "r1"},
            {"execution_path": "NO_CODE", "reasoning_framework": "LeastToMost", "rationale": "r2"},
        ]
    )

    def fake_call(module, **kwargs):
        assert kwargs.get("config") == {"n": route_mod._SELF_CONSISTENCY_VOTES}
        return raw

    monkeypatch.setattr(route_mod, "call_with_fast_lm", fake_call)

    result = route_node(_base_state(self_consistency_router=True))

    assert result["execution_path"] == "CODE"
    assert result["router_vote_summary"] == "CODE:2/NO_CODE:1"


def test_route_node_self_consistency_unanimous_vote(monkeypatch):
    """3/3 oy birleşirse vote_summary tek girdi olmalı."""
    raw = dspy.Prediction.from_completions(
        [
            {"execution_path": "NO_CODE", "reasoning_framework": "LeastToMost", "rationale": "r0"},
            {"execution_path": "NO_CODE", "reasoning_framework": "LeastToMost", "rationale": "r1"},
            {"execution_path": "NO_CODE", "reasoning_framework": "LeastToMost", "rationale": "r2"},
        ]
    )
    monkeypatch.setattr(route_mod, "call_with_fast_lm", lambda module, **kw: raw)

    result = route_node(_base_state(self_consistency_router=True))

    assert result["execution_path"] == "NO_CODE"
    assert result["router_vote_summary"] == "NO_CODE:3"


# =============================================================================
# state.py — A6 flag propagation (aynı desen ie-eval/tests/test_ablations.py'de A1-A4 için var)
# =============================================================================
def test_empty_state_self_consistency_router_default():
    from iesolver.state import empty_state

    s = empty_state("test prompt")
    assert s.get("self_consistency_router") is False


def test_empty_state_self_consistency_router_propagates():
    from iesolver.state import empty_state

    s = empty_state("test prompt", self_consistency_router=True)
    assert s["self_consistency_router"] is True
