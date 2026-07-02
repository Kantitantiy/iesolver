"""
Auto-mode birim testleri (DESIGN_REVIEW §3.1).

Bu testler LLM ÇAĞRISI YAPMAZ — yalnızca clarify_node ve graph edge'lerinin
auto_mode davranışını doğrular. E2E doğrulama için ayrı bir slow test gerekir
(NL4Opt/IndustryOR harness Faz 4.5'te gelecek).
"""

from __future__ import annotations

import pytest

from iesolver.graph import _route_after_clarify, _route_after_requirement
from iesolver.nodes.clarify import clarify_node
from iesolver.state import SolverState, empty_state


# =============================================================================
# empty_state
# =============================================================================
def test_empty_state_defaults_auto_mode_false():
    state = empty_state(raw_prompt="hello")
    assert state["auto_mode"] is False


def test_empty_state_propagates_auto_mode_flag():
    state = empty_state(raw_prompt="hello", auto_mode=True)
    assert state["auto_mode"] is True


# =============================================================================
# clarify_node — auto_mode dalı
# =============================================================================
def test_clarify_auto_mode_skips_interrupt_and_completes():
    """auto_mode=True: no interrupt, is_complete=True, assumption logged."""
    state: SolverState = {
        "auto_mode": True,
        "missing_items": ["budget cap?", "demand forecast horizon?"],
        "explicit_goal": "minimize cost",
    }
    result = clarify_node(state)

    assert result["is_complete"] is True
    assert "user_clarification" in result
    assert "AUTO_MODE" in result["user_clarification"]
    assert "budget cap?" in result["user_clarification"]

    log = result["auto_assumptions_log"]
    assert isinstance(log, list) and len(log) == 1
    assert "AUTO_MODE" in log[0]


def test_clarify_auto_mode_appends_to_existing_log():
    """Repeated clarify calls (e.g. after another requirement pass) accumulate."""
    state: SolverState = {
        "auto_mode": True,
        "missing_items": ["second gap"],
        "auto_assumptions_log": ["AUTO_MODE: earlier assumption"],
    }
    result = clarify_node(state)

    assert len(result["auto_assumptions_log"]) == 2
    assert result["auto_assumptions_log"][0] == "AUTO_MODE: earlier assumption"
    assert "second gap" in result["auto_assumptions_log"][1]


def test_clarify_auto_mode_with_empty_missing_items():
    """Missing_items may be empty; assumption text should still be well-formed."""
    state: SolverState = {"auto_mode": True}
    result = clarify_node(state)
    assert result["is_complete"] is True
    assert "AUTO_MODE" in result["user_clarification"]
    assert "<none reported>" in result["user_clarification"]


# =============================================================================
# Conditional edge predicates
# =============================================================================
def test_route_after_clarify_auto_mode_goes_to_refine():
    assert _route_after_clarify({"auto_mode": True}) == "refine"


def test_route_after_clarify_interactive_goes_to_requirement():
    assert _route_after_clarify({"auto_mode": False}) == "requirement"
    assert _route_after_clarify({}) == "requirement"   # default = interactive


def test_route_after_requirement_incomplete_goes_to_clarify():
    """auto_mode does NOT change routing here — clarify_node itself branches."""
    assert _route_after_requirement({"is_complete": False, "auto_mode": True}) == "clarify"
    assert _route_after_requirement({"is_complete": True, "auto_mode": True}) == "refine"