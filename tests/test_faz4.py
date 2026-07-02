"""
Faz 4 birim testleri — SensitivityAnalysis + ArtifactGenerator.

LLM ÇAĞRISI YAPMAZ. sandbox ve LM yardımcıları patch edilir.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from iesolver.graph import _route_after_validate
from iesolver.nodes.artifacts import artifacts_node
from iesolver.nodes.sensitivity import sensitivity_node
from iesolver.observability.metrics import merge_metrics
from iesolver.sandbox.runner import RunResult
from iesolver.state import SolverState


# =============================================================================
# _route_after_validate — Faz 4 üç-yönlü dallanma
# =============================================================================
def test_route_valid_goes_to_sensitivity():
    state: SolverState = {"is_valid": True, "retry_count": 0}
    assert _route_after_validate(state) == "sensitivity"


def test_route_invalid_with_retry_goes_to_code_branch():
    state: SolverState = {"is_valid": False, "retry_count": 1}
    assert _route_after_validate(state) == "code_branch"


def test_route_invalid_max_retry_goes_to_report():
    # MAX_RETRIES = 3; retry_count = 3 → report (atla)
    state: SolverState = {"is_valid": False, "retry_count": 3}
    assert _route_after_validate(state) == "report"


# =============================================================================
# sensitivity_node
# =============================================================================
DUMMY_SENSITIVITY = "[analysis_type: dual]\nShadow price demand: 0.0141\nReduced cost Q: 0.0"


def _make_sens_prediction(code: str, atype: str = "dual") -> MagicMock:
    pred = MagicMock()
    pred.sensitivity_code = code
    pred.analysis_type = atype
    return pred


def test_sensitivity_node_success(tmp_path):
    state: SolverState = {
        "essential_prompt": "EOQ problem",
        "final_code": "print('Q* = 707.1')",
        "execution_result": "Q* = 707.1",
    }
    pred = _make_sens_prediction("print('Shadow price: 0.0141')", "dual")
    run_ok = RunResult(success=True, stdout="Shadow price: 0.0141", stderr="")

    with (
        patch("iesolver.nodes.sensitivity.call_with_configured_lm", return_value=pred),
        patch("iesolver.nodes.sensitivity.run_code", return_value=run_ok),
    ):
        out = sensitivity_node(state)

    assert "sensitivity_results" in out
    assert "dual" in out["sensitivity_results"]
    assert "Shadow price" in out["sensitivity_results"]
    assert "metrics" in out
    assert "sensitivity" in out["metrics"]


def test_sensitivity_node_sandbox_failure():
    state: SolverState = {
        "essential_prompt": "LP problem",
        "final_code": "...",
        "execution_result": "...",
    }
    pred = _make_sens_prediction("broken code()", "perturbation")
    run_fail = RunResult(success=False, stdout="", stderr="SyntaxError: bad syntax")

    with (
        patch("iesolver.nodes.sensitivity.call_with_configured_lm", return_value=pred),
        patch("iesolver.nodes.sensitivity.run_code", return_value=run_fail),
    ):
        out = sensitivity_node(state)

    assert out["sensitivity_results"].startswith("[sensitivity_analysis_failed]")
    assert "metrics" in out


# =============================================================================
# artifacts_node
# =============================================================================
def _make_chart_prediction(code: str) -> MagicMock:
    pred = MagicMock()
    pred.chart_code = code
    return pred


def test_artifacts_node_skips_failed_sensitivity():
    state: SolverState = {
        "sensitivity_results": "[sensitivity_analysis_failed]\nError: SyntaxError",
    }
    out = artifacts_node(state)
    assert out["figures"] == []
    assert "metrics" in out


def test_artifacts_node_skips_empty_sensitivity():
    out = artifacts_node({})
    assert out["figures"] == []


def test_artifacts_node_success(tmp_path):
    fake_png = tmp_path / "tornado_chart.png"
    fake_png.write_bytes(b"\x89PNG")

    state: SolverState = {
        "sensitivity_results": "[analysis_type: dual]\nShadow price demand: 0.014",
        "explicit_goal": "Minimize EOQ cost",
    }
    pred = _make_chart_prediction("import pathlib; pathlib.Path(artifact_path).touch()")
    run_ok = RunResult(success=True, stdout="", stderr="")

    with (
        patch("iesolver.nodes.artifacts.call_with_fast_lm", return_value=pred),
        patch("iesolver.nodes.artifacts.run_code", return_value=run_ok),
        patch(
            "iesolver.nodes.artifacts.settings.artifacts_dir",
            new=tmp_path,
        ),
    ):
        out = artifacts_node(state)

    # fake_png exists → figures list non-empty
    assert out["figures"] == [tmp_path / "tornado_chart.png"]
    assert "metrics" in out


def test_artifacts_node_sandbox_failure():
    state: SolverState = {
        "sensitivity_results": "[analysis_type: perturbation]\nParam: D, -10%: 5.0",
    }
    pred = _make_chart_prediction("raise RuntimeError('fail')")
    run_fail = RunResult(success=False, stdout="", stderr="RuntimeError")

    with (
        patch("iesolver.nodes.artifacts.call_with_fast_lm", return_value=pred),
        patch("iesolver.nodes.artifacts.run_code", return_value=run_fail),
    ):
        out = artifacts_node(state)

    assert out["figures"] == []


# =============================================================================
# figures reducer (Faz 4'te eklendi: operator.add)
# =============================================================================
def test_figures_reducer_accumulates():
    """LangGraph'ın operator.add reducer'ının list concat davranışını doğrula."""
    import operator
    a = [Path("fig1.png")]
    b = [Path("fig2.png")]
    assert operator.add(a, b) == [Path("fig1.png"), Path("fig2.png")]


def test_figures_reducer_with_empty():
    import operator
    assert operator.add([], [Path("x.png")]) == [Path("x.png")]
    assert operator.add([Path("x.png")], []) == [Path("x.png")]