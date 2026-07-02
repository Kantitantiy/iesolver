"""
Metrics telemetri testleri (DESIGN_REVIEW §3.5).

instrument decorator, merge_metrics reducer ve LM usage kayıt akışını
doğrular. LLM ÇAĞRISI YAPMAZ — record_llm_usage'i doğrudan simüle eder.
"""

from __future__ import annotations

import pytest

from iesolver.observability.metrics import (
    NODE_METRIC_KEYS,
    instrument,
    merge_metrics,
    record_llm_usage,
)


# =============================================================================
# instrument decorator — temel akış
# =============================================================================
def test_instrument_attaches_metrics_slice():
    @instrument("dummy")
    def node(state):
        return {"foo": "bar"}

    out = node({})
    assert out["foo"] == "bar"
    assert "metrics" in out
    assert "dummy" in out["metrics"]

    slice_ = out["metrics"]["dummy"]
    for key in NODE_METRIC_KEYS:
        assert key in slice_, f"metrics slice missing key {key!r}"

    assert slice_["invocations"] == 1
    assert slice_["error_class"] is None
    assert slice_["tokens_in"] == 0
    assert slice_["latency_ms"] >= 0.0


def test_instrument_preserves_existing_metrics_in_partial():
    """If the node itself already returned metrics for another slice, keep them."""
    @instrument("later")
    def node(state):
        return {"metrics": {"earlier": {"latency_ms": 5.0}}}

    out = node({})
    assert "earlier" in out["metrics"]
    assert "later" in out["metrics"]


def test_instrument_records_llm_usage_from_context():
    """record_llm_usage inside the wrapped fn accumulates into the slice."""
    @instrument("with_llm")
    def node(state):
        record_llm_usage([
            {"usage": {"prompt_tokens": 100, "completion_tokens": 50}, "cost": 0.001},
            {"usage": {"prompt_tokens": 20, "completion_tokens": 10}, "cost": 0.0002},
        ])
        return {}

    out = node({})
    slice_ = out["metrics"]["with_llm"]
    assert slice_["tokens_in"] == 120
    assert slice_["tokens_out"] == 60
    assert slice_["cost_usd"] == pytest.approx(0.0012, abs=1e-6)
    assert slice_["llm_calls"] == 2


def test_instrument_captures_error_class_and_reraises():
    @instrument("failing")
    def node(state):
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        node({})


def test_record_llm_usage_outside_scope_is_noop():
    """Direct call to record_llm_usage without an instrument scope must not crash."""
    record_llm_usage([{"usage": {"prompt_tokens": 5}, "cost": 0.0001}])


# =============================================================================
# merge_metrics reducer
# =============================================================================
def test_merge_metrics_adds_new_node():
    merged = merge_metrics(
        {"intake": {"latency_ms": 10, "tokens_in": 5, "invocations": 1}},
        {"refine": {"latency_ms": 20, "tokens_in": 3, "invocations": 1}},
    )
    assert set(merged) == {"intake", "refine"}
    assert merged["refine"]["latency_ms"] == 20


def test_merge_metrics_accumulates_on_reentry():
    """code_branch retry: aynı node ikinci kez giriyor, sayısal alanlar toplansın."""
    first = {
        "code_branch": {
            "latency_ms": 100.0,
            "tokens_in": 200,
            "tokens_out": 80,
            "cost_usd": 0.001,
            "llm_calls": 3,
            "invocations": 1,
            "error_class": None,
        }
    }
    second = {
        "code_branch": {
            "latency_ms": 150.0,
            "tokens_in": 250,
            "tokens_out": 90,
            "cost_usd": 0.0015,
            "llm_calls": 3,
            "invocations": 1,
            "error_class": "SandboxTimeout",
        }
    }
    out = merge_metrics(first, second)
    slice_ = out["code_branch"]
    assert slice_["latency_ms"] == 250.0
    assert slice_["tokens_in"] == 450
    assert slice_["tokens_out"] == 170
    assert slice_["cost_usd"] == pytest.approx(0.0025)
    assert slice_["llm_calls"] == 6
    assert slice_["invocations"] == 2
    assert slice_["error_class"] == "SandboxTimeout"   # son yazan kazanır


def test_merge_metrics_handles_none_inputs():
    assert merge_metrics(None, None) == {}
    assert merge_metrics({"a": {"latency_ms": 1}}, None) == {"a": {"latency_ms": 1}}
    assert merge_metrics(None, {"a": {"latency_ms": 1}}) == {"a": {"latency_ms": 1}}