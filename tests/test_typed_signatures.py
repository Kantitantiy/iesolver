"""
Tipli DSPy 3.x signature testleri (DESIGN_REVIEW §3.4).

Signature'ın output field annotation'larının doğru tiplerde tanımlandığını
doğrular. Bu testler LLM çağırmaz; yalnızca declarative kontratı sınar —
sanitization'ı yeniden eklemek isteyen bir regresyon burada yakalanır.
"""

from __future__ import annotations

from typing import Literal, get_args, get_origin

from iesolver.signatures import (
    RequirementAnalystSignature,
    ResultValidatorSignature,
    StrategyRouterSignature,
)


# =============================================================================
# RequirementAnalystSignature
# =============================================================================
def test_requirement_analyst_is_complete_is_bool():
    fields = RequirementAnalystSignature.output_fields
    assert fields["is_complete"].annotation is bool


def test_requirement_analyst_list_outputs():
    fields = RequirementAnalystSignature.output_fields
    assert get_origin(fields["missing_items"].annotation) is list
    assert get_args(fields["missing_items"].annotation) == (str,)
    assert get_origin(fields["constraints"].annotation) is list
    assert get_args(fields["constraints"].annotation) == (str,)


def test_requirement_analyst_keeps_string_outputs():
    fields = RequirementAnalystSignature.output_fields
    assert fields["explicit_goal"].annotation is str
    assert fields["output_spec"].annotation is str


# =============================================================================
# StrategyRouterSignature
# =============================================================================
def test_strategy_router_execution_path_is_literal():
    fields = StrategyRouterSignature.output_fields
    ann = fields["execution_path"].annotation
    assert get_origin(ann) is Literal.__class__ or ann == Literal["CODE", "NO_CODE"]
    assert set(get_args(ann)) == {"CODE", "NO_CODE"}


# =============================================================================
# ResultValidatorSignature
# =============================================================================
def test_result_validator_is_valid_is_bool():
    fields = ResultValidatorSignature.output_fields
    assert fields["is_valid"].annotation is bool


def test_result_validator_confidence_score_is_int():
    fields = ResultValidatorSignature.output_fields
    assert fields["confidence_score"].annotation is int