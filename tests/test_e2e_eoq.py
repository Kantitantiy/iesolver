"""
Faz 3 E2E Smoke Test — EOQ (Economic Order Quantity) problemi.

Bu test gerçek LLM çağrısı yapar (slow marker).
Çalıştırmak için: ``uv run pytest tests/test_e2e_eoq.py -v -m e2e``

Beklenen davranış:
    1. route_node → execution_path = "CODE"
    2. code_branch → PuLP veya basit formül ile EOQ hesaplar
    3. validate_node → is_valid = True
    4. report_node → executive_summary EOQ değerini içerir

EOQ formülü: Q* = sqrt(2 * D * S / H)
    D=10000, S=50, H=2  →  Q* = sqrt(500000) ≈ 707.1
"""

import math
import pytest
from iesolver import solve, is_interrupted


EOQ_PROMPT = (
    "Calculate the Economic Order Quantity (EOQ) for the following parameters: "
    "Annual demand D = 10,000 units, Ordering cost S = $50 per order, "
    "Holding cost H = $2 per unit per year. "
    "Use the EOQ formula Q* = sqrt(2*D*S/H) and provide the exact numerical result."
)

EXPECTED_EOQ = math.sqrt(2 * 10_000 * 50 / 2)   # ≈ 707.1


@pytest.mark.slow
@pytest.mark.e2e
def test_eoq_no_interrupt():
    """Solve EOQ end-to-end; result should not require clarification."""
    result = solve(EOQ_PROMPT, thread_id="test-eoq-faz3")
    assert not is_interrupted(result), (
        f"Unexpected interrupt: {result.get('__interrupt__')}"
    )


@pytest.mark.slow
@pytest.mark.e2e
def test_eoq_execution_path_is_code():
    """EOQ should route to CODE branch (mathematical calculation)."""
    result = solve(EOQ_PROMPT, thread_id="test-eoq-path")
    assert result.get("execution_path") == "CODE", (
        f"Expected CODE, got: {result.get('execution_path')}"
    )


@pytest.mark.slow
@pytest.mark.e2e
def test_eoq_numerical_result():
    """EOQ result should be within 5% of the analytical value."""
    result = solve(EOQ_PROMPT, thread_id="test-eoq-num")

    exec_result = result.get("execution_result", "")
    assert exec_result, "execution_result is empty"

    # Sayısal değeri execution_result içinde ara
    import re
    numbers = re.findall(r"\d+\.?\d*", exec_result)
    floats = [float(n) for n in numbers]

    # 707.1'e yakın bir değer bulunmalı (±%5)
    tolerance = EXPECTED_EOQ * 0.05
    found = any(abs(f - EXPECTED_EOQ) < tolerance for f in floats)
    assert found, (
        f"EOQ ≈ {EXPECTED_EOQ:.1f} beklendi, "
        f"execution_result içindeki sayılar: {floats}\n"
        f"Tam çıktı:\n{exec_result}"
    )


@pytest.mark.slow
@pytest.mark.e2e
def test_eoq_report_has_three_layers():
    """Final report must contain all three output layers."""
    result = solve(EOQ_PROMPT, thread_id="test-eoq-report")
    assert result.get("technical_output"), "technical_output boş"
    assert result.get("executive_summary"), "executive_summary boş"
    assert result.get("action_directives"), "action_directives boş"
