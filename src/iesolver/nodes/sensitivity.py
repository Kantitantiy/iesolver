"""
iesolver.nodes.sensitivity — Faz 4C (Sensitivity Analysis).

DESIGN_REVIEW §3.6 gereği dual-öncelikli tasarım:
    LP/MILP solver'larından (PuLP, scipy) shadow price ve reduced cost
    bedava gelir; kaba kuvvet ±%5/±%10 perturbasyonu yalnızca dual bilgisi
    olmayan durumlar için fallback. Bu tercih hem OR açısından doğru
    hem de hakem beklentisini karşılar.

Akış:
    1. LLM (reasoning_lm): final_code + execution_result'a bakarak
       dual-extraction kodu veya perturbation kodu üretir.
    2. Sandbox: üretilen kodu çalıştırır → stdout = duyarlılık tablosu.
    3. Başarısız sandbox: sensitivity_results'a hata notu → artifacts_node
       boş döner ve rapor yine de yazılır.

DSPy modülü neden ChainOfThought?
    Hangi stratejinin (dual/perturbation) uygun olduğuna karar verirken
    kısa bir akıl yürütme payı gerekli; Predict yeterli olmayabilir.
"""

from __future__ import annotations

import dspy

from iesolver.lm import call_with_configured_lm
from iesolver.observability.metrics import instrument
from iesolver.sandbox.runner import run_code
from iesolver.signatures import SensitivityCodeSignature
from iesolver.state import SolverState

_sens_gen = dspy.ChainOfThought(SensitivityCodeSignature)


@instrument("sensitivity")
def sensitivity_node(state: SolverState) -> SolverState:
    """Generate and run sensitivity analysis code; store formatted results.

    Reads
    -----
    essential_prompt, final_code, execution_result

    Writes
    ------
    sensitivity_results
    """
    essential = state.get("essential_prompt", "") or ""
    final_code = state.get("final_code", "") or ""
    exec_result = state.get("execution_result", "") or ""

    # 1. LLM: duyarlılık kodu üret (dual veya perturbasyon)
    prediction = call_with_configured_lm(
        _sens_gen,
        fast_only=state.get("fast_only", False),
        essential_prompt=essential,
        final_code=final_code,
        execution_result=exec_result,
    )

    # 2. Sandbox: kodu çalıştır
    run_result = run_code(prediction.sensitivity_code)

    if run_result.success and run_result.stdout.strip():
        sensitivity_results = (
            f"[analysis_type: {prediction.analysis_type}]\n"
            f"{run_result.stdout.strip()}"
        )
    else:
        # Fallback: hata notu — artifacts_node boş döner, rapor yine yazılır
        sensitivity_results = (
            f"[sensitivity_analysis_failed]\n"
            f"Type attempted: {prediction.analysis_type}\n"
            f"Error: {run_result.error_summary or run_result.stderr[:300]}"
        )

    return {"sensitivity_results": sensitivity_results}