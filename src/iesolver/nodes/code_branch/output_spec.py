"""
iesolver.nodes.code_branch.output_spec — Phase 4B.3 (Output Spec Engineer).

Eski ``CodeEnginePlanner.define_output`` adımının LangGraph karşılığı.
Üretilecek kodun tam olarak ne print edeceğini tanımlar; böylece
ReAct motorunun çıktısını FinalReportGenerator parse edebilir.

DSPy modülü neden Predict?
    Eski ``phase_4b_code_engine.py``'daki gerekçeyi koruyoruz:
    "Çıktı formatlamada doğrudan deterministik sonuç istiyoruz."
    Format spesifikasyonu katı ve tekrarlı değil; tek atış yeterli.
"""

from __future__ import annotations

import dspy

from iesolver.lm import call_with_reasoning_lm
from iesolver.signatures import OutputSpecEngineerSignature
from iesolver.state import SolverState

_spec = dspy.Predict(OutputSpecEngineerSignature)


def output_spec_node(state: SolverState) -> SolverState:
    """Define exactly what the generated code must print.

    Reads
    -----
    essential_prompt, library_specific_constraints

    Writes
    ------
    code_output_spec
    """
    result = call_with_reasoning_lm(
        _spec,
        essential_prompt=state.get("essential_prompt", "") or "",
        library_specific_constraints=state.get("library_specific_constraints", "") or "",
    )

    return {
        "code_output_spec": result.code_output_spec,
    }
