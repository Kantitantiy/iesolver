"""
iesolver.nodes.report — Phase 5 (Final Report Generator).

Eski ``phase_5_reporter.py``'nin LangGraph karşılığı. 3 katmanlı çıktı:
Technical / Executive / Action.

DSPy modülü neden ChainOfThought?
    Eski ``phase_5_reporter.py``'deki gerekçeyi koruyoruz:
    "Modelin verileri sentezleyip 3 farklı formata dönüştürürken
    kaliteyi artırması için CoT kullanıyoruz."

Path-aware context assembly:
    Eski ``_resume_pipeline`` içinde validated_results manuel olarak
    iki yolda farklı kuruluyordu:
      - NO_CODE: raw_result alone
      - CODE:    execution_result + validation notes
    Aynı mantığı state'i okuyarak yeniden inşa ediyoruz.
"""

from __future__ import annotations

import dspy

from iesolver.lm import call_with_fast_lm
from iesolver.observability.metrics import instrument
from iesolver.signatures import FinalReportSignature
from iesolver.state import SolverState
from iesolver.text import fenced

_reporter = dspy.ChainOfThought(FinalReportSignature)


def _build_solution_context(state: SolverState) -> str:
    """Combine analytical path and/or executed code into one context blob."""
    chunks: list[str] = []
    solution_path = state.get("solution_path", "") or ""
    final_code = state.get("final_code", "") or ""

    if final_code:
        chunks.append(fenced("EXECUTED_PYTHON_CODE", final_code))
    if solution_path:
        chunks.append(fenced("ANALYTICAL_SOLUTION_PATH", solution_path))

    return "\n\n".join(chunks) if chunks else "N/A"


def _build_validated_results(state: SolverState) -> str:
    """Compose the 'validated_results' field expected by FinalReportSignature.

    Path discrimination:
      - CODE     → execution_result + QA notes
      - NO_CODE  → raw_result alone
    """
    path = state.get("execution_path", "")
    if path == "CODE":
        qa_notes = (
            f"Valid: {state.get('is_valid', 'N/A')}\n"
            f"Confidence Score: {state.get('confidence_score', 0)}/100\n"
            f"Notes: {state.get('validation_notes', '(none)')}"
        )
        return (
            f"{fenced('RAW_EXECUTION_RESULTS', state.get('execution_result', '(missing)'))}\n\n"
            f"{fenced('VALIDATION_QA_NOTES', qa_notes)}"
        )

    return state.get("raw_result", "(no result available)") or "(no result)"


@instrument("report")
def report_node(state: SolverState) -> SolverState:
    """Synthesize the 3-layered final report from path-specific state.

    Reads
    -----
    explicit_goal, execution_path, raw_result OR (execution_result +
    is_valid + confidence_score + validation_notes), solution_path,
    final_code

    Writes
    ------
    technical_output, executive_summary, action_directives
    """
    original_goal = state.get("explicit_goal", "") or "(unspecified)"
    solution_context = _build_solution_context(state)
    validated_results = _build_validated_results(state)

    result = call_with_fast_lm(
        _reporter,
        original_goal=original_goal,
        solution_path_or_code=solution_context,
        validated_results=validated_results,
    )

    return {
        "technical_output": result.technical_output,
        "executive_summary": result.executive_summary,
        "action_directives": result.action_directives,
    }
