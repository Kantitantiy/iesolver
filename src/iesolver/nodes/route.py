"""
iesolver.nodes.route — Phase 3 (Strategy Router) — Bifurcation Logic.

Eski ``phase_3_router.py``'nin LangGraph karşılığı.

DSPy modülü neden ChainOfThought?
    Eski ``phase_3_router.py``'deki gerekçeyi koruyoruz:
    "LLM önce rationale (gerekçe) yazacak, sonra execution_path seçecek."
    Bu, makalenin observability argümanına da hizmet eder.

Tipli DSPy 3.x çıktısı (DESIGN_REVIEW §3.4):
    execution_path artık ``Literal["CODE","NO_CODE"]``; DSPy adapter binary
    kararı garanti eder. Eski normalize heuristic'i ("code_based",
    "CODE_REQUIRED" gibi varyantları CODE'a çekme) kaldırıldı.
"""

from __future__ import annotations

import dspy

from iesolver.lm import call_with_fast_lm
from iesolver.observability.metrics import instrument
from iesolver.signatures import StrategyRouterSignature
from iesolver.state import SolverState

_router = dspy.ChainOfThought(StrategyRouterSignature)


@instrument("route")
def route_node(state: SolverState) -> SolverState:
    """Decide CODE vs NO_CODE; record reasoning framework + rationale.

    Reads
    -----
    essential_prompt, problem_type, strict_constraints

    Writes
    ------
    execution_path, reasoning_framework, rationale
    """
    essential = state.get("essential_prompt", "") or ""
    problem_type = state.get("problem_type", "") or ""
    constraints = state.get("strict_constraints", "") or ""

    result = call_with_fast_lm(
        _router,
        essential_prompt=essential,
        problem_type=problem_type,
        strict_constraints=constraints,
    )

    return {
        "execution_path": result.execution_path,
        "reasoning_framework": result.reasoning_framework,
        "rationale": result.rationale,
    }