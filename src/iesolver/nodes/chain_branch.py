"""
iesolver.nodes.chain_branch — Phase 4A (Analytical Engine, NO_CODE path).

Eski ``phase_4a_analytic.py``'nin LangGraph karşılığı. Least-to-most
decomposition + multi-perspective evaluation + synthesis.

DSPy modülü neden ChainOfThought?
    Eski ``phase_4a_analytic.py``'deki gerekçeyi koruyoruz:
    "Modelin derinlemesine düşünmesi için CoT kullanıyoruz."
    Faz 4 notu: doğruluğu artırmak için ``dspy.MajorityVote(_engine)``
    ile Self-Consistency uygulanabilir.
"""

from __future__ import annotations

import dspy

from iesolver.lm import call_with_fast_lm
from iesolver.observability.metrics import instrument
from iesolver.signatures import AnalyticalEngineSignature
from iesolver.state import SolverState
from iesolver.text import fenced

_engine = dspy.ChainOfThought(AnalyticalEngineSignature)


@instrument("chain_branch")
def chain_branch_node(state: SolverState) -> SolverState:
    """Synthesize a qualitative IE solution without writing code.

    Reads
    -----
    essential_prompt, problem_type, strict_constraints

    Writes
    ------
    raw_result, solution_path
    """
    essential = state.get("essential_prompt", "") or ""
    problem_type = state.get("problem_type", "") or ""
    constraints = state.get("strict_constraints", "") or ""

    result = call_with_fast_lm(
        _engine,
        essential_prompt=essential,
        problem_type=problem_type,
        strict_constraints=constraints,
    )

    solution_path_log = (
        f"{fenced('DECOMPOSITION', result.sub_problem_decomposition)}\n\n"
        f"{fenced('EXPLORATION', result.perspective_exploration)}"
    )

    return {
        "raw_result": result.raw_analytical_result,
        "solution_path": solution_path_log,
    }
