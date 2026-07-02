"""
iesolver.nodes.refine — Phase 2 (Prompt Refiner).

Eski ``phase_2_refiner.py``'nin LangGraph karşılığı.

DSPy modülü neden ChainOfThought?
    Eski ``phase_2_refiner.py``'deki gerekçeyi koruyoruz:
    "Sınıflandırma işlemlerinde LLM'in kısa bir akıl yürütme yapması,
    doğru IE kategorisini seçmesi için elzemdir."
"""

from __future__ import annotations

import dspy

from iesolver.lm import call_with_fast_lm
from iesolver.observability.metrics import instrument
from iesolver.signatures import PromptRefinerSignature
from iesolver.state import SolverState

_refiner = dspy.ChainOfThought(PromptRefinerSignature)


@instrument("refine")
def refine_node(state: SolverState) -> SolverState:
    """Distill the prompt and classify the IE problem type.

    Reads
    -----
    explicit_goal, constraints

    Writes
    ------
    essential_prompt, strict_constraints, problem_type
    """
    goal = state.get("explicit_goal", "") or ""
    # constraints artık list[str] (DESIGN_REVIEW §3.4). PromptRefiner
    # InputField'ı str beklediği için bulleted metne çeviriyoruz.
    constraints_list = state.get("constraints", []) or []
    initial_constraints = "\n".join(f"- {c}" for c in constraints_list)

    result = call_with_fast_lm(
        _refiner,
        goal=goal,
        initial_constraints=initial_constraints,
    )

    return {
        "essential_prompt": result.essential_prompt,
        "strict_constraints": result.strict_constraints,
        "problem_type": result.problem_type,
    }
