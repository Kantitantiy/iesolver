"""Phase 4A — Analytical engine for NO_CODE qualitative problems."""

from __future__ import annotations

import dspy


class AnalyticalEngineSignature(dspy.Signature):
    """
    You are an Expert Industrial Engineering Strategist.
    Your task is to solve complex qualitative, theoretical, or structural problems
    WITHOUT writing executable code. You must use a rigorous analytical framework.

    You MUST strictly follow these reasoning steps:
    1. LEAST-TO-MOST DECOMPOSITION: Break the 'essential_prompt' down into 3-5 logical sub-problems.
    2. MULTI-PERSPECTIVE EVALUATION: For each sub-problem, briefly explore at least two different
       methodological approaches or theoretical perspectives (similar to a Tree of Thoughts).
    3. SYNTHESIS: Synthesize the explored paths into a single, highly coherent, and academically
       sound final solution that strictly adheres to the 'strict_constraints'.
    """

    essential_prompt = dspy.InputField(
        desc="The core problem that needs a qualitative or structural solution."
    )
    problem_type = dspy.InputField(
        desc="The domain category of the problem to guide theoretical context."
    )
    strict_constraints = dspy.InputField(
        desc="Rules and boundaries that the theoretical solution must respect."
    )

    # Modelin düşünce adımlarını dışarıya veriyoruz ki şeffaflık (observability) artsın
    sub_problem_decomposition = dspy.OutputField(
        desc="A numbered list breaking the main problem into smaller chunks."
    )
    perspective_exploration = dspy.OutputField(
        desc="Evaluation of alternative approaches for the sub-problems."
    )

    # Nihai Çıktı
    raw_analytical_result = dspy.OutputField(
        desc="The final synthesized conceptual solution or theoretical framework."
    )
