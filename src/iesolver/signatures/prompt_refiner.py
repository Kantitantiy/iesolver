"""Phase 2 — Prompt distillation + IE problem-type classification."""

from __future__ import annotations

import dspy


class PromptRefinerSignature(dspy.Signature):
    """
    You are an expert Prompt Optimizer for an Industrial Engineering Decision Support System.
    Your task is to strip away conversational noise, irrelevant details, and ambiguity from the goal.
    You must convert the inputs into a highly precise, machine-actionable set of instructions.

    Additionally, you must classify the 'problem_type' into one of the following quantitative categories:
    - Mathematical Optimization (e.g., LP, MILP, routing, scheduling)
    - Machine Learning / Data Analytics (e.g., regression, clustering, predictive maintenance)
    - System Simulation (e.g., discrete-event simulation, queuing theory)
    - Statistical / Decision Analysis (e.g., hypothesis testing, AHP, decision trees)
    - Theoretical / Managerial Report (e.g., literature review, conceptual framework)
    """
    # NOT: Yukarıdaki kategoriler, sistemin kod yazıp yazmayacağına veya
    # PuLP, scikit-learn, Arena gibi hangi mantıkla ilerleyeceğine karar vermesini sağlayacak.

    goal = dspy.InputField(desc="The initial goal extracted from the user's prompt.")
    initial_constraints = dspy.InputField(desc="The preliminary list of constraints.")

    essential_prompt = dspy.OutputField(
        desc="A distilled, concise, and highly technical instruction prompt for the LLM execution engine."
    )
    strict_constraints = dspy.OutputField(
        desc="A finalized, formalized list of constraints that algorithms must strictly adhere to."
    )
    problem_type = dspy.OutputField(
        desc="The exact classification category of the problem from the allowed list."
    )
