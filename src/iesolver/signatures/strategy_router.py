"""Phase 3 — Bifurcation Logic: CODE vs NO_CODE execution path."""

from __future__ import annotations

from typing import Literal

import dspy


class StrategyRouterSignature(dspy.Signature):
    """
    You are the Strategic Execution Router for an Industrial Engineering AI Agent.
    Your task is to analyze the refined problem and determine the most effective computational path.

    You must decide between two strictly defined execution paths:
    1. 'CODE': Select this if the problem involves mathematical optimization (e.g., LP, MILP),
       data analytics, simulations requiring dynamic generation, or explicit algorithmic calculations.
    2. 'NO_CODE': Select this if the problem requires qualitative decision-making, literature
       reviews, theoretical frameworks, or logical reasoning without mathematical computation.

    Furthermore, recommend the appropriate reasoning framework:
    - 'ReAct' (Reasoning and Acting) for CODE paths.
    - 'TreeOfThoughts' or 'LeastToMost' for NO_CODE analytical paths.
    """
    # NOT: Bu prompt, makalede "Bifurcation Logic" (Çatallanma Mantığı) olarak geçecektir.
    # LLM'in otonom bir şekilde kendi kullanacağı aracı (Python veya sadece Mantık) seçmesini sağlıyoruz.
    # execution_path artık Literal — DSPy adapter kararı garanti eder, string
    # normalize kodu (route_node içindeki eski heuristic) artık gereksiz (DESIGN_REVIEW §3.4).

    essential_prompt: str = dspy.InputField(desc="The highly technical and distilled instruction prompt.")
    problem_type: str = dspy.InputField(desc="The classified category of the industrial engineering problem.")
    strict_constraints: str = dspy.InputField(desc="The strict rules and boundaries for the problem.")

    execution_path: Literal["CODE", "NO_CODE"] = dspy.OutputField(
        desc="Exactly one of 'CODE' or 'NO_CODE'."
    )
    reasoning_framework: str = dspy.OutputField(
        desc="The recommended framework (e.g., 'ReAct', 'TreeOfThoughts')."
    )
    rationale: str = dspy.OutputField(
        desc="A brief justification for why this execution path and framework were chosen."
    )