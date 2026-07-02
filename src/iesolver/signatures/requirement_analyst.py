"""Phase 1 — G-O-C requirement extraction + completeness gating."""

from __future__ import annotations

import dspy


class RequirementAnalystSignature(dspy.Signature):
    """
You are a meticulous Industrial Engineering Requirement Analyst.
Analyze the cleaned prompt and the provided data summary.
Apply the G-O-C (Goal, Output, Constraint) framework to extract exact specifications.

CRITICAL CONTEXT ON DATA SUMMARY:
The provided data summary is ONLY a schema preview (e.g., a sample of the first 5 rows) designed to demonstrate the data structure. It is NOT the complete dataset.
- DO NOT flag 'is_complete' as False based on row counts (e.g., seeing only 5 rows when the prompt requires 15 locations).
- Assume the full dataset contains the required number of records AS LONG AS the necessary data fields (columns) exist in the preview.

CRITICAL INSTRUCTION:
If essential structural information required to define a solvable mathematical or logical model is missing (e.g., objective function direction, specific capacity constraints, or required data fields/columns are entirely absent from the schema preview), you must set 'is_complete' to False and list the exact questions needed to proceed.
    """
    # G-O-C framework'ünü DSPy'a burada öğretiyoruz. DSPy 3.x tipli output'ları
    # sayesinde is_complete artık native bool; missing_items ve constraints
    # list[str] olarak dönerek ölçüm/test için doğrudan iterasyonlanabilir
    # (DESIGN_REVIEW §3.4).

    cleaned_prompt: str = dspy.InputField(desc="The standardized task description.")
    data_summary: str = dspy.InputField(
        desc="Statistical summary and schema of the attached data. Can be empty if no data is provided."
    )

    is_complete: bool = dspy.OutputField(
        desc="Is the information sufficient to build a formal model?",
    )
    missing_items: list[str] = dspy.OutputField(
        desc="If is_complete is False, list specific, actionable questions for the user, one per element. If True, return an empty list."
    )
    explicit_goal: str = dspy.OutputField(
        desc="The primary objective of the problem (e.g., 'Minimize transportation cost')."
    )
    constraints: list[str] = dspy.OutputField(
        desc="Strict rules, boundaries, and logical conditions extracted from the prompt, one per element."
    )
    output_spec: str = dspy.OutputField(
        desc="The exact format expected by the user (e.g., 'A pandas DataFrame with optimal assignments')."
    )