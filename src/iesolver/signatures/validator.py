"""Phase 4B.6 — Result validator / Semantic boundary verification."""

from __future__ import annotations

import dspy


class ResultValidatorSignature(dspy.Signature):
    """
    You are a strict Industrial Engineering QA (Quality Assurance) Auditor.
    Your task is to review the numerical and logical outputs of a successfully executed Python script.

    You must evaluate the 'execution_result' against the 'strict_constraints' and general engineering logic.
    Check for critical fallacies such as:
    - Negative probabilities, distances, or physical quantities.
    - Extreme outliers (e.g., infinite costs, zero production in a maximization problem).
    - Violation of any hard boundary specified in the constraints.

    Based on your analysis, provide a 'confidence_score' (0-100) indicating how robust
    and mathematically sound the result is.
    """
    # NOT: Makalede bu aşama "Semantic and Logical Boundary Verification" olarak geçebilir.
    # is_valid ve confidence_score DSPy 3.x tipli output'ları sayesinde native bool/int;
    # eski str→bool / str→int sanitization'ı validate_node'dan kaldırıldı (DESIGN_REVIEW §3.4).

    essential_prompt: str = dspy.InputField(desc="The core analytical problem.")
    strict_constraints: str = dspy.InputField(desc="The hard rules the solution was supposed to follow.")
    execution_result: str = dspy.InputField(desc="The raw printed output from the Python code execution.")

    is_valid: bool = dspy.OutputField(
        desc="Does the result make logical and mathematical sense?"
    )
    confidence_score: int = dspy.OutputField(
        desc="Integer between 0 and 100 indicating robustness of the result."
    )
    validation_notes: str = dspy.OutputField(
        desc="Brief explanation of the validation outcome, highlighting any anomalies or variance issues."
    )