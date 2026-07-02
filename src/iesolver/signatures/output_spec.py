"""Phase 4B.3 — Output specification for generated code."""

from __future__ import annotations

import dspy


class OutputSpecEngineerSignature(dspy.Signature):
    """
    You are a Code Output Architect.
    Define EXACTLY what the generated Python code must output.
    The execution environment will capture 'stdout' (print statements).
    Specify exactly what variables, metrics, or dataframes must be printed
    so the final report generator can easily parse the results.
    """

    essential_prompt = dspy.InputField()
    library_specific_constraints = dspy.InputField(
        desc="First, add import warnings; warnings.filterwarnings('ignore') to suppress any sklearn terminal warnings so they do not corrupt the JSON output. "
             "Also include your other constraints here (Clustering Configuration, Hard Feasibility Constraint, etc.)."
    )

    code_output_spec = dspy.OutputField(
        desc="Strict instructions for the code's output format (e.g., 'Print the optimal objective value and the final variable dictionary')."
    )
