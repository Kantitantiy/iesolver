"""Phase 4C — Sensitivity Analysis code generator."""

from __future__ import annotations

from typing import Literal

import dspy


class SensitivityCodeSignature(dspy.Signature):
    """
    You are an Operations Research expert generating Python code for sensitivity analysis.

    STRATEGY (in priority order):
    1. DUAL VARIABLES (preferred for LP/MILP):
       Modify the original solution code to also extract and print:
       - Shadow prices (constraint.pi for each PuLP constraint, or dual_values from scipy)
       - Reduced costs (var.duals for each PuLP variable)
       Print them as a clear table after the main solve.

    2. PERTURBATION FALLBACK (when solver duals are unavailable, e.g. heuristics):
       Write code that re-solves the problem with ±5% and ±10% changes to the key
       numeric parameters (demand, cost, capacity) and prints the impact on the
       objective value as a table with columns: Parameter, Base, -10%, -5%, +5%, +10%.

    OUTPUT FORMAT (must be parseable as a table):
    Print results with clear labels so downstream code can parse them.
    Include the original objective value as a reference row.
    Do NOT generate plots — text output only.
    """
    # DESIGN_REVIEW §3.6: dual'lar önce; perturbasyon sadece fallback.

    essential_prompt: str = dspy.InputField(desc="The core IE problem statement.")
    final_code: str = dspy.InputField(desc="The original working Python solution code.")
    execution_result: str = dspy.InputField(desc="Stdout of the original code execution.")

    sensitivity_code: str = dspy.OutputField(
        desc="Complete, self-contained Python code that performs sensitivity analysis and prints a formatted table."
    )
    analysis_type: Literal["dual", "perturbation"] = dspy.OutputField(
        desc="'dual' if extracting shadow prices/reduced costs, 'perturbation' if varying parameters."
    )