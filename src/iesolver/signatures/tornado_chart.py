"""Phase 4D — Tornado chart artifact generator."""

from __future__ import annotations

import dspy


class TornadoChartSignature(dspy.Signature):
    """
    You are a data visualization expert generating Python matplotlib code for a tornado chart.

    A tornado chart shows the sensitivity of an objective value to changes in parameters.
    Bars extend left (negative impact) and right (positive impact) from a center baseline.
    Bars are sorted by absolute impact — widest bar at top (most influential parameter).

    REQUIREMENTS:
    - Parse the sensitivity_results text to extract parameter names and their impact values.
    - Use matplotlib to draw a horizontal bar chart (tornado layout).
    - Save the figure to the exact path given in artifact_path (use plt.savefig).
    - Do NOT call plt.show() — save only.
    - Use tight_layout() and dpi=150 for publication quality.
    - If sensitivity_results contains dual/shadow price rows, use those as the impact values.
    - If sensitivity_results contains perturbation rows (±5%, ±10%), use the ±10% values.
    - Handle gracefully if parsing fails: save an empty figure with a 'No data' label.
    """

    sensitivity_results: str = dspy.InputField(
        desc="Text output from the sensitivity analysis code (shadow prices table or perturbation table)."
    )
    artifact_path: str = dspy.InputField(
        desc="Absolute file path where the PNG must be saved (e.g., '/path/to/tornado_chart.png')."
    )
    problem_title: str = dspy.InputField(
        desc="Short title for the chart (e.g., 'EOQ Sensitivity Analysis')."
    )

    chart_code: str = dspy.OutputField(
        desc="Complete, self-contained Python code using matplotlib that parses sensitivity_results and saves a tornado chart PNG to artifact_path."
    )