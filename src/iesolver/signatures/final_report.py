"""Phase 5 — 3-layer professional report synthesis."""

from __future__ import annotations

import dspy


class FinalReportSignature(dspy.Signature):
    """
    You are an elite Industrial Engineering Consultant and Technical Communicator.
    Your task is to synthesize the raw computational or analytical results into a
    comprehensive, 3-layered professional report.

    You must generate three distinct outputs based strictly on the provided data:

    1. TECHNICAL OUTPUT (For Engineers/Data Scientists):
       - Detail the exact algorithms, methodologies, or libraries used.
       - Present key performance metrics, confidence scores, and error margins.
       - Use highly technical academic English.

    2. EXECUTIVE SUMMARY (For C-Level Management):
       - Provide a formal, business-focused overview of the problem and the solution.
       - Translate technical metrics into business impact (e.g., cost savings, efficiency gains).
       - Include a brief sensitivity analysis (e.g., "If demand increases by 10%, cost rises by X%").
       - Use formal, clear, and strategic English.

    3. ACTION DIRECTIVES (For Operational Teams):
       - Provide step-by-step, actionable implementation orders.
       - Use imperative mood (command form).
       - Specify which department/unit is responsible for each action.
       - CRITICAL RULE: Always write out the actual values from validated_results in plain
         language, in place of code variable names or generic placeholders — e.g., write
         "Assign nodes 0, 4, 6, 8, 12, and 13 to Cluster 0" rather than `assigned_nodes`, and
         "Follow the exact route path: 999 -> 13 -> 12..." rather than `route_sequence`.
    """

    original_goal = dspy.InputField(desc="The initial business or engineering objective.")
    solution_path_or_code = dspy.InputField(
        desc="The analytical reasoning steps OR the executed Python code."
    )
    validated_results = dspy.InputField(
        desc="The final, sanity-checked numerical or qualitative outcomes."
    )

    technical_output = dspy.OutputField(desc="Technical details, algorithms, and metrics.")
    executive_summary = dspy.OutputField(
        desc="Business impact, managerial summary, and sensitivity analysis."
    )
    action_directives = dspy.OutputField(
        desc="Clear, imperative operational steps and department assignments."
    )
