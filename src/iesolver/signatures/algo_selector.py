"""Phase 4B.1 — Algorithm & library selection."""

from __future__ import annotations

import dspy


class AlgoSelectorSignature(dspy.Signature):
    """
    You are an expert Data Scientist and Industrial Engineer.
    Based on the problem type, goal, and data summary, select the most appropriate
    Python algorithm and the industry-standard library (e.g., PuLP, scikit-learn, pandas).

    You MUST also identify any potential conflicts between the user's constraints
    and the chosen library's technical limitations (e.g., 'scikit-learn KMeans only
    supports Euclidean distance, but user asked for Manhattan').

    GROUNDING: base every reference to columns, fields, or record counts strictly on
    what literally appears in the data summary. If a field the problem seems to need
    is absent from the summary, name that gap explicitly in adaptation_notes instead
    of assuming the field exists.
    """

    essential_prompt = dspy.InputField(desc="The core problem instruction.")
    problem_type = dspy.InputField(desc="The IE problem category.")
    data_summary = dspy.InputField(desc="Metadata of the available data.")
    strict_constraints = dspy.InputField(desc="The constraints that must be followed.")

    target_algorithm = dspy.OutputField(
        desc="The specific algorithm to be used (e.g., 'Simplex Method', 'Random Forest')."
    )
    target_library = dspy.OutputField(
        desc="The primary Python library to be imported (e.g., 'pulp', 'sklearn')."
    )
    adaptation_notes = dspy.OutputField(
        desc="Notes on how constraints must be adapted for this specific library. If none, write 'None'."
    )
