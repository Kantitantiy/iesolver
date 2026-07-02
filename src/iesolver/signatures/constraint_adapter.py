"""Phase 4B.2 — Constraint adaptation to selected library's API."""

from __future__ import annotations

import dspy


class ConstraintAdapterSignature(dspy.Signature):
    """
    You are a precise Technical Constraint Engineer.
    Your task is to rewrite the user's strict constraints so they perfectly align
    with the technical realities and limitations of the selected target library.
    Incorporate the adaptation notes directly into the final constraints.
    """

    strict_constraints = dspy.InputField()
    target_library = dspy.InputField()
    adaptation_notes = dspy.InputField()

    library_specific_constraints = dspy.OutputField(
        desc="A finalized, bulleted list of constraints perfectly tailored for the target library's syntax and limits."
    )
