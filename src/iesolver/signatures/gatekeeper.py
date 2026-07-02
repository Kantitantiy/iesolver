"""Phase 0 — Input standardization & noise stripping."""

from __future__ import annotations

import dspy


class GateKeeperSignature(dspy.Signature):
    """
    You are an expert Data Engineer and prompt standardizer for an Industrial Engineering system.
    Your task is to analyze the user's raw input prompt. You must:
    1. Strip away any unnecessary code snippets or formatting errors.
    2. Correct any character encoding issues.
    3. Maintain all technical requirements and domain-specific terminology (e.g., BOM, MRP, WIP).
    Do NOT solve the problem; only standardize the input for downstream processing.
    """
    # Buradaki docstring modeli sadece temizlik yapmaya zorluyor, problem çözmesini engelliyor.

    raw_prompt = dspy.InputField(desc="The original, unformatted request from the user.")

    cleaned_prompt = dspy.OutputField(
        desc="A standardized, grammatically correct, and clear version of the prompt."
    )
