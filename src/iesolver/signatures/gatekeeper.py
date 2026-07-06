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
    4. Preserve every number, parameter, and named entity exactly as given; limit your output to
       standardization only, and leave solving the problem to downstream stages.
    """
    # Buradaki docstring modeli sadece temizlik yapmaya zorluyor, problem çözmesini engelliyor.
    # Grounding + positive-instruction düzeltmesi (CLAUDE.md Düzeltme #2, #3):
    # "Do NOT solve" yerine "limit your output to X" — hem pozitif hem de
    # sayısal/isimsel içeriği koruma talimatı (uydurma riskini azaltır).

    raw_prompt = dspy.InputField(desc="The original, unformatted request from the user.")

    cleaned_prompt = dspy.OutputField(
        desc="A standardized, grammatically correct, and clear version of the prompt."
    )
