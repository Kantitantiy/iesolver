"""
ie_eval.validator — Deterministic solution verification (DESIGN_REVIEW §3.2).

İki katmanlı doğrulama:
    1. numerical_match: LLM-üretilen execution_result stringinden sayı
       çıkar, ground truth ile karşılaştır.
    2. check_feasibility: Ground truth'un feasibility fonksiyonunu solution
       sözlüğü üzerinde çalıştır — LLM'siz, kesin.

Bu katman "LLM kendi kendini mi notluyor?" eleştirisine karşı savunmadır:
solution_value karşılaştırması LLM-bağımsız (deterministic); LLM
validator (iesolver'ın validate_node'u) yalnızca semantik kontrolde kalır.

Çözüm ayrıştırma:
    LLM'in ürettiği execution_result serbest metin. Basit regex ile
    tüm sayıları çekiyoruz; ground truth'a %tolerans dahilinde herhangi
    biri eşleşiyorsa numerical_match=True. Bu yaklaşım NL4Opt/IndustryOR
    çoğunluğunda çalışır; karmaşık çıktı formatları için Faz 5'te
    Signature-tabanlı yapılandırılmış çıkarıcı gelecek.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ie_eval.problem import GroundTruth


# =============================================================================
# Numerical match
# =============================================================================
_NUMBER_PATTERN = re.compile(
    r"[-+]?\d+(?:[\.,]\d+)?(?:[eE][-+]?\d+)?"
)


def extract_numbers(text: str) -> list[float]:
    """Extract all numeric literals from free-form text.

    Uses a permissive regex covering ints, decimals, thousand separators,
    scientific notation. Comma-thousand-sep is heuristically normalized.
    """
    out: list[float] = []
    for match in _NUMBER_PATTERN.findall(text):
        # Handle "10,000" vs "3,14": treat comma as thousand-sep only if
        # exactly 3 digits follow it AND no decimal point present.
        candidate = match
        if "," in candidate and "." not in candidate:
            parts = candidate.split(",")
            if all(len(p) == 3 for p in parts[1:]):
                candidate = candidate.replace(",", "")
            else:
                candidate = candidate.replace(",", ".")
        try:
            out.append(float(candidate))
        except ValueError:
            continue
    return out


def numerical_match(
    expected: float,
    text: str,
    tolerance_rel: float = 1e-3,
) -> bool:
    """True iff any number in ``text`` matches ``expected`` within tolerance.

    Göreli tolerans: |actual - expected| / max(|expected|, 1) < tolerance_rel.
    max(|expected|, 1) küçük beklenen değerlerde göreli toleransın patlamamasını
    sağlar.
    """
    if not text:
        return False
    numbers = extract_numbers(text)
    denom = max(abs(expected), 1.0)
    return any(abs(n - expected) / denom < tolerance_rel for n in numbers)


# =============================================================================
# Feasibility
# =============================================================================
@dataclass(slots=True)
class FeasibilityCheck:
    """Structured feasibility outcome."""

    feasible: bool
    violations: list[str]
    checked: bool  # False = ground truth had no feasibility_fn


def check_feasibility(
    solution: dict[str, float],
    ground_truth: GroundTruth,
) -> FeasibilityCheck:
    """Deterministically verify a solution against the problem's constraints.

    Solution: karar değişkeni değerleri sözlüğü (LLM çıktısından ayrıştırılmış
    veya elle sağlanmış). Ground truth'un ``feasibility_fn``'i tanımlıysa
    çalıştırılır; değilse ``checked=False`` döner.
    """
    if ground_truth.feasibility_fn is None:
        return FeasibilityCheck(feasible=True, violations=[], checked=False)
    violations = ground_truth.feasibility_fn(solution)
    return FeasibilityCheck(
        feasible=len(violations) == 0,
        violations=violations,
        checked=True,
    )