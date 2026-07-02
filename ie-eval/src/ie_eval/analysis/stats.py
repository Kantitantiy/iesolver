"""
ie_eval.analysis.stats — İstatistiksel testler (EVALUATION_PLAN §7).

**scipy YOK** — makalede rapor edilen iki testi sıfırdan hesaplıyoruz.
Bu, ie-eval'ı hafif tutar ve reviewer'a "hangi formül kullanıldı" sorusunun
cevabını satır satır verebilir kılar.

Testler:
    * mcnemar_test(b, c)
        Eşleştirilmiş kategorik veri (aynı problem seti üzerinde iki config).
        n = b + c ≥ 25 → sürekli düzeltmeli chi-square (df=1)
        n < 25         → exact binomial (küçük n için standart öneri)
    * bootstrap_diff_ci(a_correct, b_correct)
        Percentile method; seedable, reproducible.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


# =============================================================================
# McNemar
# =============================================================================
@dataclass(slots=True)
class McNemarResult:
    """Structured McNemar output."""

    b: int                 # A doğru, B yanlış
    c: int                 # A yanlış, B doğru
    statistic: float       # chi-square statistic (with continuity correction) or NaN
    p_value: float         # two-sided
    method: str            # "chi_square_continuity" | "exact_binomial" | "trivial"


def _binom_cdf_half(k: int, n: int) -> float:
    """P(X ≤ k) for X ~ Binomial(n, 0.5). k, n ≥ 0; k ≤ n."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    # Direct summation; n is small (McNemar exact branch uses n < 25)
    total = 0.0
    for i in range(k + 1):
        total += math.comb(n, i)
    return total / (2**n)


def _chi_square_1df_sf(x: float) -> float:
    """Survival function of χ²(1) at x. Uses erfc since χ²(1) = Z²."""
    if x <= 0:
        return 1.0
    return math.erfc(math.sqrt(x / 2.0))


def mcnemar_test(b: int, c: int) -> McNemarResult:
    """Paired McNemar test. ``b``/``c`` = discordant pair counts.

    b: config A correct, config B wrong
    c: config A wrong,  config B correct

    Convention (EVALUATION_PLAN §7): α=0.05, iki taraflı.

    * n = b + c ≥ 25 → chi-square with continuity correction:
          χ² = (|b - c| - 1)² / n     df=1
    * n < 25 → exact binomial:
          p = 2 · P(X ≤ min(b, c))    X ~ Bin(n, 0.5), clipped to 1.0
    * n = 0 → trivial: no discordance, p_value=1.0
    """
    n = b + c
    if n == 0:
        return McNemarResult(b=b, c=c, statistic=float("nan"), p_value=1.0, method="trivial")

    if n < 25:
        k = min(b, c)
        p = 2.0 * _binom_cdf_half(k, n)
        p = min(p, 1.0)
        return McNemarResult(
            b=b, c=c, statistic=float("nan"), p_value=p, method="exact_binomial"
        )

    stat = (abs(b - c) - 1.0) ** 2 / n
    p = _chi_square_1df_sf(stat)
    return McNemarResult(
        b=b, c=c, statistic=stat, p_value=p, method="chi_square_continuity"
    )


# =============================================================================
# Bootstrap CI
# =============================================================================
@dataclass(slots=True)
class BootstrapCI:
    """Bootstrap CI for accuracy_A - accuracy_B (paired resampling)."""

    mean_diff: float
    lower: float
    upper: float
    ci_level: float           # e.g. 0.95
    n_iterations: int
    seed: int


def bootstrap_diff_ci(
    a_correct: list[bool],
    b_correct: list[bool],
    *,
    n_iterations: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> BootstrapCI:
    """Percentile bootstrap CI for accuracy(A) - accuracy(B) on paired data.

    Both lists must be the same length and index-aligned (problem i in same
    position in both). Resampling is over PROBLEM indices, preserving pairing
    — this matches McNemar's paired-data setup (EVALUATION_PLAN §7).

    Deterministic given ``seed``.
    """
    if len(a_correct) != len(b_correct):
        raise ValueError("a_correct and b_correct must have equal length")
    n = len(a_correct)
    if n == 0:
        return BootstrapCI(
            mean_diff=0.0, lower=0.0, upper=0.0,
            ci_level=ci_level, n_iterations=n_iterations, seed=seed,
        )

    rng = random.Random(seed)
    diffs: list[float] = []
    for _ in range(n_iterations):
        # Aynı indeksten aynı problem — pairing korunur
        acc_a = 0
        acc_b = 0
        for _j in range(n):
            i = rng.randrange(n)
            if a_correct[i]:
                acc_a += 1
            if b_correct[i]:
                acc_b += 1
        diffs.append((acc_a - acc_b) / n)

    diffs.sort()
    alpha = (1.0 - ci_level) / 2.0
    lo_idx = int(alpha * n_iterations)
    hi_idx = min(int((1.0 - alpha) * n_iterations), n_iterations - 1)
    return BootstrapCI(
        mean_diff=sum(diffs) / n_iterations,
        lower=diffs[lo_idx],
        upper=diffs[hi_idx],
        ci_level=ci_level,
        n_iterations=n_iterations,
        seed=seed,
    )
