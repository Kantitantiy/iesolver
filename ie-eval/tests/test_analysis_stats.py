"""
Statistical tests: McNemar + bootstrap CI.

Referans değerler el/scipy ile önceden hesaplandı; regresyon önleyici.
"""

from __future__ import annotations

import math

import pytest

from ie_eval.analysis.stats import bootstrap_diff_ci, mcnemar_test


# =============================================================================
# McNemar — küçük n (exact binomial)
# =============================================================================
def test_mcnemar_no_discordance_returns_p_1():
    r = mcnemar_test(b=0, c=0)
    assert r.method == "trivial"
    assert r.p_value == 1.0
    assert math.isnan(r.statistic)


def test_mcnemar_exact_b5_c0():
    """b=5, c=0 → p = 2 * P(X<=0; n=5, 0.5) = 2 * (1/32) = 0.0625."""
    r = mcnemar_test(b=5, c=0)
    assert r.method == "exact_binomial"
    assert r.p_value == pytest.approx(0.0625, rel=1e-9)


def test_mcnemar_exact_b8_c2():
    """b=8, c=2, n=10 → p = 2 * P(X<=2; n=10, 0.5)."""
    r = mcnemar_test(b=8, c=2)
    assert r.method == "exact_binomial"
    # C(10,0)+C(10,1)+C(10,2) = 1+10+45 = 56; p = 2 * 56 / 1024 = 0.109375
    assert r.p_value == pytest.approx(0.109375, rel=1e-9)


def test_mcnemar_exact_symmetric():
    """b and c swapped → same p."""
    r1 = mcnemar_test(b=7, c=3)
    r2 = mcnemar_test(b=3, c=7)
    assert r1.p_value == pytest.approx(r2.p_value)


def test_mcnemar_exact_clipped_to_1():
    """b=c=5, n=10 → 2 * P(X<=5) > 1 → clip to 1.0."""
    r = mcnemar_test(b=5, c=5)
    assert r.p_value == 1.0


# =============================================================================
# McNemar — büyük n (chi-square with continuity)
# =============================================================================
def test_mcnemar_chi_square_used_at_n25():
    r = mcnemar_test(b=20, c=5)
    assert r.method == "chi_square_continuity"
    # χ² = (|20-5|-1)² / 25 = 14² / 25 = 196 / 25 = 7.84
    assert r.statistic == pytest.approx(7.84, rel=1e-9)
    # p-value = erfc(sqrt(7.84/2)) = erfc(sqrt(3.92)) ≈ 0.00512
    assert r.p_value == pytest.approx(0.00512, abs=1e-4)


def test_mcnemar_chi_square_very_significant():
    r = mcnemar_test(b=50, c=5)
    # χ² = (44)² / 55 = 1936/55 = 35.2
    assert r.statistic == pytest.approx(35.2, rel=1e-9)
    assert r.p_value < 1e-6


def test_mcnemar_chi_square_no_diff():
    """b = c yaklaşık → p ≈ 1."""
    r = mcnemar_test(b=25, c=25)
    # χ² = (0-1)²/50 = 0.02 → p ≈ 0.888
    assert r.p_value > 0.5


# =============================================================================
# Bootstrap CI
# =============================================================================
def test_bootstrap_identical_arrays_zero_diff():
    a = [True] * 10 + [False] * 10
    b = list(a)
    ci = bootstrap_diff_ci(a, b, n_iterations=500, seed=42)
    assert ci.mean_diff == 0.0
    assert ci.lower == 0.0
    assert ci.upper == 0.0


def test_bootstrap_all_a_beats_none_b():
    """A hepsini doğru, B hepsini yanlış → diff ≡ 1.0."""
    n = 20
    a = [True] * n
    b = [False] * n
    ci = bootstrap_diff_ci(a, b, n_iterations=500, seed=42)
    assert ci.mean_diff == 1.0
    assert ci.lower == 1.0
    assert ci.upper == 1.0


def test_bootstrap_is_reproducible():
    a = [True, False, True, True, False, True, False, True]
    b = [False, True, True, False, False, True, False, False]
    ci1 = bootstrap_diff_ci(a, b, n_iterations=1000, seed=123)
    ci2 = bootstrap_diff_ci(a, b, n_iterations=1000, seed=123)
    assert ci1.mean_diff == ci2.mean_diff
    assert ci1.lower == ci2.lower
    assert ci1.upper == ci2.upper


def test_bootstrap_different_seeds_differ():
    a = [True, False] * 20
    b = [False, True] * 20
    ci1 = bootstrap_diff_ci(a, b, n_iterations=500, seed=1)
    ci2 = bootstrap_diff_ci(a, b, n_iterations=500, seed=2)
    # Farklı seed farklı sonuç vermeli (aynı olma ihtimali astronomik)
    assert (ci1.mean_diff, ci1.lower, ci1.upper) != (ci2.mean_diff, ci2.lower, ci2.upper)


def test_bootstrap_ci_contains_true_mean():
    """A: 70% doğru, B: 40% doğru → gerçek fark ~+0.3; CI bunu içermeli."""
    a = [True] * 14 + [False] * 6
    b = [True] * 8 + [False] * 12
    ci = bootstrap_diff_ci(a, b, n_iterations=5000, seed=42)
    # Gözlenen fark 0.30; %95 CI genellikle bu değeri kapsar
    assert ci.lower <= 0.30 <= ci.upper


def test_bootstrap_raises_on_length_mismatch():
    with pytest.raises(ValueError, match="equal length"):
        bootstrap_diff_ci([True, False], [True], n_iterations=100)


def test_bootstrap_empty_input():
    ci = bootstrap_diff_ci([], [], n_iterations=100)
    assert ci.mean_diff == 0.0
    assert ci.lower == 0.0
    assert ci.upper == 0.0
