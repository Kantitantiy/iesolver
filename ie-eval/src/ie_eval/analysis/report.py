"""
ie_eval.analysis.report — Human-readable text formatters.

Konfigürasyon özetleri ve karşılaştırma raporlarını sabit-genişlikli
metin olarak yazar. Kullanım:

    print(format_summary(summarize_by_config(store, "pipeline")))
    print(format_comparison(comp, mcnemar_result, bootstrap_ci))

Makale tabloları için ham veriyi yine dataclass'lardan alın; bu modülün
işi terminal/log okunabilirliği.
"""

from __future__ import annotations

from ie_eval.analysis.stats import BootstrapCI, McNemarResult
from ie_eval.analysis.summary import ComparisonSummary, ConfigSummary


# =============================================================================
# ConfigSummary
# =============================================================================
def format_summary(s: ConfigSummary) -> str:
    """Multi-line pretty-print of a single config's aggregate metrics."""
    lines = [
        f"═══ config: {s.config_id} ═══",
        f"  problems:               {s.n_problems}",
        f"  runs:                   {s.n_runs}  ({s.n_runs_per_problem}/problem)",
        f"  pass@1 (mean ± std):    {s.accuracy_mean:.3f} ± {s.accuracy_std:.3f}",
    ]
    if s.accuracy_per_run:
        per_run_str = ", ".join(f"{x:.3f}" for x in s.accuracy_per_run)
        lines.append(f"    per run:              [{per_run_str}]")

    lines += [
        f"  execution_rate:         {s.execution_rate:.3f}",
    ]
    if s.feasibility_checked > 0:
        lines.append(
            f"  feasibility_rate:       {s.feasibility_rate:.3f}  "
            f"(checked on {s.feasibility_checked} runs)"
        )
    else:
        lines.append("  feasibility_rate:       n/a (no feasibility_fn on any problem)")

    lines += [
        f"  cost (total, USD):      {s.total_cost_usd:.6f}",
        f"  tokens (in / out):      {s.total_tokens_in} / {s.total_tokens_out}",
        f"  llm_calls (total):      {s.total_llm_calls}",
        f"  elapsed (mean/median):  {s.mean_elapsed_s:.2f}s / {s.median_elapsed_s:.2f}s",
        f"  retries (mean/max):     {s.mean_retry_count:.2f} / {s.max_retry_count}",
    ]
    if s.error_class_counts:
        lines.append("  error classes:")
        for cls, n in sorted(s.error_class_counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {cls}: {n}")
    return "\n".join(lines)


# =============================================================================
# Comparison + McNemar + Bootstrap CI
# =============================================================================
def _format_p_value(p: float) -> str:
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def format_comparison(
    comp: ComparisonSummary,
    mcnemar: McNemarResult | None = None,
    bootstrap: BootstrapCI | None = None,
) -> str:
    """Multi-line comparison report with optional statistical annotations."""
    lines = [
        f"═══ comparison: {comp.config_a}  vs  {comp.config_b}  "
        f"(policy: {comp.policy}) ═══",
        f"  common problems:            {comp.n_common_problems}",
        f"  accuracy({comp.config_a}):{' ' * max(1, 15 - len(comp.config_a))}{comp.accuracy_a:.3f}",
        f"  accuracy({comp.config_b}):{' ' * max(1, 15 - len(comp.config_b))}{comp.accuracy_b:.3f}",
        f"  difference (A − B):         {comp.accuracy_diff:+.3f}",
        "",
        "  2×2 contingency:",
        f"    both correct:             {comp.both_correct}",
        f"    only {comp.config_a} correct:{' ' * max(1, 13 - len(comp.config_a))}{comp.only_a_correct}",
        f"    only {comp.config_b} correct:{' ' * max(1, 13 - len(comp.config_b))}{comp.only_b_correct}",
        f"    both wrong:               {comp.both_wrong}",
    ]

    if mcnemar is not None:
        lines += [
            "",
            f"  McNemar ({mcnemar.method}):",
            f"    discordant (b={mcnemar.b}, c={mcnemar.c})",
        ]
        if mcnemar.method == "chi_square_continuity":
            lines.append(f"    χ² = {mcnemar.statistic:.4f},  {_format_p_value(mcnemar.p_value)}")
        elif mcnemar.method == "exact_binomial":
            lines.append(f"    exact two-sided {_format_p_value(mcnemar.p_value)}")
        else:
            lines.append(f"    no discordant pairs → {_format_p_value(mcnemar.p_value)}")

    if bootstrap is not None:
        lines += [
            "",
            f"  Bootstrap {int(bootstrap.ci_level * 100)}% CI  "
            f"({bootstrap.n_iterations} iterations, seed={bootstrap.seed}):",
            f"    mean diff:    {bootstrap.mean_diff:+.4f}",
            f"    CI [lo, hi]:  [{bootstrap.lower:+.4f}, {bootstrap.upper:+.4f}]",
        ]

    return "\n".join(lines)
