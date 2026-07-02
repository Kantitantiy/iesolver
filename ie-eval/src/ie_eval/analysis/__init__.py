"""Aggregation, statistical tests, and reporting (EVALUATION_PLAN §3, §7)."""

from ie_eval.analysis.report import format_comparison, format_summary
from ie_eval.analysis.stats import (
    BootstrapCI,
    McNemarResult,
    bootstrap_diff_ci,
    mcnemar_test,
)
from ie_eval.analysis.summary import (
    ComparisonSummary,
    ConfigSummary,
    compare_configs,
    per_problem_correctness,
    summarize_by_config,
)

__all__ = [
    "BootstrapCI",
    "ComparisonSummary",
    "ConfigSummary",
    "McNemarResult",
    "bootstrap_diff_ci",
    "compare_configs",
    "format_comparison",
    "format_summary",
    "mcnemar_test",
    "per_problem_correctness",
    "summarize_by_config",
]
