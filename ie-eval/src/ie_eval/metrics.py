"""
ie_eval.metrics — Per-run metric extraction & aggregation.

Bir Problem üzerinde iesolver.solve() çalıştırıldıktan sonra final state'ten
makale tablolarını dolduracak metrikleri çıkarır. Node-başı telemetri
(latency_ms, tokens_in/out, cost_usd) iesolver'ın state["metrics"] alanında
zaten toplanıyor (DESIGN_REVIEW §3.5 — ön düzeltmeler); burada yalnızca
problem-düzeyi özet ve doğruluk metrikleri hesaplanır.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ie_eval.problem import Problem
from ie_eval.validator import FeasibilityCheck, check_feasibility, numerical_match


# =============================================================================
# ProblemMetrics — bir koşunun tüm sayısal özeti
# =============================================================================
@dataclass(slots=True)
class ProblemMetrics:
    """Aggregated metrics for a single (problem, run_idx) pair.

    Kaynaklar:
        - Doğruluk / feasibility: iesolver çıktısı + ground truth
        - Maliyet / gecikme: iesolver'ın state["metrics"] (per-node özet)
    """

    problem_id: str
    execution_rate: bool          # code çalıştı mı (execution_result var mı)
    numerical_match: bool         # objective_value tolerance içinde mi
    feasibility: FeasibilityCheck # feasibility_fn sonucu
    elapsed_s: float              # runner tarafından ölçülen uçtan uca süre

    # Aşağı seviye telemetri özeti (state["metrics"] üzerinden)
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_cost_usd: float = 0.0
    total_llm_calls: int = 0
    node_count: int = 0
    retry_count: int = 0
    error_class: str | None = None    # herhangi bir node'da yakalanan hata

    per_node: dict[str, dict[str, Any]] = field(default_factory=dict)


# =============================================================================
# Extraction — iesolver final state'inden metric çekimi
# =============================================================================
def _sum_node_field(node_metrics: dict[str, dict[str, Any]], key: str) -> float:
    total = 0.0
    for slice_ in node_metrics.values():
        val = slice_.get(key, 0) or 0
        try:
            total += float(val)
        except (TypeError, ValueError):
            continue
    return total


def _first_error_class(node_metrics: dict[str, dict[str, Any]]) -> str | None:
    for slice_ in node_metrics.values():
        err = slice_.get("error_class")
        if err:
            return str(err)
    return None


def extract_metrics(
    problem: Problem,
    state: dict[str, Any] | None,
    elapsed_s: float,
    proposed_solution: dict[str, float] | None = None,
) -> ProblemMetrics:
    """Build a ProblemMetrics record from a completed solve() call.

    Parameters
    ----------
    problem :
        The Problem that was run.
    state :
        Final SolverState from ``iesolver.solve(...)``; ``None`` on hard failure.
    elapsed_s :
        Wall-clock time of the whole solve() call.
    proposed_solution :
        Decision variable dict extracted from execution_result (if available).
        MVP: caller-provided (Faz 5'te Signature-tabanlı otomatik ayrıştırıcı).
    """
    if state is None:
        return ProblemMetrics(
            problem_id=problem.id,
            execution_rate=False,
            numerical_match=False,
            feasibility=FeasibilityCheck(feasible=False, violations=["no state"], checked=False),
            elapsed_s=elapsed_s,
            error_class="RunnerException",
        )

    exec_result = state.get("execution_result") or ""
    execution_rate = bool(exec_result.strip())

    # Numerical match — CODE path'te execution_result, NO_CODE'da raw_result
    haystack = exec_result or (state.get("raw_result") or "")
    if problem.ground_truth.objective_value is None:
        num_match = True  # değerlendirilmez, feasibility'ye bakılır
    else:
        num_match = numerical_match(
            problem.ground_truth.objective_value,
            haystack,
            tolerance_rel=problem.ground_truth.tolerance_rel,
        )

    # Feasibility — solution sözlüğü varsa
    if proposed_solution is not None:
        feas = check_feasibility(proposed_solution, problem.ground_truth)
    else:
        feas = FeasibilityCheck(feasible=True, violations=[], checked=False)

    node_metrics = state.get("metrics") or {}

    return ProblemMetrics(
        problem_id=problem.id,
        execution_rate=execution_rate,
        numerical_match=num_match,
        feasibility=feas,
        elapsed_s=elapsed_s,
        total_tokens_in=int(_sum_node_field(node_metrics, "tokens_in")),
        total_tokens_out=int(_sum_node_field(node_metrics, "tokens_out")),
        total_cost_usd=round(_sum_node_field(node_metrics, "cost_usd"), 6),
        total_llm_calls=int(_sum_node_field(node_metrics, "llm_calls")),
        node_count=len(node_metrics),
        retry_count=int(state.get("retry_count", 0) or 0),
        error_class=_first_error_class(node_metrics),
        per_node=dict(node_metrics),
    )