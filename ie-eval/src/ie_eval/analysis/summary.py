"""
ie_eval.analysis.summary — Konfigürasyon başına özet + karşılaştırma verisi.

EVALUATION_PLAN §3 gereği raporlanan birincil metrikler:
    * pass@1 (solving accuracy) → 3 koşunun ortalaması ± std
    * execution_rate
    * feasibility_rate  (feasibility fonksiyonu tanımlı olanlar üzerinden)
    * total_cost_usd, mean_latency_s
    * total_tokens_in/out

Per-problem korrektlik politikası (McNemar/bootstrap için tek 0/1 gerekir):
    * "majority" — 3 koşudan en az yarısı doğruysa doğru (default)
    * "all"      — 3 koşunun hepsi doğruysa doğru (deterministic reliability)
    * "any"      — 3 koşudan en az biri doğruysa doğru (optimistic)
    * "first"    — yalnızca run_idx=0

Store şeması per-koşu satır tutar; agregasyon bu modülde yapılır.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from ie_eval.store import ResultStore


AggregationPolicy = Literal["majority", "all", "any", "first"]

# metadata_filter tipi:
#   * dict → tüm key-value çiftleri eşleşmeli (exact match)
#   * callable → (metadata_dict) -> bool
#   * None → tümü dahil
MetadataFilter = Callable[[dict[str, Any]], bool] | dict[str, Any] | None


def _compile_metadata_filter(mf: MetadataFilter) -> Callable[[dict[str, Any]], bool]:
    """Convert the flexible metadata_filter form into a callable predicate."""
    if mf is None:
        return lambda md: True
    if callable(mf):
        return mf
    # dict → all pairs must match
    keys = list(mf.items())
    return lambda md: all(md.get(k) == v for k, v in keys)


# =============================================================================
# ConfigSummary — bir konfigürasyonun tüm özet metrikleri
# =============================================================================
@dataclass(slots=True)
class ConfigSummary:
    """Aggregate metrics for one config across all problems and runs."""

    config_id: str
    n_problems: int
    n_runs: int                             # toplam satır sayısı (problem × run_idx)
    n_runs_per_problem: int                 # varsayılan olarak sabit; değişkense max

    # Doğruluk (pass@1) — her run_idx için ayrı skor, sonra mean ± std
    accuracy_mean: float
    accuracy_std: float
    accuracy_per_run: list[float]           # her run_idx için pass@1

    # Diğer birincil metrikler — problem-run ortalaması
    execution_rate: float                   # başarılı çalışan kod / toplam
    feasibility_rate: float                 # feasible / feasibility_checked (denominator sıfırsa 0)
    feasibility_checked: int                # kaç kayıtta feasibility_fn vardı

    # Maliyet / gecikme (problem-run toplamları)
    total_cost_usd: float
    total_tokens_in: int
    total_tokens_out: int
    total_llm_calls: int
    mean_elapsed_s: float
    median_elapsed_s: float

    # Retry istatistikleri
    mean_retry_count: float
    max_retry_count: int

    # Hata sınıfı dağılımı
    error_class_counts: dict[str, int] = field(default_factory=dict)


# =============================================================================
# Yardımcılar
# =============================================================================
def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    """Population std (n divisor). N=1 → 0.0 tanımlı."""
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return math.sqrt(var)


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    sxs = sorted(xs)
    n = len(sxs)
    mid = n // 2
    return sxs[mid] if n % 2 == 1 else (sxs[mid - 1] + sxs[mid]) / 2


# =============================================================================
# summarize_by_config — bir konfigürasyonun tam özeti
# =============================================================================
def summarize_by_config(
    store: ResultStore,
    config_id: str,
    *,
    metadata_filter: MetadataFilter = None,
) -> ConfigSummary:
    """Compute the full ConfigSummary for one config_id from the store.

    ``metadata_filter``:
        * ``None``     → tüm koşular
        * ``dict``     → verilen anahtarlar problem.metadata'da bire bir eşleşmeli
                         (örn. ``{"benchmark": "NL4Opt"}``)
        * ``callable`` → ``(metadata_dict) -> bool`` esnek predikat
    """
    predicate = _compile_metadata_filter(metadata_filter)
    all_rows = store.list_runs(config_id=config_id)
    rows = [r for r in all_rows if predicate(r.get("metadata") or {})]
    if not rows:
        return ConfigSummary(
            config_id=config_id, n_problems=0, n_runs=0, n_runs_per_problem=0,
            accuracy_mean=0.0, accuracy_std=0.0, accuracy_per_run=[],
            execution_rate=0.0, feasibility_rate=0.0, feasibility_checked=0,
            total_cost_usd=0.0, total_tokens_in=0, total_tokens_out=0,
            total_llm_calls=0, mean_elapsed_s=0.0, median_elapsed_s=0.0,
            mean_retry_count=0.0, max_retry_count=0,
        )

    # Per-run_idx → per-problem korrektlik matrisi
    by_run: dict[int, dict[str, bool]] = defaultdict(dict)
    problem_ids: set[str] = set()
    for r in rows:
        by_run[r["run_idx"]][r["problem_id"]] = bool(r["numerical_match"])
        problem_ids.add(r["problem_id"])

    n_problems = len(problem_ids)
    accuracy_per_run: list[float] = []
    for run_idx in sorted(by_run):
        row_map = by_run[run_idx]
        # Bu run için beklenen tüm problemler var mı — yoksa eksik kabul
        n_correct = sum(1 for pid in problem_ids if row_map.get(pid, False))
        accuracy_per_run.append(n_correct / n_problems if n_problems else 0.0)

    # Tek geçişte diğer tüm agregasyonlar
    exec_ok = 0
    feas_ok = 0
    feas_checked = 0
    total_cost = 0.0
    total_tin = 0
    total_tout = 0
    total_calls = 0
    elapsed_list: list[float] = []
    retries: list[int] = []
    err_counts: dict[str, int] = defaultdict(int)

    for r in rows:
        if r["execution_rate"]:
            exec_ok += 1
        if r["feasibility_checked"]:
            feas_checked += 1
            if r["feasibility_ok"]:
                feas_ok += 1
        total_cost += float(r["cost_usd"] or 0.0)
        total_tin += int(r["tokens_in"] or 0)
        total_tout += int(r["tokens_out"] or 0)
        total_calls += int(r["llm_calls"] or 0)
        elapsed_list.append(float(r["elapsed_s"] or 0.0))
        retries.append(int(r["retry_count"] or 0))
        ec = r["error_class"]
        if ec:
            err_counts[str(ec)] += 1

    n_runs = len(rows)

    return ConfigSummary(
        config_id=config_id,
        n_problems=n_problems,
        n_runs=n_runs,
        n_runs_per_problem=max(len(v) for v in by_run.values()) if by_run else 0,
        accuracy_mean=_mean(accuracy_per_run),
        accuracy_std=_std(accuracy_per_run),
        accuracy_per_run=accuracy_per_run,
        execution_rate=exec_ok / n_runs,
        feasibility_rate=(feas_ok / feas_checked) if feas_checked else 0.0,
        feasibility_checked=feas_checked,
        total_cost_usd=round(total_cost, 6),
        total_tokens_in=total_tin,
        total_tokens_out=total_tout,
        total_llm_calls=total_calls,
        mean_elapsed_s=_mean(elapsed_list),
        median_elapsed_s=_median(elapsed_list),
        mean_retry_count=_mean([float(x) for x in retries]),
        max_retry_count=max(retries) if retries else 0,
        error_class_counts=dict(err_counts),
    )


# =============================================================================
# per_problem_correctness — McNemar/bootstrap girdisi
# =============================================================================
def per_problem_correctness(
    store: ResultStore,
    config_id: str,
    *,
    policy: AggregationPolicy = "majority",
    metadata_filter: MetadataFilter = None,
) -> dict[str, bool]:
    """Reduce multi-run results to a single 0/1 per problem for statistical tests.

    Returns ``{problem_id: correct}``. ``metadata_filter`` benchmark/problem_type
    kırılımı için kullanılır (bkz. ``summarize_by_config``).
    """
    predicate = _compile_metadata_filter(metadata_filter)
    rows = [
        r for r in store.list_runs(config_id=config_id)
        if predicate(r.get("metadata") or {})
    ]
    by_problem: dict[str, list[tuple[int, bool]]] = defaultdict(list)
    for r in rows:
        by_problem[r["problem_id"]].append((int(r["run_idx"]), bool(r["numerical_match"])))

    out: dict[str, bool] = {}
    for pid, runs in by_problem.items():
        marks = [correct for _, correct in runs]
        if policy == "first":
            runs.sort()
            out[pid] = runs[0][1] if runs else False
        elif policy == "any":
            out[pid] = any(marks)
        elif policy == "all":
            out[pid] = all(marks) if marks else False
        else:  # majority
            out[pid] = sum(marks) * 2 > len(marks)
    return out


# =============================================================================
# ComparisonSummary — iki config'i eşleştirilmiş problem seti üzerinde karşılaştırır
# =============================================================================
@dataclass(slots=True)
class ComparisonSummary:
    """Head-to-head comparison between two configs on their common problems."""

    config_a: str
    config_b: str
    n_common_problems: int
    accuracy_a: float
    accuracy_b: float
    accuracy_diff: float           # a - b
    both_correct: int
    only_a_correct: int
    only_b_correct: int
    both_wrong: int
    policy: AggregationPolicy


def compare_configs(
    store: ResultStore,
    config_a: str,
    config_b: str,
    *,
    policy: AggregationPolicy = "majority",
    metadata_filter: MetadataFilter = None,
) -> ComparisonSummary:
    """Build a paired 2×2 contingency for McNemar-style comparison.

    ``metadata_filter`` her iki config'e de uygulanır → kırılımlı karşılaştırma
    (örn. sadece NL4Opt üzerinde pipeline vs single_shot).
    """
    a = per_problem_correctness(store, config_a, policy=policy, metadata_filter=metadata_filter)
    b = per_problem_correctness(store, config_b, policy=policy, metadata_filter=metadata_filter)
    common = sorted(set(a) & set(b))

    both_c = only_a = only_b = both_w = 0
    for pid in common:
        ca, cb = a[pid], b[pid]
        if ca and cb:
            both_c += 1
        elif ca and not cb:
            only_a += 1
        elif not ca and cb:
            only_b += 1
        else:
            both_w += 1

    n = len(common)
    acc_a = (both_c + only_a) / n if n else 0.0
    acc_b = (both_c + only_b) / n if n else 0.0

    return ComparisonSummary(
        config_a=config_a,
        config_b=config_b,
        n_common_problems=n,
        accuracy_a=acc_a,
        accuracy_b=acc_b,
        accuracy_diff=acc_a - acc_b,
        both_correct=both_c,
        only_a_correct=only_a,
        only_b_correct=only_b,
        both_wrong=both_w,
        policy=policy,
    )
