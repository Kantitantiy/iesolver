"""
ie_eval.runner — Batch execution using ``iesolver.solve(auto_mode=True)``.

MVP (Faz 4.5):
    - Tek problem koşusu (run_one) — timing + hata yakalama
    - Bir dataset üzerinde toplu koşu (run_dataset) — sırayla
    - Sonuçlar ResultStore'a yazılır

Sonraya bırakıldı:
    - Paralellik (multi-worker) — 3-koşu ortalamayla birlikte
    - Baseline'lar (single-shot LLM) — ayrı sarmalayıcı gerekecek
    - Konfigürasyon matrisi (model x ablasyon)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from ie_eval.datasets.base import Dataset
from ie_eval.metrics import ProblemMetrics, extract_metrics
from ie_eval.problem import Problem


# =============================================================================
# RunRecord — bir koşunun ham çıktısı
# =============================================================================
@dataclass(slots=True)
class RunRecord:
    """Raw outcome of one solve() invocation on one problem."""

    problem_id: str
    config_id: str
    run_idx: int
    success: bool                       # solve() istisnasız döndü ve interrupt yok
    elapsed_s: float
    state: Optional[dict[str, Any]] = None    # iesolver final state (varsa)
    error: Optional[str] = None
    metrics: Optional[ProblemMetrics] = field(default=None)
    # Problem.metadata'nın kopyası — store'a metadata_json olarak yazılır,
    # analysis modülü benchmark/problem_type kırılımı yapabilir.
    metadata: Optional[dict[str, Any]] = None


# =============================================================================
# run_one — tek problem
# =============================================================================
def run_one(
    problem: Problem,
    *,
    config_id: str = "baseline",
    run_idx: int = 0,
    thread_id_prefix: str = "eval",
    solve_fn: Any = None,         # bağımlılık enjeksiyonu (testlerde mock)
    correctness_fn: Any = None,   # A3 ablation: (state, problem) -> bool
) -> RunRecord:
    """Run ``iesolver.solve(prompt, auto_mode=True)`` on a single problem.

    ``solve_fn`` normalde ``iesolver.solve``. Testlerde mock geçilir; böylece
    runner LLM'siz doğrulanabilir.

    ``correctness_fn`` A3 ablation içindir: ``(state, problem) -> bool``
    imzasıyla çağrılır ve ``metrics.numerical_match`` yerine geçer. ``None``
    olduğunda varsayılan deterministik kontrol devreye girer.
    """
    if solve_fn is None:
        # Lazy import: testler sırasında iesolver'ın .env / API key olmadan
        # yüklenebilmesi için modül-üst import'undan kaçınıyoruz.
        from iesolver import solve as _iesolver_solve
        solve_fn = _iesolver_solve

    thread_id = f"{thread_id_prefix}-{config_id}-{problem.id}-r{run_idx}"

    t0 = time.perf_counter()
    try:
        state = solve_fn(
            problem.prompt,
            data_path=problem.data_path,
            auto_mode=True,
            thread_id=thread_id,
        )
        elapsed = time.perf_counter() - t0
        interrupted = "__interrupt__" in (state or {})
        rec = RunRecord(
            problem_id=problem.id,
            config_id=config_id,
            run_idx=run_idx,
            success=not interrupted,
            elapsed_s=elapsed,
            state=state,
            error="interrupted" if interrupted else None,
        )
    except Exception as exc:   # noqa: BLE001
        elapsed = time.perf_counter() - t0
        rec = RunRecord(
            problem_id=problem.id,
            config_id=config_id,
            run_idx=run_idx,
            success=False,
            elapsed_s=elapsed,
            state=None,
            error=f"{type(exc).__name__}: {exc}",
        )

    rec.metrics = extract_metrics(problem, rec.state, rec.elapsed_s)
    # A3 ablation: özel correctness_fn varsa numerical_match'i override et.
    # (state, problem) -> bool imzası; None olduğunda deterministik yol devrededir.
    if correctness_fn is not None and rec.metrics is not None:
        rec.metrics.numerical_match = bool(correctness_fn(rec.state, problem))
    # Problem metadata'sını RunRecord'a ve store'a geçir (kırılım analizi için)
    rec.metadata = dict(problem.metadata) if problem.metadata else None
    return rec


# =============================================================================
# run_dataset — sırayla toplu koşu
# =============================================================================
def run_dataset(
    dataset: Dataset | Iterable[Problem],
    *,
    config_id: str = "baseline",
    n_runs: int = 1,
    on_result: Any = None,      # callback: (RunRecord) -> None
    solve_fn: Any = None,
) -> list[RunRecord]:
    """Run every problem ``n_runs`` times; return all records.

    ``on_result`` her koşudan sonra çağrılır (yaygın kullanım: ResultStore.persist).
    Bellekte tüm state'leri tutmak istemezseniz callback ile disk'e yazın ve
    dönen listedeki state'i ihmal edin.
    """
    if isinstance(dataset, Dataset):
        problems = list(dataset.load())
    else:
        problems = list(dataset)

    records: list[RunRecord] = []
    for problem in problems:
        for run_idx in range(n_runs):
            rec = run_one(
                problem,
                config_id=config_id,
                run_idx=run_idx,
                solve_fn=solve_fn,
            )
            records.append(rec)
            if on_result is not None:
                on_result(rec)
    return records