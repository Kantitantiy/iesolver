"""
ie_eval.ablations — Ablation solve_fn factory'leri (EVALUATION_PLAN §5).

Her fonksiyon, ``runner.run_one`` / ``runner.run_dataset`` 'ın beklediği
``solve_fn(prompt, data_path=None, **kw) -> dict`` imzasıyla uyumlu bir
sarmalayıcı döndürür.

Ablasyon konfigürasyonları:

| # | İsim               | Değiştirilen          | Test edilen iddia                         |
|---|--------------------|-----------------------|-------------------------------------------|
| A1 | no_refiner        | enable_refiner=False  | PromptRefiner katkısı                     |
| A2 | no_retry          | enable_validator_retry=False | Retry döngüsü katkısı             |
| A3 | llm_validator_only| correctness_fn        | Deterministik doğrulama katmanı katkısı   |
| A4 | fast_only         | fast_only=True        | Reasoning LM anahtarlaması katkısı        |
| A5 | optimized_sigs    | compiled DSPy program | MIPROv2 optimizasyonunun katkısı          |
| A6 | self_consistency  | self_consistency_router=True | Router'da self-consistency katkısı |

Kullanım örneği::

    from ie_eval.ablations import (
        make_a1_solve, make_a2_solve, make_a3_correctness_fn,
        make_a4_solve, make_a5_solve, make_a6_solve,
    )
    from ie_eval.runner import run_dataset

    # A1: PromptRefiner kapalı
    run_dataset(dataset, config_id="A1_no_refiner",
                solve_fn=make_a1_solve(auto_mode=True))

    # A3: run_one'a özel correctness_fn (solve_fn DEĞİL)
    from ie_eval import run_one
    correctness_fn = make_a3_correctness_fn()
    rec = run_one(problem, correctness_fn=correctness_fn)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# A1 — PromptRefiner devre dışı
# ---------------------------------------------------------------------------
def make_a1_solve(**default_kw: Any) -> Callable:
    """Return a solve_fn with the PromptRefiner node bypassed (A1 ablation).

    Ham prompt, ön yapılandırma adımı atlanarak doğrudan strateji
    router'ına gider. PromptRefiner'ın structured reformulation katkısını
    izole eder.

    Parameters
    ----------
    **default_kw :
        ``auto_mode``, ``thread_id`` gibi sabitler; her çağrıda
        caller tarafından override edilebilir.
    """
    from iesolver import solve as _solve

    def _solve_fn(prompt: str, data_path: Any = None, **kw: Any) -> dict:
        merged = {**default_kw, **kw}
        merged.setdefault("auto_mode", True)
        return _solve(prompt, data_path=data_path, enable_refiner=False, **merged)

    _solve_fn.__name__ = "a1_no_refiner_solve"
    return _solve_fn


# ---------------------------------------------------------------------------
# A2 — Validator retry devre dışı
# ---------------------------------------------------------------------------
def make_a2_solve(**default_kw: Any) -> Callable:
    """Return a solve_fn without the code-retry loop (A2 ablation).

    validate_node ilk denemede başarısız olursa doğrudan report'a
    geçilir; retry döngüsü aktif değildir. Retry döngüsünün başarı
    oranına katkısını ölçer.

    Parameters
    ----------
    **default_kw :
        ``auto_mode`` vb. kalıcı parametreler.
    """
    from iesolver import solve as _solve

    def _solve_fn(prompt: str, data_path: Any = None, **kw: Any) -> dict:
        merged = {**default_kw, **kw}
        merged.setdefault("auto_mode", True)
        return _solve(prompt, data_path=data_path, enable_validator_retry=False, **merged)

    _solve_fn.__name__ = "a2_no_retry_solve"
    return _solve_fn


# ---------------------------------------------------------------------------
# A3 — Deterministik doğrulama yok (yalnızca LLM validator sinyali)
# ---------------------------------------------------------------------------
def make_a3_correctness_fn() -> Callable[[dict, Any], bool]:
    """Return a correctness_fn that uses only the LLM's is_valid signal (A3 ablation).

    Normalde ie_eval.metrics, ground truth'a karşı sayısal eşleşme
    (deterministik numerical_match) uygular. Bu fonksiyon bunu atlar ve
    yalnızca iesolver pipeline'ının LLM-tabanlı validate_node kararını
    (state["is_valid"]) kullanır.

    ``run_one(..., correctness_fn=make_a3_correctness_fn())`` ile kullanılır.

    Signature: ``(state: dict, problem: Problem) -> bool``
    """
    from ie_eval.problem import Problem  # noqa: PLC0415

    def _check(state: Optional[dict], problem: "Problem") -> bool:  # noqa: F821
        if state is None:
            return False
        return bool(state.get("is_valid", False))

    _check.__name__ = "a3_llm_validator_only"
    return _check


# ---------------------------------------------------------------------------
# A4 — Fast-only (reasoning LM devre dışı)
# ---------------------------------------------------------------------------
def make_a4_solve(**default_kw: Any) -> Callable:
    """Return a solve_fn where all LLM calls use the fast model (A4 ablation).

    Kod üretimi ve duyarlılık analizi dahil tüm LLM çağrıları fast LM'e
    yönlendirilir; reasoning LM anahtarlaması devre dışıdır. Model
    anahtarlama stratejisinin katkısını izole eder.

    Parameters
    ----------
    **default_kw :
        ``auto_mode`` vb. kalıcı parametreler.
    """
    from iesolver import solve as _solve

    def _solve_fn(prompt: str, data_path: Any = None, **kw: Any) -> dict:
        merged = {**default_kw, **kw}
        merged.setdefault("auto_mode", True)
        return _solve(prompt, data_path=data_path, fast_only=True, **merged)

    _solve_fn.__name__ = "a4_fast_only_solve"
    return _solve_fn


# ---------------------------------------------------------------------------
# A5 — MIPROv2 optimize edilmiş Signature'lar
# ---------------------------------------------------------------------------
def make_a5_solve(compiled_path: Path, **default_kw: Any) -> Callable:
    """Return a solve_fn that loads MIPROv2-optimized DSPy signatures (A5 ablation).

    MIPROv2 eğitimi ayrıca çalıştırılmalıdır (bkz. ``scripts/optimize_mipro.py``).
    Çıktı dosyası (``compiled_path``) mevcut değilse ``FileNotFoundError`` fırlatır.

    Çalışma prensibi:
        1. ``iesolver._optimization.IESolverProgram()`` oluşturulur.
           Bu, live pipeline singleton'larına referans tutar.
        2. ``program.load(compiled_path)`` tüm DSPy modüllerinin prompt /
           few-shot ağırlıklarını günceller (singleton'lar in-place değişir).
        3. Sonraki ``iesolver.solve()`` çağrıları güncellenmiş prompt'larla çalışır.

    Parameters
    ----------
    compiled_path :
        ``scripts/optimize_mipro.py`` tarafından kaydedilen JSON dosyası.
    **default_kw :
        ``auto_mode`` vb. kalıcı parametreler.

    Raises
    ------
    FileNotFoundError
        ``compiled_path`` dosyası bulunamazsa.
    """
    compiled_path = Path(compiled_path)
    if not compiled_path.exists():
        raise FileNotFoundError(
            f"A5 ablation: compiled signatures not found at {compiled_path}. "
            "Run `uv run python scripts/optimize_mipro.py` first."
        )

    _applied = False

    def _solve_fn(prompt: str, data_path: Any = None, **kw: Any) -> dict:
        nonlocal _applied
        if not _applied:
            from iesolver._optimization import load_compiled_graph
            load_compiled_graph(compiled_path)   # singleton'ları in-place güncelle
            _applied = True

        from iesolver import solve as _solve
        merged = {**default_kw, **kw}
        merged.setdefault("auto_mode", True)
        return _solve(prompt, data_path=data_path, **merged)

    _solve_fn.__name__ = "a5_optimized_solve"
    return _solve_fn



# ---------------------------------------------------------------------------
# A6 — Self-consistency router (3-oy çoğunluk)
# ---------------------------------------------------------------------------
def make_a6_solve(**default_kw: Any) -> Callable:
    """Return a solve_fn with self-consistency enabled on the strategy router (A6 ablation).

    execution_path kararı 3 bağımsız örneklemenin çoğunluk oyuyla alınır
    (``dspy.majority``), tek örnekleme yerine. Pipeline'daki en yüksek
    blast-radius'lu kararın (yanlış dallanma her şeyi geçersiz kılar)
    varyansını izole eder — 3x router maliyeti karşılığında.

    Parameters
    ----------
    **default_kw :
        ``auto_mode`` vb. kalıcı parametreler.
    """
    from iesolver import solve as _solve

    def _solve_fn(prompt: str, data_path: Any = None, **kw: Any) -> dict:
        merged = {**default_kw, **kw}
        merged.setdefault("auto_mode", True)
        return _solve(prompt, data_path=data_path, self_consistency_router=True, **merged)

    _solve_fn.__name__ = "a6_self_consistency_solve"
    return _solve_fn


__all__ = [
    "make_a1_solve",
    "make_a2_solve",
    "make_a3_correctness_fn",
    "make_a4_solve",
    "make_a5_solve",
    "make_a6_solve",
]
