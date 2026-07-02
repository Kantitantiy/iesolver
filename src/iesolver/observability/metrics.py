"""
iesolver.observability.metrics — per-node telemetry (DESIGN_REVIEW §3.5).

Makale tablolarını dolduran metrikler:
    latency_ms, tokens_in, tokens_out, cost_usd, llm_calls, invocations,
    error_class.

Tasarım kararları
─────────────────
* **Contextvar bucket**: her instrumented node çağrısı bir bucket kurar;
  ``call_with_fast_lm`` / ``call_with_reasoning_lm`` DSPy history delta'sını
  bu bucket'a yazar. Contextvar seçildi çünkü LangGraph node'ları copy_context
  ile yürütür — thread-safe, propagate olur.
* **Reducer (merge_metrics)**: aynı node birden fazla kez çalışırsa
  (özellikle code_branch retry döngüsü) sayısal alanlar toplanır,
  ``invocations`` artar, ``error_class`` son yazana verilir. Böylece
  makale tablosunda "code_branch: 3 invocation, X token toplam" bilgisi
  otomatik toplanır — replay/hesaplama gerekmez.
* **Kararlı şema**: ``NODE_METRIC_KEYS`` sabit; ileride yeni alan
  eklenirse (örn. dual_vars_count Faz 4'te) buraya eklenir ve reducer
  otomatik hesaplar. Ölçüm sözleşmesi tek yerde.
"""

from __future__ import annotations

import contextvars
import functools
import time
from typing import Any, Callable, Iterable

# Sabit metrik şeması — testler ve makale tabloları buradan okur.
NODE_METRIC_KEYS: tuple[str, ...] = (
    "latency_ms",
    "tokens_in",
    "tokens_out",
    "cost_usd",
    "llm_calls",
    "invocations",
    "error_class",
)


# =============================================================================
# Contextvar bucket — LM helper'ları buraya yazar, instrument decorator okur.
# =============================================================================
_current_bucket: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "iesolver_metrics_bucket", default=None
)


def _new_bucket() -> dict[str, Any]:
    return {
        "tokens_in": 0,
        "tokens_out": 0,
        "cost_usd": 0.0,
        "llm_calls": 0,
    }


def record_llm_usage(history_entries: Iterable[dict[str, Any]]) -> None:
    """Aggregate DSPy history entries into the active node's bucket.

    ``call_with_fast_lm`` / ``call_with_reasoning_lm`` DSPy modülünü
    çalıştırdıktan sonra yeni history girdilerini (delta) buraya iletir.
    Bir instrument scope'u dışında çağrıldığında sessizce no-op olur —
    testlerde LM helper'ı doğrudan çağırmak güvenli kalır.
    """
    bucket = _current_bucket.get()
    if bucket is None:
        return
    for entry in history_entries:
        usage = entry.get("usage") or {}
        bucket["tokens_in"] += int(usage.get("prompt_tokens", 0) or 0)
        bucket["tokens_out"] += int(usage.get("completion_tokens", 0) or 0)
        cost = entry.get("cost")
        if cost is not None:
            try:
                bucket["cost_usd"] += float(cost)
            except (TypeError, ValueError):
                pass
        bucket["llm_calls"] += 1


# =============================================================================
# Reducer — aynı node yeniden çağrıldığında sayısal alanlar toplansın.
# =============================================================================
_NUMERIC_KEYS: tuple[str, ...] = (
    "latency_ms",
    "tokens_in",
    "tokens_out",
    "cost_usd",
    "llm_calls",
)


def merge_metrics(
    a: dict[str, dict[str, Any]] | None,
    b: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """LangGraph reducer for the ``metrics`` state field.

    * Yeni node → doğrudan eklenir.
    * Tekrar giren node (retry) → sayısal alanlar toplanır, invocations +1,
      error_class son yazana verilir.
    """
    out: dict[str, dict[str, Any]] = {k: dict(v) for k, v in (a or {}).items()}
    for node_name, slice_ in (b or {}).items():
        existing = out.get(node_name)
        if existing is None:
            out[node_name] = dict(slice_)
            continue
        merged = dict(existing)
        for key in _NUMERIC_KEYS:
            merged[key] = existing.get(key, 0) + slice_.get(key, 0)
        merged["invocations"] = existing.get("invocations", 1) + slice_.get("invocations", 1)
        merged["error_class"] = slice_.get("error_class")
        out[node_name] = merged
    return out


# =============================================================================
# Instrument decorator — her node'un başına eklenir.
# =============================================================================
def instrument(node_name: str) -> Callable:
    """Wrap a LangGraph node fn to record its metrics slice.

    Kullanım::

        @instrument("intake")
        def intake_node(state: SolverState) -> SolverState:
            ...

    Node'un döndürdüğü partial state'e ``{"metrics": {node_name: {...}}}``
    eklenir. Aynı node ikinci kez çağrılırsa ``merge_metrics`` reducer'ı
    sayısal alanları toplar.

    Exception davranışı: instrument yakalar → error_class'ı slice'a yazar
    → yeniden fırlatır. LangGraph error handling'ini bozmaz; yalnızca
    hata sınıfı telemetriye yansır. (Metrik slice'ı re-raise sonrası
    state'e ulaşamayacağı için, exception'lı çalışmalarda ilgili slice
    kaybolur — makale hata analizi için LangGraph checkpoint'i okumak
    gerekir. MVP kapsamında bilinçli sınırlama.)
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(state: dict[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
            bucket = _new_bucket()
            token = _current_bucket.set(bucket)
            t0 = time.perf_counter()
            error_class: str | None = None
            try:
                partial = fn(state, *args, **kwargs) or {}
            except Exception as exc:
                error_class = type(exc).__name__
                raise
            finally:
                latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                _current_bucket.reset(token)

            slice_: dict[str, Any] = {
                "latency_ms": latency_ms,
                "tokens_in": bucket["tokens_in"],
                "tokens_out": bucket["tokens_out"],
                "cost_usd": round(bucket["cost_usd"], 6),
                "llm_calls": bucket["llm_calls"],
                "invocations": 1,
                "error_class": error_class,
            }
            existing_metrics = partial.get("metrics") or {}
            return {**partial, "metrics": {**existing_metrics, node_name: slice_}}

        return wrapper

    return decorator