"""
iesolver.lm — centralized DSPy language model configuration.

Plan §7'deki "Fast/Reasoning model anahtarlaması → LangGraph node-level
``dspy.context(lm=...)``" kararının uygulaması.

DSPy 2.5.40+ (3.x dahil) gerekçesi:
    DSPy artık ``contextvars`` ile global LM ayarını tutuyor.
    LangGraph her node'u ``contextvars.copy_context().run(...)`` ile
    yürüttüğü için, ``solve()`` içinde yapılan global
    ``dspy.configure(lm=...)`` çağrısı node'lara propagate olmuyor
    (özellikle async/threaded path'lerde). Çözüm: her node kendi DSPy
    çağrısını ``with dspy.context(lm=...)`` bloğuna alır. Burada iki
    yardımcı sunuyoruz ki her node'da tek satırlık değişiklikle bu
    pattern uygulanabilsin:

    * ``call_with_fast_lm(module, **kw)``      — triage/routing/report
    * ``call_with_reasoning_lm(module, **kw)`` — kod üretimi (Faz 3)

Ek olarak ``configure_default_lm()``'i tutuyoruz: API key
yokluğunu erken yakalar, bazı düz testlerde gerekli olabilir.
"""

from __future__ import annotations

from typing import Any

import dspy
import litellm

from iesolver.config import settings

# Gemini/Anthropic etc. don't support all OpenAI params (e.g. `seed`).
# LiteLLM will silently drop unsupported params instead of raising.
litellm.drop_params = True
from iesolver.observability.metrics import record_llm_usage

# -----------------------------------------------------------------------------
# Module-level singletons (lazy)
# -----------------------------------------------------------------------------
_fast_lm: dspy.LM | None = None
_reasoning_lm: dspy.LM | None = None
_configured: bool = False


def get_fast_lm() -> dspy.LM:
    """Return (and lazily build) the fast LM for triage/routing/reporting.

    temperature=0 ve seed, DESIGN_REVIEW §3.7 gereği tekrarlanabilirlik
    için her LM örneğine uygulanır. Seed provider'a iletilir; desteklemeyen
    provider'lar (Gemini) sessizce görmezden gelir.
    """
    global _fast_lm
    if _fast_lm is None:
        _fast_lm = dspy.LM(
            settings.fast_model,
            api_key=settings.require_api_key(),
            temperature=settings.temperature,
            seed=settings.lm_seed,
        )
    return _fast_lm


def get_reasoning_lm() -> dspy.LM:
    """Return (and lazily build) the heavy LM for code generation (Faz 3).

    Aynı temperature/seed politikası: deterministic output → tekrarlanabilir
    benchmark sonuçları (DESIGN_REVIEW §3.7).
    """
    global _reasoning_lm
    if _reasoning_lm is None:
        _reasoning_lm = dspy.LM(
            settings.reasoning_model,
            api_key=settings.require_api_key(),
            temperature=settings.temperature,
            seed=settings.lm_seed,
        )
    return _reasoning_lm


def configure_default_lm() -> None:
    """Configure DSPy global LM to the fast model. Idempotent.

    ``solve()`` her çağrıda bunu tetikler; ikinci çağrıda no-op olur.
    Node'lar yine de ``dspy.context()`` ile kendi LM scope'larını
    açar — bu çağrı yalnızca API key sanity check ve global default
    görevi görür.
    """
    global _configured
    if _configured:
        return
    dspy.configure(lm=get_fast_lm())
    _configured = True


def _invoke_and_record(lm: dspy.LM, module: Any, /, **kwargs: Any) -> Any:
    """Run ``module`` under ``lm``; push new history entries into metrics bucket."""
    before = len(lm.history)
    with dspy.context(lm=lm):
        result = module(**kwargs)
    # DSPy history entries added during this call — could be multiple
    # (ChainOfThought emits rationale + answer calls).
    record_llm_usage(lm.history[before:])
    return result


def call_with_fast_lm(module: Any, /, **kwargs: Any) -> Any:
    """Invoke a DSPy module under the fast LM context.

    DSPy module çağrısını ``dspy.context(lm=...)`` ile sarmalar;
    sonuç ``dspy.Prediction`` objesi olarak döner, context dışında
    okunabilir (Prediction LM'e bağımlı değil, sadece veri taşır).

    Ayrıca aktif ``@instrument`` scope'u varsa DSPy history delta'sını
    node metrics bucket'ına yazar (DESIGN_REVIEW §3.5).

    Parameters
    ----------
    module :
        Inşa edilmiş bir DSPy modülü (``ChainOfThought``, ``Predict``, ...).
    **kwargs :
        Modülün Signature'ının ``InputField``'ları.

    Returns
    -------
    Any
        ``dspy.Prediction`` — Signature'ın ``OutputField``'larını taşır.
    """
    return _invoke_and_record(get_fast_lm(), module, **kwargs)


def call_with_reasoning_lm(module: Any, /, **kwargs: Any) -> Any:
    """Invoke a DSPy module under the reasoning (heavy) LM context.

    Faz 3'te kod üretimi node'larında kullanılır. Token/maliyet metrikleri
    ``@instrument`` scope'unda ise otomatik toplanır.
    """
    return _invoke_and_record(get_reasoning_lm(), module, **kwargs)


def call_with_configured_lm(module: Any, /, *, fast_only: bool = False, **kwargs: Any) -> Any:
    """Invoke a DSPy module; route to fast or reasoning LM based on fast_only.

    A4 ablation (EVALUATION_PLAN §5): ``fast_only=True`` olduğunda reasoning
    LM çağrıları da fast LM'e yönlendirilir. Normal akışta ``fast_only=False``
    → ``call_with_reasoning_lm`` davranışı.

    Node'larda şu şekilde kullanılır::

        result = call_with_configured_lm(
            _module, fast_only=state.get("fast_only", False), **kwargs
        )

    Parameters
    ----------
    module :
        DSPy module instance.
    fast_only :
        When ``True``, delegate to :func:`call_with_fast_lm` regardless of
        the module's intended tier.
    **kwargs :
        Forwarded to the DSPy module's input fields.
    """
    if fast_only:
        return call_with_fast_lm(module, **kwargs)
    return call_with_reasoning_lm(module, **kwargs)


def reset_lm_cache() -> None:
    """Drop cached LM instances. Used in tests and when API key rotates."""
    global _fast_lm, _reasoning_lm, _configured
    _fast_lm = None
    _reasoning_lm = None
    _configured = False


__all__ = [
    "call_with_configured_lm",
    "call_with_fast_lm",
    "call_with_reasoning_lm",
    "configure_default_lm",
    "get_fast_lm",
    "get_reasoning_lm",
    "reset_lm_cache",
]
