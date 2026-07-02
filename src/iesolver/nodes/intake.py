"""
iesolver.nodes.intake — Phase 0 (GateKeeper) — gerçek DSPy implementasyonu.

Eski ``phase_0_gatekeeper.py``'nın LangGraph karşılığı. İki iş yapar:

1. Veri dosyasını ``DataBundle``'a yükler (deterministik, LLM yok).
2. Ham prompt'u ``GateKeeperSignature`` üzerinden temizler (LLM).

DSPy modülü neden ChainOfThought?
    Eski ``phase_0_gatekeeper.py``'daki gerekçeyi koruyoruz:
    "Modelin ham prompt içindeki hangi kısmın gereksiz kod, hangi
    kısmın IE problemi olduğunu anlaması için kısa bir 'düşünme'
    payına ihtiyacı var."

LM bağlama:
    DSPy modülünün çağrısı ``call_with_fast_lm`` ile sarmalanır.
    Bu, ``with dspy.context(lm=...)`` pattern'inin tek-satırlık
    karşılığı; LangGraph contextvars izolasyonunu aşar (bkz. lm.py).
"""

from __future__ import annotations

import dspy

from iesolver.io.data_loader import load_data
from iesolver.lm import call_with_fast_lm
from iesolver.observability.metrics import instrument
from iesolver.signatures import GateKeeperSignature
from iesolver.state import SolverState

_gatekeeper = dspy.ChainOfThought(GateKeeperSignature)


@instrument("intake")
def intake_node(state: SolverState) -> SolverState:
    """Standardize the prompt and profile the (optional) data file.

    Reads
    -----
    raw_prompt, data_path

    Writes
    ------
    cleaned_prompt, data_bundle, data_summary
    """
    raw = state.get("raw_prompt", "") or ""
    data_path = state.get("data_path")

    # 1. Deterministik kısım: veri yüklemesi
    bundle = load_data(data_path)
    data_summary = bundle.summary()

    # 2. LLM kısmı: prompt sanitization (per-node LM context)
    prediction = call_with_fast_lm(_gatekeeper, raw_prompt=raw)
    cleaned = prediction.cleaned_prompt

    return {
        "cleaned_prompt": cleaned,
        "data_bundle": bundle,
        "data_summary": data_summary,
    }
