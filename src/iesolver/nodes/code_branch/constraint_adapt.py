"""
iesolver.nodes.code_branch.constraint_adapt — Phase 4B.2 (Constraint Adapter).

Eski ``CodeEnginePlanner.adapt_constraints`` adımının LangGraph karşılığı.
Kullanıcının soyut kısıtlarını seçilen kütüphanenin API gerçeklikleriyle
uyumlu hale getirir.

DSPy modülü neden Predict (CoT değil)?
    Eski ``phase_4b_code_engine.py``'daki gerekçeyi koruyoruz:
    "Kısıt adaptasyonunda doğrudan deterministik sonuç istiyoruz."
    adaptation_notes zaten AlgoSelector'dan geliyor; burada sadece
    metin dönüşümü var, ek muhakeme gerekmiyor.

LM seçimi: call_with_reasoning_lm
    CODE branch'inde tutarlılık için reasoning LM.
    Predict + reasoning_lm = hızlı + kaliteli dönüşüm.
"""

from __future__ import annotations

import dspy

from iesolver.lm import call_with_reasoning_lm
from iesolver.signatures import ConstraintAdapterSignature
from iesolver.state import SolverState

_adapter = dspy.Predict(ConstraintAdapterSignature)


def constraint_adapt_node(state: SolverState) -> SolverState:
    """Rewrite constraints to match the target library's API.

    Reads
    -----
    strict_constraints, target_library, _adaptation_notes (internal)

    Writes
    ------
    library_specific_constraints
    """
    result = call_with_reasoning_lm(
        _adapter,
        strict_constraints=state.get("strict_constraints", "") or "",
        target_library=state.get("target_library", "") or "",
        adaptation_notes=state.get("_adaptation_notes", "None") or "None",  # type: ignore[typeddict-item]
    )

    return {
        "library_specific_constraints": result.library_specific_constraints,
    }
