"""
iesolver.nodes.code_branch.algo_select — Phase 4B.1 (Algorithm Selector).

Eski ``CodeEnginePlanner.select_algo`` adımının LangGraph karşılığı.
Hangi algoritma ve Python kütüphanesinin kullanılacağına karar verir;
kütüphane ile kullanıcı kısıtları arasındaki olası çatışmaları tespit eder.

DSPy modülü neden ChainOfThought?
    Eski ``phase_4b_code_engine.py``'daki gerekçeyi koruyoruz:
    "Algoritma seçerken modelin biraz mantık yürütmesini istiyoruz."
    Özellikle library-constraint uyuşmazlıklarını (ör. scikit-learn
    KMeans'ın Öklid metriği zorlaması) tespit etmek için akıl
    yürütme şart.

LM seçimi: call_with_reasoning_lm
    CODE path'teki tüm node'lar reasoning LM kullanır (eski kodda
    "Pro modele geçiş" adımı). AlgoSelector yanlış bir seçim yaparsa
    downstream tüm adımlar mahvolur — kaliteli muhakeme kritik.
"""

from __future__ import annotations

import dspy

from iesolver.lm import call_with_reasoning_lm
from iesolver.signatures import AlgoSelectorSignature
from iesolver.state import SolverState

_selector = dspy.ChainOfThought(AlgoSelectorSignature)


def algo_select_node(state: SolverState) -> SolverState:
    """Select algorithm and library; flag constraint-library conflicts.

    Reads
    -----
    essential_prompt, problem_type, data_summary, strict_constraints

    Writes
    ------
    target_algorithm, target_library
    Internal: adaptation_notes (passed via state for constraint_adapt)
    """
    result = call_with_reasoning_lm(
        _selector,
        essential_prompt=state.get("essential_prompt", "") or "",
        problem_type=state.get("problem_type", "") or "",
        data_summary=state.get("data_summary", "No data provided."),
        strict_constraints=state.get("strict_constraints", "") or "",
    )

    return {
        "target_algorithm": result.target_algorithm,
        "target_library": result.target_library,
        # adaptation_notes SolverState'te deklareli değil; code_branch
        # içi geçici alan. __init__.py'daki pipeline bunu sırayla iletir.
        "_adaptation_notes": result.adaptation_notes,  # type: ignore[typeddict-unknown-key]
    }
