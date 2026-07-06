"""
iesolver.nodes.route — Phase 3 (Strategy Router) — Bifurcation Logic.

Eski ``phase_3_router.py``'nin LangGraph karşılığı.

DSPy modülü neden ChainOfThought?
    Eski ``phase_3_router.py``'deki gerekçeyi koruyoruz:
    "LLM önce rationale (gerekçe) yazacak, sonra execution_path seçecek."
    Bu, makalenin observability argümanına da hizmet eder.

Tipli DSPy 3.x çıktısı (DESIGN_REVIEW §3.4):
    execution_path artık ``Literal["CODE","NO_CODE"]``; DSPy adapter binary
    kararı garanti eder. Eski normalize heuristic'i ("code_based",
    "CODE_REQUIRED" gibi varyantları CODE'a çekme) kaldırıldı.

Self-Consistency (A6 ablasyonu — CLAUDE.md Düzeltme #5):
    execution_path pipeline'daki en yüksek blast-radius'lu tekil karardır:
    yanlış dallanma downstream'deki her adımı geçersiz kılar. Tek örnekleme
    yerine ``self_consistency_router=True`` olduğunda 3 bağımsız örnekleme
    alınır ve ``dspy.majority`` ile çoğunluk oyu uygulanır (Wang et al. 2022,
    Self-Consistency). Varsayılan davranış (``False``) tek örneklemedir —
    maliyet artışı yalnızca bu bayrak açıkken oluşur.
"""

from __future__ import annotations

from collections import Counter

import dspy

from iesolver.lm import call_with_fast_lm
from iesolver.observability.metrics import instrument
from iesolver.signatures import StrategyRouterSignature
from iesolver.state import SolverState

_router = dspy.ChainOfThought(StrategyRouterSignature)

# A6: kaç bağımsız örnekleme alınacağı. DSPy, n>1 + temperature<=0.15
# olduğunda örnekler arası çeşitlilik için otomatik olarak temperature=0.7'ye
# çıkarır (dspy.predict.Predict._forward_preprocess) — ayrıca ayarlamaya gerek yok.
_SELF_CONSISTENCY_VOTES = 3


@instrument("route")
def route_node(state: SolverState) -> SolverState:
    """Decide CODE vs NO_CODE; record reasoning framework + rationale.

    Reads
    -----
    essential_prompt, problem_type, strict_constraints, self_consistency_router

    Writes
    ------
    execution_path, reasoning_framework, rationale
    Additionally when self_consistency_router=True: router_vote_summary
    """
    essential = state.get("essential_prompt", "") or ""
    problem_type = state.get("problem_type", "") or ""
    constraints = state.get("strict_constraints", "") or ""

    if state.get("self_consistency_router", False):
        raw = call_with_fast_lm(
            _router,
            essential_prompt=essential,
            problem_type=problem_type,
            strict_constraints=constraints,
            config={"n": _SELF_CONSISTENCY_VOTES},
        )
        votes = Counter(raw.completions["execution_path"])
        result = dspy.majority(raw, field="execution_path")

        return {
            "execution_path": result.execution_path,
            "reasoning_framework": result.reasoning_framework,
            "rationale": result.rationale,
            "router_vote_summary": "/".join(f"{path}:{n}" for path, n in votes.most_common()),
        }

    result = call_with_fast_lm(
        _router,
        essential_prompt=essential,
        problem_type=problem_type,
        strict_constraints=constraints,
    )

    return {
        "execution_path": result.execution_path,
        "reasoning_framework": result.reasoning_framework,
        "rationale": result.rationale,
    }