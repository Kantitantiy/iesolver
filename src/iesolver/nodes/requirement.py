"""
iesolver.nodes.requirement — Phase 1 (Requirement Analyst).

Eski ``phase_1_triage.py``'nin LangGraph karşılığı. G-O-C framework
ile gereksinim çıkarımı; eksik bilgi varsa ``is_complete=False``
döndürerek human-in-loop interrupt'ı tetikler.

DSPy modülü neden Predict (CoT değil)?
    Eski ``phase_1_triage.py``'deki gerekçeyi koruyoruz:
    "Phase 1'de LLM'in uzun uzun düşünmesini istemiyoruz; katı,
    yapısal cevap istiyoruz." CoT bu görevde gereksiz token harcar.

Human-in-loop entegrasyonu:
    ``user_clarification`` state'te varsa cleaned_prompt'a karıştırılır.
    Aksi halde sonsuz döngü olurdu (LLM aynı eksikliği tekrar raporlar).

Tipli DSPy 3.x çıktıları (DESIGN_REVIEW §3.4):
    is_complete → native bool; missing_items ve constraints → list[str].
    Eski ``str → bool`` sanitization deseni kaldırıldı; DSPy adapter tip
    zorlamasını doğrudan yapar.
"""

from __future__ import annotations

import dspy

from iesolver.lm import call_with_fast_lm
from iesolver.observability.metrics import instrument
from iesolver.signatures import RequirementAnalystSignature
from iesolver.state import SolverState

_analyst = dspy.Predict(RequirementAnalystSignature)


@instrument("requirement")
def requirement_node(state: SolverState) -> SolverState:
    """Run G-O-C analysis; flag missing structural info.

    Reads
    -----
    cleaned_prompt, data_summary, user_clarification (opsiyonel)

    Writes
    ------
    is_complete, missing_items, explicit_goal, constraints, output_spec
    """
    cleaned = state.get("cleaned_prompt", "") or ""
    data_summary = state.get("data_summary", "No data provided.")
    clarification = state.get("user_clarification", "") or ""

    if clarification:
        effective_prompt = f"{cleaned}\n\n[USER CLARIFICATION]: {clarification}"
    else:
        effective_prompt = cleaned

    result = call_with_fast_lm(
        _analyst,
        cleaned_prompt=effective_prompt,
        data_summary=data_summary,
    )

    return {
        "is_complete": result.is_complete,
        "missing_items": result.missing_items,
        "explicit_goal": result.explicit_goal,
        "constraints": result.constraints,
        "output_spec": result.output_spec,
    }