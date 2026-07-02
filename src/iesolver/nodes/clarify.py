"""
iesolver.nodes.clarify — human-in-loop veya otomatik varsayım node'u.

İki mod:
    - Interactive (varsayılan): ``interrupt()`` ile grafiği durdurur ve
      kullanıcı cevabını bekler. UI/CLI bunu ``Command(resume=...)`` ile
      sürdürür.
    - auto_mode (batch/benchmark): Interrupt YERINE varsayılan bir
      varsayım metni üretir, ``auto_assumptions_log``'a yazar ve
      ``is_complete=True`` ile döngüyü kapatır. NL4Opt/IndustryOR gibi
      tam otomatik değerlendirme koşuları için zorunlu (DESIGN_REVIEW §3.1).

Not: auto_mode altında grafik conditional edge sayesinde ``requirement``'a
geri dönmez, doğrudan ``refine``'a ilerler; aksi halde LLM'in aynı
eksikliği tekrar raporlaması sonsuz döngü oluştururdu.
"""

from __future__ import annotations

from langgraph.types import interrupt

from iesolver.observability.metrics import instrument
from iesolver.state import SolverState


@instrument("clarify")
def clarify_node(state: SolverState) -> SolverState:
    """Collect user clarification via interrupt, or auto-assume in batch mode."""
    missing_list = state.get("missing_items", []) or []
    # DSPy 3.x tipli çıktı sonrası missing_items list[str] (DESIGN_REVIEW §3.4).
    missing_text = "; ".join(missing_list) if missing_list else "<none reported>"

    if state.get("auto_mode", False):
        assumption = (
            "AUTO_MODE: proceeding with reasonable defaults. "
            f"Missing items were: {missing_text}"
        )
        prior_log = state.get("auto_assumptions_log", []) or []
        return {
            "user_clarification": assumption,
            "is_complete": True,
            "auto_assumptions_log": [*prior_log, assumption],
        }

    payload = {
        "type": "missing_info",
        "missing_items": missing_list,
        "explicit_goal_so_far": state.get("explicit_goal", ""),
        "message": "Additional information is required to proceed.",
    }
    user_answer = interrupt(payload)
    return {"user_clarification": str(user_answer)}