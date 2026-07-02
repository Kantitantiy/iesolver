"""
iesolver.graph — LangGraph workflow assembly.

Faz 4 topolojisi (Faz 3'ü genişletir):
─────────────────────────────────────────

    START → intake → requirement ──[eksik]──▶ clarify ─┬─[interactive]→ requirement (loop)
                          │ [tam]                        │
                          ▼                              └─[auto_mode]──▶ refine
                        refine ◀────────────────────────────────────────────┘
                          │
                          ▼
                        route ──[NO_CODE]──▶ chain_branch ──────────────────────┐
                          │                                                       │
                          └──[CODE]──▶ code_branch                               │
                                            │                                    │
                                            ▼                                    │
                                        validate                                 │
                                       /       \                                 │
                            [geçersiz+retry]  [geçerli]   [geçersiz+max_retry]  │
                                  │               │              │               │
                            code_branch     sensitivity       report ◀───────────┘
                            (max 3 kez)          │               ▲
                                                 ▼               │
                                            artifacts ───────────┘
                                                                  ▼
                                                                 END

Retry döngüsü (Plan §5 Faz 3):
    ``validate`` → is_valid=False AND retry_count < MAX_RETRIES
    → ``code_branch``'e dön (code_branch retry_count'u artırır).
    3 başarısız denemeden sonra zorunlu olarak rapor yazılır.
    Bu, makalede "Autonomous Error Recovery with Bounded Retries"
    olarak konumlanabilir.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from iesolver.config import settings
from iesolver.nodes import (
    artifacts_node,
    chain_branch_node,
    clarify_node,
    code_branch_node,
    intake_node,
    refine_node,
    report_node,
    requirement_node,
    route_node,
    sensitivity_node,
    validate_node,
)
from iesolver.state import SolverState

MAX_RETRIES = 3


# =============================================================================
# Conditional edge predicates
# =============================================================================
def _route_after_requirement(state: SolverState) -> str:
    """Complete → refine (or route if A1); incomplete → clarify."""
    if not state.get("is_complete"):
        return "clarify"
    # A1 ablation: enable_refiner=False → PromptRefiner node'u atla
    if not state.get("enable_refiner", True):
        return "route"
    return "refine"


def _route_after_clarify(state: SolverState) -> str:
    """Auto-mode → refine/route (döngüyü kapat); interactive → requirement (yeniden analiz).

    auto_mode altında clarify_node zaten ``is_complete=True`` yazar; grafiği
    requirement'a geri döndürmek gereksiz LLM çağrısı üretir ve — daha kötüsü —
    LLM aynı eksikliği tekrar raporlarsa sonsuz döngü oluşur.

    A1 ablation: enable_refiner=False olduğunda auto_mode path "refine" değil
    "route"'a gider (PromptRefiner atlanır).
    """
    if not state.get("auto_mode", False):
        return "requirement"
    # A1 ablation: auto_mode=True iken de refiner atlanabilir
    if not state.get("enable_refiner", True):
        return "route"
    return "refine"


def _route_after_router(state: SolverState) -> str:
    return "chain_branch" if state.get("execution_path") == "NO_CODE" else "code_branch"


def _route_after_validate(state: SolverState) -> str:
    """Three-way branch after validation.

    retry_count code_branch_node tarafından artırılır (her geçişte +1).

    * is_valid=False + retry kalan + A2 aktif  → code_branch (tekrar dene)
    * is_valid=True                             → sensitivity
    * is_valid=False + max retry / A2 kapalı   → report
    """
    is_valid = state.get("is_valid", True)
    retry_count = int(state.get("retry_count", 0) or 0)
    # A2 ablation: enable_validator_retry=False → retry döngüsü devre dışı
    enable_retry = state.get("enable_validator_retry", True)

    if not is_valid and enable_retry and retry_count < MAX_RETRIES:
        return "code_branch"
    if is_valid:
        return "sensitivity"
    return "report"  # geçersiz + (limit aşıldı veya A2 kapalı)


# =============================================================================
# Graph builder
# =============================================================================
def build_graph(checkpointer: SqliteSaver | None = None):
    """Compile and return the iesolver LangGraph (Faz 3)."""
    builder: StateGraph = StateGraph(SolverState)

    # ---- Nodes ----
    builder.add_node("intake", intake_node)
    builder.add_node("requirement", requirement_node)
    builder.add_node("clarify", clarify_node)
    builder.add_node("refine", refine_node)
    builder.add_node("route", route_node)
    builder.add_node("chain_branch", chain_branch_node)
    builder.add_node("code_branch", code_branch_node)
    builder.add_node("validate", validate_node)
    builder.add_node("sensitivity", sensitivity_node)
    builder.add_node("artifacts", artifacts_node)
    builder.add_node("report", report_node)

    # ---- Edges ----
    builder.add_edge(START, "intake")
    builder.add_edge("intake", "requirement")

    builder.add_conditional_edges(
        "requirement",
        _route_after_requirement,
        # A1: "route" eklendi (enable_refiner=False hali)
        {"refine": "refine", "clarify": "clarify", "route": "route"},
    )
    builder.add_conditional_edges(
        "clarify",
        _route_after_clarify,
        # A1: "route" eklendi (enable_refiner=False + auto_mode=True hali)
        {"requirement": "requirement", "refine": "refine", "route": "route"},
    )
    builder.add_edge("refine", "route")

    builder.add_conditional_edges(
        "route",
        _route_after_router,
        {"chain_branch": "chain_branch", "code_branch": "code_branch"},
    )

    # NO_CODE path
    builder.add_edge("chain_branch", "report")

    # CODE path: generate → validate → (retry | sensitivity | report)
    builder.add_edge("code_branch", "validate")
    builder.add_conditional_edges(
        "validate",
        _route_after_validate,
        {"code_branch": "code_branch", "sensitivity": "sensitivity", "report": "report"},
    )

    # Faz 4: sensitivity → artifacts → report
    builder.add_edge("sensitivity", "artifacts")
    builder.add_edge("artifacts", "report")

    builder.add_edge("report", END)

    return (
        builder.compile(checkpointer=checkpointer)
        if checkpointer is not None
        else builder.compile()
    )


# =============================================================================
# Checkpointer context
# =============================================================================
@contextmanager
def open_checkpointer(db_path: Path | None = None) -> Iterator[SqliteSaver]:
    """Yield a SqliteSaver; guarantees connection cleanup on exit."""
    path = Path(db_path) if db_path is not None else settings.checkpoint_db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(path)) as saver:
        yield saver


__all__ = ["build_graph", "open_checkpointer"]
