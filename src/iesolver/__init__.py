"""
iesolver — end-to-end LLM-driven solver for industrial engineering problems.

Public API
----------
::

    from iesolver import solve

    result = solve("EOQ for D=10000, S=50, H=2?")
    print(result["executive_summary"])

Faz 1'de ``solve()`` dummy node'lar üzerinden uçtan uca akar; ardından
Faz 2'de DSPy çağrıları, Faz 3'te kod motoru ve Faz 5'te 3-formatlı rapor
derlemesi devreye girer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4
from langgraph.types import Command

from iesolver.config import settings
from iesolver.graph import build_graph, open_checkpointer
from iesolver.lm import get_fast_lm, get_reasoning_lm
from iesolver.report import ReportWriter, write_report
from iesolver.sandbox.runner import RunResult, run_code
from iesolver.state import DataBundle, SolverState, empty_state

__version__ = "0.1.0"


def solve(
    prompt: str,
    data_path: Path | str | None = None,
    *,
    auto_mode: bool = False,
    thread_id: str | None = None,
    checkpoint_db: Path | None = None,
    # Ablation flags (EVALUATION_PLAN §5)
    enable_refiner: bool = True,
    enable_validator_retry: bool = True,
    fast_only: bool = False,
) -> SolverState:
    """Run the iesolver workflow end-to-end.

    Parameters
    ----------
    prompt :
        Natural-language problem description.
    data_path :
        Optional single data file (``.csv`` / ``.xlsx`` / ``.sqlite``).
    auto_mode :
        When ``True``, missing-info clarifications do **not** pause the graph
        via ``interrupt()``; instead a default assumption is logged into
        ``auto_assumptions_log`` and the pipeline proceeds. Required for
        batch/benchmark runs (NL4Opt, IndustryOR).
    thread_id :
        LangGraph thread id. Reuse the same id to **resume** a previous
        checkpointed run (replay after interrupt or failure). When ``None``,
        a fresh UUID is minted.
    checkpoint_db :
        Override path for the SqliteSaver database. Defaults to
        ``settings.checkpoint_db_path``.
    enable_refiner :
        **A1 ablation**. When ``False``, the PromptRefiner node is skipped;
        the problem statement routes directly to the strategy router without
        structural reformulation.
    enable_validator_retry :
        **A2 ablation**. When ``False``, a failed validation result does not
        trigger the code-retry loop; the pipeline falls through to the report
        node after the first attempt.
    fast_only :
        **A4 ablation**. When ``True``, all LLM calls — including those
        normally routed to the heavier reasoning model — use the fast LM.

    Returns
    -------
    SolverState
        Final merged state. In Faz 1 this includes the dummy
        ``technical_output`` / ``executive_summary`` / ``action_directives``
        triple.

    Examples
    --------
    >>> result = solve("hello world")
    >>> "executive_summary" in result
    True
    """
    settings.ensure_directories()

    seed = empty_state(
        raw_prompt=prompt,
        data_path=Path(data_path) if data_path is not None else None,
        auto_mode=auto_mode,
        enable_refiner=enable_refiner,
        enable_validator_retry=enable_validator_retry,
        fast_only=fast_only,
    )

    config: dict[str, Any] = {
        "configurable": {"thread_id": thread_id or str(uuid4())}
    }

    with open_checkpointer(checkpoint_db) as saver:
        graph = build_graph(checkpointer=saver)
        final_state = graph.invoke(seed, config=config)

    return final_state  # type: ignore[return-value]

def is_interrupted(state: SolverState) -> bool:
    """Return True iff the graph paused awaiting user clarification."""
    return "__interrupt__" in state  # type: ignore[operator]


__all__ = [
    "DataBundle",
    "ReportWriter",
    "RunResult",
    "SolverState",
    "__version__",
    "get_fast_lm",
    "get_reasoning_lm",
    "is_interrupted",
    "run_code",
    "solve",
    "write_report",
]
