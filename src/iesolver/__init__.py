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

from collections.abc import Generator
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
    self_consistency_router: bool = False,
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
    self_consistency_router :
        **A6 ablation**. When ``True``, the CODE/NO_CODE routing decision is
        sampled 3 times and resolved by majority vote instead of a single
        completion, trading 3x router cost for lower variance on the single
        highest-blast-radius decision in the pipeline.

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
        self_consistency_router=self_consistency_router,
    )

    config: dict[str, Any] = {
        "configurable": {"thread_id": thread_id or str(uuid4())}
    }

    with open_checkpointer(checkpoint_db) as saver:
        graph = build_graph(checkpointer=saver)
        final_state = graph.invoke(seed, config=config)

    return final_state  # type: ignore[return-value]

def stream_solve(
    prompt: str,
    data_path: Path | str | None = None,
    *,
    auto_mode: bool = True,
    thread_id: str | None = None,
    checkpoint_db: Path | None = None,
    enable_refiner: bool = True,
    enable_validator_retry: bool = True,
    fast_only: bool = False,
    self_consistency_router: bool = False,
) -> Generator[tuple[str, dict[str, Any]], None, None]:
    """Like solve(), but yields (node_name, partial_state) after each node completes.

    Her node tamamlandığında o node'un yazdığı alanları anında döndürür.
    Aşama aşama ilerlemeyi terminalde görmek için kullanın.

    Usage::

        for node_name, partial in stream_solve("EOQ problemi..."):
            print(f"✓ {node_name}: {list(partial.keys())}")

    Parameters
    ----------
    prompt, data_path, auto_mode, thread_id, checkpoint_db :
        solve() ile aynı.

    Yields
    ------
    (node_name, partial_state) :
        node_name  — tamamlanan LangGraph node'unun adı
        partial_state — o node'un state'e yazdığı alanlar (sadece o node'un çıktısı)
    """
    settings.ensure_directories()
    seed = empty_state(
        raw_prompt=prompt,
        data_path=Path(data_path) if data_path is not None else None,
        auto_mode=auto_mode,
        enable_refiner=enable_refiner,
        enable_validator_retry=enable_validator_retry,
        fast_only=fast_only,
        self_consistency_router=self_consistency_router,
    )
    config: dict[str, Any] = {
        "configurable": {"thread_id": thread_id or str(uuid4())}
    }
    with open_checkpointer(checkpoint_db) as saver:
        graph = build_graph(checkpointer=saver)
        for chunk in graph.stream(seed, config=config, stream_mode="updates"):
            for node_name, partial_state in chunk.items():
                yield node_name, partial_state


def show_llm_history(n: int = 3) -> None:
    """Print the last n LLM calls: prompts sent and responses received.

    LLM'e gönderilen tam prompt'ları ve alınan yanıtları terminale basar.
    Her aşamadan sonra nelerin gönderilip alındığını görmek için kullanın.

    Parameters
    ----------
    n :
        Kaç LLM çağrısı gösterilsin (en yeniden geriye doğru). Default: 3.

    Usage::

        state = solve("EOQ problemi...")
        show_llm_history(n=5)   # son 5 çağrıyı göster
    """
    fast_lm = get_fast_lm()
    reasoning_lm = get_reasoning_lm()

    # Her ikisinin geçmişini birleştir, zamana göre sırala
    combined: list[dict[str, Any]] = []
    combined.extend(fast_lm.history or [])
    combined.extend(reasoning_lm.history or [])

    if not combined:
        print("Henüz LLM çağrısı yapılmadı.")
        return

    recent = combined[-n:]
    sep = "─" * 70

    for i, entry in enumerate(recent, 1):
        model = entry.get("model", "?")
        messages = entry.get("messages", [])
        response = entry.get("response", {})
        usage = entry.get("usage") or {}

        print(f"\n{sep}")
        print(f"  [{i}/{len(recent)}] Model: {model}")
        print(f"  Tokens: {usage.get('prompt_tokens', '?')} giriş / "
              f"{usage.get('completion_tokens', '?')} çıkış")
        print(sep)

        # Gönderilen mesajlar
        for msg in messages:
            role = msg.get("role", "?").upper()
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
            print(f"\n  ── {role} ──")
            print(f"  {str(content)[:600]}{'...' if len(str(content)) > 600 else ''}")

        # Alınan yanıt
        choices = response.get("choices", []) if isinstance(response, dict) else []
        if choices:
            reply = choices[0].get("message", {}).get("content", "")
            print(f"\n  ── ASSISTANT ──")
            print(f"  {str(reply)[:800]}{'...' if len(str(reply)) > 800 else ''}")

    print(f"\n{sep}\n")


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
    "show_llm_history",
    "solve",
    "stream_solve",
    "write_report",
]
