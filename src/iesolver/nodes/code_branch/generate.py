"""
iesolver.nodes.code_branch.generate — Phase 4B.4 (ReAct Code Generator).

Eski ``CodeGeneratorAndExecutor``'ın LangGraph karşılığı. DSPy ReAct döngüsü
ile kod yazar, sandbox'ta çalıştırır, hata alırsa düzelterek tekrar dener.

DSPy modülü neden ReAct?
    "Kodu yazar, Tool'u kullanarak çalıştırır, hatayı görürse kendi kendini
    düzelterek tekrar dener (Human-in-the-loop gerektirmeden)."
    Bu, makalede "Autonomous Error Recovery" argümanının somut karşılığıdır.

Tool entegrasyonu:
    Eski kodda ``PythonREPL`` tool kullanılıyordu. Burada sandbox runner'ı
    DSPy tool formatına adapte ediyoruz: DSPy ReAct bir tool'u ``Callable``
    olarak alır, tool çıktısını observation olarak okur, sonraki adımı planlar.

LM: call_with_reasoning_lm
    Eski kodda bu aşamada "Pro modele geçiş" yapılıyordu. Yeni mimaride
    call_with_reasoning_lm ile aynı etki, ama thread-safe.

Retry mantığı:
    DSPy ReAct ``max_iters`` parametresiyle kendi içinde retry yapar.
    Dış retry (4B.5 → 4B.4) ise validate_node'un is_valid=False dönmesi
    durumunda graph.py'daki conditional edge tarafından yönetilir.
"""

from __future__ import annotations

import dspy

from iesolver.lm import call_with_configured_lm
from iesolver.sandbox.runner import RunResult, run_code
from iesolver.signatures import ReActCodeSignature
from iesolver.state import SolverState


def _python_executor(code: str) -> str:
    """DSPy tool: execute Python code in sandbox, return stdout or error.

    DSPy ReAct bir tool'u ``(input: str) -> str`` imzasıyla çağırır.
    Bu fonksiyon ``run_code`` ile sandbox'ı köprüler; çıktıyı string
    olarak döndürür ki ReAct observation olarak okuyabilsin.
    """
    result: RunResult = run_code(code)
    if result.success:
        return result.stdout or "(no output)"
    return (
        f"ERROR (exit={result.exit_code}):\n"
        f"{result.stderr or result.error_summary}"
    )


# DSPy ReAct: Signature + tool listesi + max iterasyon
# max_iters = settings.max_retries (3): her iterasyonda
# "düşün → araç çağır → gözlemle" döngüsü.
_react = dspy.ReAct(
    ReActCodeSignature,
    tools=[_python_executor],
    max_iters=3,
)


def generate_node(state: SolverState) -> SolverState:
    """Write and iteratively debug code using ReAct + sandbox.

    Reads
    -----
    essential_prompt, target_algorithm, target_library,
    library_specific_constraints, code_output_spec

    Writes
    ------
    final_code, execution_result
    """
    result = call_with_configured_lm(
        _react,
        fast_only=state.get("fast_only", False),
        essential_prompt=state.get("essential_prompt", "") or "",
        target_algorithm=state.get("target_algorithm", "") or "",
        target_library=state.get("target_library", "") or "",
        library_specific_constraints=state.get("library_specific_constraints", "") or "",
        code_output_spec=state.get("code_output_spec", "") or "",
    )

    return {
        "final_code": result.final_working_code,
        "execution_result": result.execution_result,
    }
