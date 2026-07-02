"""
ie_eval.baselines — Single-shot LLM comparison targets (EVALUATION_PLAN §4).

Amaç: iesolver pipeline'ının katma değerini izole etmek. Aynı LM'e "problemi
çöz, kod yaz" tek prompt gönderir; kodu sandbox'ta çalıştırır; sonucu döndürür.
Ne PromptRefiner, ne bifurcation, ne validate/retry, ne sensitivity — dümdüz
tek atış.

İki varyant:
    single_shot_solve(prompt)                     — çıplak tek atış
    single_shot_solve(prompt, use_cot=True)       — chain-of-thought promptla

CoT baseline'ı EVAL_PLAN §4.2 gereği ayrı bir baseline: pipeline kazancının
"sadece CoT etkisi olmadığını" göstermek için gerekli.

Sözleşme:
    Fonksiyon imzası ``runner.run_one``'un beklediği ``solve_fn`` ile
    uyumlu: (prompt, data_path=None, auto_mode=True, thread_id=None)
    → state dict.

    State dict şeması iesolver.SolverState'in kritik alanlarını taşır:
        - execution_result   → numerical_match için
        - final_code         → hata analizi için
        - execution_path     → "CODE" (bilgi)
        - metrics            → {"single_shot": {...}} — tek slice

Sandbox ve LM iesolver'ın **public API**'sinden alınır — iç modüllere
dokunma yasağı korunur.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Callable

from iesolver import get_fast_lm, run_code as _default_run_code


# =============================================================================
# Prompt şablonları
# =============================================================================
_BASE_PROMPT_TEMPLATE = """\
Solve the following operations research / industrial engineering problem.
Write a complete, self-contained Python script that computes the answer
and prints it clearly labeled as the FINAL numerical result on the last line.
You may use standard libraries such as pulp, scipy, numpy, or math.
Return ONLY the Python code — no explanation, no markdown fences.

PROBLEM:
{prompt}
"""

_COT_PROMPT_TEMPLATE = """\
Solve the following operations research / industrial engineering problem.

Think step by step (write your reasoning as Python # comments inside the code):
  1. Identify decision variables
  2. Formulate the objective function
  3. List the constraints
  4. Choose an appropriate solver (pulp, scipy, numpy, or closed-form)

Then write a complete, self-contained Python script that computes the answer
and prints it clearly labeled as the FINAL numerical result on the last line.
Return ONLY the Python code (with your reasoning as comments) — no markdown fences.

PROBLEM:
{prompt}
"""


# =============================================================================
# Kod çıkarımı — LLM markdown fence sararsa
# =============================================================================
_CODE_FENCE_RE = re.compile(
    r"```(?:python|py)?\s*\n?(.*?)\n?```",
    re.DOTALL | re.IGNORECASE,
)


def _extract_code(text: str) -> str:
    """Strip markdown code fences if present; return raw code otherwise."""
    match = _CODE_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


# =============================================================================
# Ana baseline fonksiyonu
# =============================================================================
def single_shot_solve(
    prompt: str,
    *,
    use_cot: bool = False,
    data_path: Path | None = None,   # noqa: ARG001  — solve_fn uyumu için; kullanılmıyor
    auto_mode: bool = True,           # noqa: ARG001  — solve_fn uyumu için; kullanılmıyor
    thread_id: str | None = None,     # noqa: ARG001  — checkpoint yok
    lm: Any = None,
    run_code_fn: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """One-shot LLM → code → sandbox → state.

    Parameters
    ----------
    prompt :
        Natural-language problem statement (same format iesolver.solve gets).
    use_cot :
        True → chain-of-thought prompt template. EVAL_PLAN §4.2 baseline.
    data_path, auto_mode, thread_id :
        solve_fn ABI uyumluluğu için kabul edilir; baseline'da anlamsız,
        yok sayılır. (Data file support requires DataBundle — pipeline özelliği.)
    lm :
        DSPy LM örneği. Testlerde mock; production'da None → get_fast_lm().
    run_code_fn :
        Sandbox executor. Testlerde mock; production'da None → iesolver.run_code.

    Returns
    -------
    dict
        SolverState-şekilli minimum sözleşme: execution_result, final_code,
        execution_path, metrics.
    """
    lm = lm if lm is not None else get_fast_lm()
    run_code_fn = run_code_fn if run_code_fn is not None else _default_run_code

    template = _COT_PROMPT_TEMPLATE if use_cot else _BASE_PROMPT_TEMPLATE
    full_prompt = template.format(prompt=prompt)

    # ---- LLM call ----
    t_start = time.perf_counter()
    hist_before = len(getattr(lm, "history", []))
    outputs = lm(prompt=full_prompt)

    raw_output = outputs[0] if outputs else ""
    if not isinstance(raw_output, str):
        raw_output = str(raw_output)

    code = _extract_code(raw_output)

    # ---- Sandbox ----
    run_result = run_code_fn(code)
    t_end = time.perf_counter()

    execution_result = run_result.stdout if getattr(run_result, "success", False) else ""

    # ---- Aggregate DSPy LM history delta ----
    history = getattr(lm, "history", [])
    new_entries = history[hist_before:]
    tokens_in = 0
    tokens_out = 0
    cost_usd = 0.0
    for entry in new_entries:
        usage = (entry.get("usage") if isinstance(entry, dict) else None) or {}
        tokens_in += int(usage.get("prompt_tokens", 0) or 0)
        tokens_out += int(usage.get("completion_tokens", 0) or 0)
        cost = entry.get("cost") if isinstance(entry, dict) else None
        if cost is not None:
            try:
                cost_usd += float(cost)
            except (TypeError, ValueError):
                pass

    error_class: str | None = None
    if not getattr(run_result, "success", False):
        # timed_out veya stderr — hata sınıfını kabaca ayır
        if getattr(run_result, "timed_out", False):
            error_class = "SandboxTimeout"
        else:
            error_class = "SandboxFailure"

    metrics_slice = {
        "latency_ms": round((t_end - t_start) * 1000.0, 2),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": round(cost_usd, 6),
        "llm_calls": len(new_entries),
        "invocations": 1,
        "error_class": error_class,
    }

    return {
        "raw_prompt": prompt,
        "final_code": code,
        "execution_result": execution_result,
        "execution_path": "CODE",
        "metrics": {"single_shot": metrics_slice},
    }


# =============================================================================
# CoT convenience
# =============================================================================
def single_shot_cot_solve(prompt: str, **kwargs: Any) -> dict[str, Any]:
    """Chain-of-thought variant (EVAL_PLAN §4.2). Equivalent to
    ``single_shot_solve(prompt, use_cot=True, **kwargs)``.
    """
    return single_shot_solve(prompt, use_cot=True, **kwargs)


__all__ = ["single_shot_cot_solve", "single_shot_solve"]
