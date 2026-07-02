"""
Single-shot baseline tests (EVALUATION_PLAN §4).

LLM ÇAĞRISI YAPMAZ. LM ve sandbox mock'lanır; sözleşme + kod ayrıştırma
+ runner entegrasyonu doğrulanır.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ie_eval.baselines import _extract_code, single_shot_cot_solve, single_shot_solve
from ie_eval.problem import GroundTruth, Problem
from ie_eval.runner import run_one


# =============================================================================
# Fake LM ve sandbox
# =============================================================================
class FakeLM:
    """DSPy LM'i taklit eden minimum örnek.

    __call__(prompt=...) → list[str]  (DSPy sözleşmesi)
    history: list[dict]                (usage / cost girdileri)
    """

    def __init__(self, response: str, usage: dict | None = None, cost: float | None = None):
        self._response = response
        self._usage = usage or {"prompt_tokens": 50, "completion_tokens": 20}
        self._cost = cost
        self.history: list[dict] = []
        self.calls: list[str] = []

    def __call__(self, prompt: str | None = None, messages=None, **kwargs) -> list[str]:
        self.calls.append(prompt or "")
        self.history.append({
            "prompt": prompt,
            "usage": self._usage,
            "cost": self._cost,
        })
        return [self._response]


@dataclass
class FakeRunResult:
    success: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    timed_out: bool = False
    error_summary: str = ""


def _make_run_code_fn(result: FakeRunResult):
    seen_code: list[str] = []
    def _run(code: str) -> FakeRunResult:
        seen_code.append(code)
        return result
    _run.seen = seen_code   # test için sızıntı
    return _run


# =============================================================================
# _extract_code — markdown fence stripping
# =============================================================================
def test_extract_code_plain():
    assert _extract_code("print(1)") == "print(1)"


def test_extract_code_python_fence():
    text = "```python\nprint(42)\n```"
    assert _extract_code(text) == "print(42)"


def test_extract_code_bare_fence():
    text = "```\nx = 1\n```"
    assert _extract_code(text) == "x = 1"


def test_extract_code_preamble_stripped():
    """LLM önce açıklama sonra kod verirse — sadece kodu alalım."""
    text = "Here is the code:\n```python\ncost = 100\nprint(cost)\n```\nDone."
    assert _extract_code(text) == "cost = 100\nprint(cost)"


# =============================================================================
# single_shot_solve — temel akış
# =============================================================================
def test_single_shot_base_prompt_no_cot():
    lm = FakeLM(response="print('Q = 707.1')")
    run_fn = _make_run_code_fn(FakeRunResult(success=True, stdout="Q = 707.1"))

    state = single_shot_solve("Compute EOQ", lm=lm, run_code_fn=run_fn)

    assert state["execution_result"] == "Q = 707.1"
    assert state["final_code"] == "print('Q = 707.1')"
    assert state["execution_path"] == "CODE"
    assert "single_shot" in state["metrics"]

    slice_ = state["metrics"]["single_shot"]
    assert slice_["tokens_in"] == 50
    assert slice_["tokens_out"] == 20
    assert slice_["llm_calls"] == 1
    assert slice_["error_class"] is None

    # Kullanılan prompt CoT şablonu OLMAMALI
    assert "step by step" not in lm.calls[0].lower()
    assert "Compute EOQ" in lm.calls[0]


def test_single_shot_cot_variant():
    lm = FakeLM(response="# reasoning here\nprint(42)")
    run_fn = _make_run_code_fn(FakeRunResult(success=True, stdout="42"))

    state = single_shot_solve("problem X", use_cot=True, lm=lm, run_code_fn=run_fn)
    assert state["execution_result"] == "42"

    # CoT şablonu kullanılmalı
    assert "step by step" in lm.calls[0].lower()
    assert "decision variables" in lm.calls[0].lower()


def test_single_shot_cot_convenience():
    lm = FakeLM(response="print(1)")
    run_fn = _make_run_code_fn(FakeRunResult(success=True, stdout="1"))

    state = single_shot_cot_solve("problem", lm=lm, run_code_fn=run_fn)
    assert "step by step" in lm.calls[0].lower()
    assert state["execution_result"] == "1"


# =============================================================================
# Sandbox hataları — error_class doğru sınıflansın
# =============================================================================
def test_sandbox_failure_populates_error_class():
    lm = FakeLM(response="raise ValueError('x')")
    run_fn = _make_run_code_fn(FakeRunResult(success=False, stderr="ValueError: x"))

    state = single_shot_solve("p", lm=lm, run_code_fn=run_fn)
    assert state["execution_result"] == ""
    assert state["metrics"]["single_shot"]["error_class"] == "SandboxFailure"


def test_sandbox_timeout_labelled_separately():
    lm = FakeLM(response="while True: pass")
    run_fn = _make_run_code_fn(FakeRunResult(success=False, timed_out=True,
                                              stderr="Process killed"))

    state = single_shot_solve("p", lm=lm, run_code_fn=run_fn)
    assert state["metrics"]["single_shot"]["error_class"] == "SandboxTimeout"


# =============================================================================
# Markdown fence sarılmış LLM çıktısı sandbox'a temiz kod olarak gitmeli
# =============================================================================
def test_fenced_llm_output_is_unwrapped_before_sandbox():
    lm = FakeLM(response="```python\nprint('clean')\n```")
    run_fn = _make_run_code_fn(FakeRunResult(success=True, stdout="clean"))

    single_shot_solve("p", lm=lm, run_code_fn=run_fn)

    executed_code = run_fn.seen[-1]
    assert executed_code == "print('clean')"


# =============================================================================
# Cost aggregation
# =============================================================================
def test_cost_summed_from_lm_history():
    lm = FakeLM(response="print(1)", cost=0.0012)
    run_fn = _make_run_code_fn(FakeRunResult(success=True, stdout="1"))

    state = single_shot_solve("p", lm=lm, run_code_fn=run_fn)
    assert state["metrics"]["single_shot"]["cost_usd"] == 0.0012


def test_cost_missing_is_zero_not_crash():
    lm = FakeLM(response="print(1)", cost=None)
    run_fn = _make_run_code_fn(FakeRunResult(success=True, stdout="1"))

    state = single_shot_solve("p", lm=lm, run_code_fn=run_fn)
    assert state["metrics"]["single_shot"]["cost_usd"] == 0.0


# =============================================================================
# Runner entegrasyonu — baseline solve_fn olarak geçirilebilmeli
# =============================================================================
def test_baseline_works_as_solve_fn_in_runner():
    lm = FakeLM(response="print('answer = 42')")
    run_fn = _make_run_code_fn(FakeRunResult(success=True, stdout="answer = 42"))

    def baseline_solve_fn(prompt, data_path=None, auto_mode=False, thread_id=None):
        return single_shot_solve(prompt, lm=lm, run_code_fn=run_fn)

    problem = Problem(
        id="p1",
        prompt="find answer",
        ground_truth=GroundTruth(objective_value=42.0, tolerance_rel=1e-3),
    )
    rec = run_one(problem, config_id="single_shot", solve_fn=baseline_solve_fn)

    assert rec.success
    assert rec.metrics.execution_rate
    assert rec.metrics.numerical_match
    assert rec.metrics.total_tokens_in == 50
    assert rec.metrics.node_count == 1
    # Baseline retry döngüsü yok
    assert rec.metrics.retry_count == 0


def test_baseline_ignores_data_path_and_auto_mode():
    """solve_fn ABI: data_path/auto_mode kabul edilmeli ama sessizce yok sayılmalı."""
    lm = FakeLM(response="print(0)")
    run_fn = _make_run_code_fn(FakeRunResult(success=True, stdout="0"))

    state = single_shot_solve(
        "p",
        data_path="/does/not/exist",   # noqa
        auto_mode=True,
        thread_id="ignored",
        lm=lm,
        run_code_fn=run_fn,
    )
    assert state["execution_result"] == "0"
