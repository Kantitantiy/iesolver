"""Runner + metrics extraction tests. iesolver LLM ÇAĞRISI YAPMAZ (mock)."""

from __future__ import annotations

from ie_eval.datasets.ie_case import ie_case_dataset
from ie_eval.problem import GroundTruth, Problem
from ie_eval.runner import run_dataset, run_one


# =============================================================================
# Fake solve — testler için deterministik state döndürür
# =============================================================================
def _make_fake_solve(state_map: dict[str, dict]):
    """Return a solve_fn that returns state_map[problem_prompt] on call."""
    def _solve(prompt, data_path=None, auto_mode=False, thread_id=None):
        # prompt ilk sözcüğü key olarak kullanalım
        for prompt_marker, state in state_map.items():
            if prompt_marker in prompt:
                return state
        return {"raw_prompt": prompt}
    return _solve


# =============================================================================
# run_one — başarılı akış
# =============================================================================
def test_run_one_success_records_metrics():
    problem = Problem(
        id="fake",
        prompt="THE FAKE PROMPT",
        ground_truth=GroundTruth(objective_value=42.0, tolerance_rel=1e-2),
    )
    state = {
        "execution_result": "The optimum is 42.0",
        "retry_count": 1,
        "metrics": {
            "intake": {"latency_ms": 10.0, "tokens_in": 100, "tokens_out": 50,
                       "cost_usd": 0.001, "llm_calls": 1, "invocations": 1,
                       "error_class": None},
            "report": {"latency_ms": 20.0, "tokens_in": 200, "tokens_out": 80,
                       "cost_usd": 0.002, "llm_calls": 2, "invocations": 1,
                       "error_class": None},
        },
    }
    fake_solve = _make_fake_solve({"FAKE PROMPT": state})

    rec = run_one(problem, config_id="test", solve_fn=fake_solve)

    assert rec.success
    assert rec.metrics is not None
    assert rec.metrics.execution_rate
    assert rec.metrics.numerical_match      # 42.0 vardı execution_result'ta
    assert rec.metrics.total_tokens_in == 300
    assert rec.metrics.total_tokens_out == 130
    assert rec.metrics.node_count == 2
    assert rec.metrics.retry_count == 1


def test_run_one_numerical_mismatch():
    problem = Problem(
        id="fake",
        prompt="THE FAKE PROMPT",
        ground_truth=GroundTruth(objective_value=42.0, tolerance_rel=1e-3),
    )
    state = {"execution_result": "wrong answer 100", "metrics": {}}
    fake_solve = _make_fake_solve({"FAKE PROMPT": state})

    rec = run_one(problem, solve_fn=fake_solve)
    assert rec.success
    assert not rec.metrics.numerical_match


def test_run_one_interrupt_flags_failure():
    problem = Problem(id="fake", prompt="THE FAKE PROMPT", ground_truth=GroundTruth())
    state = {"__interrupt__": ["missing info"], "execution_result": ""}
    fake_solve = _make_fake_solve({"FAKE PROMPT": state})

    rec = run_one(problem, solve_fn=fake_solve)
    assert not rec.success
    assert rec.error == "interrupted"


def test_run_one_exception_captured():
    problem = Problem(id="fake", prompt="X", ground_truth=GroundTruth())

    def broken_solve(*args, **kwargs):
        raise RuntimeError("solver crashed")

    rec = run_one(problem, solve_fn=broken_solve)
    assert not rec.success
    assert rec.state is None
    assert "RuntimeError" in (rec.error or "")
    assert rec.metrics is not None
    assert rec.metrics.error_class == "RunnerException"


# =============================================================================
# run_dataset — batch çağrı
# =============================================================================
def test_run_dataset_calls_each_problem_n_times():
    problems = [
        Problem(id="p1", prompt="ONE", ground_truth=GroundTruth()),
        Problem(id="p2", prompt="TWO", ground_truth=GroundTruth()),
    ]
    state_map = {
        "ONE": {"execution_result": "ok", "metrics": {}},
        "TWO": {"execution_result": "ok", "metrics": {}},
    }
    fake_solve = _make_fake_solve(state_map)

    seen = []
    def cb(rec):
        seen.append((rec.problem_id, rec.run_idx))

    recs = run_dataset(problems, n_runs=2, on_result=cb, solve_fn=fake_solve)

    assert len(recs) == 4
    assert seen == [("p1", 0), ("p1", 1), ("p2", 0), ("p2", 1)]


def test_run_dataset_accepts_dataset_object():
    """Dataset protocol implementer'lar da geçebilmeli. IE-Case seti (6 problem)."""
    fake_solve = _make_fake_solve({})   # boş state map — sadece kaç kez çağrıldığını sayıyoruz

    recs = run_dataset(ie_case_dataset, n_runs=1, solve_fn=fake_solve)
    assert len(recs) == 6
    ids = {r.problem_id for r in recs}
    assert ids == {
        "eoq-basic", "transport-2x3", "multi-product-inventory",
        "transport-3x2-csv", "assignment-3x3-sqlite", "abc-classification",
    }
    # Runner tüm problemleri yürütmeli (fake solve default state döner)
    assert all(r.success for r in recs)