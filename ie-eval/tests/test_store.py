"""ResultStore persistence tests."""

from __future__ import annotations

from ie_eval.metrics import ProblemMetrics
from ie_eval.problem import GroundTruth, Problem
from ie_eval.runner import RunRecord, run_one
from ie_eval.store import ResultStore
from ie_eval.validator import FeasibilityCheck


def _make_record(problem_id="p1", config_id="baseline", run_idx=0) -> RunRecord:
    metrics = ProblemMetrics(
        problem_id=problem_id,
        execution_rate=True,
        numerical_match=True,
        feasibility=FeasibilityCheck(feasible=True, violations=[], checked=True),
        elapsed_s=1.23,
        total_tokens_in=500,
        total_tokens_out=200,
        total_cost_usd=0.005,
        total_llm_calls=8,
        node_count=3,
        retry_count=0,
        error_class=None,
        per_node={
            "intake": {"latency_ms": 10.0, "tokens_in": 100, "tokens_out": 50,
                       "cost_usd": 0.001, "llm_calls": 1, "invocations": 1,
                       "error_class": None},
            "code_branch": {"latency_ms": 500.0, "tokens_in": 300, "tokens_out": 100,
                            "cost_usd": 0.003, "llm_calls": 5, "invocations": 2,
                            "error_class": None},
            "report": {"latency_ms": 30.0, "tokens_in": 100, "tokens_out": 50,
                       "cost_usd": 0.001, "llm_calls": 2, "invocations": 1,
                       "error_class": None},
        },
    )
    return RunRecord(
        problem_id=problem_id, config_id=config_id, run_idx=run_idx,
        success=True, elapsed_s=1.23, state={}, metrics=metrics,
    )


def test_store_persist_and_read(tmp_path):
    store = ResultStore(tmp_path / "results.sqlite")
    rec = _make_record()

    run_id = store.persist(rec)
    assert run_id > 0
    assert store.count() == 1

    rows = store.list_runs()
    assert len(rows) == 1
    row = rows[0]
    assert row["problem_id"] == "p1"
    assert row["config_id"] == "baseline"
    assert row["numerical_match"] == 1
    assert row["tokens_in"] == 500
    assert row["node_count"] == 3


def test_store_filters_by_config(tmp_path):
    store = ResultStore(tmp_path / "r.sqlite")
    store.persist(_make_record(problem_id="p1", config_id="A"))
    store.persist(_make_record(problem_id="p1", config_id="B"))
    store.persist(_make_record(problem_id="p2", config_id="A"))

    assert store.count() == 3
    assert len(store.list_runs(config_id="A")) == 2
    assert len(store.list_runs(config_id="B")) == 1


def test_store_persists_node_metrics(tmp_path):
    import sqlite3
    store = ResultStore(tmp_path / "r.sqlite")
    run_id = store.persist(_make_record())

    conn = sqlite3.connect(str(store.db_path))
    rows = conn.execute(
        "SELECT node_name, tokens_in, invocations FROM node_metrics WHERE run_id = ? ORDER BY node_name",
        (run_id,),
    ).fetchall()
    conn.close()

    assert len(rows) == 3
    names = [r[0] for r in rows]
    assert "code_branch" in names
    # code_branch invocations=2 (retry döngüsü)
    code_row = next(r for r in rows if r[0] == "code_branch")
    assert code_row[2] == 2


def test_store_end_to_end_with_runner_callback(tmp_path):
    """Runner callback → store.persist entegrasyonu."""
    problem = Problem(
        id="e2e",
        prompt="THE FAKE PROMPT",
        ground_truth=GroundTruth(objective_value=100.0, tolerance_rel=1e-2),
    )
    state = {
        "execution_result": "Answer is 100",
        "metrics": {"intake": {"latency_ms": 5.0, "tokens_in": 10, "tokens_out": 5,
                                "cost_usd": 0.0001, "llm_calls": 1, "invocations": 1,
                                "error_class": None}},
    }

    def fake_solve(prompt, data_path=None, auto_mode=False, thread_id=None):
        return state

    store = ResultStore(tmp_path / "r.sqlite")
    rec = run_one(problem, solve_fn=fake_solve)
    store.persist(rec)

    assert store.count() == 1
    row = store.list_runs()[0]
    assert row["execution_rate"] == 1
    assert row["numerical_match"] == 1