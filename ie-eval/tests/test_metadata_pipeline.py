"""
End-to-end metadata pipeline: Problem.metadata → runner → store → analysis.

DESIGN: kırılım analizi (benchmark/problem_type) için problem.metadata'nın
uçtan uca korunması gerekir.
"""

from __future__ import annotations

from ie_eval.analysis import (
    compare_configs,
    per_problem_correctness,
    summarize_by_config,
)
from ie_eval.problem import GroundTruth, Problem
from ie_eval.runner import run_one
from ie_eval.store import ResultStore


def _make_problem(pid: str, benchmark: str, problem_type: str, expected: float = 1.0) -> Problem:
    return Problem(
        id=pid,
        prompt=f"solve {pid}",
        ground_truth=GroundTruth(objective_value=expected, tolerance_rel=1e-2),
        metadata={"benchmark": benchmark, "problem_type": problem_type},
    )


def _fake_solve(answer: float):
    def _solve(prompt, data_path=None, auto_mode=False, thread_id=None):
        return {"execution_result": f"answer = {answer}", "metrics": {}}
    return _solve


# =============================================================================
# Runner → RunRecord.metadata dolduruluyor mu?
# =============================================================================
def test_runner_populates_metadata_from_problem():
    problem = _make_problem("p1", "NL4Opt", "LP")
    rec = run_one(problem, config_id="test", solve_fn=_fake_solve(1.0))
    assert rec.metadata is not None
    assert rec.metadata["benchmark"] == "NL4Opt"
    assert rec.metadata["problem_type"] == "LP"


# =============================================================================
# Store → metadata_json roundtrip
# =============================================================================
def test_store_persists_and_returns_metadata(tmp_path):
    store = ResultStore(tmp_path / "meta.sqlite")
    problem = _make_problem("p1", "NL4Opt", "LP")
    rec = run_one(problem, config_id="c1", solve_fn=_fake_solve(1.0))
    store.persist(rec)

    rows = store.list_runs()
    assert len(rows) == 1
    row = rows[0]
    assert "metadata" in row
    assert row["metadata"]["benchmark"] == "NL4Opt"
    assert row["metadata"]["problem_type"] == "LP"
    # Ham JSON string de korunmalı
    assert row["metadata_json"] is not None


def test_store_empty_metadata_returns_empty_dict(tmp_path):
    """RunRecord.metadata=None ise list_runs metadata={}"""
    store = ResultStore(tmp_path / "empty.sqlite")
    problem = Problem(id="p", prompt="x", ground_truth=GroundTruth())
    problem.metadata = {}   # boş
    rec = run_one(problem, config_id="c", solve_fn=_fake_solve(0.0))
    store.persist(rec)

    row = store.list_runs()[0]
    assert row["metadata"] == {}


# =============================================================================
# Analysis metadata_filter — dict form
# =============================================================================
def test_summarize_metadata_filter_dict(tmp_path):
    store = ResultStore(tmp_path / "mf.sqlite")

    # 3 NL4Opt problem (doğru), 2 IndustryOR (yanlış)
    for i in range(3):
        p = _make_problem(f"nlp{i}", "NL4Opt", "LP", expected=1.0)
        store.persist(run_one(p, config_id="pipeline", solve_fn=_fake_solve(1.0)))
    for i in range(2):
        p = _make_problem(f"iop{i}", "IndustryOR", "MILP", expected=1.0)
        store.persist(run_one(p, config_id="pipeline", solve_fn=_fake_solve(99.0)))  # yanlış

    # Filtresiz: 3 doğru / 5 → 0.6
    s_all = summarize_by_config(store, "pipeline")
    assert s_all.n_problems == 5
    assert s_all.accuracy_mean == 0.6

    # NL4Opt sadece → 3 doğru / 3
    s_nl = summarize_by_config(store, "pipeline",
                                metadata_filter={"benchmark": "NL4Opt"})
    assert s_nl.n_problems == 3
    assert s_nl.accuracy_mean == 1.0

    # IndustryOR sadece → 0 doğru / 2
    s_io = summarize_by_config(store, "pipeline",
                                metadata_filter={"benchmark": "IndustryOR"})
    assert s_io.n_problems == 2
    assert s_io.accuracy_mean == 0.0


def test_summarize_metadata_filter_callable(tmp_path):
    """Callable form: karmaşık predikat (örn. problem_type IN belirli set)."""
    store = ResultStore(tmp_path / "cf.sqlite")
    for pid, ptype in [("a", "LP"), ("b", "MILP"), ("c", "NLP")]:
        p = _make_problem(pid, "X", ptype)
        store.persist(run_one(p, config_id="c1", solve_fn=_fake_solve(1.0)))

    s = summarize_by_config(store, "c1",
                             metadata_filter=lambda md: md.get("problem_type") in {"LP", "MILP"})
    assert s.n_problems == 2


def test_per_problem_correctness_respects_filter(tmp_path):
    store = ResultStore(tmp_path / "pf.sqlite")
    store.persist(run_one(_make_problem("a", "NL4Opt", "LP"),
                          config_id="c", solve_fn=_fake_solve(1.0)))
    store.persist(run_one(_make_problem("b", "IndustryOR", "LP"),
                          config_id="c", solve_fn=_fake_solve(1.0)))
    out = per_problem_correctness(store, "c", metadata_filter={"benchmark": "NL4Opt"})
    assert set(out.keys()) == {"a"}


def test_compare_configs_metadata_filter_isolates_benchmark(tmp_path):
    """Kırılım analizi: pipeline vs single_shot yalnızca NL4Opt üzerinde."""
    store = ResultStore(tmp_path / "cmp.sqlite")

    # NL4Opt: pipeline hepsini, single_shot yarısını doğrur
    for i in range(4):
        p = _make_problem(f"nl{i}", "NL4Opt", "LP")
        store.persist(run_one(p, config_id="pipeline", solve_fn=_fake_solve(1.0)))
        store.persist(run_one(p, config_id="single_shot",
                              solve_fn=_fake_solve(1.0 if i % 2 == 0 else 999.0)))

    # IndustryOR: her ikisi de karışık — filtre çalışıyorsa bu sayılmayacak
    for i in range(4):
        p = _make_problem(f"io{i}", "IndustryOR", "MILP")
        store.persist(run_one(p, config_id="pipeline", solve_fn=_fake_solve(999.0)))
        store.persist(run_one(p, config_id="single_shot", solve_fn=_fake_solve(999.0)))

    comp = compare_configs(store, "pipeline", "single_shot", policy="first",
                            metadata_filter={"benchmark": "NL4Opt"})
    assert comp.n_common_problems == 4
    assert comp.accuracy_a == 1.0
    assert comp.accuracy_b == 0.5


# =============================================================================
# Backward-compat: metadata_json sütunu olmayan eski DB'de bile çalışmalı
# =============================================================================
def test_store_migration_on_pre_existing_db(tmp_path):
    """Eski schema'lı DB'yi aç → ALTER TABLE otomatik uygulanmalı."""
    import sqlite3
    db_path = tmp_path / "old.sqlite"

    # Elle eski (metadata_json'suz) şema oluştur
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            config_id TEXT NOT NULL,
            problem_id TEXT NOT NULL,
            run_idx INTEGER NOT NULL,
            success INTEGER NOT NULL,
            elapsed_s REAL NOT NULL,
            execution_rate INTEGER NOT NULL DEFAULT 0,
            numerical_match INTEGER NOT NULL DEFAULT 0,
            feasibility_checked INTEGER NOT NULL DEFAULT 0,
            feasibility_ok INTEGER NOT NULL DEFAULT 0,
            retry_count INTEGER NOT NULL DEFAULT 0,
            error_class TEXT,
            error_message TEXT,
            tokens_in INTEGER NOT NULL DEFAULT 0,
            tokens_out INTEGER NOT NULL DEFAULT 0,
            cost_usd REAL NOT NULL DEFAULT 0.0,
            llm_calls INTEGER NOT NULL DEFAULT 0,
            node_count INTEGER NOT NULL DEFAULT 0,
            violations_json TEXT,
            created_at REAL NOT NULL
        );
    """)
    conn.commit()
    conn.close()

    # Store aç → migration otomatik uygulanmalı
    store = ResultStore(db_path)

    # Migration sonrası yeni satır yazılabilmeli
    p = _make_problem("mig-01", "NL4Opt", "LP")
    store.persist(run_one(p, config_id="c", solve_fn=_fake_solve(1.0)))
    rows = store.list_runs()
    assert rows[0]["metadata"]["benchmark"] == "NL4Opt"
