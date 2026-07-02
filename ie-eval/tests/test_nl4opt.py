"""NL4Opt adapter tests.

Sentetik JSONL/JSON verisi üretir; gerçek NL4Opt dosyası gerektirmez.
Amaç: şema esnekliği ve MVP hata durumları.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ie_eval.datasets.nl4opt import NL4OptDataset
from ie_eval.problem import Problem


# =============================================================================
# Fixtures
# =============================================================================
def _write_jsonl(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "nl4opt_sample.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def _write_json_array(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "nl4opt_sample.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


# =============================================================================
# Şema toleransı — yaygın alan adları
# =============================================================================
def test_loads_standard_schema(tmp_path):
    rows = [
        {
            "id": "nl4opt_test_042",
            "document": "A factory produces two products A and B. Maximize profit...",
            "optimal_value": 123.4,
            "problem_type": "LP",
        }
    ]
    ds = NL4OptDataset(path=_write_jsonl(tmp_path, rows))
    problems = list(ds.load())

    assert len(problems) == 1
    p = problems[0]
    assert p.id == "nl4opt_test_042"
    assert "factory produces" in p.prompt
    assert p.ground_truth.objective_value == 123.4
    assert p.ground_truth.tolerance_rel == 1e-4
    assert p.metadata["benchmark"] == "NL4Opt"
    assert p.metadata["cleaning"] == "cleaned"
    assert p.metadata["problem_type"] == "LP"


def test_loads_alternative_field_names(tmp_path):
    """gold_optimal_value + question alanlarını da tanısın."""
    rows = [
        {"problem_id": "alt-01", "question": "Solve...", "gold_optimal_value": 42.0},
        {"identifier": "alt-02", "prompt": "Another...", "answer": "17"},   # answer: str
    ]
    ds = NL4OptDataset(path=_write_jsonl(tmp_path, rows))
    problems = list(ds.load())

    assert len(problems) == 2
    assert problems[0].id == "alt-01"
    assert problems[0].ground_truth.objective_value == 42.0
    assert problems[1].id == "alt-02"
    assert problems[1].ground_truth.objective_value == 17.0  # str → float coerce


# =============================================================================
# require_optimal davranışı
# =============================================================================
def test_require_optimal_skips_missing(tmp_path):
    rows = [
        {"id": "ok", "document": "solve this", "optimal_value": 1.0},
        {"id": "bad", "document": "no optimal"},   # optimal yok
    ]
    ds = NL4OptDataset(path=_write_jsonl(tmp_path, rows), require_optimal=True)
    problems = list(ds.load())
    assert [p.id for p in problems] == ["ok"]


def test_require_optimal_false_keeps_missing(tmp_path):
    rows = [
        {"id": "ok", "document": "x", "optimal_value": 1.0},
        {"id": "bad", "document": "no optimal"},
    ]
    ds = NL4OptDataset(path=_write_jsonl(tmp_path, rows), require_optimal=False)
    problems = list(ds.load())
    assert len(problems) == 2
    assert problems[1].ground_truth.objective_value is None


# =============================================================================
# limit
# =============================================================================
def test_limit_stops_iteration(tmp_path):
    rows = [{"id": f"p{i}", "document": f"P{i}", "optimal_value": float(i)} for i in range(10)]
    ds = NL4OptDataset(path=_write_jsonl(tmp_path, rows), limit=3)
    problems = list(ds.load())
    assert len(problems) == 3
    assert [p.id for p in problems] == ["p0", "p1", "p2"]


# =============================================================================
# JSON array formatı
# =============================================================================
def test_loads_json_array_format(tmp_path):
    rows = [{"id": "a1", "document": "text", "optimal_value": 5.0}]
    ds = NL4OptDataset(path=_write_json_array(tmp_path, rows))
    problems = list(ds.load())
    assert len(problems) == 1
    assert problems[0].id == "a1"


# =============================================================================
# Hatalı satır — bozuk JSONL
# =============================================================================
def test_raises_on_malformed_json(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "ok"}\nnot-a-json\n', encoding="utf-8")
    ds = NL4OptDataset(path=path, require_optimal=False)
    with pytest.raises(ValueError, match="invalid JSON line"):
        list(ds.load())


# =============================================================================
# Yaygın metadata etiketleri (problem_type, difficulty, ...)
# =============================================================================
def test_extra_metadata_forwarded(tmp_path):
    rows = [
        {
            "id": "meta-01",
            "document": "text",
            "optimal_value": 1.0,
            "problem_type": "LP",
            "difficulty": "hard",
            "domain": "logistics",
            "unrelated_field": "ignored",
        }
    ]
    ds = NL4OptDataset(path=_write_jsonl(tmp_path, rows))
    p = next(iter(ds.load()))
    assert p.metadata["problem_type"] == "LP"
    assert p.metadata["difficulty"] == "hard"
    assert p.metadata["domain"] == "logistics"
    assert "unrelated_field" not in p.metadata


def test_cleaning_flag_propagated(tmp_path):
    rows = [{"id": "c-01", "document": "t", "optimal_value": 1.0}]
    ds_orig = NL4OptDataset(path=_write_jsonl(tmp_path, rows), cleaning="original")
    p = next(iter(ds_orig.load()))
    assert p.metadata["cleaning"] == "original"


# =============================================================================
# Fallback ID üretimi
# =============================================================================
def test_generates_fallback_id_when_missing(tmp_path):
    rows = [{"document": "no id here", "optimal_value": 1.0}]
    ds = NL4OptDataset(path=_write_jsonl(tmp_path, rows))
    p = next(iter(ds.load()))
    assert p.id.startswith("nl4opt-")


# =============================================================================
# Dataset protocol conformance
# =============================================================================
def test_conforms_to_dataset_protocol(tmp_path):
    from ie_eval.datasets.base import Dataset
    rows = [{"id": "x", "document": "t", "optimal_value": 1.0}]
    ds = NL4OptDataset(path=_write_jsonl(tmp_path, rows))
    assert isinstance(ds, Dataset)
    assert ds.name == "NL4Opt"


# =============================================================================
# Runner ile entegrasyon (mock solve_fn)
# =============================================================================
def test_integrates_with_runner(tmp_path):
    from ie_eval.runner import run_dataset

    rows = [
        {"id": "int-01", "document": "Problem alpha", "optimal_value": 100.0},
        {"id": "int-02", "document": "Problem beta",  "optimal_value": 250.0},
    ]
    ds = NL4OptDataset(path=_write_jsonl(tmp_path, rows))

    def fake_solve(prompt, data_path=None, auto_mode=False, thread_id=None):
        # Farklı problemlere farklı sonuçlar döndür
        if "alpha" in prompt:
            return {"execution_result": "answer = 100.0", "metrics": {}}
        return {"execution_result": "answer = 999", "metrics": {}}

    recs = run_dataset(ds, n_runs=1, solve_fn=fake_solve)
    assert len(recs) == 2
    assert recs[0].metrics.numerical_match     # alpha eşleşti
    assert not recs[1].metrics.numerical_match # beta eşleşmedi


def test_empty_prompt_row_is_skipped(tmp_path):
    rows = [
        {"id": "good", "document": "solve me", "optimal_value": 1.0},
        {"id": "bad", "document": "", "optimal_value": 1.0},
        {"id": "worse", "optimal_value": 1.0},   # document alanı yok
    ]
    ds = NL4OptDataset(path=_write_jsonl(tmp_path, rows))
    problems = list(ds.load())
    assert [p.id for p in problems] == ["good"]
