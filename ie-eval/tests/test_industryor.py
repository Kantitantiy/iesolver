"""
IndustryOR adapter tests.

NL4Opt ile aynı ortak taban → temel şema testleri paralel. Buradaki farklar:
    * benchmark_name "IndustryOR"
    * id_prefix "industryor-"
    * sector/industry metadata forward
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ie_eval.datasets import IndustryORDataset
from ie_eval.datasets.base import Dataset


def _write_jsonl(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "industryor_sample.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


# =============================================================================
# Temel şema
# =============================================================================
def test_industryor_standard_schema(tmp_path):
    rows = [
        {
            "id": "industryor_007",
            "question": "A distribution center serves 5 stores...",
            "answer": 3400.0,
            "sector": "logistics",
            "difficulty": "hard",
        }
    ]
    ds = IndustryORDataset(path=_write_jsonl(tmp_path, rows))
    problems = list(ds.load())

    assert len(problems) == 1
    p = problems[0]
    assert p.id == "industryor_007"
    assert p.ground_truth.objective_value == 3400.0
    assert p.metadata["benchmark"] == "IndustryOR"
    assert p.metadata["cleaning"] == "cleaned"
    assert p.metadata["sector"] == "logistics"
    assert p.metadata["difficulty"] == "hard"


def test_industryor_forwards_industry_and_sector(tmp_path):
    """IndustryOR-özgü etiketler NL4Opt'ta yokken burada metadata'ya taşınmalı."""
    rows = [{
        "id": "s-01", "document": "text", "optimal_value": 1.0,
        "sector": "energy", "industry": "renewables",
    }]
    ds = IndustryORDataset(path=_write_jsonl(tmp_path, rows))
    p = next(iter(ds.load()))
    assert p.metadata["sector"] == "energy"
    assert p.metadata["industry"] == "renewables"


def test_industryor_fallback_id_prefix(tmp_path):
    rows = [{"document": "no id", "optimal_value": 1.0}]
    ds = IndustryORDataset(path=_write_jsonl(tmp_path, rows))
    p = next(iter(ds.load()))
    assert p.id.startswith("industryor-")


def test_industryor_dataset_protocol_conformance(tmp_path):
    rows = [{"id": "x", "document": "t", "optimal_value": 1.0}]
    ds = IndustryORDataset(path=_write_jsonl(tmp_path, rows))
    assert isinstance(ds, Dataset)
    assert ds.name == "IndustryOR"


# =============================================================================
# Cleaning flag ve tolerance
# =============================================================================
def test_industryor_cleaning_original_flag(tmp_path):
    rows = [{"id": "c-01", "document": "t", "optimal_value": 1.0}]
    ds = IndustryORDataset(path=_write_jsonl(tmp_path, rows), cleaning="original")
    p = next(iter(ds.load()))
    assert p.metadata["cleaning"] == "original"


def test_industryor_default_tolerance_is_1e_minus_4(tmp_path):
    rows = [{"id": "t-01", "document": "t", "optimal_value": 1.0}]
    ds = IndustryORDataset(path=_write_jsonl(tmp_path, rows))
    p = next(iter(ds.load()))
    assert p.ground_truth.tolerance_rel == 1e-4


# =============================================================================
# require_optimal + limit paylaşılan taban davranışı
# =============================================================================
def test_industryor_require_optimal_skips_missing(tmp_path):
    rows = [
        {"id": "ok", "document": "x", "answer": 1.0},
        {"id": "bad", "document": "no optimal"},
    ]
    ds = IndustryORDataset(path=_write_jsonl(tmp_path, rows), require_optimal=True)
    problems = list(ds.load())
    assert [p.id for p in problems] == ["ok"]


def test_industryor_limit(tmp_path):
    rows = [{"id": f"p{i}", "document": f"P{i}", "answer": float(i)} for i in range(5)]
    ds = IndustryORDataset(path=_write_jsonl(tmp_path, rows), limit=2)
    problems = list(ds.load())
    assert len(problems) == 2


# =============================================================================
# Ortak parse hataları
# =============================================================================
def test_industryor_malformed_jsonl_raises(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id":"ok"}\ninvalid\n', encoding="utf-8")
    ds = IndustryORDataset(path=path, require_optimal=False)
    with pytest.raises(ValueError, match="invalid JSON line"):
        list(ds.load())


def test_industryor_json_array_format(tmp_path):
    """Top-level array formatı da kabul edilmeli."""
    path = tmp_path / "arr.json"
    path.write_text(json.dumps([{"id": "a", "document": "t", "answer": 5}]), encoding="utf-8")
    ds = IndustryORDataset(path=path)
    problems = list(ds.load())
    assert len(problems) == 1
