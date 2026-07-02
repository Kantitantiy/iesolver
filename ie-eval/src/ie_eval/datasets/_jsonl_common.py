"""
Ortak permissive JSONL/JSON loader — NL4Opt ve IndustryOR ortak kullanıyor.

Neden ayrı modül?
    Her iki benchmark da doğal-dil-problem + optimal-değer sözleşmesi taşıyor
    ve alan adları sürüme göre değişiyor. Aynı okuma/parse mantığını iki
    yerde tutmak sağırlığa yol açar; buradaki tek nokta değişince her iki
    adaptör yeni sürümü tanır. Adaptör sınıfları yalnızca benchmark-özgü
    varsayılanları (id_prefix, extra_metadata_keys, benchmark_name) taşır.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from ie_eval.problem import GroundTruth, Problem


# =============================================================================
# Şema tarama sabitleri
# =============================================================================
_ID_KEYS: tuple[str, ...] = ("id", "problem_id", "identifier", "uuid")
_PROMPT_KEYS: tuple[str, ...] = (
    "document", "question", "prompt", "text", "problem", "description",
    "natural_language",
)
_OPTIMAL_KEYS: tuple[str, ...] = (
    "optimal_value", "gold_optimal_value", "obj_val", "optimal", "answer",
    "solution", "objective", "objective_value",
)


def _first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return None


def _coerce_float(value: Any) -> float | None:
    """Cast to float; None on failure (some rows carry str-serialized numbers)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (ValueError, TypeError):
        return None


# =============================================================================
# Row iterator
# =============================================================================
def iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a JSONL or JSON-array file.

    * ``.jsonl`` — line-per-object (ilk non-whitespace `[` değilse otomatik)
    * ``.json``  — top-level array
    """
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()

    if stripped.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{path}: expected JSON array at top level")
        yield from data
        return

    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{line_no}: invalid JSON line — {e}") from e


# =============================================================================
# Row → Problem
# =============================================================================
def row_to_problem(
    row: dict[str, Any],
    fallback_idx: int,
    *,
    benchmark_name: str,
    cleaning: str,
    tolerance_rel: float,
    require_optimal: bool,
    id_prefix: str,
    source_path: Path,
    extra_metadata_keys: tuple[str, ...] = (),
) -> Problem | None:
    """Convert one raw JSON row into a ``Problem`` (or None if unusable).

    None döndürür (satırı atlar) durumlar:
        * ``prompt`` alanı yok veya boş
        * ``require_optimal=True`` ve optimal_value yok / parse edilemedi
    """
    raw_prompt = _first_present(row, _PROMPT_KEYS)
    if not raw_prompt or not isinstance(raw_prompt, str) or not raw_prompt.strip():
        return None

    optimal = _coerce_float(_first_present(row, _OPTIMAL_KEYS))
    if require_optimal and optimal is None:
        return None

    raw_id = _first_present(row, _ID_KEYS)
    problem_id = str(raw_id) if raw_id is not None else f"{id_prefix}-{fallback_idx:04d}"

    gt = GroundTruth(
        objective_value=optimal,
        tolerance_rel=tolerance_rel,
        solution={},
        feasibility_fn=None,
    )

    metadata: dict[str, Any] = {
        "benchmark": benchmark_name,
        "cleaning": cleaning,
        "source_path": str(source_path),
        "source_row": fallback_idx,
    }
    for k in extra_metadata_keys:
        if k in row and row[k] is not None:
            metadata[k] = row[k]

    return Problem(
        id=problem_id,
        prompt=raw_prompt.strip(),
        data_path=None,
        ground_truth=gt,
        metadata=metadata,
    )
