"""
ie_eval.problem — Benchmark problem specification.

Bir Problem: girdi (prompt + opsiyonel veri dosyası) + değerlendirme
sözleşmesi (ground truth + feasibility check). Her benchmark set aynı
soyutlamayı kullanır; runner benchmark'a bakmaz, yalnızca Problem'e bakar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


# =============================================================================
# GroundTruth — beklenen çözümü ve feasibility check'i taşır
# =============================================================================
@dataclass(slots=True)
class GroundTruth:
    """Expected solution + feasibility contract for a problem.

    Attributes
    ----------
    objective_value :
        Beklenen amaç fonksiyonu değeri. ``None`` → sadece feasibility bakılır
        (NO_CODE veya kavramsal sorular için).
    tolerance_rel :
        Göreli tolerans (default 1e-3 = %0.1). EVALUATION_PLAN §3'te 1e-4;
        MVP için biraz daha gevşek.
    solution :
        Karar değişkenlerinin tam değerleri (varsa). Feasibility check'te
        constraint fonksiyonlarına iletilir.
    feasibility_fn :
        Deterministik feasibility fonksiyonu (DESIGN_REVIEW §3.2):
        ``(solution_dict) -> list[str]`` — ihlal listesi; boş = feasible.
        None → sadece objective_value karşılaştırması yapılır.
    """

    objective_value: Optional[float] = None
    tolerance_rel: float = 1e-3
    solution: dict[str, float] = field(default_factory=dict)
    feasibility_fn: Optional[Callable[[dict[str, float]], list[str]]] = None


# =============================================================================
# Problem — bir benchmark girdisi
# =============================================================================
@dataclass(slots=True)
class Problem:
    """A single benchmark problem specification.

    Attributes
    ----------
    id :
        Benchmark içinde benzersiz kimlik ("eoq-basic", "nl4opt-042", ...).
    prompt :
        Doğal dil problem açıklaması — iesolver.solve()'a doğrudan iletilir.
    data_path :
        Opsiyonel tek veri dosyası (csv/xlsx/sqlite). Çoğu benchmark None.
    ground_truth :
        Değerlendirme sözleşmesi (objective + feasibility).
    metadata :
        Analiz için serbest form etiketler:
        {"benchmark": "IE-Case", "problem_type": "EOQ", "difficulty": "easy"}
    """

    id: str
    prompt: str
    data_path: Optional[Path] = None
    ground_truth: GroundTruth = field(default_factory=GroundTruth)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def benchmark(self) -> str:
        """Convenience: return metadata['benchmark'] or 'unknown'."""
        return str(self.metadata.get("benchmark", "unknown"))