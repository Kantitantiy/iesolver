"""
IndustryOR benchmark adapter (EVALUATION_PLAN §2).

IndustryOR (Meituan/Tsinghua): 13 sektörden gerçek OR problemleri.
Zorluk kanıtı — SOTA ~%37; NL4Opt'a göre çok daha katı bir set.
Ayrışma burada görünür (rakipler bu setle zorlanıyor).

Kaynak:
    * Orijinal: HuggingFace / GitHub (arama: "IndustryOR benchmark")
    * Cleaned altküme: 50 düzeltme + 23 geçersiz örnek dışlanmış
      (EVALUATION_PLAN §2 referansı; OptiMind supplementary)

Şema toleransı ve parse mantığı ``_jsonl_common`` modülünde. IndustryOR'a
özgü metadata alanları: ``sector``, ``industry``, ``difficulty``,
``domain``, ``problem_type``.

NL4Opt ile fark:
    * ``id_prefix``: "industryor" vs "nl4opt"
    * ``extra_metadata_keys``: "sector" ve "industry" eklendi (IndustryOR
      taksonomisinin çekirdek etiketleri; kırılım analizinde kullanılır)
    * ``benchmark`` metadata değeri: "IndustryOR"
    * Aynı ``tolerance_rel=1e-4`` (EVALUATION_PLAN §3)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from ie_eval.datasets._jsonl_common import iter_rows, row_to_problem
from ie_eval.problem import Problem


_INDUSTRYOR_METADATA_KEYS: tuple[str, ...] = (
    "sector", "industry", "problem_type", "type", "difficulty", "domain", "category",
)


@dataclass(slots=True)
class IndustryORDataset:
    """Iterable IndustryOR loader from a JSONL/JSON file.

    Attributes
    ----------
    path :
        Path to the JSONL/JSON file.
    cleaning :
        Metadata etiketi: "cleaned" (önerilen — 23 invalid dışlanmış, 50
        düzeltilmiş) veya "original". Adaptör veriyi değiştirmez.
    limit :
        İlk N problemi yükle (debug/subset).
    require_optimal :
        True ise objective_value'su olmayan satırlar atlanır. Q1 için True.
    tolerance_rel :
        EVALUATION_PLAN §3 gereği ``1e-4``.
    """

    path: Path
    cleaning: Literal["cleaned", "original"] = "cleaned"
    limit: int | None = None
    require_optimal: bool = True
    tolerance_rel: float = 1e-4

    name: str = "IndustryOR"

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    def load(self) -> Iterable[Problem]:
        """Yield ``Problem``s from the JSONL/JSON file."""
        yielded = 0
        for idx, row in enumerate(iter_rows(self.path)):
            if not isinstance(row, dict):
                continue
            problem = row_to_problem(
                row, idx,
                benchmark_name="IndustryOR",
                cleaning=self.cleaning,
                tolerance_rel=self.tolerance_rel,
                require_optimal=self.require_optimal,
                id_prefix="industryor",
                source_path=self.path,
                extra_metadata_keys=_INDUSTRYOR_METADATA_KEYS,
            )
            if problem is None:
                continue
            yield problem
            yielded += 1
            if self.limit is not None and yielded >= self.limit:
                return
