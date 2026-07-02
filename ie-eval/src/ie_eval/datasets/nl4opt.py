"""
NL4Opt benchmark adapter (EVALUATION_PLAN §2).

NL4Opt (NeurIPS 2022 shared task): doğal dil → LP formülasyonu. 289 test
problemi kabul görmüş standarttır; rakiplerin çoğu (OptiMUS, CoE, LLMOPT)
bu set üzerinde rapor eder.

**Kritik**: EVALUATION_PLAN §2 orijinal set değil, **temizlenmiş sürüm**
kullanılmasını gerektirir (literatür 16 etiket hatası raporluyor). Adaptör
temiz/orijinal ayrımı yapmaz — `cleaning` metadata'sına yazılır, veri
kaynağı seçimi kullanıcının sorumluluğu. Q1 makale metninde hangi
sürümün kullanıldığı açıkça belirtilecek.

Kaynak önerileri (kullanıcı elle indirir):
    * OptiMind supplementary (arxiv 2509.22979) — temizlenmiş sürüm
    * Original: https://github.com/nl4opt/nl4opt-competition
    * HuggingFace: mirror varyantları (şema farklı olabilir)

Şema toleransı ve parse mantığı ``_jsonl_common`` modülünde. Bu adaptör
yalnızca NL4Opt-özgü varsayılanları tutar.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from ie_eval.datasets._jsonl_common import iter_rows, row_to_problem
from ie_eval.problem import Problem


_NL4OPT_METADATA_KEYS: tuple[str, ...] = (
    "problem_type", "type", "difficulty", "domain", "category",
)


@dataclass(slots=True)
class NL4OptDataset:
    """Iterable NL4Opt loader from a JSONL/JSON file.

    Attributes
    ----------
    path :
        Path to the JSONL (satır başı obje) veya JSON (top-level array) dosyası.
    cleaning :
        Metadata etiketi: "cleaned" (önerilen) veya "original". Adaptör
        veriyi değiştirmez; ``Problem.metadata["cleaning"]`` alanına yazar.
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

    name: str = "NL4Opt"

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
                benchmark_name="NL4Opt",
                cleaning=self.cleaning,
                tolerance_rel=self.tolerance_rel,
                require_optimal=self.require_optimal,
                id_prefix="nl4opt",
                source_path=self.path,
                extra_metadata_keys=_NL4OPT_METADATA_KEYS,
            )
            if problem is None:
                continue
            yield problem
            yielded += 1
            if self.limit is not None and yielded >= self.limit:
                return
