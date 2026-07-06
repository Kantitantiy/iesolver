"""
iesolver.state — shared workflow state and data abstraction.

Bu modül, LangGraph pipeline'ı boyunca akan iki temel veri yapısını tanımlar:

1. ``DataBundle``  — csv / xlsx çok-sayfa / sqlite girişlerini tek bir
   in-memory soyutlamaya indirger. Downstream node'lar (özellikle GateKeeper'ın
   "data summary" üretici kısmı ve AlgoSelector) artık kaynak formata bakmaz;
   yalnızca ``DataBundle`` ile konuşur.

2. ``SolverState`` — pipeline boyunca taşınan paylaşımlı durum.
   ``TypedDict(total=False)`` olarak modellendi: her node yalnızca yazdığı
   alanları döndürür, LangGraph partial-merge ile durumu birleştirir.
   Bu, makalenin "state-as-contract" argümanının somut karşılığıdır:
   tip-güvenli, çağrı sıralamasından bağımsız, optimize edilebilir.

Mimari karar — TypedDict yerine Pydantic BaseModel neden değil?
    Pydantic her partial update'te re-validation maliyeti getirir. LangGraph
    döngülerinde (özellikle 4B'nin retry loop'unda) bu maliyet birikir.
    TypedDict statik tip ipucu sağlar, runtime'da dict kadar hafiftir.
    Validation gereken sınırlarda (kullanıcı girdisi, dosya okuma) Pydantic
    ``config.py`` ve ``io/data_loader.py`` içinde devreye girer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import operator
from typing import Annotated, Any, Literal, TypedDict

import pandas as pd

from iesolver.observability.metrics import merge_metrics
from iesolver.text import fenced

# -----------------------------------------------------------------------------
# Yardımcı tip takma adları / type aliases
# -----------------------------------------------------------------------------
SourceType = Literal["csv", "xlsx", "sqlite", "none"]
ExecutionPath = Literal["CODE", "NO_CODE"]
OutputFormat = Literal["pdf", "docx", "html"]


# =============================================================================
# DataBundle
# =============================================================================
@dataclass(slots=True)
class DataBundle:
    """Normalized container for tabular input passed to the solver.

    Three accepted input formats collapse into the same structure:

    * **CSV**   →  ``tables = {"data": <DataFrame>}``
    * **XLSX**  →  ``tables = {<sheet_name>: <DataFrame>, ...}``
    * **SQLite** →  ``tables = {<table_name>: <DataFrame>, ...}``
    * **None**  →  ``tables = {}`` (prompt-only çalışma)

    Bu soyutlama, downstream node'ların kaynak formatla ilgilenmemesini sağlar.
    Eski koddaki ``DataProfiler.generate_summary`` görevini ``summary()``
    metodu üstlenir.

    Attributes
    ----------
    tables :
        Mapping from logical table/sheet name to its DataFrame.
    source_path :
        Original file path on disk; ``None`` if solver invoked without data.
    source_type :
        Discriminator for the original input format.
    """

    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    source_path: Path | None = None
    source_type: SourceType = "none"

    # ---- properties ---------------------------------------------------------
    @property
    def is_empty(self) -> bool:
        """True iff no tables are bound to this bundle."""
        return len(self.tables) == 0

    @property
    def table_names(self) -> list[str]:
        """Return logical table names in insertion order."""
        return list(self.tables.keys())

    # ---- helpers ------------------------------------------------------------
    def get(self, name: str) -> pd.DataFrame:
        """Fetch a table by name; raises KeyError with a helpful message.

        Parameters
        ----------
        name :
            Table key as it appears in :attr:`tables`.

        Raises
        ------
        KeyError
            When ``name`` is not present in the bundle.
        """
        if name not in self.tables:
            available = ", ".join(self.table_names) or "<empty>"
            raise KeyError(
                f"Table {name!r} not found in DataBundle. Available: {available}"
            )
        return self.tables[name]

    def summary(self, max_rows: int = 5, max_cols: int = 20, max_preview_chars: int = 2000) -> str:
        """Produce a token-friendly textual summary for LLM context.

        Bu metot eski sistemdeki ``DataProfiler.generate_summary`` ile
        aynı sözleşmeyi sunar: her tablo için şekil, sütun tipleri,
        eksik değer sayıları ve küçük bir baş-satır önizlemesi.
        Çıktı doğrudan LLM prompt'una gömülmek üzere tasarlandı, bu
        nedenle sütun sayısı sert üstten sınırlandırılır.

        Guardrail (CLAUDE.md Düzeltme #7): hücre içerikleri kullanıcının
        yüklediği dosyadan gelir, dolayısıyla güvenilmez. Her tablo bloğu
        ``fenced(..., untrusted=True)`` ile talimatlardan ayrılır ve önizleme
        metni ``max_preview_chars`` ile sert şekilde sınırlanır — bir hücreye
        gömülmüş uzun bir "talimat" metninin prompt'u ele geçirmesi engellenir.

        Parameters
        ----------
        max_rows :
            Number of head rows to include per table.
        max_cols :
            Hard cap on columns shown to keep the summary token-bounded.
        max_preview_chars :
            Hard cap on the rendered head-row preview per table.

        Returns
        -------
        str
            Multi-section summary, or a placeholder when bundle is empty.
        """
        if self.is_empty:
            return "No data provided."

        header = f"Source: {self.source_type}"
        if self.source_path is not None:
            header += f" ({self.source_path})"

        chunks: list[str] = [header]
        for name, df in self.tables.items():
            rows, cols = df.shape
            shown = df.columns[:max_cols].tolist()
            dtypes = {c: str(df[c].dtype) for c in shown}
            missing = {c: int(df[c].isna().sum()) for c in shown}
            head_preview = df.head(max_rows).to_string(index=False)
            if len(head_preview) > max_preview_chars:
                head_preview = head_preview[:max_preview_chars] + " …(truncated)"

            truncated_note = (
                f" (showing first {max_cols} of {cols} cols)"
                if cols > max_cols
                else ""
            )

            chunks.append(
                fenced(
                    f"TABLE: {name}",
                    f"Shape: {rows} rows x {cols} cols{truncated_note}\n"
                    f"dtypes: {dtypes}\n"
                    f"missing: {missing}\n"
                    f"head:\n{head_preview}",
                    untrusted=True,
                )
            )

        return "\n".join(chunks)


# =============================================================================
# SolverState
# =============================================================================
class SolverState(TypedDict, total=False):
    """Shared, partial-update state flowing through the LangGraph workflow.

    Her node yalnızca yazdığı alanları döndürür; LangGraph okunan ve yazılan
    alanları birleştirerek bir sonraki node'a aktarır. ``total=False`` zorunlu:
    aksi halde node'lar tüm anahtarları doldurmak zorunda kalır.

    Alan grupları :doc:`PLAN.md` §3 ile birebir hizalanmıştır; yeni faz
    açıldıkça (yeni node'lar eklendikçe) alanlar bu sınıfa eklenir, mevcut
    node'lar bozulmaz — TypedDict bunu garanti eder.
    """

    # ----- Girdi / Input -----------------------------------------------------
    raw_prompt: str
    data_path: Path | None
    data_bundle: DataBundle | None

    # ----- Runtime flags -----------------------------------------------------
    # auto_mode: batch/benchmark koşularında interrupt yerine varsayılan
    # varsayımlarla devam et. DESIGN_REVIEW §3.1 gereği; NL4Opt/IndustryOR
    # gibi tam otomatik değerlendirme için zorunlu.
    auto_mode: bool
    # Auto-mode altında clarify_node'un ürettiği varsayımların log listesi;
    # metrics/error-analysis için okunur.
    auto_assumptions_log: list[str]

    # ----- Ablation flags (EVALUATION_PLAN §5) --------------------------------
    # A1: False → PromptRefiner node atlanır; ham prompt doğrudan router'a gider.
    enable_refiner: bool
    # A2: False → validate sonrası retry döngüsü devre dışı; tek geçiş.
    enable_validator_retry: bool
    # A4: True → call_with_reasoning_lm da fast LM kullanır (reasoning LM kapalı).
    fast_only: bool
    # A6 (CLAUDE.md Düzeltme #5): True → StrategyRouter kararı 3 örnekleme +
    # çoğunluk oyu (dspy.majority) ile alınır; en riskli tekil karar olan
    # execution_path'in tekil-örnekleme varyansını azaltır.
    self_consistency_router: bool

    # ----- Aşama 0 — GateKeeper ---------------------------------------------
    cleaned_prompt: str
    data_summary: str

    # ----- Aşama 1 — Requirement Analyst ------------------------------------
    is_complete: bool
    # DSPy 3.x tipli çıktılar (DESIGN_REVIEW §3.4): list[str] iterasyonlanabilir,
    # ölçüm ve testte doğrudan uzunluk/eleman kontrolü mümkün.
    missing_items: list[str]
    explicit_goal: str
    constraints: list[str]
    output_spec: str
    user_clarification: str   # human-in-loop interrupt'tan dönen cevap

    # ----- Aşama 2 — Prompt Refiner -----------------------------------------
    essential_prompt: str
    strict_constraints: str
    problem_type: str

    # ----- Aşama 3 — Strategy Router ----------------------------------------
    execution_path: ExecutionPath
    reasoning_framework: str
    rationale: str
    # A6: yalnızca self_consistency_router=True iken doldurulur; ör. "CODE:2/NO_CODE:1".
    router_vote_summary: str

    # ----- Aşama 4A — Analytical (NO_CODE) ----------------------------------
    raw_result: str
    solution_path: str

    # ----- Aşama 4B — Code Engine (CODE) ------------------------------------
    target_algorithm: str
    target_library: str
    library_specific_constraints: str
    code_output_spec: str
    final_code: str
    execution_result: str
    is_valid: bool
    confidence_score: int
    validation_notes: str
    retry_count: int

    # ----- Faz 4: Sensitivity + Artifacts -----------------------------------
    sensitivity_results: str | None
    # DESIGN_REVIEW §3.6: dual-first. Faz 4.5'te birden fazla artifact
    # gelebileceği için operator.add reducer ile birikiyor.
    figures: Annotated[list[Path], operator.add]

    # ----- Aşama 5 — 3-Katmanlı Rapor ---------------------------------------
    technical_output: str
    executive_summary: str
    action_directives: str

    # ----- Compile — Faz 5 --------------------------------------------------
    output_path: Path | None
    output_format: OutputFormat

    # ----- Telemetry (DESIGN_REVIEW §3.5) -----------------------------------
    # Per-node metrikleri (latency_ms, tokens_in/out, cost_usd, llm_calls,
    # invocations, error_class). Custom reducer aynı node'un birden fazla
    # çağrısını (özellikle code_branch retry döngüsü) sayısal alanlar
    # üzerinden toplar; ``invocations`` çağrı sayısını taşır.
    metrics: Annotated[dict[str, dict[str, Any]], merge_metrics]


# =============================================================================
# Boş state üretici (test ve replay için)
# =============================================================================
def empty_state(
    raw_prompt: str = "",
    data_path: Path | None = None,
    *,
    auto_mode: bool = False,
    enable_refiner: bool = True,
    enable_validator_retry: bool = True,
    fast_only: bool = False,
    self_consistency_router: bool = False,
) -> SolverState:
    """Return a minimal :class:`SolverState` seeded with the user's inputs.

    Bu yardımcı, smoke test'ler ve LangGraph ``invoke()`` çağrıları için
    tip-güvenli bir başlangıç durumu üretir. Diğer alanlar bilinçli olarak
    boş bırakılır; node'lar üretildikçe doldurulur.

    Parameters
    ----------
    raw_prompt :
        User's natural-language problem statement.
    data_path :
        Optional path to a single data file (csv / xlsx / sqlite).
    auto_mode :
        When ``True``, ``clarify_node`` skips ``interrupt()`` and proceeds
        with logged default assumptions. Required for batch/benchmark runs.
    enable_refiner :
        A1 ablation flag. When ``False``, PromptRefiner node is bypassed;
        the (optionally clarified) prompt routes directly to the strategy router.
    enable_validator_retry :
        A2 ablation flag. When ``False``, failed validation does not trigger
        the code retry loop; the pipeline proceeds to the report node immediately.
    fast_only :
        A4 ablation flag. When ``True``, ``call_with_reasoning_lm`` delegates
        to the fast LM, disabling the reasoning model switch.
    self_consistency_router :
        A6 ablation flag. When ``True``, the strategy router samples 3
        completions and takes the majority vote on ``execution_path``
        instead of a single sample.

    Returns
    -------
    SolverState
        A state dict containing the seeded input fields.
    """
    state: SolverState = {
        "raw_prompt": raw_prompt,
        "auto_mode": auto_mode,
        "enable_refiner": enable_refiner,
        "enable_validator_retry": enable_validator_retry,
        "fast_only": fast_only,
        "self_consistency_router": self_consistency_router,
    }
    if data_path is not None:
        state["data_path"] = data_path
    return state


__all__ = [
    "DataBundle",
    "ExecutionPath",
    "OutputFormat",
    "SolverState",
    "SourceType",
    "empty_state",
]
