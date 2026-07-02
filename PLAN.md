# IE-Solver — Proje Planı

> Bu dosya `ie-solver` projesinin tek referans haritasıdır. Yeni bir Claude sohbeti açıldığında **ilk önce bu dosya okunur**, sonra geliştirmeye devam edilir.

---

## 1. Vizyon

Endüstri mühendisliği problemlerini uçtan uca çözen genel amaçlı bir Python kütüphanesi.

- **Girdi**: tek bir prompt + opsiyonel tek bir veri dosyası (csv | xlsx çok-sayfa | sqlite)
- **Çıktı**: 3 katmanlı rapor — Teknik / Yönetici / Aksiyon — PDF, DOCX, HTML formatlarında
- **Hedef**: akademik makale yayını
- **Kütüphane adı**: `iesolver`
- **Public API**: `iesolver.solve(prompt, data_path=None, *, auto_mode=False, enable_refiner=True, enable_validator_retry=True, fast_only=False) -> SolverState`

---

## 2. Mimari İlkeler

### 2.1 Rol Ayrımı (DSPy ⊕ LangGraph)

| Katman | Sorumluluk | Karşılığı |
|---|---|---|
| **DSPy** | "Bir LLM çağrısı ne yapmalı?" | Signature (I/O kontratı) + Module (reasoning) |
| **LangGraph** | "Aşamalar nasıl bağlanır?" | TypedDict state, conditional edges, checkpoint, döngü, interrupt |

Her LangGraph node bir DSPy Module çalıştırır. State LangGraph'ta; akıl yürütme DSPy'da. Bu ayrım makalenin methodology argümanıdır: **"reasoning units" vs "workflow engine"**.

### 2.2 Modül Sınırları

- `signatures/` — saf DSPy Signature sınıfları (test edilebilir, optimize edilebilir, declarative I/O)
- `nodes/` — LangGraph node fonksiyonları; Signature'ları çağırır, `SolverState`'i yazar; sanitization burada
- `domains/` — takılıp çıkarılabilir domain bilgisi (IE ilk pack; finans/lojistik sonra)

### 2.3 Tek Dosyalı Veri Girişi

Üç format tek soyutlamaya indirgenir:

```python
@dataclass
class DataBundle:
    tables: dict[str, pd.DataFrame]   # xlsx çok-sayfa → birden çok; csv → {"data": df}; sqlite → her tablo
    source_path: Path | None
    source_type: Literal["csv", "xlsx", "sqlite", "none"]
    def summary(self) -> str: ...     # LLM context için kısa özet
```

### 2.4 Çıktı Stratejisi

Aynı içerikten 3 format: PDF (WeasyPrint), DOCX (python-docx), HTML (Jinja2). Her birinde aynı 3 bölüm. Format seçimi `solve()` parametresi veya UI'dan.

### 2.5 Gözlenebilirlik

- LangGraph `SqliteSaver` ile node-başı checkpoint
- Stream events → Streamlit UI'da canlı durum
- Eski koddaki manuel `_log_to_file` yapısal olarak yeniden doğar (eski mantık opsiyonel kalır)

---

## 3. State Şeması

```python
from typing import TypedDict, Literal
from pathlib import Path

class SolverState(TypedDict, total=False):
    # Girdi
    raw_prompt: str
    data_path: Path | None
    data_bundle: DataBundle | None

    # Runtime flags
    auto_mode: bool                  # §3.1: interrupt() yerine log-and-continue
    auto_assumptions_log: list[str]  # auto_mode varsayım kaydı

    # Ablation flags (EVALUATION_PLAN §5)
    enable_refiner: bool             # A1: False → PromptRefiner atlanır
    enable_validator_retry: bool     # A2: False → retry döngüsü kapalı
    fast_only: bool                  # A4: True → reasoning LM de fast LM kullanır

    # Aşama 0 — GateKeeper
    cleaned_prompt: str
    data_summary: str

    # Aşama 1 — Requirement Analyst
    is_complete: bool                # §3.4 tipli DSPy output
    missing_items: list[str]         # §3.4 tipli DSPy output
    explicit_goal: str
    constraints: list[str]           # §3.4 tipli DSPy output
    output_spec: str
    user_clarification: str

    # Aşama 2 — Prompt Refiner
    essential_prompt: str
    strict_constraints: str
    problem_type: str

    # Aşama 3 — Strategy Router
    execution_path: Literal["CODE", "NO_CODE"]   # §3.4 tipli DSPy output
    reasoning_framework: str
    rationale: str

    # Aşama 4A — Analytical (NO_CODE)
    raw_result: str
    solution_path: str

    # Aşama 4B — Code Engine (CODE)
    target_algorithm: str
    target_library: str
    library_specific_constraints: str
    code_output_spec: str
    final_code: str
    execution_result: str
    is_valid: bool                   # §3.4 tipli DSPy output
    confidence_score: int            # §3.4 tipli DSPy output
    validation_notes: str
    retry_count: int

    # Faz 4 — Sensitivity + Artifacts
    sensitivity_results: str | None
    figures: Annotated[list[Path], operator.add]  # reducer: birden fazla artifact

    # Aşama 5 — Raporlar
    technical_output: str
    executive_summary: str
    action_directives: str

    # Compile
    output_path: Path | None
    output_format: Literal["pdf", "docx", "html"]

    # Telemetri (§3.5 per-node metrics)
    metrics: Annotated[dict[str, dict[str, Any]], merge_metrics]
```

---

## 4. Klasör Yapısı

```
v4/                                      ← uv workspace kökü
├── pyproject.toml                       # workspace tanımı
├── uv.lock                              # pinli bağımlılıklar (deterministik)
├── PLAN.md                              ← bu dosya
├── EVALUATION_PLAN.MD                   # Q1 makale deney protokolü
├── src/
│   └── iesolver/                        # kütüphane (src-layout)
│       ├── __init__.py                  # public API: solve(), is_interrupted(), get_fast_lm(), ...
│       ├── state.py                     # SolverState, DataBundle, empty_state()
│       ├── graph.py                     # build_graph(), open_checkpointer()
│       ├── config.py                    # Pydantic settings (temperature, seed, modeller)
│       ├── lm.py                        # call_with_fast_lm / reasoning_lm / configured_lm
│       ├── signatures/                  # 12 DSPy Signature
│       │   ├── __init__.py
│       │   ├── gatekeeper.py
│       │   ├── requirement_analyst.py
│       │   ├── prompt_refiner.py
│       │   ├── strategy_router.py
│       │   ├── analytical_engine.py
│       │   ├── algo_selector.py
│       │   ├── constraint_adapter.py
│       │   ├── output_spec.py
│       │   ├── react_code.py
│       │   ├── validator.py
│       │   ├── sensitivity.py           # Faz 4
│       │   ├── tornado_chart.py         # Faz 4
│       │   └── final_report.py
│       ├── nodes/
│       │   ├── __init__.py
│       │   ├── intake.py                # Aşama 0
│       │   ├── requirement.py           # Aşama 1
│       │   ├── clarify.py               # human-in-loop / auto_mode
│       │   ├── refine.py                # Aşama 2 (A1 ile atlanabilir)
│       │   ├── route.py                 # Aşama 3
│       │   ├── chain_branch.py          # Aşama 4A
│       │   ├── code_branch/             # Aşama 4B (alt-akış)
│       │   │   ├── __init__.py          # code_branch_node
│       │   │   ├── algo_select.py
│       │   │   ├── constraint_adapt.py
│       │   │   ├── output_spec.py
│       │   │   └── generate.py          # ReAct + sandbox
│       │   ├── validate.py              # (A2 ile retry atlanabilir)
│       │   ├── sensitivity.py           # Faz 4: dual-first
│       │   ├── artifacts.py             # Faz 4: tornado chart
│       │   └── report.py                # Aşama 5
│       ├── io/
│       │   └── data_loader.py           # csv|xlsx|sqlite → DataBundle
│       ├── sandbox/
│       │   └── runner.py                # subprocess + timeout; RunResult
│       └── observability/
│           └── metrics.py               # @instrument, merge_metrics, record_llm_usage
├── ie-eval/                             # uv workspace member — evaluation harness
│   ├── pyproject.toml
│   └── src/ie_eval/
│       ├── __init__.py                  # Problem, run_one, run_dataset, ResultStore, ablasyon factory'leri
│       ├── problem.py                   # Problem, GroundTruth dataclass'ları
│       ├── runner.py                    # run_one (correctness_fn destekli), run_dataset
│       ├── validator.py                 # numerical_match, check_feasibility
│       ├── metrics.py                   # ProblemMetrics, extract_metrics
│       ├── store.py                     # ResultStore — SQLite (metadata_json sütunlu)
│       ├── baselines.py                 # single_shot_solve, single_shot_cot_solve
│       ├── ablations.py                 # make_a1..a5_solve, make_a3_correctness_fn
│       ├── datasets/
│       │   ├── base.py                  # Dataset protokolü
│       │   ├── ie_case.py               # IE-Case 6 problem (fixture üretici)
│       │   ├── nl4opt.py                # NL4Opt JSONL adaptörü
│       │   ├── industryor.py            # IndustryOR JSONL adaptörü
│       │   └── _jsonl_common.py         # ortak JSONL yardımcıları
│       └── analysis/
│           ├── summary.py               # summarize_by_config, per_problem_correctness, compare_configs
│           ├── stats.py                 # mcnemar_test, bootstrap_diff_ci (scipy-free)
│           └── report.py                # format_summary, format_comparison
├── tests/                               # iesolver kütüphane testleri
│   └── ...
└── ie-eval/tests/                       # ie-eval harness testleri (143 test)
    ├── test_ablations.py
    ├── test_analysis_stats.py
    ├── test_analysis_summary.py
    ├── test_baselines.py
    ├── test_ie_case_extended.py
    ├── test_industryor.py
    ├── test_metadata_pipeline.py
    ├── test_nl4opt.py
    ├── test_runner.py
    ├── test_store.py
    └── test_validator.py
```

---

## 5. 5 Fazlık Yol Haritası

Her fazın sonunda **çalışan bir smoke test** olacak. Fazlar arası birikim.

### Faz 1 — Çatı (Foundation)
**Çıktı**: çalışan minimum iskelet
- `pyproject.toml` (DSPy, LangGraph, pandas, openpyxl, pydantic-settings, uv)
- `state.py`, `config.py`, `graph.py`
- `io/data_loader.py` (csv/xlsx-çok-sayfa/sqlite → DataBundle)
- 3 dummy node (intake → refine → report) — gerçek LLM çağırmaz, state'i taşır
- `SqliteSaver` aktif
- `iesolver.solve("hello")` uçtan uca akar

**Smoke test**: `solve("merhaba")` çalışır, state diske yazılır, replay edilebilir.

### Faz 2 — Signature Aktarımı
**Çıktı**: 12 Signature yeni iskelete taşınmış, gerçek LLM çağrıları yapılıyor
- Eski `signatures.py` 11 dosyaya bölünür (docstring'ler aynen)
- Eski Module'ların `forward` mantığı (özellikle sanitization) `nodes/` altına taşınır
- Conditional edges: `missing_items` → human-in-loop (interrupt), `NO_CODE` vs `CODE`
- Aşama 4A çalışır; Aşama 4B henüz tek dummy (Faz 3'te dolar)

**Smoke test**: bir NO_CODE problemi (örnek: "BOM nedir, nasıl hazırlanır" — kavramsal) uçtan uca akar; rapor üretilir (henüz düz metin).

### Faz 3 — Kod Branch'i
**Çıktı**: Aşama 4B tam çalışır
- AlgoSelector → ConstraintAdapter → OutputSpec → Generate → Execute → Validate
- Sandbox: subprocess + timeout
- Retry döngüleri (4B.5 → 4B.4 max 3; library error → 4B.2)
- Per-node LM context: fast varsayılan, kod üretiminde reasoning model

**Smoke test**: bir LP problemi (EOQ veya tek depo transportation) doğru sayısal sonuç verir.

### Faz 4 — Yeni Node'lar ✅
**Çıktı**: Sensitivity + Artifacts + Ön düzeltmeler
- `SensitivityAnalysis` node: dual-first (shadow price/reduced cost), perturbation fallback (±%5, ±%10)
- `ArtifactGenerator` node: matplotlib tornado chart, `state["figures"]`'a Path
- Ön düzeltmeler: auto_mode (§3.1), tipli DSPy 3.x output'lar (§3.4), per-node metrics (§3.5), temperature=0+seed (§3.7)
- Graph: 3-yönlü `_route_after_validate` (retry | sensitivity | report)

**Smoke test**: aynı LP problemi, çıktıda tornado chart üretir. (e2e: LLM çağrısı gerektirir)

### Faz 4.5 — Evaluation Harness + Ablasyonlar ✅
**Çıktı**: `ie-eval/` paketi — uçtan uca değerlendirme altyapısı

**Tamamlananlar:**
- IE-Case benchmark seti: 6 problem (EOQ, transport-2×3, multi-product-inventory.xlsx, transport-3×2.csv, assignment-3×3.sqlite, ABC-classification NO_CODE)
- NL4Opt adaptörü (temizlenmiş JSONL, tolerans 1e-4)
- IndustryOR adaptörü (sector/industry metadata)
- Deterministik validator: `numerical_match`, `check_feasibility`, `FeasibilityCheck`
- Runner: `run_one(correctness_fn=...)`, `run_dataset(n_runs=...)`
- SQLite store: `metadata_json` sütunlu backward-compat şema
- Baseline'lar: `single_shot_solve`, `single_shot_cot_solve`
- Analysis: `summarize_by_config`, `compare_configs`, `per_problem_correctness`, `metadata_filter` kırılımı
- İstatistik: `mcnemar_test` (scipy-free, exact + χ²), `bootstrap_diff_ci` (seedable)
- **Ablasyonlar A1–A5** (EVALUATION_PLAN §5):
  - `iesolver.solve()` yeni param'ları: `enable_refiner`, `enable_validator_retry`, `fast_only`
  - `ie_eval/ablations.py`: `make_a1/a2/a3/a4/a5_solve`, `make_a3_correctness_fn`
  - A5 (MIPROv2): framework hazır, `scripts/optimize_mipro.py` bekliyor

**Test durumu**: 143 test, tümü geçiyor

**Bekleyen:**
- NL4Opt / IndustryOR temizlenmiş JSONL dosyaları (kullanıcı aksiyonu — OptiMind supplementary)
- E2E gerçek LLM koşusu IE-Case üzerinde (ücretli API kotası gerekli)
- A5: MIPROv2 optimizasyon script'i

### Faz 5 — Çıktı + UI *(makale sonrasına ertelendi)*
**Çıktı**: production-ready
- `ReportWriter`: PDF (WeasyPrint), DOCX (python-docx), HTML (Jinja2)
- Streamlit UI: `text_area` + `file_uploader` + canlı durum (LangGraph stream events)
- 2–3 örnek IE problemi e2e test
- README, makale taslağı için figür arşivi

---

## 6. Sabitlenmiş Kararlar

| Karar | Seçim | Gerekçe |
|---|---|---|
| Workflow engine | LangGraph | Typed state, conditional edges, checkpoint, interrupt |
| Reasoning units | DSPy | Eski koddan korunan yatırım, Signature optimize edilebilir |
| Veri girişi | Tek dosya: csv / xlsx çok-sayfa / sqlite | Eski sistemle uyum |
| Çıktı formatı | PDF + DOCX + HTML | Yönetici / akademik / web için ayrı kullanım |
| Domain stratejisi | Pluggable pack (IE ilk) | Mimari genel kalır, makale "extensible" argümanı kazanır |
| Persistence | SqliteSaver | Replay + ücretsiz API kotası koruması |
| UI | Streamlit (kütüphane dışı, `apps/` altında) | Library/UI ayrımı temiz |
| Sandbox (başlangıç) | subprocess + timeout | Basit; Docker'a sonra geçilebilir |
| Package manager | uv | Hızlı, modern, kilit dosyası deterministik |
| Python | 3.11+ | TypedDict total=False, match-case |
| Yorum dili | Türkçe (mimari kararlar) + İngilizce (docstring) | Eski kodla tutarlı, makale için iki dilli |

---

## 7. Eski Koddan Birebir Taşınan

- `signatures.py` → 11 Signature sınıfı + docstring'leri (G-O-C, Bifurcation Logic dahil)
- Sanitization desenleri: `str → bool` ve `str → int` dönüşümleri (`is_complete`, `is_valid`, `confidence_score`)
- ChainOfThought vs Predict seçimleri (her aşamada eski koddaki gerekçe korunur)
- Fast/Reasoning model anahtarlaması (eski `_resume_pipeline` içindeki) → LangGraph node-level `dspy.context(lm=...)`
- DataProfiler kavramı (ilgili dosya yüklenince entegre edilir)
- PythonREPL kavramı (ilgili dosya yüklenince entegre edilir)

## 8. Yeni Olan

- LangGraph orchestration (eski: prosedürel `_resume_pipeline()`)
- TypedDict state (eski: `IEAgent` instance attribute'ları)
- Checkpointing (eski: manuel markdown log)
- Human-in-loop via `interrupt()` (eski: `provide_missing_info()` method)
- auto_mode: interrupt yerine log-and-continue (benchmark koşuları için)
- Tipli DSPy 3.x output'lar (`bool`, `int`, `Literal`, `list[str]`) — string sanitization kalktı
- Per-node telemetri (`@instrument` decorator, `merge_metrics` reducer)
- SensitivityAnalysis node (dual-first LP sensitivity)
- ArtifactGenerator node (matplotlib tornado chart)
- Ablasyon flag'leri (`enable_refiner`, `enable_validator_retry`, `fast_only`) — `solve()` parametreleri
- `call_with_configured_lm` (A4: fast_only routing)
- `ie-eval/` evaluation harness: IE-Case + NL4Opt + IndustryOR adaptörleri, deterministik validator, runner, store, analysis, ablasyon factory'leri
- PDF/DOCX/HTML üç-formatlı rapor (eski: stdout) *(Faz 5)*
- Streamlit UI (eski: CLI) *(Faz 5)*
- Tek-dosyalı veri loader (csv/xlsx-çok-sayfa/sqlite tek soyutlama)

---

## 9. API & Maliyet

- Şu an: ücretsiz Gemini tier (`gemini-3.1-flash-lite-preview`)
- Faz 1+2 bu kotada sorunsuz
- Faz 3+ için reasoning_model ihtiyacı artarsa upgrade gerekebilir
- Checkpointing baştan açık → başarısız run'lar API kotasını çöpe atmaz, replay ücretsiz

---

