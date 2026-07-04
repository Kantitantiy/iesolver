# iesolver — Tam Sistem Dokümantasyonu

> Bu dosya, projeyi hiç görmemiş bir yapay zeka için yazılmıştır.
> Mimari, veri akışı, her bileşen, optimizasyon ve test altyapısı eksiksiz açıklanır.

---

## 1. Proje Özeti

**iesolver**, doğal dil olarak ifade edilen Endüstri Mühendisliği (IE) ve Yöneylem Araştırması (OR) problemlerini uçtan uca çözen bir LLM ajanı kütüphanesidir. Kullanıcı bir metin yazar; sistem problemi anlar, uygun çözüm yolunu seçer, Python kodu üretir, çalıştırır, doğrular ve üç katmanlı profesyonel rapor üretir.

**Teknoloji stack'i:**
- **DSPy 3.x** — LLM çağrıları için optimize edilebilir Signature modülleri
- **LangGraph** — durum makinesi orchestration, checkpoint, human-in-loop interrupt
- **LiteLLM / Gemini** — model bağımsız LLM erişim katmanı
- **fpdf2 / python-docx / Jinja2** — üç formatlı rapor çıktısı
- **uv** — deterministik Python paket yönetimi, workspace desteği

**Akademik bağlam:** Q1 dergi makalesinin kod tabanı. Temel iddia: DSPy Signature optimizasyonu + LangGraph orchestration, OR problemlerinde tek-atış LLM baseline'larını anlamlı ölçüde geçer. `ie-eval/` bu iddiayı ölçer.

---

## 2. Dizin Yapısı

```
v4/
├── src/iesolver/               ← Çekirdek kütüphane (Python paketi)
│   ├── __init__.py             ← Public API: solve(), write_report(), ReportWriter
│   ├── config.py               ← Pydantic-backed runtime ayarları (.env okur)
│   ├── state.py                ← SolverState TypedDict + DataBundle dataclass
│   ├── graph.py                ← LangGraph pipeline kurulumu + edge predicates
│   ├── lm.py                   ← LM singleton yönetimi, call_with_fast/reasoning_lm
│   ├── _optimization.py        ← MIPROv2 altyapısı: IESolverProgram, load_compiled_graph
│   ├── signatures/             ← 12 DSPy Signature sınıfı (her biri bir LLM görevi)
│   ├── nodes/                  ← 11 LangGraph node (her biri bir pipeline aşaması)
│   │   └── code_branch/        ← 4 node: algo_select, constraint_adapt, output_spec, generate
│   ├── io/                     ← DataBundle yükleyici (csv/xlsx/sqlite → tek soyutlama)
│   ├── observability/          ← Per-node telemetri: @instrument decorator, merge_metrics
│   ├── sandbox/                ← subprocess kod yürütücü (timeout + güvenli izolasyon)
│   └── report/                 ← Faz 5a: HTML/DOCX/PDF rapor üretici
│       ├── _html.py            ← Jinja2 + markdown-it-py → self-contained HTML
│       ├── _docx.py            ← python-docx DOCX üretici
│       ├── _pdf.py             ← fpdf2 PDF üretici (platform bağımsız)
│       └── templates/
│           └── report.html.j2  ← Responsive HTML şablon (print CSS dahil)
│
├── ie-eval/                    ← Değerlendirme harness (ayrı uv workspace üyesi)
│   └── src/ie_eval/
│       ├── __init__.py         ← Public re-export
│       ├── problem.py          ← Problem + GroundTruth dataclass'ları
│       ├── runner.py           ← run_one() + run_dataset() — toplu koşu
│       ├── store.py            ← SQLite sonuç deposu (RunRecord kalıcılığı)
│       ├── metrics.py          ← ProblemMetrics + extract_metrics()
│       ├── validator.py        ← numerical_match() deterministik doğrulayıcı
│       ├── ablations.py        ← A1-A5 solve_fn/correctness_fn factory'leri
│       ├── baselines.py        ← Single-shot + CoT baseline solve_fn'leri
│       ├── datasets/           ← IE-Case, NL4Opt, IndustryOR adaptörleri
│       └── analysis/           ← İstatistik: McNemar, bootstrap CI, özet rapor
│
├── scripts/
│   └── optimize_mipro.py       ← MIPROv2 CLI (A5 ablation eğitimi)
│
├── tests/                      ← iesolver birim + entegrasyon testleri (58 test)
├── ie-eval/tests/              ← ie-eval testleri (144 test)
│
├── pyproject.toml              ← iesolver paket tanımı + uv workspace kök
├── PLAN.md                     ← 5-fazlı yol haritası
├── DESIGN_REVIEW.md            ← Mimari karar gerekçeleri
├── EVALUATION_PLAN.MD          ← Ablasyon ve deney tasarımı
└── METHODOLOGY_NOTES.md        ← Akademik bağlam notları
```

---

## 3. LangGraph Pipeline Topolojisi

Pipeline 11 node ve iki ana çözüm yolundan oluşur:

```
START
  │
  ▼
[intake]           ← GateKeeper: prompt temizleme, veri profili, ön denetim
  │
  ▼
[requirement]      ← RequirementAnalyst: is_complete?, missing_items, explicit_goal
  │
  ├─[eksik]──► [clarify] ──[interactive]──► requirement (döngü)
  │                      └─[auto_mode]────► refine (ya da route, A1 aktifse)
  │
  ▼ [tam]
[refine]           ← PromptRefiner: essential_prompt, strict_constraints, problem_type
  │                  (A1 ablation: bu node atlanır)
  ▼
[route]            ← StrategyRouter: execution_path = CODE | NO_CODE
  │
  ├─[NO_CODE]──► [chain_branch] ──────────────────────────────────┐
  │              (analytical, formül tabanlı)                     │
  │                                                               │
  └─[CODE]─────► [code_branch]                                    │
                    │  (algo_select → constraint_adapt →          │
                    │   output_spec → generate/ReAct)             │
                    ▼                                             │
                 [validate]                                       │
                  /      \                                        │
       [geçersiz+retry] [geçersiz+limit/A2]  [geçerli]            │
              │                  │               │                │
        code_branch           [report] ◄─────── [sensitivity]     │
        (max 3 kez)                              │                │
                                              [artifacts]         │
                                                  │               │
                                               [report] ◄─────────┘
                                                  │
                                                 END
```

### Node → Signature eşlemesi

| Node | DSPy Modülü | Signature | LM Tier |
|---|---|---|---|
| `intake` | `ChainOfThought` | `GatekeeperSignature` | Fast |
| `requirement` | `Predict` | `RequirementAnalystSignature` | Fast |
| `clarify` | interrupt / log | — | — |
| `refine` | `ChainOfThought` | `PromptRefinerSignature` | Fast |
| `route` | `ChainOfThought` | `StrategyRouterSignature` | Fast |
| `chain_branch` | `ChainOfThought` | `AnalyticalEngineSignature` | Fast |
| `code_branch/algo_select` | `ChainOfThought` | `AlgoSelectorSignature` | Fast |
| `code_branch/constraint_adapt` | `Predict` | `ConstraintAdapterSignature` | Fast |
| `code_branch/output_spec` | `Predict` | `OutputSpecSignature` | Fast |
| `code_branch/generate` | `ReAct` | `ReActCodeGeneratorSignature` | **Reasoning** |
| `validate` | `ChainOfThought` | `ValidatorSignature` | Fast |
| `sensitivity` | `ChainOfThought` | `SensitivitySignature` | **Reasoning** |
| `artifacts` | `ChainOfThought` | `TornadoChartSignature` | Fast |
| `report` | `ChainOfThought` | `FinalReportSignature` | Fast |

---

## 4. SolverState — Paylaşımlı Durum Şeması

`SolverState`, `TypedDict(total=False)` olarak tanımlanmıştır. Her node yalnızca yazdığı alanları döndürür; LangGraph partial-merge ile birleştirir. `total=False` zorunludur: aksi halde her node tüm alanları doldurmak zorunda kalır.

```python
class SolverState(TypedDict, total=False):
    # Girdi
    raw_prompt: str
    data_path: Path | None
    data_bundle: DataBundle | None

    # Runtime flags
    auto_mode: bool                    # True → interrupt yok, batch koşusu
    auto_assumptions_log: list[str]    # clarify_node varsayımları

    # Ablation flags (EVALUATION_PLAN §5)
    enable_refiner: bool               # A1: False → refine node atlanır
    enable_validator_retry: bool       # A2: False → retry döngüsü kapalı
    fast_only: bool                    # A4: True → tüm LLM çağrıları fast LM

    # Aşama 0 — GateKeeper
    cleaned_prompt: str
    data_summary: str

    # Aşama 1 — Requirement
    is_complete: bool
    missing_items: list[str]
    explicit_goal: str
    constraints: list[str]
    output_spec: str
    user_clarification: str

    # Aşama 2 — Refiner
    essential_prompt: str
    strict_constraints: str
    problem_type: str

    # Aşama 3 — Router
    execution_path: Literal["CODE", "NO_CODE"]
    reasoning_framework: str
    rationale: str

    # Aşama 4A — NO_CODE
    raw_result: str
    solution_path: str

    # Aşama 4B — CODE
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

    # Faz 4 — Sensitivity + Artifacts
    sensitivity_results: str | None
    figures: Annotated[list[Path], operator.add]  # reducer: biriktir

    # Faz 5 — Rapor
    technical_output: str
    executive_summary: str
    action_directives: str
    output_path: Path | None
    output_format: Literal["pdf", "docx", "html"]

    # Telemetri
    metrics: Annotated[dict[str, dict], merge_metrics]  # reducer: topla
```

### DataBundle

```python
@dataclass(slots=True)
class DataBundle:
    tables: dict[str, pd.DataFrame]  # "data" (csv), sheet_name (xlsx), table_name (sqlite)
    source_path: Path | None
    source_type: Literal["csv", "xlsx", "sqlite", "none"]

    def summary(self, max_rows=5, max_cols=20) -> str: ...
    # Token-dostu metin özeti; LLM prompt'una doğrudan gömülür
```

`DataBundle.summary()` çıktısı `data_summary` alanına yazılır ve GateKeeper ile AlgoSelector tarafından okunur.

---

## 5. DSPy Signature Sistemi

Her DSPy Signature bir LLM görevini tip-güvenli şekilde tanımlar. `InputField` → `OutputField` sözleşmesi MIPROv2 optimizasyonu için gereklidir.

### Tipli OutputField'lar (DSPy 3.x)

```python
class RequirementAnalystSignature(dspy.Signature):
    # InputField'lar
    raw_prompt: str = dspy.InputField(...)
    data_summary: str = dspy.InputField(...)

    # Tipli OutputField'lar — string sanitization kaldırıldı
    is_complete: bool = dspy.OutputField(...)         # doğrudan bool
    missing_items: list[str] = dspy.OutputField(...)  # doğrudan liste
    confidence_score: int = dspy.OutputField(...)     # doğrudan int
    explicit_goal: str = dspy.OutputField(...)
    constraints: list[str] = dspy.OutputField(...)
    output_spec: str = dspy.OutputField(...)
```

Eski sistemdeki `str → bool` string sanitization kaldırıldı; DSPy 3.x bunu otomatik halleder.

### 12 Signature Listesi

| Dosya | Sınıf | Amaç |
|---|---|---|
| `gatekeeper.py` | `GatekeeperSignature` | Prompt temizleme, veri profili, alan tespiti |
| `requirement_analyst.py` | `RequirementAnalystSignature` | Eksik bilgi tespiti, goal/constraint çıkarımı |
| `prompt_refiner.py` | `PromptRefinerSignature` | Yapısal yeniden formülasyon |
| `strategy_router.py` | `StrategyRouterSignature` | CODE/NO_CODE karar bifurcation |
| `analytical_engine.py` | `AnalyticalEngineSignature` | NO_CODE: analitik çözüm yolu |
| `algo_selector.py` | `AlgoSelectorSignature` | Algoritma + kütüphane seçimi (PuLP, scipy...) |
| `constraint_adapter.py` | `ConstraintAdapterSignature` | Kütüphane-spesifik kısıt dönüşümü |
| `output_spec.py` | `OutputSpecSignature` | Kod çıktı formatı belirleme |
| `react_code.py` | `ReActCodeGeneratorSignature` | ReAct döngüsünde Python kodu üretimi |
| `validator.py` | `ValidatorSignature` | Çalıştırma sonucu doğrulama |
| `sensitivity.py` | `SensitivitySignature` | LP sensitivity analizi (dual-first) |
| `tornado_chart.py` | `TornadoChartSignature` | Tornado chart matplotlib kodu üretimi |
| `final_report.py` | `FinalReportSignature` | 3 katmanlı rapor sentezi |

---

## 6. LM Yönetimi — `lm.py`

### İki LM Tier

```python
# Fast LM — triage/routing/report node'ları için (ucuz, hızlı)
_fast_lm: dspy.LM = dspy.LM(
    settings.fast_model,          # default: gemini/gemini-3.1-flash-lite-preview
    api_key=...,
    temperature=0.0,              # deterministik (DESIGN_REVIEW §3.7)
    seed=42,                      # LiteLLM'e iletilir; desteklemeyen provider sessizce atlar
)

# Reasoning LM — kod üretimi ve sensitivity için (ağır, pahalı)
_reasoning_lm: dspy.LM = dspy.LM(settings.reasoning_model, ...)
```

**Kritik karar:** LangGraph her node'u `copy_context()` ile çalıştırır; global `dspy.configure(lm=...)` node'lara propagate olmaz. Bu nedenle her LLM çağrısı `with dspy.context(lm=...)` bloğuna alınır:

```python
def call_with_fast_lm(module, **kwargs):
    before = len(fast_lm.history)
    with dspy.context(lm=fast_lm):
        result = module(**kwargs)
    record_llm_usage(fast_lm.history[before:])  # telemetri
    return result

def call_with_reasoning_lm(module, **kwargs): ...

# A4 ablation router — fast_only=True ise reasoning çağrılarını fast'e yönlendirir
def call_with_configured_lm(module, *, fast_only=False, **kwargs):
    if fast_only:
        return call_with_fast_lm(module, **kwargs)
    return call_with_reasoning_lm(module, **kwargs)
```

**litellm.drop_params = True** — `lm.py` import anında set edilir; Gemini'nin desteklemediği `seed` gibi parametreler LiteLLM tarafından sessizce atılır.

---

## 7. Observability — Per-Node Telemetri

Her node `@instrument("node_name")` decorator'ı taşır:

```python
@instrument("intake")
def intake_node(state: SolverState) -> SolverState:
    ...
```

Decorator:
1. Contextvar `_current_bucket` ile bir ölçüm bucket'ı açar
2. Node'u çalıştırır; `call_with_fast/reasoning_lm` DSPy history delta'sını bucket'a yazar
3. `latency_ms, tokens_in, tokens_out, cost_usd, llm_calls, invocations, error_class` ölçer
4. Partial state'e `{"metrics": {"intake": {...}}}` ekler

### merge_metrics Reducer

Aynı node'un birden fazla çağrısı (retry döngüsü) sayısal olarak toplanır:

```python
# Annotated field — LangGraph bu reducer'ı kullanır
metrics: Annotated[dict[str, dict], merge_metrics]
```

`code_branch` 3 kez çalışırsa `metrics["code_branch"]["llm_calls"]` toplam çağrı sayısını verir.

### Metrik şeması (NODE_METRIC_KEYS)

```
latency_ms | tokens_in | tokens_out | cost_usd | llm_calls | invocations | error_class
```

---

## 8. Sandbox — Güvenli Kod Yürütme

```python
@dataclass
class RunResult:
    success: bool
    stdout: str
    stderr: str

def run_code(code: str, timeout: int = 60, workdir: Path = ...) -> RunResult:
    # subprocess.run ile izole Python process'i
    # timeout: settings.timeout_seconds (default 60s)
    # stdout/stderr yakalanır, exit code kontrol edilir
```

Üretilen Python kodu `generate` node'undan sonra sandbox'ta çalıştırılır. `RunResult.stdout` → `execution_result` alanına yazılır. `validate` node'u bu çıktıyı LLM ile doğrular.

---

## 9. Çözüm Akışı — Adım Adım

### `solve()` — Public Entry Point

```python
from iesolver import solve

state = solve(
    prompt="EOQ: D=10000, S=50, H=2 — Q* nedir?",
    data_path=None,          # opsiyonel: csv/xlsx/sqlite
    auto_mode=True,          # batch koşusu için
    thread_id="run-001",     # resume için aynı id tekrar kullanılır
    checkpoint_db=Path(".../.iesolver/ckpt.sqlite"),
    # Ablation flags
    enable_refiner=True,     # A1
    enable_validator_retry=True,  # A2
    fast_only=False,         # A4
)
# state["executive_summary"], state["execution_result"], state["metrics"] okunabilir
```

### İç akış:

1. `empty_state(raw_prompt, ..., ablation_flags)` → seed dict oluştur
2. `SqliteSaver` checkpoint bağlantısı aç
3. `build_graph(checkpointer).invoke(seed, config={"configurable": {"thread_id": ...}})`
4. LangGraph pipeline düğümleri sırayla / koşullu çalışır
5. Final `SolverState` dön

### Resume (Checkpoint)

`thread_id` aynı tutularak `solve()` tekrar çağrılırsa, LangGraph checkpoint'ten kaldığı yerden devam eder. Bu, human-in-loop interrupt sonrası yanıt sağlandığında kullanılır.

---

## 10. Konfigürasyon — `config.py`

Tüm ayarlar `IESolverSettings(BaseSettings)` içinde, `.env` dosyasından okunur:

| Alan | Tip | Default | Açıklama |
|---|---|---|---|
| `google_api_key` | str | `""` | Gemini API anahtarı |
| `fast_model` | str | `gemini/gemini-3.1-flash-lite-preview` | Triage/routing LM |
| `reasoning_model` | str | `gemini/gemini-3.1-flash-lite-preview` | Kod üretim LM |
| `temperature` | float | `0.0` | Deterministik (tüm LM'lere uygulanır) |
| `lm_seed` | int | `42` | LiteLLM seed (destekleyen provider'larda) |
| `max_retries` | int | `3` | Validate→retry döngüsü üst sınırı |
| `timeout_seconds` | int | `60` | Sandbox subprocess timeout |
| `budget_limit_usd` | float | `5.0` | Günlük API maliyet limiti |
| `checkpoint_db_path` | Path | `.iesolver/checkpoints.sqlite` | LangGraph SqliteSaver |
| `artifacts_dir` | Path | `.iesolver/artifacts` | Matplotlib/Plotly PNG'leri |
| `output_dir` | Path | `.iesolver/outputs` | PDF/DOCX/HTML raporlar |
| `sandbox_workdir` | Path | `.iesolver/sandbox` | Subprocess scratch alanı |
| `default_output_format` | str | `"html"` | Varsayılan rapor formatı |

`.env` örneği:
```ini
GOOGLE_API_KEY=AIza...
FAST_MODEL=gemini/gemini-3.1-flash-lite-preview
REASONING_MODEL=gemini/gemini-2.0-flash-thinking-exp
TEMPERATURE=0.0
LM_SEED=42
```

---

## 11. Rapor Üretici — Faz 5a

```python
from iesolver import write_report

write_report(state, "output/report.html", format="html")
write_report(state, "output/report.docx", format="docx")
write_report(state, "output/report.pdf",  format="pdf")

# Sınıf arayüzü
from iesolver import ReportWriter
ReportWriter(state).write("output/report.pdf", format="pdf")
```

### Üç Format

**HTML** (`_html.py`):
- Jinja2 şablon (`templates/report.html.j2`)
- `markdown-it-py` ile LLM çıktılarını HTML'e render eder
- Figürler base64 ile gömülür → self-contained tek dosya
- Responsive CSS, print media query (tarayıcıdan PDF yazdırılabilir)
- Bölümler: Executive Summary, Technical Analysis, Sensitivity, Action Directives, Charts, Metrics

**DOCX** (`_docx.py`):
- `python-docx` ile Heading/Paragraph stilleri
- Metrik tablosu (Word tablo stilinde)
- `Path.exists()` kontrolü sonrası figür embed
- Markdown işaretleri regex ile temizlenir

**PDF** (`_pdf.py`):
- `fpdf2` — saf Python, GTK/Pango/Cairo gerekmez, Windows'ta çalışır
- Helvetica core font; Latin-1 dışı karakterler `encode('latin-1', 'replace')` ile normalize edilir
- Başlık bloğu (mavi dolgu), bölüm başlıkları (gri arka plan), metrik tablosu
- Sayfa altbilgisi otomatik

### State → Rapor alanları

| State alanı | Raporda kullanımı |
|---|---|
| `explicit_goal` | Rapor başlığı altı — problem özeti |
| `executive_summary` | "Executive Summary" bölümü |
| `technical_output` | "Technical Analysis" bölümü |
| `action_directives` | "Action Directives" bölümü |
| `sensitivity_results` | "Sensitivity Analysis" bölümü (varsa) |
| `figures` | "Charts" bölümü (PNG Path listesi) |
| `execution_result` | Başlık bloğundaki "Optimal Result" pill'i |
| `is_valid` | "Validated / Not Validated" badge |
| `metrics` | "Performance Metrics" tablosu |

---

## 12. ie-eval — Değerlendirme Harness'i

`ie-eval/`, `iesolver`'ı ayrı bir uv workspace üyesi olarak kullanır. Tek import noktası:
```python
from iesolver import solve  # iç modüllere dokunma yasağı
```

### Problem / GroundTruth

```python
@dataclass
class GroundTruth:
    objective_value: float | None  # None → NO_CODE / kavramsal
    tolerance_rel: float = 1e-3    # göreli tolerans (%0.1)
    solution: dict[str, float]     # karar değişkenleri (feasibility check için)
    feasibility_fn: Callable | None  # (solution_dict) -> list[str] (ihlaller)

@dataclass
class Problem:
    id: str
    prompt: str
    data_path: Path | None
    ground_truth: GroundTruth
    metadata: dict  # {"benchmark": "IE-Case", "problem_type": "EOQ", ...}
```

### IE-Case Benchmark (6 Problem)

| Problem | Tip | Veri Dosyası |
|---|---|---|
| EOQ | Analitik | Yok |
| Transport 2×3 | LP (CODE) | Yok |
| Multi-product Inventory | LP/MIP | `multi_product_inventory.xlsx` |
| Transport 3×2 | LP | `transportation_3x2.csv` |
| Assignment 3×3 | IP | `assignment_3x3.sqlite` |
| ABC Classification | NO_CODE (analitik) | Yok |

### Desteklenen Dataset'ler

| Dataset | Adaptör | Format |
|---|---|---|
| IE-Case (yerleşik) | `ie_case.py` | Hardcoded Problem listesi |
| NL4Opt | `nl4opt.py` | JSONL (OptiMind supplementary) |
| IndustryOR | `industryor.py` | JSONL (OptiMind supplementary) |

JSONL common altyapı (`_jsonl_common.py`) her iki adaptör tarafından paylaşılır.

### Runner

```python
# Tek problem koşusu
rec = run_one(
    problem,
    config_id="baseline",
    run_idx=0,
    solve_fn=None,           # None → iesolver.solve (production)
    correctness_fn=None,     # A3 ablation override
)

# Toplu koşu
records = run_dataset(
    dataset,                 # Dataset nesnesi veya Problem listesi
    config_id="A1_no_refiner",
    n_runs=1,
    on_result=store.persist, # her sonuçta callback
    solve_fn=make_a1_solve(auto_mode=True),
)
```

### Metrikler

```python
@dataclass
class ProblemMetrics:
    numerical_match: bool    # |output - gt| / |gt| ≤ tolerance_rel
    feasible: bool | None    # feasibility_fn sonucu (varsa)
    elapsed_s: float
    # iesolver per-node metrikleri (varsa):
    total_tokens_in: int
    total_tokens_out: int
    total_cost_usd: float
    total_llm_calls: int
    retry_count: int
```

**Deterministik numerical_match** (`validator.py`):
- `execution_result` string'inden ilk float değeri çıkarır (regex)
- `|pred - gt| / max(|gt|, 1e-9) ≤ tolerance_rel` kontrolü

### Sonuç Deposu (SQLite)

```python
store = ResultStore("results.sqlite")
store.persist(run_record)
records = store.query(config_id="baseline", benchmark="NL4Opt")
```

Backward-compatible `ALTER TABLE` ile metadata_json sütunu eklenmiştir.

### İstatistiksel Analiz (scipy-free)

```python
from ie_eval.analysis.stats import mcnemar_test, bootstrap_diff_ci

# McNemar testi (eşleştirilmiş problem seti üzerinde iki config karşılaştırması)
result = mcnemar_test(b=12, c=5)
# b = A doğru B yanlış; c = A yanlış B doğru
# n ≥ 25 → chi-square with continuity correction
# n < 25  → exact binomial (math.erfc/math.comb ile, scipy yok)

# Bootstrap CI (percentile yöntemi)
ci = bootstrap_diff_ci(a_correct=[True, False, ...], b_correct=[...], seed=42)
```

---

## 13. Ablasyon Sistemi — EVALUATION_PLAN §5

Beş ablasyon, makale RQ2'yi ("hangi bileşen ne kadar katkıda bulunuyor?") cevaplar.

### A1 — PromptRefiner Kapalı

```python
solve_fn = make_a1_solve(auto_mode=True)
# enable_refiner=False → refine node atlanır
# Graph: requirement/clarify → route (refine bypass edilir)
```

**Etki:** `_route_after_requirement` ve `_route_after_clarify` `enable_refiner=False` gördüğünde `"refine"` yerine `"route"` döner. LangGraph statik grafında her iki destination da tanımlı olduğu için dinamik bypass güvenlidir.

### A2 — Validator Retry Kapalı

```python
solve_fn = make_a2_solve(auto_mode=True)
# enable_validator_retry=False → is_valid=False olsa bile code_branch'e dönülmez
```

**Etki:** `_route_after_validate`: `enable_retry=False` → her zaman `"report"` döner (tek geçiş).

### A3 — Yalnızca LLM Validator Sinyali

```python
correctness_fn = make_a3_correctness_fn()
rec = run_one(problem, correctness_fn=correctness_fn)
# state["is_valid"] → True/False; deterministik numerical_match bypass edilir
```

**Etki:** `run_one` içinde `correctness_fn` verilmişse `metrics.numerical_match` override edilir. `iesolver` pipeline değişmez; yalnızca değerlendirme mantığı değişir.

### A4 — Fast-Only (Reasoning LM Kapalı)

```python
solve_fn = make_a4_solve(auto_mode=True)
# fast_only=True → call_with_configured_lm her çağrıda fast LM kullanır
```

**Etki:** `generate.py` ve `sensitivity.py` node'ları `call_with_configured_lm(_module, fast_only=state.get("fast_only", False))` çağırır. `fast_only=True` olduğunda reasoning LM çağrısı fast LM'e yönlendirilir.

### A5 — MIPROv2 Optimize Edilmiş Signature'lar

```python
solve_fn = make_a5_solve(compiled_path=Path("compiled/iesolver_mipro.json"))
# İlk çağrıda load_compiled_graph(path) singleton'ları in-place günceller
```

**Etki:** DSPy module singleton'ları (`_gatekeeper`, `_analyst`, vb.) in-place güncellenir. Sonraki `solve()` çağrıları MIPROv2 tarafından optimize edilmiş prompt/few-shot örneklerle çalışır.

### Ablasyon Factory Özeyi

| Factory | Signature | Ne değişir |
|---|---|---|
| `make_a1_solve()` | `(prompt, data_path, **kw) -> dict` | `enable_refiner=False` |
| `make_a2_solve()` | `(prompt, data_path, **kw) -> dict` | `enable_validator_retry=False` |
| `make_a3_correctness_fn()` | `(state, problem) -> bool` | `is_valid` sinyali kullanılır |
| `make_a4_solve()` | `(prompt, data_path, **kw) -> dict` | `fast_only=True` |
| `make_a5_solve(path)` | `(prompt, data_path, **kw) -> dict` | Compiled DSPy yüklenir |

### Baseline'lar

```python
from ie_eval.baselines import make_single_shot_solve

# Düz tek-atış (PromptRefiner, bifurcation, retry yok)
baseline_fn = make_single_shot_solve(use_cot=False)

# Chain-of-Thought baseline (tek prompt içinde adım adım düşünme)
cot_fn = make_single_shot_solve(use_cot=True)

records = run_dataset(dataset, config_id="single_shot", solve_fn=baseline_fn)
```

---

## 14. MIPROv2 Optimizasyonu — A5 Altyapısı

### IESolverProgram

```python
class IESolverProgram(dspy.Module):
    """Tüm iesolver DSPy singleton'larını tek eğitilebilir program olarak açar."""

    def __init__(self):
        super().__init__()
        from iesolver.nodes.intake import _gatekeeper
        from iesolver.nodes.requirement import _analyst
        # ... tüm 11 singleton

        self.gatekeeper = _gatekeeper   # AYNI Python nesnesi — alias, kopya değil
        self.analyst = _analyst
        # ...

    def forward(self, prompt: str, data_path=None) -> dspy.Prediction:
        from iesolver import solve
        with tempfile.TemporaryDirectory() as tmpdir:
            state = solve(
                prompt, data_path=data_path, auto_mode=True,
                thread_id=f"mipro-{uuid4().hex[:8]}",
                checkpoint_db=Path(tmpdir) / "ckpt.sqlite",
            )
        return dspy.Prediction(
            execution_result=state.get("execution_result", ""),
            is_valid=bool(state.get("is_valid", False)),
            executive_summary=state.get("executive_summary", ""),
        )
```

**Singleton paylaşım prensibi:** `self.gatekeeper = _gatekeeper` — `is` operatörü `True` verir. MIPROv2 `self.gatekeeper` üzerinden prompt güncellediğinde `call_with_fast_lm(_gatekeeper, ...)` de güncellenmiş prompt'u kullanır.

### optimize_mipro.py CLI

```bash
# Dry-run (setup doğrulama, LLM çağrısı yok)
uv run python scripts/optimize_mipro.py \
    --train-data data/nl4opt_train_cleaned.jsonl \
    --output compiled/iesolver_mipro.json \
    --dry-run

# Pilot koşu (~$2 tahmini maliyet)
uv run python scripts/optimize_mipro.py \
    --train-data data/nl4opt_train_cleaned.jsonl \
    --output compiled/iesolver_mipro.json \
    --max-train 40 --num-candidates 5 --num-trials 10

# Tam koşu (makale için, ~$10)
uv run python scripts/optimize_mipro.py \
    --train-data data/nl4opt_train_cleaned.jsonl \
    --dev-data data/nl4opt_dev_cleaned.jsonl \
    --output compiled/iesolver_mipro_full.json \
    --num-candidates 15 --num-trials 30 \
    --teacher-model gemini/gemini-2.0-pro-exp
```

**Correctness metric:**
```python
def correctness_metric(example, pred, trace=None) -> float:
    # example.optimal_value varsa → numerical_match
    # None ise (NO_CODE) → execution_result boş değilse 1.0
```

**MIPROv2 parametreleri:**
- `num_candidates`: Her DSPy modülü için aday prompt sayısı (teacher LM tarafından üretilir)
- `num_trials`: Bayesian optimizasyon döngüsü sayısı (Optuna)
- `max_labeled_demos / max_bootstrapped_demos`: Few-shot örnek sayısı
- `num_threads=1`: Thread güvenliği (singleton'lar paylaşılıyor)

---

## 15. Test Altyapısı

### Test Sayıları

| Suite | Dosya | Test Sayısı |
|---|---|---|
| iesolver — e2e + smoke | `tests/test_e2e_eoq.py` | 4 |
| iesolver — auto_mode | `tests/test_auto_mode.py` | 8 |
| iesolver — Faz 4 | `tests/test_faz4.py` | 11 |
| iesolver — metrics | `tests/test_metrics.py` | 8 |
| iesolver — reproducibility | `tests/test_reproducibility.py` | 6 |
| iesolver — typed signatures | `tests/test_typed_signatures.py` | 6 |
| iesolver — report writer | `tests/test_report_writer.py` | 15 |
| **iesolver toplamı** | | **58** |
| ie-eval — ablations | `ie-eval/tests/test_ablations.py` | 26 |
| ie-eval — analysis stats | `ie-eval/tests/test_analysis_stats.py` | — |
| ie-eval — analysis summary | `ie-eval/tests/test_analysis_summary.py` | — |
| ie-eval — baselines | `ie-eval/tests/test_baselines.py` | — |
| ie-eval — IE-Case extended | `ie-eval/tests/test_ie_case_extended.py` | — |
| ie-eval — IndustryOR | `ie-eval/tests/test_industryor.py` | — |
| ie-eval — metadata pipeline | `ie-eval/tests/test_metadata_pipeline.py` | — |
| ie-eval — NL4Opt | `ie-eval/tests/test_nl4opt.py` | — |
| ie-eval — runner | `ie-eval/tests/test_runner.py` | — |
| ie-eval — store | `ie-eval/tests/test_store.py` | — |
| ie-eval — validator | `ie-eval/tests/test_validator.py` | — |
| **ie-eval toplamı** | | **144** |
| **GENEL TOPLAM** | | **202** |

### Test Çalıştırma

```bash
# iesolver testleri (LLM çağrısı yapar — API key gerekli)
uv run pytest tests/ -v

# ie-eval testleri (LLM çağrısı yok — mock/fake)
uv run pytest ie-eval/ -v

# Tamamı
uv run pytest tests/ ie-eval/ -q
```

### Test Stratejileri

**`@instrument` monkeypatch:** Sensitivity ve generate testleri `call_with_configured_lm` patch'ler, gerçek LLM çağrısı yapmaz:
```python
with patch("iesolver.nodes.sensitivity.call_with_configured_lm", return_value=pred):
    out = sensitivity_node(state)
```

**Graph routing testleri (A1/A2):** `_route_after_requirement`, `_route_after_validate` fonksiyonları doğrudan state dict ile test edilir — tam graph çalıştırılmaz:
```python
assert _route_after_requirement({"is_complete": True, "enable_refiner": False}) == "route"
```

**A3 correctness_fn:** `run_one` fake `solve_fn` ve `correctness_fn` ile çağrılır — LLM yok:
```python
fake_solve = lambda prompt, data_path=None, **kw: {"execution_result": "42.0", "is_valid": False}
rec = run_one(problem, solve_fn=fake_solve, correctness_fn=make_a3_correctness_fn())
```

**Report writer:** `tmp_path` fixture, gerçek dosya üretimi; içerik assertions:
```python
def test_html_contains_sensitivity_section(tmp_path):
    out = write_report(SAMPLE_STATE, tmp_path / "r.html", format="html")
    assert "Sensitivity Analysis" in out.read_text(encoding="utf-8")
```

**E2E EOQ testi** (`test_e2e_eoq.py`): Gerçek Gemini API çağrısı yapar. `litellm.drop_params=True` sayesinde `seed` parametresi Gemini'ye sessizce iletilmez.

---

## 16. Bilinen Kısıtlar ve Açık Noktalar

| Konu | Durum | Not |
|---|---|---|
| NL4Opt/IndustryOR JSONL verileri | Kullanıcı aksiyonu | OptiMind supplementary'den indirilmeli |
| E2E LLM benchmark koşusu | Bekliyor | Ücretli API kotası gerekli |
| A5 MIPROv2 eğitimi | Veri gelince | `--dry-run` ile setup doğrulandı |
| WeasyPrint PDF | Kapalı (Windows GTK yok) | fpdf2 ile değiştirildi |
| Paralel benchmark koşusu | Yok | `run_dataset` sıralı; `num_threads=1` zorunlu (singleton) |
| Streamlit UI (Faz 5b) | Ertelendi | Makale sonrasına bırakıldı |
| ReportWriter figür desteği (PDF) | Eksik | fpdf2 ile PNG embed henüz yok |

---

## 17. Dependency Listesi

### iesolver (zorunlu)

```
dspy>=3.0.0           — LLM Signature sistemi, ChainOfThought/Predict/ReAct
langgraph>=0.2.0      — Workflow graph, conditional edges, interrupt
langgraph-checkpoint-sqlite>=2.0.0  — SqliteSaver checkpoint backend
pandas>=2.1.0         — DataBundle tablo işleme
openpyxl>=3.1.0       — xlsx çok-sayfa okuma
pydantic>=2.5.0       — Veri doğrulama (DataBundle sınırları)
pydantic-settings>=2.1.0  — IESolverSettings (.env okuma)
litellm               — Model-bağımsız LLM erişim (dspy bağımlılığı)
```

### iesolver[report] (opsiyonel, Faz 5a)

```
fpdf2>=2.7.0          — PDF üretici (saf Python)
python-docx>=1.1.0    — DOCX üretici
jinja2>=3.1.2         — HTML şablon motoru
```

### iesolver[viz] (opsiyonel, Faz 4)

```
matplotlib>=3.8.0     — Tornado chart üretimi
plotly>=5.18.0        — İnteraktif grafik (gelecek)
```

### ie-eval (geliştirme)

```
pytest>=7.4.0
pytest-cov, pytest-asyncio, typeguard
```

---

## 18. Git Geçmişi (Özet)

```
b5db81c  feat: Faz 5a ReportWriter — HTML/DOCX/PDF 3-format rapor ciktisi
cd342b2  feat: iesolver + ie-eval Faz 1-4.5 tam implementasyon
```

---

## 19. Hızlı Başlangıç

```bash
# 1. Bağımlılıkları kur
uv sync --all-packages --extra dev --extra report --extra viz

# 2. API anahtarını ayarla
echo "GOOGLE_API_KEY=AIza..." > .env

# 3. Testleri çalıştır
uv run pytest tests/ ie-eval/ -q

# 4. EOQ problemi çöz
uv run python -c "
from iesolver import solve, write_report
state = solve('EOQ: yıllık talep 10000, sipariş maliyeti 50 TL, stok tutma maliyeti 2 TL/birim. Optimal sipariş miktarı nedir?', auto_mode=True)
print(state['executive_summary'])
write_report(state, 'output/report.html', format='html')
"
```

## DSPy Signature optimizasyonunu nerede nasıl yapıyoruz?                                                                                                                                                                                                                                                            
  ⎿  ⧉ Selected 1 lines from ..\..\..\..\..\..\mnt\c\Data\OneDrive - arel.edu.tr\Ders\20-Makale\Dspy\v4\SYSTEM.md in PyCharm

● Bu projedeki optimizasyon tamamen MIPROv2 ile yapılıyor ve A5 ablasyonuna karşılık geliyor. İki katman var:

  ---
  1. Ne optimize ediliyor?
```
  Her DSPy Signature'ın iki öğrenilebilir bileşeni var:
  
  - Instruction — "Bu bir EOQ problemi. Aşağıdaki kısıtları dikkat et..." gibi görev tanımı
  - Few-shot demo'lar — LLM'e örnek (prompt → beklenen çıktı) çiftleri

  MIPROv2 bunları otomatik arar; siz sadece bir doğruluk metriği ve eğitim verisi veriyorsunuz.
```
  ---
  2. Nasıl çalışıyor?
```
  NL4Opt train split (JSONL)
          │
          ▼
  IESolverProgram.forward(prompt) → solve() → execution_result
          │
          ▼
  correctness_metric(example, pred) → 0.0 veya 1.0
          │
          ▼
  MIPROv2 Bayesian optimizasyon (Optuna)
    ├── Teacher LM → num_candidates aday instruction üretir (her modül için)
    ├── Bootstrap → eğitim verisi üzerinde few-shot örnek toplar
    └── num_trials deneme → en yüksek accuracy veren kombinasyonu seçer
          │
          ▼
  compiled_program.save("compiled/iesolver_mipro.json")
```
  ---
  3. Kritik teknik karar — Singleton paylaşımı
```
  Tüm node'lar modül yüklenince singleton yaratır:

  # nodes/intake.py
  _gatekeeper = dspy.ChainOfThought(GatekeeperSignature)

  IESolverProgram bunları kopya değil, alias olarak tutar:

  class IESolverProgram(dspy.Module):
      def __init__(self):
          from iesolver.nodes.intake import _gatekeeper
          self.gatekeeper = _gatekeeper   # aynı Python nesnesi

  MIPROv2 self.gatekeeper.predict.signature üzerinde instruction güncellediğinde, call_with_fast_lm(_gatekeeper, ...) da güncellenmiş instruction'ı kullanır — ayrı bir nesne yok.

  program.save(path) bu güncellenmiş instruction + demo'ları JSON'a yazar.
```
  ---
  4. Nerede çalıştırılıyor?
```bash
  # Adım 1: Dry-run (setup kontrolü, LLM çağrısı yok)
  uv run python scripts/optimize_mipro.py \
      --train-data data/nl4opt_train_cleaned.jsonl \
      --output compiled/iesolver_mipro.json \
      --dry-run

  # Adım 2: Pilot (~$2, ~400 LLM çağrısı)
  uv run python scripts/optimize_mipro.py \
      --train-data data/nl4opt_train_cleaned.jsonl \
      --output compiled/iesolver_mipro.json \
      --max-train 40 --num-candidates 5 --num-trials 10

  # Adım 3: Tam koşu (makale için, ~$10, Pro-class teacher)
  uv run python scripts/optimize_mipro.py \
      --train-data data/nl4opt_train_cleaned.jsonl \
      --dev-data   data/nl4opt_dev_cleaned.jsonl \
      --output compiled/iesolver_mipro_full.json \
      --num-candidates 15 --num-trials 30 \
      --teacher-model gemini/gemini-2.0-pro-exp
```
  ---
  5. Optimize edilmiş program nasıl kullanılıyor?
```python
  # A5 ablasyonu — benchmark koşusunda
  from ie_eval.ablations import make_a5_solve
  from ie_eval.runner import run_dataset

  solve_fn = make_a5_solve(compiled_path=Path("compiled/iesolver_mipro.json"))
  # İlk çağrıda: load_compiled_graph() → singleton'ları in-place günceller
  # Sonraki solve() çağrıları optimize edilmiş instruction'larla çalışır

  records = run_dataset(dataset, config_id="A5_mipro", solve_fn=solve_fn)
```
  ---
  6. Şu an neden A5 koşusu yok?
```
  NL4Opt temizlenmiş JSONL verisi henüz yok (OptiMind supplementary'den gelmesi gerekiyor). Veri gelince --dry-run ile setup doğrulanacak, ardından pilot koşu yapılacak.
```
