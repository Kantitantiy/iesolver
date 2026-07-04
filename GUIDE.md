# iesolver v4 — Araştırmacı Rehberi

Bu rehber; iesolver v4'ü hiç görmemiş bir araştırmacı için yazılmıştır.
Sistemi nasıl çalıştıracağınızı, LLM konuşmalarını nasıl izleyeceğinizi,
prompt'ları nasıl geliştireceğinizi ve deney tasarımını nasıl uygulayacağınızı
adım adım açıklar.

---

## İçindekiler

1. [v3 ile v4 Farkı](#1-v3-ile-v4-farkı)
2. [Kurulum](#2-kurulum)
3. [Bir Problem Tanımlama](#3-bir-problem-tanımlama)
4. [Programı Başlatma](#4-programı-başlatma)
5. [Aşama Aşama İlerleyişi İzleme](#5-aşama-aşama-i̇lerleyişi-i̇zleme)
6. [LLM Konuşmalarını Görme](#6-llm-konuşmalarını-görme)
7. [Prompt'ları Nasıl Geliştirilir](#7-promptları-nasıl-geliştirilir)
8. [Deney Tasarımı](#8-deney-tasarımı)
9. [Test Suite](#9-test-suite)
10. [Dosya Referans Tablosu](#10-dosya-referans-tablosu)

---

## 1. v3 ile v4 Farkı

| Özellik | v3 | v4 |
|---|---|---|
| Mimari | Monolitik Python script | LangGraph DAG + DSPy node'lar |
| Çalıştırma | Tek fonksiyon çağrısı | `solve()` / `stream_solve()` |
| Akış izleme | Print statement'lar | `stream_solve()` → node bazlı yield |
| LLM geçmişi | Loglama | `show_llm_history()` + `lm.history` |
| Prompt düzenleme | Hardcoded f-string'ler | DSPy Signature'ları (`src/iesolver/signatures/`) |
| Prompt optimizasyonu | Manuel | MIPROv2 otomasyonu (`scripts/optimize_mipro.py`) |
| Rapor | Tek formatlı metin | HTML + DOCX + PDF (Faz 5) |
| Ablasyon testleri | Yok | A1-A5 bayrakları + ie-eval paketi |
| Veri desteği | CSV | CSV / XLSX / SQLite |
| Tekrarlanabilirlik | Yok | `temperature=0, seed=42` + checkpoint |

**v3'teki akışınız v4'te şu şekilde karşılanır:**

```
v3: problem_tanımla() → çalıştır() → raporu_gör()
v4: stream_solve("problem metni", data_path="veri.csv")
         → her node tamamlandığında (node_adı, partial_state) yield eder
         → write_report(final_state, "rapor.html", format="html")
```

---

## 2. Kurulum

```bash
# Projenin kök dizininde:
cd "v4/"

# Temel bağımlılıklar + rapor modülü:
uv sync --extra report

# Tüm eklentilerle (geliştirme ortamı):
uv sync --all-extras

# API anahtarınızı ayarlayın (.env dosyası veya ortam değişkeni):
# .env dosyası oluşturun:
echo IESOLVER_API_KEY=sk-... > .env
echo IESOLVER_FAST_MODEL=gemini/gemini-2.0-flash >> .env
echo IESOLVER_REASONING_MODEL=gemini/gemini-2.0-flash >> .env
```

Desteklenen model formatları (LiteLLM üzerinden):
- `gemini/gemini-2.0-flash`
- `openai/gpt-4o-mini`
- `anthropic/claude-3-5-haiku-20241022`

---

## 3. Bir Problem Tanımlama

### Sadece metin (veri yok)

```python
problem = "EOQ problemi: Yıllık talep D=10000 birim, sipariş maliyeti S=50 TL, birim tutma maliyeti H=2 TL/yıl. Optimal sipariş miktarını hesapla."
```

### Veri dosyasıyla

```python
problem = "Ekteki CSV'deki lokasyonlar için ulaşım maliyetini minimize eden taşıma planını bul."
data    = "data/transport.csv"
```

Kabul edilen veri formatları:
- `.csv` → tek tablo (`tables["data"]`)
- `.xlsx` → her sayfa ayrı tablo (`tables["Sheet1"]`, ...)
- `.sqlite` → her tablo ayrı (`tables["orders"]`, ...)

### İyi bir problem metni nasıl olmalı?

Sistem G-O-C çerçevesini uygular: **G**oal (Hedef), **O**utput (Çıktı), **C**onstraint (Kısıt).

```
İyi: "D=10000, S=50, H=2 parametreleriyle EOQ modelini çöz. Optimal sipariş
     miktarını ve toplam yıllık maliyeti döndür."

Zayıf: "Stok optimizasyonu yap."
```

Metninizde hedef, beklenen çıktı ve kısıtlar açık değilse, sistem
`clarify` node'unda durup ek bilgi ister (interaktif mod) veya
varsayım yaparak devam eder (auto mod).

---

## 4. Programı Başlatma

### A) Terminal üzerinden (en basit yol)

```bash
# Temel kullanım:
uv run python scripts/run_problem.py "EOQ: D=10000, S=50, H=2"

# Veri dosyasıyla:
uv run python scripts/run_problem.py "LP problemi" --data data/input.csv

# PDF rapor:
uv run python scripts/run_problem.py "EOQ problemi" --format pdf

# LLM geçmişini de göster:
uv run python scripts/run_problem.py "EOQ" --show-llm-log

# Interrupt olmadan (batch/test modu):
uv run python scripts/run_problem.py "EOQ" --auto

# Ablasyon testleri:
uv run python scripts/run_problem.py "EOQ" --no-refiner   # A1
uv run python scripts/run_problem.py "EOQ" --no-retry     # A2
uv run python scripts/run_problem.py "EOQ" --fast-only    # A4
```

### B) Python'dan (notebook veya script)

```python
from iesolver import solve, stream_solve, show_llm_history
from iesolver.report import write_report

# Basit çalıştırma (tüm akış tamamlanınca döner):
state = solve("EOQ: D=10000, S=50, H=2")
print(state["executive_summary"])
print(state["execution_result"])

# Raporu kaydet:
write_report(state, "raporum.html", format="html")
write_report(state, "raporum.pdf",  format="pdf")
write_report(state, "raporum.docx", format="docx")
```

### C) Kesintili çalıştırma (checkpoint ile devam)

```python
import uuid
thread = str(uuid.uuid4())

# İlk çalıştırma — interrupt olursa durur:
state = solve("LP optimizasyon", thread_id=thread, auto_mode=False)

# Eğer interrupt olduysa, kullanıcı cevabını ekleyip devam edin:
from langgraph.types import Command
# (LangGraph interrupt akışı — gelişmiş kullanım)
```

---

## 5. Aşama Aşama İlerleyişi İzleme

`stream_solve()`, her LangGraph node tamamlandığında `(node_adı, partial_state)` üretir.
v3'teki adım adım görünümü buradan gelir.

### Temel kullanım

```python
from iesolver import stream_solve

for node_name, partial in stream_solve("EOQ: D=10000, S=50, H=2"):
    keys = [k for k in partial if not k.startswith("_")]
    print(f"  ✓ {node_name:<25} → {keys}")
```

**Beklenen çıktı:**

```
  ✓ intake                   → ['raw_prompt', 'data_bundle', 'data_summary']
  ✓ requirement              → ['explicit_goal', 'constraints', 'is_complete']
  ✓ refiner                  → ['refined_prompt']
  ✓ route                    → ['execution_path']
  ✓ code_branch              → ['generated_code', 'retry_count']
  ✓ validate                 → ['is_valid', 'validation_notes']
  ✓ sensitivity              → ['sensitivity_results']
  ✓ artifacts                → ['figures']
  ✓ report                   → ['executive_summary', 'technical_output', 'action_directives']
```

### Belirli alanları yakalamak

```python
final_state = {}
for node_name, partial in stream_solve("EOQ problemi"):
    final_state.update(partial)
    if node_name == "validate":
        print("Doğrulama:", partial.get("is_valid"), partial.get("validation_notes"))
    if node_name == "report":
        print(partial.get("executive_summary", "")[:300])
```

### Node sırası ve açıklaması

| Node | Ne yapar | Hangi alanlara yazar |
|---|---|---|
| `intake` | Ham prompt + veri dosyasını okur, özet çıkarır | `data_bundle`, `data_summary` |
| `requirement` | G-O-C çerçevesiyle gereksinimler çıkarır | `explicit_goal`, `constraints`, `is_complete` |
| `clarify` | Eksik bilgi varsa interrupt/soru üretir | `clarification_questions` |
| `refiner` | Prompt'u yapısal DSPy formatına dönüştürür | `refined_prompt` |
| `route` | CODE vs NO_CODE kararı verir | `execution_path` |
| `code_branch` | Algoritma seçer, kod üretir (ReAct) | `generated_code`, `algo_name` |
| `validate` | Kodu çalıştırır, sonucu doğrular | `is_valid`, `execution_result` |
| `sensitivity` | Duyarlılık analizi yapar | `sensitivity_results` |
| `artifacts` | Grafik/figür oluşturur | `figures` |
| `chain_branch` | NO_CODE yolu: analitik çözüm | `technical_output` |
| `report` | 3-bölümlü rapor yazar | `executive_summary`, `technical_output`, `action_directives` |

---

## 6. LLM Konuşmalarını Görme

Her LLM çağrısı `lm.history` listesine eklenir.
`show_llm_history(n)` son `n` çağrıyı ayrıntılı basar.

### Temel kullanım

```python
from iesolver import solve, show_llm_history

state = solve("EOQ: D=10000, S=50, H=2")
show_llm_history(n=5)   # Son 5 LLM çağrısını göster
```

**Çıktı formatı:**

```
──────────────────────────────────────────────────────────────────────
  [1/5] Model: gemini/gemini-2.0-flash
  Tokens: 342 giriş / 87 çıkış
──────────────────────────────────────────────────────────────────────

  ── SYSTEM ──
  You are a meticulous Industrial Engineering Requirement Analyst...

  ── USER ──
  Cleaned prompt: EOQ: D=10000, S=50, H=2...
  Data summary: (veri yok)

  ── ASSISTANT ──
  {"is_complete": true, "explicit_goal": "Minimize total inventory cost..."}
```

### Terminal üzerinden (run_problem.py ile)

```bash
uv run python scripts/run_problem.py "EOQ" --show-llm-log --llm-log-n 10
```

### Doğrudan history'ye erişim

```python
from iesolver.lm import get_fast_lm, get_reasoning_lm

fast_lm      = get_fast_lm()
reasoning_lm = get_reasoning_lm()

# Her entry: {"model": "...", "messages": [...], "response": {...}, "usage": {...}}
for entry in fast_lm.history[-3:]:
    print(entry["model"], entry.get("usage", {}))
```

---

## 7. Prompt'ları Nasıl Geliştirilir

### Hangi dosyayı düzenlemeliyim?

Tüm LLM talimatları `src/iesolver/signatures/` klasöründedir.
Her `.py` dosyası bir DSPy `Signature` sınıfı içerir.

| Signature dosyası | Hangi node kullanır | Ne kontrol eder |
|---|---|---|
| `gatekeeper.py` | `intake` | Ham prompt temizleme, veri özetleme |
| `requirement_analyst.py` | `requirement` | G-O-C çerçevesi, eksik bilgi tespiti |
| `prompt_refiner.py` | `refiner` | Yapısal prompt dönüştürme |
| `strategy_router.py` | `route` | CODE/NO_CODE kararı mantığı |
| `algo_selector.py` | `code_branch/algo_select` | Algoritma seçim kriterleri |
| `constraint_adapter.py` | `code_branch/constraint_adapt` | Kısıt tanımlama |
| `output_spec.py` | `code_branch/output_spec` | Beklenen çıktı formatı |
| `react_code.py` | `code_branch/generate` | Kod üretimi (ReAct döngüsü) |
| `validator.py` | `validate` | Doğrulama kriterleri |
| `sensitivity.py` | `sensitivity` | Duyarlılık analizi yönergesi |
| `final_report.py` | `report` | Rapor formatı ve tonu |
| `analytical_engine.py` | `chain_branch` | NO_CODE analitik çözüm |

### Signature nasıl düzenlenir?

```python
# src/iesolver/signatures/requirement_analyst.py

class RequirementAnalystSignature(dspy.Signature):
    """
    [BURASI SYSTEM PROMPT — LLM'e ne yapması gerektiği]
    DSPy bu docstring'i system prompt olarak gönderir.
    """
    # InputField'lar: node'dan gelen veriler
    cleaned_prompt: str = dspy.InputField(desc="...")
    data_summary:   str = dspy.InputField(desc="...")

    # OutputField'lar: LLM'in döndürmesi beklenenler
    is_complete:    bool       = dspy.OutputField(desc="...")
    explicit_goal:  str        = dspy.OutputField(desc="...")
    constraints:    list[str]  = dspy.OutputField(desc="...")
```

**Düzenleme noktaları:**
1. `"""..."""` docstring → sistem direktifi (en etkili)
2. `InputField(desc="...")` → LLM'e alan açıklaması
3. `OutputField(desc="...")` → beklenen format açıklaması

### Değişikliği test etme

```bash
# Signature'ı düzenledikten sonra:
uv run python scripts/run_problem.py "EOQ: D=10000, S=50, H=2" --show-llm-log
```

Çıktıda değişen system prompt'u ve LLM'in yeni yanıtını görürsünüz.

### Otomatik Prompt Optimizasyonu (MIPROv2)

DSPy'ın MIPROv2 algoritması, Signature prompt'larını otomatik olarak optimize edebilir:

```bash
uv run python scripts/optimize_mipro.py \
    --dataset datasets/nl4opt_sample.jsonl \
    --out optimized_program.pkl \
    --trials 20
```

Optimize edilmiş program daha sonra `solve()` içinde yüklenebilir
(`_optimization.py` aracılığıyla). Bu, makalenin ana araştırma katkısıdır.

---

## 8. Deney Tasarımı

### Ablasyon Bayrakları (EVALUATION_PLAN §5)

| Bayrak | Python | Terminal | Ablasyon |
|---|---|---|---|
| `enable_refiner=False` | `solve("...", enable_refiner=False)` | `--no-refiner` | A1: PromptRefiner etkisi |
| `enable_validator_retry=False` | `solve("...", enable_validator_retry=False)` | `--no-retry` | A2: Retry döngüsü etkisi |
| *(A3 = optimizasyon yok)* | varsayılan | varsayılan | A3: MIPROv2 optimizasyonu etkisi |
| `fast_only=True` | `solve("...", fast_only=True)` | `--fast-only` | A4: Model tier seçimi |
| *(A5 = no-code path)* | `execution_path="NO_CODE"` yönlendirme | — | A5: Kod branchi etkisi |

### Toplu Değerlendirme (ie-eval paketi)

```bash
cd ie-eval/
uv run python -m ieeval.runner \
    --dataset datasets/nl4opt_sample.jsonl \
    --config configs/full.yaml \
    --out results/

# Sadece ablasyon A1 (refiner yok):
uv run python -m ieeval.runner --config configs/ablation_a1.yaml
```

### Metrikler

Sistem her node için otomatik olarak toplar:

```python
state = solve("EOQ problemi")
print(state["metrics"])
# {
#   "intake":   {"llm_calls": 1, "tokens_in": 342, "tokens_out": 87,
#                "cost_usd": 0.0001, "latency_ms": 450},
#   "report":   {"llm_calls": 1, "tokens_in": 800, "tokens_out": 250, ...},
#   ...
# }
```

Raporlarda "Performance Metrics" tablosu olarak otomatik eklenir.

### Tekrarlanabilirlik

Tüm çalıştırmalar `temperature=0, seed=42` ile yapılır (`.env` ile değiştirilebilir).
LangGraph checkpoint SQLite'a kaydedilir (`checkpoints/solver.db`).
Aynı `thread_id` ile çağrı yapılırsa önceki durum yüklenir.

```python
# Aynı thread_id ile ikinci çalıştırma → checkpoint'ten devam
state2 = solve("EOQ", thread_id="benim-thread-123")
```

---

## 9. Test Suite

```bash
# Tüm testler:
uv run pytest

# Sadece rapor testleri:
uv run pytest tests/test_report_writer.py -v

# Sadece Faz 4 (duyarlılık):
uv run pytest tests/test_faz4.py -v

# Yavaş testler hariç (LLM çağrısı yapmaz):
uv run pytest -m "not slow"

# ie-eval testleri:
cd ie-eval/ && uv run pytest
```

| Test dosyası | Ne test eder |
|---|---|
| `test_report_writer.py` | HTML/DOCX/PDF üretimi (15 test) |
| `test_faz4.py` | Duyarlılık analizi, artifact üretimi |
| `test_typed_signatures.py` | DSPy Signature tip kontrolü |
| `test_metrics.py` | Node metrik toplama |
| `test_auto_mode.py` | Auto mode / interrupt davranışı |
| `test_reproducibility.py` | Seed/determinizm |
| `test_e2e_eoq.py` | Uçtan uca EOQ smoke testi (gerçek LLM) |

Test sırasında üretilen dosyalar (PDF, DOCX, HTML) şuraya kaydedilir:
`C:\Users\<kullanıcı>\AppData\Local\Temp\pytest-of-<kullanıcı>\pytest-NN\`

---

## 10. Dosya Referans Tablosu

### Giriş Noktaları

| Dosya | Amaç |
|---|---|
| `src/iesolver/__init__.py` | Public API: `solve()`, `stream_solve()`, `show_llm_history()`, `write_report()` |
| `scripts/run_problem.py` | Terminal CLI: problem ver → aşama aşama izle → rapor al |
| `scripts/optimize_mipro.py` | MIPROv2 ile DSPy Signature'larını otomatik optimize et |

### Çekirdek Modüller

| Dosya | Amaç |
|---|---|
| `src/iesolver/config.py` | `Settings` sınıfı: API key, model isimleri, dizin yolları (`.env`'den) |
| `src/iesolver/state.py` | `SolverState` (TypedDict) + `DataBundle` — pipeline'daki ortak veri |
| `src/iesolver/graph.py` | LangGraph DAG tanımı: node'lar, edge'ler, conditional routing |
| `src/iesolver/lm.py` | LM yönetimi: `call_with_fast_lm()`, `call_with_reasoning_lm()`, `call_with_configured_lm()` |

### Node'lar (İş Mantığı)

| Dosya | Node | Görev |
|---|---|---|
| `src/iesolver/nodes/intake.py` | `intake` | Ham prompt temizleme + veri yükleme |
| `src/iesolver/nodes/requirement.py` | `requirement` | G-O-C gereksinim çıkarımı |
| `src/iesolver/nodes/clarify.py` | `clarify` | Eksik bilgi → interrupt veya varsayım |
| `src/iesolver/nodes/refine.py` | `refiner` | Prompt → yapısal DSPy formatı |
| `src/iesolver/nodes/route.py` | `route` | CODE / NO_CODE karar |
| `src/iesolver/nodes/code_branch/algo_select.py` | `code_branch` (1/4) | Algoritma seçimi |
| `src/iesolver/nodes/code_branch/constraint_adapt.py` | `code_branch` (2/4) | Kısıt adaptasyonu |
| `src/iesolver/nodes/code_branch/output_spec.py` | `code_branch` (3/4) | Çıktı tanımı |
| `src/iesolver/nodes/code_branch/generate.py` | `code_branch` (4/4) | ReAct kod üretimi |
| `src/iesolver/nodes/validate.py` | `validate` | Sandbox çalıştırma + doğrulama |
| `src/iesolver/nodes/chain_branch.py` | `chain_branch` | NO_CODE analitik çözüm |
| `src/iesolver/nodes/sensitivity.py` | `sensitivity` | Duyarlılık analizi |
| `src/iesolver/nodes/artifacts.py` | `artifacts` | Grafik/figür üretimi |
| `src/iesolver/nodes/report.py` | `report` | 3-bölümlü rapor metni |

### DSPy Signature'ları (Prompt Mühendisliği)

| Dosya | Kontrol ettiği prompt |
|---|---|
| `src/iesolver/signatures/gatekeeper.py` | Girdi temizleme direktifi |
| `src/iesolver/signatures/requirement_analyst.py` | G-O-C çerçevesi + tamamlık değerlendirmesi |
| `src/iesolver/signatures/prompt_refiner.py` | Yapısal dönüştürme direktifi |
| `src/iesolver/signatures/strategy_router.py` | CODE/NO_CODE karar mantığı |
| `src/iesolver/signatures/algo_selector.py` | Algoritma seçim kriterleri |
| `src/iesolver/signatures/constraint_adapter.py` | Kısıt formülasyon direktifi |
| `src/iesolver/signatures/output_spec.py` | Çıktı format direktifi |
| `src/iesolver/signatures/react_code.py` | ReAct döngüsü kod üretim direktifi |
| `src/iesolver/signatures/validator.py` | Doğrulama kriterlerini direktifi |
| `src/iesolver/signatures/sensitivity.py` | Duyarlılık analizi direktifi |
| `src/iesolver/signatures/tornado_chart.py` | Tornado chart veri direktifi |
| `src/iesolver/signatures/analytical_engine.py` | NO_CODE analitik çözüm direktifi |
| `src/iesolver/signatures/final_report.py` | Rapor bölümü yazım direktifi |

### Rapor Modülü

| Dosya | Amaç |
|---|---|
| `src/iesolver/report/__init__.py` | `ReportWriter`, `write_report` export |
| `src/iesolver/report/writer.py` | `ReportWriter` sınıfı — format yönlendirmesi |
| `src/iesolver/report/_html.py` | Jinja2 + markdown-it-py → HTML renderer |
| `src/iesolver/report/_docx.py` | python-docx → DOCX renderer |
| `src/iesolver/report/_pdf.py` | fpdf2 → PDF renderer (saf Python, GTK gerekmez) |
| `src/iesolver/report/templates/report.html.j2` | HTML rapor Jinja2 şablonu |

### Altyapı

| Dosya | Amaç |
|---|---|
| `src/iesolver/io/data_loader.py` | CSV/XLSX/SQLite → `DataBundle` dönüşümü |
| `src/iesolver/sandbox/runner.py` | Üretilen Python kodunu izole sandbox'ta çalıştırır |
| `src/iesolver/observability/metrics.py` | Node başına LLM çağrı sayısı, token, maliyet, gecikme |
| `src/iesolver/_optimization.py` | Optimize edilmiş DSPy programını yükleme/kaydetme |

### Değerlendirme (ie-eval paketi)

| Dosya | Amaç |
|---|---|
| `ie-eval/` | Ayrı uv workspace paketi — benchmark runner |
| `ie-eval/ieeval/runner.py` | NL4Opt / IndustryOR dataset'leri üzerinde toplu değerlendirme |

### Konfigürasyon

| Dosya | Amaç |
|---|---|
| `.env` | `IESOLVER_API_KEY`, `IESOLVER_FAST_MODEL`, `IESOLVER_REASONING_MODEL` |
| `pyproject.toml` | Proje bağımlılıkları, ruff, mypy, pytest ayarları |
| `PLAN.md` | 5 fazlı yol haritası ve teknik kararlar |
| `METHODOLOGY_NOTES.md` | Makale için mimari kararlar ve gerekçeler |
| `SYSTEM.md` | Sistem mimarisinin tam AI-odaklı dokümantasyonu |
| `EVALUATION_PLAN.md` | Deney tasarımı, metrikler, ablasyon planı |
