# iesolver — Methodology Notes for Academic Paper
### Faz 1 + Faz 2 kararları · Ham materyal · Makale bölümü taslağı

> **Kullanım**: Bu dosya her yeni Faz sonunda güncellenir.
> Makalenin *Methodology* ve *System Design* bölümleri buradan beslenir.
> Türkçe notlar → makale yazımında İngilizce'ye çevrilecek.

---

## 1. Genel Mimari Argümanı

### 1.1 DSPy ⊕ LangGraph Rol Ayrımı

Sistemin merkezinde iki framework'ün bilinçli iş bölümü yatar:

| Katman | Framework | Sorumluluk | Akademik Karşılığı |
|--------|-----------|------------|-------------------|
| Reasoning Units | **DSPy** | "Bir LLM çağrısı ne yapmalı?" | Signature (I/O kontratı) + declarative prompt |
| Workflow Engine | **LangGraph** | "Aşamalar nasıl bağlanır?" | Typed state, conditional edges, checkpoint, interrupt |

**Makale argümanı**: Mevcut agentic sistemlerin çoğu ya prompt'u doğrudan koda gömer (zor test edilir, optimize edilemez) ya da orkestrasyon mantığını LLM'e bırakır (deterministik değil). iesolver bu iki endişeyi ayrı katmanlara yönlendirerek her birini bağımsız optimize edilebilir kılar.

Her LangGraph node tam olarak bir DSPy Module çalıştırır. State LangGraph'ta; akıl yürütme DSPy'da. Bu ayrım implementasyonda şu biçimde netleşir:

- `signatures/` — saf DSPy Signature sınıfları: test edilebilir, `dspy.MIPRO` / `dspy.BootstrapFewShot` ile optimize edilebilir, declarative I/O.
- `nodes/` — LangGraph node fonksiyonları: Signature'ları çağırır, `SolverState`'i yazar, sanitization burada yapılır.
- `graph.py` — salt topoloji: node'ları ve conditional edge'leri birbirine bağlar; ne prompt ne de iş mantığı içerir.

---

## 2. State Şeması Tasarım Kararı

### 2.1 TypedDict(total=False) — Pydantic BaseModel yerine neden?

`SolverState`, Python `TypedDict(total=False)` olarak modellendi. Alternatif Pydantic `BaseModel`'i bilinçli olarak reddettik:

- **Pydantic**: Her partial update'te re-validation maliyeti getirir. LangGraph döngülerinde (özellikle Faz 3'ün retry loop'unda, `max_retries=3`) bu maliyet birikerek ölçülebilir gecikmeye dönüşür.
- **TypedDict**: Statik tip ipucu sağlar (mypy), runtime'da `dict` kadar hafiftir, ve LangGraph'ın partial-merge semantiğiyle doğal uyum içindedir.

Validation gereken sınırlarda (kullanıcı girdisi, dosya okuma) Pydantic `config.py` ve `io/data_loader.py` içinde devreye girer. Bu, makalede **"Pydantic at boundaries, TypedDict in flight"** argümanı olarak konumlanabilir.

### 2.2 "State as Contract" argümanı

Her node yalnızca yazdığı alanları döndürür; LangGraph okunan ve yazılan alanları birleştirir. Bu tasarım:
- Node'ları bağımsız test edilebilir kılar (sadece yazdığı alan grubunu doğrula).
- Yeni node eklemeyi kırılgansız yapar (TypedDict `total=False` → mevcut node'lar etkilenmez).
- Checkpoint replay'i mümkün kılar (her node'un delta'sı ayrı kayıt altında).

---

## 3. Veri Soyutlama Katmanı

### 3.1 DataBundle — Format-Agnostic Reasoning

Üç farklı giriş formatı (CSV, çok-sayfalı XLSX, SQLite) tek bir `DataBundle` soyutlamasına indirgenir:

```
CSV    → tables = {"data": DataFrame}
XLSX   → tables = {sheet_name: DataFrame, ...}
SQLite → tables = {table_name: DataFrame, ...}
```

**Makale argümanı**: Downstream node'lar (GateKeeper'ın data contextualization kısmı, AlgoSelector) kaynak formata bakmaz; yalnızca `DataBundle.summary()` çıktısını LLM bağlamına ekler. Bu, "format-agnostic reasoning units" argümanının somut karşılığıdır: LLM'e verinin nasıl depolandığı değil, ne içerdiği iletilir.

`DataBundle.summary()` metodu eski `DataProfiler.generate_summary()` ile aynı sözleşmeyi sunar: token-dostu istatistiksel öz (şekil, dtype, missing values, ilk N satır önizleme). Bu, LLM'in veriyi "görmesi" için gereken minimum context'i sağlar, aynı zamanda token israfını önler.

---

## 4. DSPy Modül Seçim Gerekçeleri (per node)

Her aşamada `Predict` ile `ChainOfThought` arasındaki tercih performans + maliyet dengesini yansıtır:

| Aşama | Node | Modül | Gerekçe |
|-------|------|-------|---------|
| 0 | GateKeeper | `ChainOfThought` | Ham girdide yapısal gürültü (kod parçaları, encoding hataları) ayrıştırma, kısa düşünme payı gerekli |
| 1 | RequirementAnalyst | **`Predict`** | G-O-C framework katı yapısal çıktı ister; CoT burada gereksiz token harcar |
| 2 | PromptRefiner | `ChainOfThought` | problem_type sınıflandırması 5 kategoriden birine zorlanır; sınıflandırma görevleri akıl yürütmeden güç alır |
| 3 | StrategyRouter | `ChainOfThought` | Bifurcation Logic: "önce rationale, sonra execution_path" — kararın gerekçesi state'te kayıt altına alınır (observability) |
| 4A | AnalyticalEngine | `ChainOfThought` | NO_CODE: Least-to-most decomposition + multi-perspective evaluation; derinlemesine düşünme zorunlu |
| 4B | AlgoSelector | `ChainOfThought` | Algoritma + kütüphane seçimi teknik nüans içerir; library-constraint uyuşmazlıklarını akıl yürüterek tespit etmeli |
| 4B | ConstraintAdapter | `Predict` | Kısıt adaptasyonu deterministik metin dönüşümü; rationale gerekmiyor |
| 4B | OutputSpecEngineer | `Predict` | Kod çıktı formatını belirtme; format katı, tek atışta yeterli |
| 4B | ReActCode | `ReAct` | Yaz-çalıştır-hata al-düzelt döngüsü; ReAct doğal seçim |
| 4B | ResultValidator | `ChainOfThought` | "Nasıl bu kanıya vardım?" sorusu: sınır ihlali tespiti akıl yürütmeden güç alır |
| 5 | FinalReportGenerator | `ChainOfThought` | 3 audience için 3 farklı ton; sentez kalitesini CoT artırır |

**Makale argümanı**: Bu seçimler rastgele değil; her biri eski koddan korunan deneysel bilgi. `Predict` seçimleri token maliyetini düşürür; `ChainOfThought` seçimleri kaliteyi artırır. Bu iş bölümü, gelecekteki `dspy.MIPRO` optimizasyon turlarında her modülün bağımsız tune edilmesini mümkün kılar — monolitik prompt'larda bu mümkün değildir.

---

## 5. Bifurcation Logic (Çatallanma Mantığı)

Phase 3 Strategy Router iki kesin yoldan birini seçer:

- **CODE**: Matematiksel optimizasyon (LP, MILP), ML/veri analitiği, algoritma gerektiren hesaplama.
- **NO_CODE**: Niteliksel karar verme, literatür taraması, teorik çerçeveleme.

Bu ayrım LangGraph conditional edge olarak hayata geçer:

```python
def _route_after_router(state) -> str:
    return "chain_branch" if state["execution_path"] == "NO_CODE" else "code_branch"
```

**Makale argümanı**: Mevcut genel amaçlı agentic sistemler her problemi "araç çağrısı gerekebilir" kararıyla yükler. iesolver problemi önce sınıflandırır, sonra uygun execution path'e yönlendirir. Bu, gereksiz sandbox başlatma maliyetini, yanlış algoritma seçimini ve halüsinasyon riskini azaltır.

**Sanitization**: StrategyRouter'ın `execution_path` çıktısı normalize edilir: "Code Path", "code-based", "CODE_REQUIRED" gibi varyantlar "CODE"'a indirgenir. Bu, LangGraph conditional edge'in deterministik çalışması için zorunludur.

---

## 6. G-O-C Framework Uygulaması

Phase 1 (RequirementAnalyst) üç boyutlu bir yapısal analiz yapar:

- **Goal (Hedef)**: Birincil amaç fonksiyonu.
- **Output (Çıktı)**: Beklenen sonuç formatı.
- **Constraint (Kısıt)**: Algoritmik ve mantıksal sınırlar.

Ek olarak `is_complete` bayrağı ile **bilgi yeterliliği kapısı** oluşturulur. Solvable matematiksel model kurabilmek için zorunlu yapısal bilgi eksikse (`is_complete=False`) sistem human-in-loop'a sapar.

**Önemli tasarım kararı**: Signature docstring'de `data_summary`'nin yalnızca şema önizlemesi olduğu vurgulanır; LLM'in "5 satır var ama 15 konum gerekiyor, eksik" gibi yanlış tespitler yapmasını engeller. Bu tür docstring mühendisliği, Signature'ın test edilebilirliğini gösterir.

---

## 7. Human-in-Loop Mekanizması

Eski sistemde `provide_missing_info()` metodu prosedürel olarak çalışıyordu:
```python
# Eski: IEAgent instance'ında
def provide_missing_info(self, user_answers):
    self.current_prompt += f"\n[USER UPDATE]: {user_answers}"
    return self._resume_pipeline()
```

Yeni sistemde LangGraph `interrupt()` + checkpoint ile:

```
requirement → [is_complete=False] → clarify (interrupt) → kullanıcı cevap verir
                                    ↓
                              Command(resume=answer)
                                    ↓
                             → requirement (tekrar çalışır)
```

**Makale argümanı**: `interrupt()` + checkpoint mimarisi üç avantaj sağlar:
1. **Durum tutarlılığı**: Tüm pipeline state'i checkpoint'te saklanır; kullanıcı cevabı beklenirken API kotası tüketilmez.
2. **Replay güvenliği**: Kesintiden sonra tam olarak kaldığı yerden devam eder.
3. **Separation of concerns**: `clarify_node` sadece pause/resume mekanizmasını yönetir; `requirement_node` saf G-O-C analizi yapar.

`user_clarification` alanı `cleaned_prompt`'a `[USER CLARIFICATION]:` etiketiyle katılır — RequirementAnalyst ikinci turda bu bilgiyi "yeni context" olarak görür.

---

## 8. LM Context Binding at Orchestration Boundaries

### 8.1 Sorun

DSPy 2.5.40+ sürümünden itibaren LM ayarı `contextvars.ContextVar` üzerinde tutulmaya başlandı. LangGraph her node'u `contextvars.copy_context().run(...)` ile yürütür. Bu izolasyon, `solve()` içinde yapılan global `dspy.configure(lm=...)` çağrısının node'lara propagate olmamasına yol açar (GitHub DSPy #1867).

### 8.2 Çözüm

Her node kendi DSPy çağrısını per-call `dspy.context(lm=...)` bloğuyla sarmalar. `lm.py`'daki `call_with_fast_lm()` ve `call_with_reasoning_lm()` yardımcıları bu pattern'i tek satıra indirger:

```python
# Her node içinde:
result = call_with_fast_lm(_module, input_field=value)

# Kod üretimi node'larında (Faz 3):
result = call_with_reasoning_lm(_code_module, ...)
```

Bu aynı zamanda **fast LM / reasoning LM anahtarlamasını** da çözer:
- Triage, routing, reporting → `get_fast_lm()` (Gemini Flash Lite)
- Kod üretimi (Faz 3, Phase 4B) → `get_reasoning_lm()` (Gemini Pro)

Global state mutate etmeden, paralel/replay senaryolarında güvenli.

**Makale argümanı**: "LM context binding at orchestration boundaries" — workflow engine'in execution model'i (contextvars) ile reasoning unit'in state model'inin çakışmasını sistematik biçimde çözen bir entegrasyon kararı. `call_with_*_lm` abstraction'ı, gelecekte farklı LM sağlayıcılarına geçişi tek noktadan mümkün kılar.

---

## 9. Sanitization at Orchestration Boundaries

### 9.1 Faz 2 Yaklaşımı (artık geçersiz — tarihsel kayıt)

DSPy eski sürümlerinde `OutputField` boolean ve integer alanlarını string olarak döndürürdü. Faz 2'de manuel sanitization uygulandı:

| Alan | DSPy çıktısı | Beklenen tip | Dönüşüm |
|------|-------------|--------------|---------|
| `is_complete` | `"True"` / `"False"` | `bool` | `str.lower() in {"true","1","yes"}` |
| `is_valid` | `"True"` / `"False"` | `bool` | aynı |
| `confidence_score` | `"85"` | `int` | `int(str(...))`, hata → `0` |
| `execution_path` | `"Code Path"` | `Literal["CODE","NO_CODE"]` | normalize: uppercase + keyword match |

### 9.2 DSPy 3.x Tipli Output'lar (DESIGN_REVIEW §3.4 — mevcut yaklaşım)

DSPy 3.x, Signature `OutputField` tip annotation'larını doğrudan uygular. Tüm manuel sanitization kaldırıldı:

```python
# signatures/requirement_analyst.py
is_complete: bool = dspy.OutputField(...)
missing_items: list[str] = dspy.OutputField(...)
constraints: list[str] = dspy.OutputField(...)

# signatures/strategy_router.py
execution_path: Literal["CODE", "NO_CODE"] = dspy.OutputField(...)

# signatures/validator.py
is_valid: bool = dspy.OutputField(...)
confidence_score: int = dspy.OutputField(...)
```

**Makale argümanı**: DSPy 3.x'in tip koercisyonu, "Signature as type contract" argümanını güçlendirir — I/O sözleşmesi artık yalnızca doküman değil, çalışma zamanında uygulanır. Sanitization kod debti ortadan kalktı; node'lar tip-güvenli state yazar. Bu DESIGN_REVIEW §3.4 kapsamında belgelenmiştir.

---

## 10. Gözlenebilirlik (Observability)

### 10.1 Checkpoint Tabanlı Replay

`SqliteSaver` ile her node geçişi diske yazılır. Pratik faydaları:

- **API kota koruması**: Başarısız run'lar kaldığı yerden devam eder; başa dönülmez.
- **Ücretsiz tier güvencesi**: Faz 1+2 ücretsiz Gemini tier'da sorunsuz; checkpoint olmadan her hata token israfına dönüşür.
- **Replay**: Aynı `thread_id` ile tekrar `invoke()` → checkpoint'ten yükle, yeniden hesaplama yok.

### 10.2 Solution Path Loglama

`chain_branch_node`, modelin düşünce zincirini (`sub_problem_decomposition` + `perspective_exploration`) `solution_path` alanında birleştirerek state'e yazar. Bu:
- Final raporda "nasıl bu sonuca ulaşıldı" bölümünü besler.
- Debug ve paper yazımı için tam izlenebilirlik sağlar.

---

## 11. Domain Pack Soyutlaması (Gelecek)

`domains/` klasörü `DomainPack` protokolü üzerinden takılıp çıkarılabilir domain bilgisi sunar:
- IE ilk pack (şu an).
- Finans, lojistik, sağlık sonra.

**Makale argümanı**: Sistemin genel amaçlı tasarımı, makalenin "extensible" argümanını güçlendirir. `StrategyRouterSignature` ve `AlgoSelectorSignature` docstring'leri domain pack'ten gelen terminoloji ile zenginleştirilebilir.

---

## 12. Terminoloji Referansı (Makale İçin)

| Sistem Bileşeni | Makale'deki İsim (önerilen) |
|-----------------|---------------------------|
| GateKeeper | Input Standardization Layer |
| G-O-C analizi | Goal-Output-Constraint Framework |
| Bifurcation Logic | Execution Path Bifurcation |
| CODE path | Computational Execution Branch |
| NO_CODE path | Analytical Reasoning Branch |
| call_with_fast_lm | LM Context Binding |
| call_with_configured_lm | Ablation-Aware LM Dispatch |
| DataBundle | Format-Agnostic Data Abstraction |
| SqliteSaver checkpoint | Stateful Replay Mechanism |
| interrupt() + clarify | Human-in-Loop Clarification Protocol |
| DSPy typed OutputField | Type Contract at Signature Boundary |
| solution_path log | Reasoning Trace for Transparency |
| enable_refiner=False | A1: No Prompt Restructuring Ablation |
| enable_validator_retry=False | A2: Single-Shot Validation Ablation |
| correctness_fn override | A3: Deterministic Validation Bypass |
| fast_only=True | A4: Uniform Fast-LM Ablation |
| MIPROv2 compiled program | A5: Optimized Signature Ablation |
| self_consistency_router=True | A6: Router Self-Consistency Ablation |
| ie-eval harness | Decoupled Evaluation Framework |
| numerical_match | Deterministic Objective Verification |
| McNemar test | Paired Correctness Significance Test |
| bootstrap CI | Percentile-Method Accuracy Interval |

---

## 13. Faz 3 — Kod Motoru (Code Branch)

### 13.1 Sandbox Tasarımı

Üretilen Python kodu `subprocess.run` ile izole bir process'te çalıştırılır.

**Gerekçeler:**
- `exec()` veya `eval()` ana process'i kirletir: LLM kodu global state'i değiştirebilir, import sistemi bozabilir.
- Subprocess sınırı, üretilen kodun yan etkilerini (dosya silme, ağ çağrısı) ana agent'tan yalıtır.
- `timeout` ile askıda kalma ve sonsuz döngü önlenir; `sys.executable` ile aynı venv kullanılır — `import pulp` gibi ifadeler proje bağımlılıklarına erişebilir.

**`RunResult` sözleşmesi:** `success`, `stdout`, `stderr`, `exit_code`, `timed_out`, `error_summary`. Node'lar `if result.success:` ile basitçe dallanır; makale terminolojisinde "Computational Sandbox Layer".

**Makale notu**: Docker'a geçiş sonraya ertelendi. Şimdiki subprocess sınırı "least-privilege execution" kategorisinde yeterince savunulabilir.

### 13.2 ReAct Retry Döngüsü

İki katmanlı retry mekanizması:

1. **DSPy ReAct iç döngüsü** (`max_iters=3`): tek bir `generate_node` çağrısı içinde "düşün → araç çağır (sandbox) → gözlemle" tekrarı. LLM kendi hatasını görerek kodu düzeltir — "Autonomous Error Recovery".

2. **Dış döngü** (`max_retries=3`, graph.py): `validate_node` is_valid=False dönerse conditional edge `code_branch`'e geri döner. Bu, "dışarıdan bakan eleştiri" → "yeniden üretim" döngüsünü modelliyor: semantik hata (negatif stok gibi) ReAct'ın göremeyeceği türden olabilir.

`retry_count` `SolverState`'te taşınır; her `code_branch_node` geçişinde `+1`. Limit aşımında `report` node'una gidilir: kısmi/hatalı sonuç bile raporlanır, sessiz başarısızlık yok.

### 13.3 Per-node Reasoning LM Switching

DSPy 3.x + LangGraph'ta `dspy.configure(lm=...)` contextvars aracılığıyla tutuluyor ve LangGraph her node'u `copy_context().run(...)` ile yürütüyor. Bu yüzden global `configure()` node'lara propagate olmuyor (özellikle async/threaded path).

**Çözüm**: `call_with_fast_lm()` / `call_with_reasoning_lm()` helper'ları her DSPy çağrısını `with dspy.context(lm=...)` bloğuna alır. Kod üretimi (`generate_node`, `sensitivity_node`) `reasoning_lm` kullanır; geri kalan node'lar `fast_lm` kullanır. Bu, token maliyetini minimize ederken kritik kod üretiminde kaliteyi korur.

### 13.4 ResultValidator: "Semantic and Logical Boundary Verification"

`validate_node`, kod çıktısının sayısal ve mantıksal geçerliliğini iki aşamada doğrular:

1. **LLM (ChainOfThought)**: Negatif fiziksel nicelik, limit ihlali, aşırı aykırı değer gibi IE-özgü sınır kontrolü. `is_valid: bool`, `confidence_score: int`, `validation_notes: str` — DSPy 3.x tipli output'lar.

2. *(Faz 4.5'te gelecek)* Programatik feasibility check: çözümü kısıtlara geri-ikame eden deterministik doğrulama.

**Makale argümanı**: "LLM kendi kendini mi notluyor?" sorusuna karşı savunma: LLM validatörü *semantik* kontrolü yapar (doğru problem mi çözüldü?); deterministik katman *matematiksel* doğruluğu sağlar (Faz 4.5). Bu iki katmanlı yaklaşım bağımsız bir metodoloji katkısı olarak sunulabilir.

---

## 14. Faz 4 — Sensitivity Analysis + Artifacts

### 14.1 Dual-First Sensitivite Stratejisi (DESIGN_REVIEW §3.6)

LP/MILP problemlerinde shadow price (constraint.pi) ve reduced cost (var.duals) solver'dan bedava gelir. PuLP veya scipy kullanıldığında bu değerler tek ekstra satırla elde edilebilir.

**Strateji hiyerarşisi** (`SensitivityCodeSignature`):
1. **Dual extraction** (tercih): Orijinal koda dual raporlama satırları eklenir, yeniden çalıştırılır.
2. **Perturbation fallback**: Dual desteklemeyen solver (heuristik, custom) için ±%5/±%10 parametre değişiminin amaç fonksiyonu üzerindeki etkisi tablolanır.

`analysis_type: Literal["dual", "perturbation"]` state'e kaydedilir → makalenin error-analysis bölümünde "kaç problemde dual elde edildi vs perturbation kullanıldı" istatistiği çıkarılabilir.

### 14.2 Artifact Üretimi ve Zarif Degradasyon

`artifacts_node` sensitivity tablosundan matplotlib tornado chart üretir. Kritik tasarım kararı: **hiçbir başarısızlık pipeline'ı durdurmamalı**.

- Sensitivity sandbox başarısız → `sensitivity_results` başarısızlık marker'ı taşır → `artifacts_node` boş döner.
- Chart sandbox başarısız veya PNG yazılmadı → `figures=[]` → `report_node` çalışmaya devam eder.

`figures: Annotated[list[Path], operator.add]` reducer'ı, ileride birden fazla artifact node'u eklendiğinde figure listesini otomatik birleştirir — kod değişikliği gerektirmez.

### 14.3 Graph Topolojisi Değişikliği

`_route_after_validate` üç-yönlü dal:
- `is_valid=False` + retry kalan → `code_branch` (mevcut)
- **`is_valid=True` → `sensitivity`** (yeni)
- `is_valid=False` + limit aşımı → `report` (mevcut; artık sensitivity atlanır)

NO_CODE path (`chain_branch → report`) değişmedi: sensitivity LP/MILP'e özgü; kavramsal problemler için uygulanamaz.

---

## 15. Tekrarlanabilirlik Politikası (DESIGN_REVIEW §3.7)

Q1 dergide tekrarlanabilirlik standardı:

| Bileşen | Karar |
|---|---|
| LLM temperature | `0.0` (config.py `temperature` alanı) |
| LLM seed | `42` (config.py `lm_seed`; provider desteklemiyorsa sessizce görmezden gelinir) |
| Bağımlılık pinleri | `uv.lock` (uv deterministik; her `uv sync` aynı ortamı üretir) |
| Sandbox kütüphaneler | `sys.executable` → aynı venv → `uv.lock` kapsar |
| Prompt/Signature versioning | git tag (örn. `v0.2.0-faz4`); Signature dosyaları `signatures/` altında ayrı — diff'lenebilir, optimize edilebilir |
| Checkpoint replay | `SqliteSaver` + `thread_id` → aynı girdiye her zaman aynı checkpoint'ten başlanabilir |

**Makale argümanı**: "Tüm deneyleri `temperature=0` ile yürüttük; `uv.lock` ve git tag ile tam ortam yeniden üretilebilir."

---

---

## 16. Faz 4.5 — Evaluation Harness ve Ablasyon Tasarımı

### 16.1 ie-eval: Harness Mimarisi

`ie-eval/` paketi, `iesolver` ile yalnızca public API üzerinden konuşur (`from iesolver import solve, is_interrupted, SolverState, DataBundle`). Bu "kütüphane sınırı" makalede şu şekilde konumlanabilir: **"iesolver bir araştırma kütüphanesidir; deney altyapısı onun hiçbir iç modülüne dokunmaz — bu, harness'ın kütüphanenin API kararlılığını test ettiğini gösterir."**

Ana bileşenler:

| Bileşen | Sorumluluk |
|---|---|
| `Problem` / `GroundTruth` | Problem tanımı ve referans çözüm (objective_value, solution, feasibility_fn) |
| `ie_case.py` (6 problem) | IE-Case benchmark: xlsx, csv, sqlite, NO_CODE — fixture idempotent yazılır |
| `numerical_match` | Permissive regex + göreli tolerans; `"10,000"` ve `"3,14"` destekli |
| `check_feasibility` | Çözümü kısıtlara geri-ikame eden deterministik doğrulama |
| `run_one(correctness_fn=...)` | Tek problem koşusu; A3 için `correctness_fn` override |
| `ResultStore` | `metadata_json` sütunlu SQLite; backward-compat `ALTER TABLE` |
| `summarize_by_config` | pass@1, accuracy_mean±std, execution_rate, total_cost_usd |
| `compare_configs` | 2×2 McNemar tablosu (both_correct, only_a, only_b, both_wrong) |
| `metadata_filter` | dict veya callable predicate ile benchmark/problem_type kırılımı |

### 16.2 Benchmark Veri Seti Tasarımı

**IE-Case** (kendi katkımız, 6 problem):
- EOQ (inline, analitik GT): ≈707.107
- Transport 2×3 (inline, MODI: 960)
- Multi-product inventory (xlsx 2-sayfa, analitik GT: ≈897.83)
- Transport 3×2 (csv, MODI: 500)
- Assignment 3×3 (sqlite, Hungarian/brute-force: 9)
- ABC Classification (NO_CODE, `objective_value=None`)

**Makale argümanı**: NL4Opt/IndustryOR'da veri dosyası yoktur; DataBundle argümanı yalnızca IE-Case'de kullanılabilir. Bu, kütüphanenin gerçek IE değerini (veri dosyası → optimizasyon → duyarlılık → rapor) gösteren tek benchmark'tır. IE-Case, makalede bağımsız katkı olarak sunulabilir.

### 16.3 Ablasyon Tasarımı (EVALUATION_PLAN §5)

Her ablasyon, `iesolver.solve()` imzasına eklenen bir flag veya `run_one`'a geçilen bir `correctness_fn` ile hayata geçer. Tasarım ilkeleri:

**Minimal izolatör prensibi**: Her ablasyon tam olarak *bir* bileşeni değiştirir; geri kalan pipeline aynı kalır. Bu, sonuçların bileşen katkısına doğrudan atfedilebilmesini sağlar.

| Ablasyon | Mekanizma | Değişen Kod |
|---|---|---|
| A1: No Refiner | `enable_refiner=False` → `_route_after_requirement`'da "route" dalı | `graph.py` conditional edge |
| A2: No Retry | `enable_validator_retry=False` → `_route_after_validate`'de `code_branch` dalı dışlanır | `graph.py` conditional edge |
| A3: LLM Validator Only | `correctness_fn=make_a3_correctness_fn()` → `run_one`'da `numerical_match` override | `runner.py` post-extract override |
| A4: Fast-Only | `fast_only=True` → `call_with_configured_lm` fast_lm'ye yönlendirir | `lm.py` + `generate.py` + `sensitivity.py` |
| A5: MIPROv2 | `make_a5_solve(compiled_path)` → DSPy compiled program yüklenir | *(script bekliyor)* |
| A6: Router Self-Consistency | `self_consistency_router=True` → router 3 örnekleme + `dspy.majority` çoğunluk oyu | `route.py` |

**A3'ün özel konumu**: A3, `iesolver`'ı değiştirmez; yalnızca *harness'ın doğruluk ölçümünü* değiştirir. Bu, "deterministik doğrulama katmanının değeri ne?" sorusunu cevaplamanın en temiz yoludur: pipeline aynı çalışır, sadece değerlendirme sinyali değişir.

**A5 (MIPROv2) için hazırlık**: DSPy module singleton'ları (`_react`, `_sens_gen`, vb.) `load()` metodu ile optimize edilmiş ağırlıkları kabul edebilir. Optimizasyon NL4Opt train split'inde yapılacak; `scripts/optimize_mipro.py` yazıldığında `make_a5_solve(compiled_path)` aktive edilir.

**A6'nın gerekçesi**: `execution_path`, pipeline'daki en yüksek blast-radius'lu tekil karardır — yanlış dallanma (CODE vs NO_CODE) downstream'deki her adımı geçersiz kılar, retry döngüsü bunu telafi edemez (route düğümü retry kapsamı dışındadır). A6, tek örneklemenin bu kararda ne kadar varyans taşıdığını 3x maliyet karşılığında ölçer; kazanç anlamlıysa üretimde varsayılan olarak açılması önerilebilir.

### 16.4 İstatistiksel Çerçeve

**Neden McNemar?** Problem başına eşleştirilmiş doğru/yanlış çiftleri karşılaştırılıyor (aynı problem, iki konfigürasyon). Bu, paired binary data için McNemar'ı t-test veya χ²'ye göre daha uygun kılar.

**scipy-free implementasyon**:
- n < 25: `math.comb` ile exact binomial `p = 2 × Σ C(n,k) × 0.5^n` (k: 0..min(b,c))
- n ≥ 25: `math.erfc(sqrt(stat/2))` ile continuity-corrected χ² (df=1)

**Bootstrap CI**: Percentile method, paired resampling (problem sıralaması korunur), seedable (`seed=42` → tekrarlanabilir). scipy/statsmodels bağımlılığı yok.

**Makale argümanı**: Tüm istatistiksel testler bağımlılık-serbest uygulandı — bu makaleyi çalıştırmak için ek istatistik kütüphanesi gerekmez. Hem McNemar hem de bootstrap sıfırdan yazıldı; her ikisi de belgelenmiş test senaryolarıyla doğrulandı.

### 16.5 Baseline Tasarımı

`single_shot_solve` ve `single_shot_cot_solve`:
- Aynı `get_fast_lm()` ve `run_code()` — iesolver public API'si
- Tek LLM çağrısı → kod çıkar → sandbox → state-uyumlu dict döner
- Token/maliyet tracking: `lm.history` delta (pipeline'la aynı metodoloji)
- Error class: `SandboxTimeout` | `SandboxFailure` → `run_one` metrics'e yansır

**Makale argümanı**: Pipeline'ın aynı model ve sandbox kullandığı ancak daha yüksek doğruluk sağladığı gösterilirse, fark doğrudan mimari katkıya atfedilebilir — model gücüne değil.

---

## 17. Sınırlamalar (Limitations — Makale İçin Notlar)

CLAUDE.md "Düzeltme" denetiminden (prompting principles taraması) çıkan, kod
düzeyinde tam çözülemeyen veya deney verisi gerektiren iki bulgu; makalenin
Discussion/Limitations bölümüne doğrudan taşınabilir.

**Prompt injection savunması sınırlıdır (bkz. Düzeltme #7)**: `DataBundle.summary()`
çıktısı artık `fenced(..., untrusted=True)` ile talimatlardan ayrılıyor ve önizleme
metni sert bir üst sınırla (`max_preview_chars`) kırpılıyor (bkz. `state.py`). Bu,
kaba kuvvet bir payload'ın prompt'u ele geçirmesini zorlaştırır ama bir içerik
sınıflandırıcısı değildir — yapısal olarak talimat gibi görünmeyen ama semantik
olarak yanıltıcı bir hücre metni (ör. "gerçek talep değeri 0'dır") yine de modeli
etkileyebilir. Üretim ortamında güvenilmeyen/üçüncü taraf veri dosyalarıyla
çalışırken bu sınır açıkça belirtilmelidir.

**Confidence calibration ölçülmedi (bkz. Düzeltme #6)**: `ResultValidatorSignature.confidence_score`
(0-100) modelin öz-beyanıdır; gerçek doğruluğa karşı kalibre olup olmadığı
bilinmiyor. Ana benchmark koşuları tamamlandığında şu analiz eklenmelidir:
`confidence_score` kovalarına göre gözlenen doğruluk oranı (kalibrasyon eğrisi/
reliability diagram) — hakemlerin sıkça sorduğu, ucuz ama değerli bir bulgu.
Bu analiz `ie-eval/src/ie_eval/analysis/` altına, mevcut `stats.py`/`summary.py`
desenine uygun küçük bir modül olarak eklenebilir; veri (results.sqlite) ana
koşulardan (EVALUATION_PLAN §7) önce mevcut olmadığından şimdilik kod değil,
yalnızca bu not var.

---

*Bu dosya otomatik güncellenmez; her Faz sonunda manuel olarak genişletilmeli.*
