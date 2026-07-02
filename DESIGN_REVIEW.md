# IE-Solver — Tasarım İncelemesi (Faz 4 Öncesi)

> Tarih: 2026-07-02 · Kapsam: PLAN.md mimarisi + Faz 1–3 durumu · Hedef: Q1 makale + atıf

---

## 1. Genel Hüküm

Mimari sağlam. DSPy (reasoning units) ⊕ LangGraph (workflow engine) ayrımı, TypedDict state, checkpointing, CODE/NO_CODE bifurcation ve retry döngüsü doğru kurulmuş. **Baştan yazma gereksiz** — aşağıdaki değişikliklerin tamamı mevcut iskelete eklemedir.

Asıl risk mimaride değil, **makale stratejisinde**: plan bir mühendislik ürünü tanımlıyor ama Q1 dergiyi ikna edecek deney tasarımı planda hiç yok. Faz 5'teki "2–3 örnek problem e2e test" hakem gözünde kanıt değildir.

---

## 2. Kritik Boşluk: Değerlendirme Tasarımı Yok

Bu alan artık kalabalık. Doğrudan rakipler: OptiMUS (ICML 2024, modeler→coder→debugger→evaluator ajan hattı), Chain-of-Experts (ICLR 2024), OptimAI, LLMOPT, SolverLLM, ORPilot ve 2026'da Microsoft'un OptiMind'ı. "LLM ile OR problemi çözen pipeline" tek başına yeni değil; hakem ilk sorusu "bunlardan farkın ne ve sayılarla nerede duruyorsun?" olacak.

Standart benchmark'lar mevcut ve hakemler bunları bekler:

| Benchmark | İçerik | SOTA civarı |
|---|---|---|
| NL4Opt | 289 test LP kelime problemi | ~%89 |
| ComplexOR | Çok adımlı OR problemleri | ~%77 |
| IndustryOR | 13 sektörden 100 gerçek problem | ~%37 (hâlâ zor) |
| MAMO, OptMATH, OptiBench | Solver-doğrulamalı setler | — |

**Gerekli deney seti:**
- En az 2 benchmark (NL4Opt + IndustryOR önerilir) üzerinde doğruluk, execution rate, maliyet, gecikme
- Baseline'lar: tek atış LLM (aynı model, pipeline'sız), mümkünse OptiMUS/CoE'nin rapor edilen sayıları
- Ablasyonlar: refiner'sız, validator-retry'sız, fast-only vs fast+reasoning, DSPy optimizer'lı vs optimizersız
- ≥2 model ailesiyle tekrar (model-agnostiklik iddiası için; DSPy bunu zaten kolaylaştırıyor)

**Özgünlük konumlandırması:** Rakipler yalnızca "doğal dil → formülasyon → çözüm" yapıyor. Sizin farkınız uçtan uca IE kapsamı: tek dosyalı veri girişi (DataBundle), human-in-loop, duyarlılık analizi, 3 katmanlı rapor üretimi, domain pack genişletilebilirliği. Makaleyi "optimization modeling agent" olarak değil **"end-to-end decision-support pipeline for IE"** olarak çerçeveleyin; benchmark bölümünde modeling doğruluğunu, case study bölümünde uçtan uca farkı gösterin.

---

## 3. Tasarım Düzeltmeleri (öncelik sırasıyla)

### 3.1 Headless/batch mod — ZORUNLU
`interrupt()` human-in-loop'u benchmark koşusunu imkânsız kılar. `solve(..., auto_mode=True)` bayrağı: eksik bilgi varsa clarify'a gitmek yerine varsayılan varsayımları loglayıp devam etsin. Bu olmadan hiçbir toplu deney koşamazsınız.

### 3.2 Deterministik doğrulama katmanı
`validate_node` şu an LLM öz-eleştirisi. Hakem "LLM kendi kendini mi notluyor?" der. Eklenecekler:
- Çözümü kısıtlara programatik geri-ikame (feasibility check — LLM'siz, kesin)
- Solver status kontrolü (OPTIMAL/INFEASIBLE/UNBOUNDED)
- LLM validator'ı yalnızca semantik kontrol için tutun (doğru problem mi çözülmüş?)

Bu ikili katman (deterministic + semantic validation) başlı başına metodoloji katkısı olarak yazılabilir.

### 3.3 DSPy optimizer kullanımı
DSPy'ın varlık sebebi Signature optimizasyonu (MIPROv2 / BootstrapFewShot). Plan bunu hiç kullanmıyor; hakem "neden DSPy, neden düz prompt değil?" diye sorar. Küçük bir train setiyle en az 1–2 kritik Signature'ı (AlgoSelector, ReAct code) optimize edip öncesi/sonrası doğruluk tablosu üretin. Hem DSPy tercihinin gerekçesi hem güçlü bir ablasyon olur.

### 3.4 Tipli çıktılar — sanitization'ı kaldırın
`str → bool` / `str → int` dönüşümleri kırılgan. DSPy 3.x tipli output destekliyor: `is_complete: bool`, `confidence_score: int`, `execution_path: Literal["CODE","NO_CODE"]` doğrudan Signature output tipi olsun. `missing_items`, `constraints` gibi alanlar `str` yerine `list[str]` olmalı (ölçüm ve test için).

### 3.5 Telemetri state'e girsin
Makale tabloları için node başına: token sayısı, maliyet, gecikme, retry sayısı, hata sınıfı. `SolverState`'e `metrics: dict` ekleyin; checkpoint'ler zaten var, üstüne yazması ucuz. Hata taksonomisi (formulation error / code error / solver error / validation error) makalenin error-analysis bölümünü bedavaya getirir.

### 3.6 Duyarlılık analizi: önce dual, sonra perturbasyon
LP/MILP'de shadow price ve reduced cost solver'dan bedava gelir. Kaba kuvvet ±%5/±%10 perturbasyonu yalnızca dual bilgisi olmayan problemler için fallback yapın. Hem hesap tasarrufu hem OR açısından doğru yaklaşım — hakemler fark eder.

### 3.7 Tekrarlanabilirlik
temperature=0 + seed, `uv.lock` sabit, sandbox kütüphane sürümleri pinli, prompt/Signature sürümleme (git tag yeter). Q1 dergide artık standart soru.

### 3.8 API bütçesi
Ücretsiz Gemini kotası benchmark koşularını kaldırmaz (289 NL4Opt problemi × node sayısı × ablasyonlar). Deney fazından önce ücretli kota planlayın; checkpointing sayesinde yarıda kalan koşular replay edilebilir, bu maliyeti düşürür.

---

## 4. Revize Faz Planı

| Faz | İçerik | Not |
|---|---|---|
| 4 | Sensitivity + Artifacts | Planlandığı gibi, ama 3.6'daki dual-öncelikli tasarımla |
| **4.5 (YENİ)** | **Evaluation harness** | Batch runner (auto_mode), NL4Opt/IndustryOR adaptörleri, metrik toplayıcı, baseline sarmalayıcı, deterministik validator | 
| 5a | ReportWriter (PDF/DOCX/HTML) | Makaledeki case study çıktıları için gerekli |
| 5b | Streamlit UI | **Makaleye katkısı sıfır — yayın sonrasına ertelenebilir** |

Faz 4.5 makale için Faz 5'in tamamından daha değerlidir.

---

## 5. Claude Code Sorusuna Cevap

Baştan yazmayın. Değişiklikler eklemeli: yeni node'lar (deterministic validator, eval harness), bir bayrak (auto_mode), Signature output tiplerinin düzeltilmesi, state'e metrics alanı. Mevcut Faz 1–3 kodu bu inceleme sonrası da geçerli kalır.

---

## Kaynaklar

- [OptiMUS (ICML 2024)](https://dl.acm.org/doi/10.5555/3692070.3692094)
- [OptimAI: Optimization from Natural Language Using LLM-Powered AI Agents](https://arxiv.org/html/2504.16918v3)
- [AlphaOPT: Self-Improving LLM Experience Library](https://arxiv.org/html/2510.18428v4)
- [SolverLLM: Test-Time Scaling for Optimization](https://www.arxiv.org/pdf/2510.16916v2)
- [ORPilot: Production-Oriented Agentic LLM-for-OR Tool](https://arxiv.org/pdf/2605.02728)
- [ConstraintBench](https://arxiv.org/html/2602.22465)
- [OptiVerse: Comprehensive Benchmark](https://arxiv.org/html/2604.21510)
- [Toward a Trustworthy Optimization Modeling Agent](https://arxiv.org/pdf/2508.03117)
- [Memory-Enhanced LLM Agents with Decentralized Debate for Optimization Modeling](https://arxiv.org/pdf/2604.25847)
