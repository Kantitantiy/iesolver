# IE-Solver — Değerlendirme Tasarımı (Q1 Makale)

> DESIGN_REVIEW §2'nin somutlaştırılmış hali. Faz 4.5 bu planı implemente eder.
> Makalenin *Experiments* bölümü buradan yazılır.

---

## 1. Araştırma Soruları

| # | Soru | Cevaplayan deney |
|---|---|---|
| RQ1 | Yapılandırılmış pipeline, tek atış LLM'e göre çözüm doğruluğunu artırıyor mu? | Benchmark + baseline karşılaştırması |
| RQ2 | Hangi bileşen ne kadar katkı sağlıyor? | Ablasyon çalışması |
| RQ3 | Mimari model-agnostik mi? | ≥2 model ailesiyle tekrar |
| RQ4 | Doğruluk kazancının maliyet/gecikme bedeli ne? | Telemetri analizi (metrics alanı) |
| RQ5 | Uçtan uca IE değeri (veri girişi → duyarlılık → rapor) ölçülebilir mi? | Vaka çalışmaları + uzman değerlendirmesi |

RQ5 makalenin farklılaşma noktası: rakipler (OptiMUS, CoE, LLMOPT) yalnızca RQ1–RQ4 alanında yarışıyor.

---

## 2. Benchmark'lar

**⚠️ Orijinal setler değil, temizlenmiş sürümler kullanılacak** — literatür orijinallerde etiket hataları raporluyor (NL4Opt'ta 16 düzeltme; IndustryOR'da 50 düzeltme + 23 geçersiz örnek). Temizlenmiş sürüm kullanımı artık standart protokol; makalede hangi sürümün kullanıldığı açıkça belirtilecek.

| Set | Boyut | Rol |
|---|---|---|
| NL4Opt (temiz test) | 289 LP | Ana karşılaştırma — herkes rapor ediyor |
| IndustryOR (temiz altküme) | ~42–77 | Zorluk kanıtı — SOTA ~%37, ayrışma burada görünür |
| ComplexOR | ~37 | Çok adımlı problemler (opsiyonel, 3. set) |
| **IE-Case (kendi setimiz)** | 5–8 problem | RQ5: veri dosyalı, uçtan uca senaryolar |

**IE-Case seti** (yeni katkı olarak sunulabilir): EOQ + çok ürünlü envanter (xlsx çok-sayfa), transportation LP (csv), job shop scheduling, atama problemi (sqlite), bir NO_CODE kavramsal soru. Her biri: veri dosyası + ground truth çözüm + uzman referans raporu. NL4Opt/IndustryOR'da veri dosyası yok — DataBundle argümanı yalnızca burada gösterilebilir.

NL4Opt'un **train split'i** (test'e değmeyen kısmı) DSPy MIPROv2 optimizasyonu için kullanılacak (bkz. §5, A5).

---

## 3. Metrikler

| Metrik | Tanım | Kaynak |
|---|---|---|
| Solving accuracy (birincil) | Optimal değer eşleşmesi, göreli tolerans 1e-4 | Deterministik karşılaştırma |
| Execution rate | Üretilen kod hatasız çalıştı mı | Sandbox |
| Feasibility rate | Çözüm kısıtları sağlıyor mu (programatik geri-ikame) | Deterministik validator |
| Router accuracy | CODE/NO_CODE kararı doğru mu (etiketli altküme) | Elle etiketleme |
| Maliyet | Token + $ / problem, node kırılımlı | metrics alanı |
| Gecikme | Uçtan uca süre + node kırılımlı | metrics alanı |
| Retry oranı | Ortalama retry, retry sonrası kurtarma oranı | metrics alanı |

Raporlama: **pass@1, 3 bağımsız koşunun ortalaması ± std**. API'ler temperature=0'da bile tam deterministik değil; tek koşu raporlamak hakem eleştirisi çeker.

---

## 4. Baseline'lar

1. **Tek atış LLM** (aynı model, pipeline'sız): "problemi çöz, kod yaz" tek prompt → kod çalıştır. En kritik baseline — pipeline'ın katma değerini izole eder.
2. **Tek atış + CoT**: aynı şey, chain-of-thought promptla. (Pipeline kazancının "sadece CoT etkisi" olmadığını gösterir.)
3. **Yayınlanmış sayılar**: OptiMUS, Chain-of-Experts, LLMOPT'un kendi makalelerindeki sonuçları tablo olarak (aynı benchmark, farklı model olduğu dipnotla). Yeniden koşturmak ideal ama zorunlu değil; OptiMUS açık kaynaksa ve bütçe elverirse bir set üzerinde yeniden koştur.

---

## 5. Ablasyonlar (RQ2)

| # | Çıkarılan/değiştirilen | Test ettiği iddia |
|---|---|---|
| A1 | PromptRefiner yok (ham prompt → router) | Yapılandırılmış ön işleme katkısı |
| A2 | Validator retry yok (tek deneme) | Retry döngüsü katkısı |
| A3 | Deterministik validation yok (yalnız LLM validator) | İkili doğrulama katmanı argümanı |
| A4 | Fast-only (reasoning LM anahtarlaması kapalı) | Model anahtarlama gerekçesi |
| A5 | MIPROv2 ile optimize edilmiş Signature'lar vs ham | "Neden DSPy?" sorusunun cevabı |
| A6 | StrategyRouter'da self-consistency (3-örnekleme + çoğunluk oyu) vs tek örnekleme | En yüksek blast-radius'lu tekil kararın (execution_path) varyansı; 3x router maliyeti karşılığında doğruluk kazancı var mı |

Ablasyonlar yalnızca **bir benchmark** (NL4Opt temiz test) + **tek model** üzerinde koşulur — kombinatorik patlamayı önler.

---

## 6. Model-Agnostiklik (RQ3)

Ana konfigürasyon: Gemini (Flash = fast, Pro = reasoning). İkinci aile: GPT-4o-mini/GPT-4o veya Claude Haiku/Sonnet — bütçeye göre. Yalnızca ana benchmark koşusu tekrarlanır (ablasyonlar tekrarlanmaz). DSPy sayesinde değişiklik `lm.py`'da tek nokta.

---

## 7. İstatistik

- 3 koşu × her konfigürasyon; ortalama ± std
- İkili karşılaştırmalar (pipeline vs baseline, ablasyon vs tam sistem): **McNemar testi** (aynı problem setinde eşleştirilmiş doğru/yanlış — doğru test budur), α=0.05
- Etki büyüklüğü: doğruluk farkı + %95 güven aralığı (bootstrap)

## 8. Protokol Sabitleri

- `auto_mode=True` (human-in-loop kapalı; clarify tetiklenirse "varsayım yapıldı" olarak loglanır ve raporlanır)
- temperature=0, sabit seed'ler, `uv.lock` sabit, sandbox kütüphane sürümleri pinli
- Timeout: problem başına sabit (ör. 300 sn), max_retries=3
- Tüm checkpoint DB'leri + üretilen kodlar arşivlenir; kod ve trace'ler yayınla birlikte açılır (reproducibility bölümü)

## 9. Hata Analizi

Her başarısız örnek 5 sınıftan birine etiketlenir: **anlama** (yanlış problem) / **formülasyon** (yanlış model) / **kodlama** (çalışmayan kod) / **çözücü** (yanlış kütüphane-parametre) / **doğrulama** (yanlış çözüm kabul edildi). Sınıf dağılımı + benchmark bazında karşılaştırma + 3–4 nitel örnek → makalenin error analysis bölümü. Etiketleme yarı otomatik: metrics alanındaki hata sınıfı + elle kontrol.

## 10. Vaka Çalışmaları + Uzman Değerlendirmesi (RQ5)

IE-Case setinin 2–3'ü makalede ayrıntılı anlatılır: girdi dosyası → pipeline izi → duyarlılık (tornado) → 3 katmanlı rapor. Ek güç: **2–3 IE akademisyeni** üretilen raporları 5'li Likert ile puanlar (doğruluk, eksiksizlik, karar desteği değeri) — küçük ama hakem nezdinde etkili. Karşılaştırma noktası: aynı problemin ders kitabı/uzman çözümü.

## 11. Bütçe Kestirimi (kaba)

Problem başına ~8–12 LLM çağrısı. Ana koşu: 289 × 3 koşu × ~10 çağrı ≈ 8.700 çağrı/model-konfigürasyonu. Ablasyonlar (5 × 289 × 3) ≈ 13.000 çağrı. Toplam iki model ailesiyle ~35–40 bin çağrı; Flash-sınıfı modellerde düşük maliyet, reasoning çağrıları belirleyici. **Ücretsiz tier bu hacmi kaldırmaz** — deney öncesi ücretli kota şart; checkpointing yarım koşuları kurtararak maliyeti düşürür.

## 12. Faz 4.5 İmplementasyon Haritası

```
v4/
├── iesolver/            (kütüphane — değişiklik yok)
└── ie-eval/             (ayrı proje: kütüphaneyi dışarıdan kullanır ✓ tasarım hedefinle uyumlu)
    ├── datasets/        # NL4Opt/IndustryOR temiz sürüm adaptörleri + IE-Case seti
    ├── runner.py        # batch koşucu: auto_mode, timeout, 3-koşu, konfigürasyon matrisi
    ├── baselines.py     # tek atış + CoT baseline sarmalayıcıları
    ├── metrics.py       # doğruluk/feasibility/maliyet toplayıcı → results.sqlite
    ├── analysis/        # McNemar, bootstrap CI, tablo/figür üretimi
    └── results/         # koşu arşivi (checkpoint DB + kodlar)
```

`ie-eval`'ın ayrı proje olması "iesolver bir kütüphanedir" tasarımının ilk gerçek testi — bu da makalede bir cümlelik güzel bir kanıt.

## 13. Açık Kararlar

1. **Bütçe**: ikinci model ailesi hangisi, ablasyonlar hangi modelde? (maliyet tavanına göre)
2. **OptiMUS yeniden koşulacak mı** yoksa yayınlanmış sayılar mı? (2–3 gün iş farkı)
3. **Hedef dergi**: Computers & Industrial Engineering / Expert Systems with Applications / Computers & OR / IJPR — hepsi Q1; ESWA sistem makalelerine en açık olanı, EJOR en zorlusu. Çerçeveleme ("end-to-end decision support") ESWA ve C&IE ile en uyumlu.
4. Uzman değerlendirmesi yapılacak mı (etik kurul gerekmez ama zaman ister)?

---

## Kaynaklar

- [A Survey of Optimization Modeling Meets LLMs (2025)](https://arxiv.org/html/2508.10047) — protokol ve veri temizliği referansı
- [ORAgentBench: End-to-End OR Agent Benchmark (2026)](https://arxiv.org/html/2606.19787v1) — uçtan uca çerçeveyle doğrudan ilgili, related work'e eklenecek
- [OptiMind (Microsoft)](https://arxiv.org/pdf/2509.22979) — temizlenmiş benchmark sürümleri
- [Step-Opt](https://arxiv.org/pdf/2506.17637) — yapılandırılmış doğrulama
- [ConstraintBench](https://arxiv.org/pdf/2602.22465)
