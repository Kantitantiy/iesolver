# IE-Solver — Makale Yol Haritası (Kod Sonrası → Q1 Yayın)

> Bu doküman, kodlama bittikten sonra makale yayınına kadar yapılacak **her işi** sıfırdan anlatır.
> Referans dosyalar: PLAN.md · DESIGN_REVIEW.md · EVALUATION_PLAN.md · METHODOLOGY_NOTES.md

---

## Özet Tablo

| # | İş | Çıktısı | Tahmini süre | Bağımlılık |
|---|-----|---------|--------------|------------|
| 1 | Sistem incelemesi + kendi iyileştirmelerin | Gözden geçirilmiş kod, notlar | 1–2 hafta | — |
| 2 | API kredi başvuruları | OpenAI/Anthropic/Google başvuruları | 1 gün (bekleme haftalar) | — (hemen başla) |
| 3 | Benchmark verilerini edinme ve temizleme | `ie-eval/datasets/` hazır | 2–3 gün | — |
| 4 | IE-Case setini oluşturma | 5–8 problem + veri + ground truth | 1 hafta | 1 |
| 5 | Pilot koşu (küçük örneklem) | Çalışan harness, düzeltilmiş hatalar | 3–5 gün | 1, 3 |
| 6 | DSPy MIPROv2 optimizasyonu | Optimize Signature'lar (A5 ablasyonu) | 3–4 gün | 5 |
| 7 | Ana benchmark koşuları | Ham sonuçlar (results.sqlite) | 3–5 gün | 5, 6, bütçe |
| 8 | Ablasyon koşuları | 5 ablasyon sonucu | 3–5 gün | 7 |
| 9 | İkinci model ailesi koşusu | RQ3 sonuçları | 2–3 gün | 7, 2 |
| 10 | İstatistiksel analiz + tablo/figür üretimi | Makale tabloları ve figürleri | 3–4 gün | 7, 8, 9 |
| 11 | Hata analizi (etiketleme) | Hata taksonomisi tablosu + nitel örnekler | 3–4 gün | 7 |
| 12 | Vaka çalışmaları + uzman değerlendirmesi | RQ5 bulguları | 1–2 hafta | 4, 7 |
| 13 | Tekrarlanabilirlik paketi | Public GitHub repo + Zenodo DOI | 2–3 gün | 7–12 |
| 14 | Makale yazımı | Tam taslak (İngilizce) | 3–4 hafta | 10, 11, 12 |
| 15 | Dergi seçimi + formatlama | Dergi formatında makale | 2–3 gün | 14 |
| 16 | İç okuma + dil kontrolü | Gönderime hazır makale | 1 hafta | 15 |
| 17 | Gönderim | Submission + cover letter | 1 gün | 16 |
| 18 | Revizyon süreci | Kabul 🎉 | 3–9 ay (bekleme) | 17 |

Kritik yol: 1 → 5 → 7 → 10 → 14 → 17. Toplam aktif çalışma ~3–4 ay.
**2 numara beklemeli iş — bugün başlat.** 4 ve 6, ana koşularla paralel yürüyebilir.

---

## 1. Sistem İncelemesi + Kendi İyileştirmelerin

**Ne**: Kodu satır satır kendi gözünle anlaman ve kendi dokunuşlarını yapman. Bu hem hâkimiyet (savunma/sunumda "bunu ben tasarladım" diyebilmek) hem kalite için şart.

**Nasıl**:
1. `graph.py`'dan başla — topolojiyi kağıda çiz, PLAN.md'deki akışla karşılaştır.
2. Her node'u tek tek oku: hangi state alanlarını okuyor, hangilerini yazıyor? Bir tabloya dök (bu tablo makaleye de girer).
3. Signature docstring'lerini oku — bunlar sistemin "prompt'ları"; iyileştirme yapacağın en etkili yer burası.
4. Testleri çalıştır, bir problemi adım adım debug modda izle (PyCharm breakpoint veya checkpoint DB'sini inceleyerek).
5. İyileştirme fikirlerini `IMPROVEMENTS.md`'ye not et; küçükleri hemen yap, büyükleri deneylerden SONRAYA bırak (deney başladıktan sonra kod değişirse tüm koşular geçersizleşir — **code freeze** kavramı).

**Çıktı**: anladığın ve sahiplendiğin kod + IMPROVEMENTS.md + deney öncesi son commit'e git tag: `v1.0-experiments`.

---

## 2. API Kredi Başvuruları (BUGÜN BAŞLAT)

**Ne**: Deney maliyetini (~$300–800) karşılamak için ücretsiz akademik kredi.

**Nasıl**:
1. [OpenAI Researcher Access Program](https://openai.com/form/researcher-access-program/) — $1.000'a kadar; projeyi 3–4 paragrafta anlat.
2. Anthropic araştırma kredileri — benzer başvuru.
3. Google Cloud research credits — Gemini ana modelin olduğu için en önemlisi.
4. DeepSeek akademik erişim (zaten biliyorsun).

Başvuru metni için EVALUATION_PLAN.md'deki RQ'ları özetleyen tek sayfalık İngilizce özet yaz — üçüne de aynı metin uyarlanır.

**Çıktı**: 3–4 başvuru gönderilmiş. Cevap beklerken diğer işlere devam.

---

## 3. Benchmark Verilerini Edinme ve Temizleme

**Ne**: NL4Opt ve IndustryOR'un **temizlenmiş** sürümlerini indirip `ie-eval/datasets/` altına adaptörleriyle koymak.

**Nasıl**:
1. Temiz sürümler literatürde paylaşılıyor (OptiMind ve survey makalelerinin GitHub repoları). Orijinali DEĞİL bunları kullan — orijinallerde etiket hataları var.
2. Her problem için standart kayıt: `{id, problem_text, ground_truth_value, tolerance, kaynak_sürüm}`.
3. Adaptör: bu kayıtları `iesolver.solve()`'un beklediği girdiye çeviren küçük Python modülü.
4. 5–10 problemi elle kontrol et: ground truth gerçekten doğru mu? (Bir LP'yi elle veya PuLP ile çöz, karşılaştır.)
5. Kullandığın sürümün commit hash'ini not et — makalede "we use the cleaned version of X (commit abc123)" yazacaksın.

**Çıktı**: `datasets/` klasörü + sürüm dokümantasyonu.

---

## 4. IE-Case Setini Oluşturma

**Ne**: Kendi 5–8 problemlik, veri dosyalı, uçtan uca test setin. Makalenin özgün katkılarından biri.

**Nasıl**:
1. Problemler: EOQ (csv), çok ürünlü envanter (xlsx çok-sayfa), transportation LP (csv), atama problemi (sqlite), job shop (xlsx), 1 NO_CODE kavramsal soru. Ders kitaplarından uyarla ama sayıları değiştir (telif + ezber riski: LLM kitap problemini ezbere biliyor olabilir!).
2. Her problem için: problem metni + veri dosyası + ground truth (elle/PuLP ile çözülmüş) + kısa uzman referans çözümü.
3. Bir README ile seti dokümante et — bu set yayınla birlikte açılacak.

**Çıktı**: `datasets/ie_case/` + README. **Sayıları değiştirilmiş olması makalede açıkça belirtilecek** (data contamination önlemi — hakemler soruyor).

---

## 5. Pilot Koşu

**Ne**: Tam protokolü küçük ölçekte (20–30 problem × 1 koşu) çalıştırıp altyapı hatalarını ayıklamak. **Bu adımı atlama** — ana koşuda hata bulmak paranı ve zamanını yakar.

**Nasıl**:
1. `auto_mode=True`, temperature=0, timeout ve max_retries sabitle.
2. NL4Opt'tan rastgele 20 + IndustryOR'dan 5 + IE-Case'ten 2 problem seç.
3. Koş; şunları kontrol et: metrics alanı doluyor mu, sonuç karşılaştırması (tolerans) doğru çalışıyor mu, checkpoint'ler yazılıyor mu, hiçbir problem pipeline'ı kilitleyemiyor mu.
4. Ücretsiz tier'da (Groq/OpenRouter/NVIDIA) koşabilirsin — burada gecikme ölçümü önemli değil.
5. Bulduğun her hatayı düzelt, pilotu tekrarla. İki temiz pilot koşu = ana koşuya hazırsın.

**Çıktı**: sorunsuz çalışan harness + kesinleşmiş protokol konfigürasyonu (`config_experiments.yaml` gibi tek dosyada).

---

## 6. DSPy MIPROv2 Optimizasyonu

**Ne**: A5 ablasyonu için Signature'ları otomatik optimize etmek. "Neden DSPy?" sorusunun deneysel cevabı.

**Nasıl**:
1. NL4Opt **train split**'inden 50–100 problem ayır (test 289'una asla dokunma — data leakage!).
2. Metrik fonksiyonu yaz: `çözüm doğru mu → 1/0`.
3. `dspy.MIPROv2` ile en kritik 2 Signature'ı optimize et: AlgoSelector ve ReAct code. (Hepsini optimize etmek pahalı ve gereksiz.)
4. Optimize edilmiş prompt'ları ayrı dosyaya kaydet (`compiled_signatures/`) — deneyde "ham vs optimize" iki konfigürasyon olarak koşulacak.

**Çıktı**: optimize Signature seti + optimizasyonun kendi maliyet kaydı (makalede raporlanır).

---

## 7. Ana Benchmark Koşuları

**Ne**: Asıl deney. NL4Opt (289) + IndustryOR (~50) × 3 koşu × ana konfigürasyon (Gemini, optimize Signature'lar) + tek atış ve CoT baseline'ları.

**Nasıl**:
1. **Önce code freeze**: `v1.0-experiments` tag'inden itibaren pipeline koduna dokunma. Hata çıkarsa düzelt, tag'i güncelle, etkilenen koşuları baştan al.
2. Ücretli Gemini kotasıyla koş; rate limit'te backoff (fallback YOK — EVALUATION_PLAN §8).
3. Her koşu ayrı `run_id` ile results.sqlite'a; checkpoint DB'leri arşivle.
4. Baseline'ları da aynı 3-koşu protokolüyle koş (aynı model, pipeline'sız).
5. Günlük ilerlemeyi kontrol et; bir problemin 3 koşusu da timeout ise not düş, hata analizine girer.

**Çıktı**: ~2.600 problem-koşusu sonucu. Makalenin ana tablosunun ham verisi.

---

## 8. Ablasyon Koşuları

**Ne**: 5 ablasyon (A1–A5, EVALUATION_PLAN §5) × NL4Opt × 3 koşu, tek model.

**Nasıl**: Her ablasyon bir config bayrağı olmalı (`disable_refiner=True` gibi) — kod kopyalamak değil. Bütçe sıkışırsa 100 problemlik tabakalı örneklem kullan (kolay/orta/zor dengeli) ve bunu makalede belirt.

**Çıktı**: ablasyon tablosu ham verisi — "her bileşen ne katıyor" grafiği buradan çıkar.

---

## 9. İkinci Model Ailesi Koşusu

**Ne**: RQ3 (model-agnostiklik). Ana koşunun aynısı, Claude Haiku veya GPT-5.4 Mini ile (kredi başvurusu sonucuna göre seç). Yalnızca ana benchmark; ablasyon tekrarlanmaz. `lm.py`'da model değişikliği + kısa pilot + koş.

**Çıktı**: ikinci ailenin sonuç seti.

---

## 10. İstatistiksel Analiz + Tablo/Figür Üretimi

**Ne**: Ham sonuçları makale tablolarına ve figürlerine dönüştürmek.

**Nasıl**:
1. `analysis/` altında script'ler — elle Excel işlemi YOK, her tablo koddan üretilmeli (revizyonda deney tekrarı gerekirse tablolar tek komutla yenilenir).
2. Ana tablo: sistem vs baseline'lar vs literatür sayıları; doğruluk ortalama±std, execution rate, feasibility rate.
3. McNemar testi (pipeline vs baseline, eşleştirilmiş) + bootstrap %95 CI.
4. Figürler: ablasyon çubuk grafiği, maliyet-doğruluk saçılımı, node bazlı maliyet kırılımı, hata taksonomisi dağılımı. Matplotlib, tek stil, PDF vektör çıktı.

**Çıktı**: `analysis/output/` altında makaleye girecek tüm tablo (LaTeX) ve figürler.

---

## 11. Hata Analizi

**Ne**: Başarısız örnekleri 5 sınıfa etiketlemek (anlama / formülasyon / kodlama / çözücü / doğrulama) ve nitel incelemek.

**Nasıl**:
1. Başarısızları listele; metrics'teki otomatik hata sınıfı ilk etiket.
2. Her birinin trace'ini (checkpoint + üretilen kod) aç, elle doğrula/düzelt — LLM'in nerede saptığını gör.
3. 3–4 öğretici vakayı derinlemesine yaz (yanlış giden + neden). Hakemler bu bölümü sever; "sistemin sınırlarını biliyorlar" izlenimi verir.
4. Doğrulama sınıfı özellikle önemli: "yanlış çözüm kabul edildi" vakaları A3 ablasyonuyla birlikte okunur.

**Çıktı**: hata dağılım tablosu + nitel vaka metinleri (makalenin Discussion malzemesi).

---

## 12. Vaka Çalışmaları + Uzman Değerlendirmesi

**Ne**: RQ5 — rakiplerin yapamadığı uçtan uca gösterim.

**Nasıl**:
1. IE-Case'ten 2–3 problemi tam pipeline'la koş: veri dosyası → çözüm → duyarlılık (tornado chart) → 3 katmanlı rapor.
2. Her adımın çıktısını makale için arşivle (figures/, raporların PDF'leri).
3. Uzman değerlendirmesi: 2–3 IE akademisyenine üretilen raporları gönder; 5'li Likert ile puanlasınlar (doğruluk / eksiksizlik / karar desteği değeri) + serbest yorum. Basit bir form yeterli. Katılımcı bilgilendirme cümlesi ekle; dergiye göre etik beyanı gerekebilir — dergi seçiminde kontrol et (İş 15).

**Çıktı**: vaka anlatıları + uzman puanları tablosu.

---

## 13. Tekrarlanabilirlik Paketi

**Ne**: Kod + veri + sonuçların herkesçe tekrar edilebilir arşivi. Q1 dergilerde artık fiilen zorunlu; kabul şansını somut artırır.

**Nasıl**:
1. Public GitHub repo: iesolver + ie-eval + IE-Case seti + analysis script'leri + README (kurulum, koşum, tablo üretimi adımları).
2. API anahtarları hariç her şey; `uv.lock` dahil.
3. Zenodo'ya bağla → DOI al; makalede "code and data available at DOI:..." yaz.
4. Lisans seç (MIT veya Apache-2.0 yaygın).

**Çıktı**: DOI'li public arşiv.

---

## 14. Makale Yazımı

**Ne**: İngilizce tam taslak. En uzun iş — küçümseme.

**Yapı** (sistem makalesi standardı):
1. **Introduction** — problem, boşluk, katkı listesi (4–5 madde: mimari, DataBundle, ikili doğrulama, IE-Case seti, deneysel bulgular)
2. **Related Work** — LLM-for-OR sistemleri (OptiMUS, CoE, LLMOPT, OptiMind, ORAgentBench...), benchmark'lar, agentic framework'ler. Survey makalesinden başla, 40–60 referans normal.
3. **Methodology** — METHODOLOGY_NOTES.md buraya çevrilir; §12'deki terminoloji tablosunu kullan. Mimari şeması + state akış figürü çiz.
4. **Experimental Setup** — EVALUATION_PLAN buraya çevrilir: RQ'lar, setler, metrikler, protokol.
5. **Results** — İş 10'un tabloları/figürleri + her RQ'ya açık cevap.
6. **Discussion** — hata analizi, sınırlar (limitations bölümü ŞART: veri kirliliği riski, LLM stokastikliği, benchmark kapsamı), tehditler (threats to validity).
7. **Conclusion + Future Work**

**Pratik**: Önce Methodology + Setup yaz (malzeme hazır), sonra Results, en son Intro ve Abstract. Overleaf'te LaTeX kullan; dergi şablonunu baştan al. Claude'dan çeviri/dil desteği alabilirsin ama her cümleyi kendin doğrula — makale senin sesin.

**Çıktı**: tam taslak.

---

## 15. Dergi Seçimi + Formatlama

**Ne**: Hedef dergiyi kesinleştirip makaleyi formatına sokmak.

**Adaylar** (hepsi Q1): **Expert Systems with Applications** (sistem makalelerine en açık, hızlı süreç) → önerilen ilk hedef; **Computers & Industrial Engineering** (IE kimliği güçlü); Computers & OR, IJPR (daha OR-teorik, daha zor). Karar kriterleri: kapsam uyumu, ortalama süreç süresi, open access ücreti (kurum anlaşmanı kontrol et — TÜBİTAK/kurum kapsıyor olabilir).

**Nasıl**: Dergi sitesinden "Guide for Authors" oku; şablon, kelime limiti, highlight'lar, beyanlar (etik, çıkar çatışması, AI kullanım beyanı — **LLM tabanlı araç kullandığını yazım sürecinde de kullandıysan beyan et, dergiler artık istiyor**).

**Çıktı**: dergi formatında makale + gerekli beyanlar.

---

## 16. İç Okuma + Dil Kontrolü

**Ne**: Gönderim öncesi kalite turu.

**Nasıl**: (1) Makaleyi 2–3 gün dinlendir, sonra baştan oku. (2) Danışman/meslektaş okuması — özellikle deney bölümüne "hangi soruyu sorardın?" diye sor. (3) Dil: native-level düzeltme (kurumun editing servisi varsa kullan). (4) Kontrol listesi: her figüre metinde atıf var mı, her sayı tablodan doğrulanıyor mu, referans formatı tutarlı mı, DOI linki çalışıyor mu.

**Çıktı**: gönderime hazır PDF.

---

## 17. Gönderim

**Nasıl**: Dergi sistemine (genelde Editorial Manager) kayıt; cover letter yaz (yarım sayfa: problem, katkı, neden bu dergi); önerilen hakem listesi istenirse related work'teki yazarlardan 3–5 isim (danışmanına danış); tüm beyanları doldur; gönder. **Aynı anda tek dergiye** — çoklu gönderim etik ihlaldir.

**Çıktı**: submission ID. 🎉 (ilk tur)

---

## 18. Revizyon Süreci

**Ne beklemeli**: İlk cevap 2–5 ay. Olası sonuçlar: desk reject (kapsam dışı — üzülme, sıradaki dergiye), major revision (EN OLASI ve iyi haber!), minor revision, kabul (ilk turda nadir).

**Major revision gelirse**:
1. Her hakem yorumunu numaralandır, "response to reviewers" dokümanında tek tek cevapla (yorum → cevap → makalede yapılan değişiklik).
2. Ek deney isterlerse harness hazır — tabloların koddan üretilmesi burada hayat kurtarır (İş 10'daki kural).
3. Kırıcı olma, savunmacı olma; her eleştiriden makaleyi güçlendirecek bir şey çıkar.
4. Verilen süreye (genelde 1–3 ay) uy.

**Çıktı**: kabul → proof kontrolü → yayın → atıf toplama dönemi başlar. Atıf için: repo README'sine "cite us" bloğu, ResearchGate/X duyurusu, IE-Case setinin başkalarınca kullanımı en güçlü atıf mıknatısıdır.

---

## Kritik Kurallar (Özet)

1. **Code freeze**: deney başladıktan sonra pipeline koduna dokunulmaz; dokunulursa etkilenen koşular baştan alınır.
2. **Test setine dokunma**: optimizasyon/geliştirme yalnızca train split'te.
3. **Her tablo koddan üretilir**: elle sayı taşımak yasak.
4. **Her şey sürümlü**: veri sürümü, kod tag'i, model adı+tarihi, prompt sürümü.
5. **Kredi başvuruları bugün**: en uzun bekleme onlarda.
