# iesolver Projesi — Claude Code Context

## Önce Oku
1. PLAN.md — proje planı ve 5 fazlı yol haritası
2. METHODOLOGY_NOTES.md — mimari kararlar ve makale notları

## Mevcut Durum
- Faz 1 ✅ Tamamlandı (çatı, dummy node'lar, smoke test)
- Faz 2 ✅ Tamamlandı (12 Signature, 6 gerçek node, conditional edges, interrupt)
- Faz 3 🔄 Başlanacak (Code Branch: sandbox + ReAct + retry döngüleri)

## Kodlama Standartları
- Python 3.11+, uv, src-layout (src/iesolver/)
- DSPy node'ları call_with_fast_lm() / call_with_reasoning_lm() ile çağrılır
- Türkçe mimari yorum, İngilizce docstring
- Her node SolverState okur → partial SolverState döner
- __init__.py dosyaları unutulmasın

## Faz 3 Hedefi
nodes/code_branch/ paketini gerçek implementasyonla doldur:
algo_select → constraint_adapt → output_spec → generate(ReAct) → execute(sandbox) → validate
Smoke test: EOQ problemi doğru sayısal sonuç vermeli.

## Kritik Teknik Karar
DSPy 3.x + LangGraph contextvars çakışması:
Her DSPy çağrısı call_with_fast_lm() veya call_with_reasoning_lm() ile sarmalanmalı.
Direkt _module(**kwargs) çağrısı "No LM is loaded" hatası verir.