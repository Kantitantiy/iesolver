"""
iesolver.nodes.code_branch — Phase 4B pipeline (Code Engine).

Faz 2'deki stub'ın yerini alan gerçek implementasyon. Dört adım
sırayla yürütülür; LangGraph node'larına bölünmesi yerine bu paketi
tek bir node fonksiyonu olarak sunuyoruz. Gerekçe:

1. AlgoSelector → ConstraintAdapter → OutputSpec ara state'leri
   SolverState'te kalıcı alanlara dönmeli (retry döngüsünde lazım).
2. Ama bu adımların ayrı node olması, graph.py'ı gereksiz karmaşık
   yapar (5 yeni edge). Tek node → state merge → daha temiz topoloji.
3. Faz 4'te bu pipeline'ı sub-graph'a çevirmek istersek,
   sadece bu dosyayı değiştiririz; dış arayüz (code_branch_node)
   değişmez.

Dış graph'ın tükettiği state alanları:
    INPUT : essential_prompt, problem_type, data_summary,
            strict_constraints
    OUTPUT: target_algorithm, target_library,
            library_specific_constraints, code_output_spec,
            final_code, execution_result
    (validate ayrı node → is_valid, confidence_score, validation_notes)

Retry döngüsü (Plan §5 Faz 3):
    validate_node is_valid=False → graph.py'daki conditional edge
    bu fonksiyona geri döner (max retry_count=3).
    Her retry'da retry_count state'te artırılır.
"""

from __future__ import annotations

from iesolver.nodes.code_branch.algo_select import algo_select_node
from iesolver.nodes.code_branch.constraint_adapt import constraint_adapt_node
from iesolver.nodes.code_branch.generate import generate_node
from iesolver.nodes.code_branch.output_spec import output_spec_node
from iesolver.observability.metrics import instrument
from iesolver.state import SolverState


@instrument("code_branch")
def code_branch_node(state: SolverState) -> SolverState:
    """Run the full 4B sub-pipeline: select → adapt → spec → generate.

    validate_node is ayrı LangGraph node olarak graph.py'da çağrılır.
    """
    # 4B.1 — Algoritma ve kütüphane seçimi
    s1 = algo_select_node(state)
    merged: SolverState = {**state, **s1}  # type: ignore[misc]

    # 4B.2 — Kısıtları kütüphaneye adapte et
    s2 = constraint_adapt_node(merged)
    merged = {**merged, **s2}  # type: ignore[misc]

    # 4B.3 — Çıktı spesifikasyonu
    s3 = output_spec_node(merged)
    merged = {**merged, **s3}  # type: ignore[misc]

    # 4B.4 — ReAct: kod üret + sandbox'ta çalıştır
    s4 = generate_node(merged)
    merged = {**merged, **s4}  # type: ignore[misc]

    # Retry sayacını artır (ilk geçişte state'te yoksa 0'dan başlar)
    current_retry = int(state.get("retry_count", 0) or 0)

    return {
        "target_algorithm": merged.get("target_algorithm", ""),
        "target_library": merged.get("target_library", ""),
        "library_specific_constraints": merged.get("library_specific_constraints", ""),
        "code_output_spec": merged.get("code_output_spec", ""),
        "final_code": merged.get("final_code", ""),
        "execution_result": merged.get("execution_result", ""),
        "retry_count": current_retry + 1,
    }


__all__ = ["code_branch_node"]
