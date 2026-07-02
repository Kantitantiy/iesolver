"""
iesolver.nodes.validate — Phase 4B.6 (Result Validator).

Eski ``ResultValidator``'ın LangGraph karşılığı. Kodun ürettiği sayısal
ve mantıksal çıktıları IE sınır testlerinden geçirir.

Makale notu — "Semantic and Logical Boundary Verification":
    Bir LP problemi negatif stok miktarı mı döndürdü? Olasılık değeri
    1'i mi aştı? Kümeler boş mu? Bu kontroller saf matematiksel
    doğrulama değil; alan bilgisi gerektiren semantik sınama. LLM
    bu muhakemeyi ChainOfThought ile yapıyor.

DSPy modülü neden ChainOfThought?
    Eski ``phase_4b_code_engine.py``'daki gerekçeyi koruyoruz:
    "Sayısal sonuçları yorumlarken modelin 'Nasıl bu kanıya vardım?'
    sorusunu kendi içinde yanıtlaması için CoT kullanıyoruz."

Tipli DSPy 3.x çıktıları (DESIGN_REVIEW §3.4):
    is_valid → native bool; confidence_score → native int.
    Eski str→bool / str→int sanitization deseni kaldırıldı.

Retry kararı:
    Bu node'un yazdığı ``is_valid`` alanını graph.py'daki
    ``_route_after_validate`` fonksiyonu okur:
      - is_valid=False AND retry_count < max_retries → code_branch'e dön
      - is_valid=False AND retry_count >= max_retries → raporu yine de yaz
      - is_valid=True  → rapor node'una geç
"""

from __future__ import annotations

import dspy

from iesolver.lm import call_with_reasoning_lm
from iesolver.observability.metrics import instrument
from iesolver.signatures import ResultValidatorSignature
from iesolver.state import SolverState

_validator = dspy.ChainOfThought(ResultValidatorSignature)


@instrument("validate")
def validate_node(state: SolverState) -> SolverState:
    """Validate execution result against IE logic and domain constraints.

    Reads
    -----
    essential_prompt, strict_constraints, execution_result

    Writes
    ------
    is_valid, confidence_score, validation_notes
    """
    result = call_with_reasoning_lm(
        _validator,
        essential_prompt=state.get("essential_prompt", "") or "",
        strict_constraints=state.get("strict_constraints", "") or "",
        execution_result=state.get("execution_result", "") or "(no result)",
    )

    return {
        "is_valid": result.is_valid,
        "confidence_score": result.confidence_score,
        "validation_notes": result.validation_notes,
    }