"""
iesolver.nodes.artifacts — Faz 4D (Artifact Generator).

sensitivity_results'taki duyarlılık tablosundan matplotlib tornado chart
üretir. Grafik artifacts_dir'e PNG olarak kaydedilir; path state["figures"]
listesine eklenir.

Tasarım kararları:
    * Artifact path deterministik: thread_id içermez; EOQ her çalıştırmada
      aynı dosyanın üzerine yazar. Faz 4.5 batch runner'da thread_id bazlı
      isimlendirme gerekecek — o zaman config'den alınır.
    * Sandbox'ın dosya yazmasına izin verilir: matplotlib chart kodunu
      artifacts_dir altındaki bir hedefe savefig() ile kaydeder. Bu, ana
      process'in güvenli klasörüdür; sandbox'ın genel dosya sistemi erişimi
      kısıtlanmamıştır (Faz 3'ten gelen bilinçli sınırlama: subprocess,
      exec() değil).
    * Başarısız sandbox veya dosya yokluğu: figures=[] ile zarif çıkış.
      Rapor yine de yazılır; make-span raporunda "no artifacts" notu düşülür.

DSPy modülü neden Predict (CoT değil)?
    Matplotlib kodu üretimi yapısal bir görev; "neden bu şekli seçtim"
    akıl yürütmesi token harcar ve kaliteyi artırmaz. Predict yeterli.
"""

from __future__ import annotations

import dspy

from iesolver.config import settings
from iesolver.lm import call_with_fast_lm
from iesolver.observability.metrics import instrument
from iesolver.sandbox.runner import run_code
from iesolver.signatures import TornadoChartSignature
from iesolver.state import SolverState

_chart_gen = dspy.Predict(TornadoChartSignature)


@instrument("artifacts")
def artifacts_node(state: SolverState) -> SolverState:
    """Generate a tornado chart PNG from sensitivity analysis results.

    Reads
    -----
    sensitivity_results, explicit_goal (chart title)

    Writes
    ------
    figures  (Annotated[list[Path], operator.add] reducer)
    """
    sensitivity_results = state.get("sensitivity_results") or ""

    # Duyarlılık analizi başarısız olduysa grafik oluşturma
    if not sensitivity_results or sensitivity_results.startswith("[sensitivity_analysis_failed]"):
        return {"figures": []}

    artifact_path = settings.artifacts_dir / "tornado_chart.png"
    problem_title = (state.get("explicit_goal") or "Sensitivity Analysis")[:80]

    # LLM: tornado chart kodu üret
    prediction = call_with_fast_lm(
        _chart_gen,
        sensitivity_results=sensitivity_results,
        artifact_path=str(artifact_path),
        problem_title=problem_title,
    )

    # Sandbox: kodu çalıştır → artifact_path'e PNG kaydet
    run_result = run_code(prediction.chart_code)

    if run_result.success and artifact_path.exists():
        return {"figures": [artifact_path]}

    # Sandbox başarısız veya dosya yazılmadı — boş liste, rapor etkilenmiyor
    return {"figures": []}