"""
iesolver.sandbox.runner — güvenli Python kod çalıştırma motoru.

Faz 3'ün deterministik çekirdeği. LLM'in ürettiği kodu izole bir
subprocess içinde çalıştırır; timeout ile askıda kalma ve sonsuz döngü
riskini önler. Şimdilik subprocess tabanlı (PLAN.md §6: "Basit;
Docker'a sonra geçilebilir").

Tasarım kararları:
    * ``subprocess.run`` + ``timeout``: En basit sandbox. Üretilen kod
      ana process'i etkileyemez; sadece stdout/stderr yakalar.
    * ``tempfile.NamedTemporaryFile``: Kod dosyası geçici dizine yazılır,
      çalıştırılır, silinir — disk kirliliği yok.
    * Ayrı Python executable: ``sys.executable`` ile aynı venv'deki
      Python kullanılır; üretilen kodun ``import pulp`` gibi ifadeleri
      proje bağımlılıklarına erişebilir.
    * ``RunResult`` dataclass: Başarı/hata durumu tek bir nesnede taşınır;
      node'lar ``if result.success:`` ile basitçe dallara ayrılabilir.

Makale notu — "Computational Sandbox Layer":
    LLM'in ürettiği kod doğrudan ``exec()`` ile çalıştırılmaz (güvenlik +
    izolasyon). Subprocess boundary, ana agent state'ini üretilen kodun
    yan etkilerinden (dosya silme, network çağrısı vs.) korur.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from iesolver.config import settings


@dataclass(slots=True)
class RunResult:
    """Outcome of a single sandbox execution.

    Attributes
    ----------
    success :
        True iff the process exited with code 0 within the timeout.
    stdout :
        Captured standard output (the "result" LLM reads).
    stderr :
        Captured standard error (tracebacks, warnings).
    exit_code :
        Raw process exit code; 0 = success, non-zero = error, None = timeout.
    timed_out :
        True when the process was killed due to timeout.
    """

    success: bool
    stdout: str
    stderr: str
    exit_code: int | None = None
    timed_out: bool = False
    error_summary: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        # Kısa hata özeti: node'ların log/retry kararı için
        if self.timed_out:
            self.error_summary = (
                f"Execution timed out after {settings.timeout_seconds}s."
            )
        elif not self.success and self.stderr:
            # Traceback'in son satırı genellikle en bilgilendirici kısım
            lines = [l for l in self.stderr.strip().splitlines() if l.strip()]
            self.error_summary = lines[-1] if lines else self.stderr[:200]


def run_code(code: str, timeout: int | None = None) -> RunResult:
    """Execute ``code`` in an isolated subprocess and return the result.

    Parameters
    ----------
    code :
        Valid Python source code as a string.
    timeout :
        Maximum execution time in seconds. Defaults to
        ``settings.timeout_seconds``.

    Returns
    -------
    RunResult
        Structured outcome; caller checks ``result.success`` to branch.
    """
    effective_timeout = timeout if timeout is not None else settings.timeout_seconds

    # Kodu geçici dosyaya yaz; NamedTemporaryFile Windows'ta delete=False
    # gerektirir çünkü açık dosya tekrar açılamaz.
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            prefix="iesolver_sandbox_",
            dir=settings.sandbox_workdir,
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(code)
            tmp_path = Path(tmp.name)

        proc = subprocess.run(
            [sys.executable, str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=effective_timeout,
        )

        return RunResult(
            success=proc.returncode == 0,
            stdout=proc.stdout,
            stderr=proc.stderr,
            exit_code=proc.returncode,
        )

    except subprocess.TimeoutExpired:
        return RunResult(
            success=False,
            stdout="",
            stderr=f"Process killed after {effective_timeout}s timeout.",
            timed_out=True,
        )

    except Exception as exc:  # noqa: BLE001
        return RunResult(
            success=False,
            stdout="",
            stderr=str(exc),
            exit_code=-1,
        )

    finally:
        # Geçici dosyayı her koşulda temizle
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass  # Temizlik başarısız olursa sessizce geç


__all__ = ["RunResult", "run_code"]
