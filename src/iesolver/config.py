"""
iesolver.config — runtime settings, Pydantic-backed.

Eski projedeki ``AgentSettings`` bu modülde ``IESolverSettings`` adıyla
yeniden doğar. Genişletmeler:

* ``checkpoint_db_path`` — LangGraph SqliteSaver'ın yazacağı dosya.
* ``artifacts_dir`` — Faz 4 figürleri (matplotlib/plotly PNG'leri) buraya yazılır.
* ``output_dir`` — Faz 5 raporları (PDF/DOCX/HTML) buraya derlenir.
* ``sandbox_workdir`` — Faz 3 subprocess sandbox'ının scratch alanı.
* ``default_output_format`` — UI'dan format gelmezse hangi format üretilsin.

Mimari karar — google_api_key neden default="":
    Faz 1'de hiçbir node DSPy çağırmaz (dummy plumbing). API key zorunlu
    olsaydı, .env yoksa import-time'da ImportError fırlardı ve smoke test
    çalıştırılamazdı. Faz 2'de gerçek LLM çağrıları başlayınca
    ``require_api_key()`` çağrılır ve eksik key burada yakalanır.

Tüm değişkenler .env'den case-insensitive okunur; eski ``GOOGLE_API_KEY``
ortam değişkeni hiç dokunmadan çalışmaya devam eder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class IESolverSettings(BaseSettings):
    """Single source of truth for runtime configuration.

    Reads from a ``.env`` file at project root (override with
    ``IESolverSettings(_env_file=...)``). Field names are case-insensitive
    when matching environment variables.
    """

    # ---- API --------------------------------------------------------------
    google_api_key: str = Field(
        default="",
        description="Google Generative AI API key. Optional in Faz 1, "
                    "required from Faz 2 (real LLM calls) onward.",
    )

    # ---- Model routing ----------------------------------------------------
    fast_model: str = Field(
        default="gemini/gemini-3.1-flash-lite-preview",
        description="Default LM for triage/routing/reporting nodes.",
    )
    reasoning_model: str = Field(
        default="gemini/gemini-3.1-flash-lite-preview",
        description="Heavy LM swapped in for Phase 4B code generation.",
    )

    # ---- Reproducibility (DESIGN_REVIEW §3.7) --------------------------------
    # Q1 dergi standard: tüm LLM çağrıları temperature=0 ile deterministik.
    # lm_seed bazı provider'larda (OpenAI) ek garanti sağlar; Gemini'de
    # sessizce görmezden gelinir — ancak geçirmek zararsız ve ileride model
    # değiştirince otomatik devreye girer.
    temperature: float = Field(
        default=0.0,
        description="LLM sampling temperature. 0.0 = deterministic (greedy). "
                    "Apply to all LM instances for reproducibility.",
    )
    lm_seed: int = Field(
        default=42,
        description="Seed forwarded to LiteLLM/provider for reproducibility. "
                    "Ignored silently if the provider does not support it.",
    )

    # ---- Safety & limits --------------------------------------------------
    max_retries: int = Field(
        default=3,
        description="Max ReAct iterations in Faz 4B code generation loop.",
    )
    timeout_seconds: int = Field(
        default=60,
        description="Sandbox subprocess execution timeout (seconds).",
    )
    budget_limit_usd: float = Field(
        default=5.0,
        description="Daily hard stop on LLM spend to prevent bill shock.",
    )

    # ---- Persistence & artifacts -----------------------------------------
    checkpoint_db_path: Path = Field(
        default=Path(".iesolver/checkpoints.sqlite"),
        description="SQLite file backing LangGraph SqliteSaver.",
    )
    artifacts_dir: Path = Field(
        default=Path(".iesolver/artifacts"),
        description="Generated figures (matplotlib/plotly) live here.",
    )
    output_dir: Path = Field(
        default=Path(".iesolver/outputs"),
        description="Final PDF/DOCX/HTML reports are written here.",
    )
    sandbox_workdir: Path = Field(
        default=Path(".iesolver/sandbox"),
        description="Scratch dir for subprocess code execution (Faz 3).",
    )

    # ---- UI / output ------------------------------------------------------
    default_output_format: Literal["pdf", "docx", "html"] = Field(
        default="html",
        description="Used when caller does not specify a format.",
    )

    # ---- Pydantic config --------------------------------------------------
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Prefix YOK: eski projedeki "GOOGLE_API_KEY" değişkeni
        # olduğu gibi okunsun, geriye uyumluluk korunsun.
    )

    # ---- Helpers ----------------------------------------------------------
    def ensure_directories(self) -> None:
        """Create any persistence directories that don't yet exist.

        ``solve()`` ilk çağrıda çağırır; idempotent.
        """
        self.checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sandbox_workdir.mkdir(parents=True, exist_ok=True)

    def require_api_key(self) -> str:
        """Return the API key or raise a clear error if missing.

        Faz 2+ node'ları DSPy çağrısı yapmadan hemen önce bunu çağırır.
        """
        if not self.google_api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Add it to .env at project root "
                "or export it in your shell before calling solve()."
            )
        return self.google_api_key


# Modül-seviyesi singleton: tüm modüller ``from iesolver.config import settings``
# diyerek aynı örneğe erişir. Test sırasında monkeypatch ile değiştirilebilir.
settings = IESolverSettings()


__all__ = ["IESolverSettings", "settings"]
