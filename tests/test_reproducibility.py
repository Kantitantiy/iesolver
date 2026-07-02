"""
Tekrarlanabilirlik testleri (DESIGN_REVIEW §3.7).

LLM ÇAĞRISI YAPMAZ — settings defaults ve LM init kwargs'ı doğrular.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from iesolver.config import IESolverSettings


# =============================================================================
# Settings defaults
# =============================================================================
def test_temperature_default_is_zero():
    s = IESolverSettings(google_api_key="x")
    assert s.temperature == 0.0


def test_lm_seed_default():
    s = IESolverSettings(google_api_key="x")
    assert isinstance(s.lm_seed, int)
    assert s.lm_seed == 42


def test_temperature_env_override(monkeypatch):
    monkeypatch.setenv("TEMPERATURE", "0.2")
    s = IESolverSettings(google_api_key="x")
    assert s.temperature == 0.2


# =============================================================================
# LM factory — temperature ve seed DSPy'a iletiliyor mu?
# =============================================================================
def test_get_fast_lm_passes_temperature_and_seed(monkeypatch):
    """get_fast_lm() dspy.LM'i doğru kwargs ile örneklemeli."""
    from iesolver import lm as lm_module

    # Singleton'ı sıfırla
    lm_module._fast_lm = None

    captured: dict = {}

    def fake_lm(model, **kwargs):
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(lm_module, "settings", IESolverSettings(google_api_key="test-key"))
    with patch("iesolver.lm.dspy.LM", side_effect=fake_lm):
        lm_module.get_fast_lm()

    assert captured.get("temperature") == 0.0
    assert captured.get("seed") == 42

    lm_module._fast_lm = None   # teardown


def test_get_reasoning_lm_passes_temperature_and_seed(monkeypatch):
    from iesolver import lm as lm_module

    lm_module._reasoning_lm = None

    captured: dict = {}

    def fake_lm(model, **kwargs):
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(lm_module, "settings", IESolverSettings(google_api_key="test-key"))
    with patch("iesolver.lm.dspy.LM", side_effect=fake_lm):
        lm_module.get_reasoning_lm()

    assert captured.get("temperature") == 0.0
    assert captured.get("seed") == 42

    lm_module._reasoning_lm = None   # teardown


# =============================================================================
# uv.lock var mı?
# =============================================================================
def test_uv_lock_exists():
    """Projenin uv.lock'u kayıt altında olmalı — bağımlılık pinleri için."""
    from pathlib import Path
    lock = Path(__file__).parent.parent / "uv.lock"
    assert lock.exists(), "uv.lock bulunamadı; 'uv lock' çalıştırın."