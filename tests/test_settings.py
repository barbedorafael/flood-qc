from __future__ import annotations

from common.settings import load_settings


def test_load_settings_has_expected_sections() -> None:
    settings = load_settings()
    assert "paths" in settings
    assert settings["paths"]["history_db"] == "data/history.sqlite"
    assert settings["run"]["mode"] == "operational"