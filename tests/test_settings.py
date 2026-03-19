from __future__ import annotations

import pytest

from common import settings as settings_module


DEFAULT_CONFIG = """\
run:
  reference_time: \"2026-03-11T00:00:00\"

ingest:
  request_days: 7
  timeout_seconds: 15

summaries:
  forecast_days: [1, 3, 10, 30]
  accum_hours: [24, 72, 240, 720]
  selected_mini_ids: [\"7601\"]

mgb:
  output_days_before: 30
  output_days_after: 15
"""


CUSTOM_CONFIG = """\
ingest:
  timeout_seconds: 30

summaries:
  selected_mini_ids: [\"7601\", \"7612\"]

mgb:
  output_days_after: 20
"""


EMPTY_CUSTOM = """\
# local overrides
"""


def write_config(tmp_path, *, default_text: str | None = DEFAULT_CONFIG, custom_text: str | None = EMPTY_CUSTOM) -> None:
    if default_text is not None:
        (tmp_path / "default.yaml").write_text(default_text, encoding="utf-8")
    if custom_text is not None:
        (tmp_path / "custom.yaml").write_text(custom_text, encoding="utf-8")


def test_load_settings_merges_default_and_custom(tmp_path, monkeypatch) -> None:
    write_config(tmp_path, custom_text=CUSTOM_CONFIG)
    monkeypatch.setattr(settings_module, "CONFIG_DIR", tmp_path)

    settings = settings_module.load_settings()

    assert settings["run"]["reference_time"] == "2026-03-11T00:00:00"
    assert settings["ingest"]["request_days"] == 7
    assert settings["ingest"]["timeout_seconds"] == 30
    assert settings["summaries"]["forecast_days"] == [1, 3, 10, 30]
    assert settings["summaries"]["accum_hours"] == [24, 72, 240, 720]
    assert settings["summaries"]["selected_mini_ids"] == ["7601", "7612"]
    assert settings["mgb"]["output_days_before"] == 30
    assert settings["mgb"]["output_days_after"] == 20


def test_load_settings_accepts_now_and_yesterday(tmp_path, monkeypatch) -> None:
    write_config(
        tmp_path,
        default_text="""\
run:
  reference_time: \"now\"

ingest:
  request_days: 7
  timeout_seconds: 15

summaries:
  forecast_days: [1]
  accum_hours: [24]
  selected_mini_ids: []

mgb:
  output_days_before: 30
  output_days_after: 15
""",
    )
    monkeypatch.setattr(settings_module, "CONFIG_DIR", tmp_path)

    settings = settings_module.load_settings()

    assert settings["run"]["reference_time"] == "now"

    write_config(
        tmp_path,
        default_text="""\
run:
  reference_time: "yesterday"

ingest:
  request_days: 7
  timeout_seconds: 15

summaries:
  forecast_days: [1]
  accum_hours: [24]
  selected_mini_ids: []

mgb:
  output_days_before: 30
  output_days_after: 15
""",
    )

    settings = settings_module.load_settings()

    assert settings["run"]["reference_time"] == "yesterday"


@pytest.mark.parametrize(
    ("default_text", "custom_text", "expected_error"),
    [
        (None, EMPTY_CUSTOM, "Arquivo de config nao encontrado"),
        (DEFAULT_CONFIG, None, "Arquivo de config nao encontrado"),
        (
            """\
run:
  reference_time: \"2026-03-11T00:00:00\"

ingest:
  request_days: 7

summaries:
  forecast_days: [1]
  accum_hours: [24]
  selected_mini_ids: []

mgb:
  output_days_before: 30
  output_days_after: 15
""",
            EMPTY_CUSTOM,
            "chaves obrigatorias",
        ),
        (
            """\
run:
  reference_time: \"2026-03-11T00:00:00\"
  mode: \"operational\"

ingest:
  request_days: 7
  timeout_seconds: 15

summaries:
  forecast_days: [1]
  accum_hours: [24]
  selected_mini_ids: []

mgb:
  output_days_before: 30
  output_days_after: 15
""",
            EMPTY_CUSTOM,
            "chaves nao suportadas",
        ),
        (
            """\
run:
  reference_time: \"\"

ingest:
  request_days: 7
  timeout_seconds: 15

summaries:
  forecast_days: [1]
  accum_hours: [24]
  selected_mini_ids: []

mgb:
  output_days_before: 30
  output_days_after: 15
""",
            EMPTY_CUSTOM,
            "nao pode ser vazio",
        ),
        (
            """\
run:
  reference_time: \"2026-03-11T00:00:00\"

ingest:
  request_days: 7
  timeout_seconds: 15

summaries:
  forecast_days: [1]
  accum_hours: [24]
  selected_mini_ids: []

mgb:
  output_days_before: 0
  output_days_after: 15
""",
            EMPTY_CUSTOM,
            "inteiro >= 1",
        ),
        (
            """\
run:
  reference_time: \"2026-03-11T00:00:00\"

ingest:
  request_days: 7
  timeout_seconds: 15

summaries:
  forecast_days: [1]
  accum_hours: [24]
  selected_mini_ids: []

mgb:
  output_days_before: 30
  output_days_after: 15
  source: \"runner\"
""",
            EMPTY_CUSTOM,
            "chaves nao suportadas",
        ),
        ("- item", EMPTY_CUSTOM, "esperado um objeto YAML"),
    ],
)
def test_load_settings_rejects_invalid_configs(
    tmp_path,
    monkeypatch,
    default_text: str | None,
    custom_text: str | None,
    expected_error: str,
) -> None:
    write_config(tmp_path, default_text=default_text, custom_text=custom_text)
    monkeypatch.setattr(settings_module, "CONFIG_DIR", tmp_path)

    with pytest.raises((FileNotFoundError, ValueError), match=expected_error):
        settings_module.load_settings()
