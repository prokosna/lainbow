"""Tests for typed settings, describe, formatting, and validation."""

import os
from pathlib import Path

import pytest

from lainbow.core.settings import (
    EngineSettings,
    ServerSettings,
    SettingRow,
    SettingsError,
    describe,
    format_settings,
    load_settings,
    start,
)


@pytest.fixture(autouse=True)
def clean_lainbow_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure no LAINBOW_* environment variables leak into tests."""
    for key in list(os.environ.keys()):
        if key.startswith("LAINBOW_"):
            monkeypatch.delenv(key, raising=False)


def test_describe_field_order_and_masking(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "LAINBOW_DATABASE_URL", "postgresql://user:secretpass@127.0.0.1:5432/lainbow"
    )
    settings = load_settings(ServerSettings)
    rows = describe(settings)

    # Field order
    env_vars = [r.env_var for r in rows]
    expected_order = [
        "LAINBOW_DATABASE_URL",
        "LAINBOW_MUSIC_DIR",
        "LAINBOW_HOST",
        "LAINBOW_PORT",
        "LAINBOW_LOG_LEVEL",
    ]
    assert env_vars == expected_order

    # Masking
    db_row = next(r for r in rows if r.env_var == "LAINBOW_DATABASE_URL")
    assert db_row.value == "********"
    assert "secretpass" not in db_row.value

    # Engine settings order and masking
    monkeypatch.setenv("LAINBOW_SERVER_URL", "http://127.0.0.1:8000")
    monkeypatch.setenv("LAINBOW_TOKEN", "supersecrettoken")
    engine_settings = load_settings(EngineSettings)
    engine_rows = describe(engine_settings)
    engine_env_vars = [r.env_var for r in engine_rows]
    assert engine_env_vars == [
        "LAINBOW_SERVER_URL",
        "LAINBOW_TOKEN",
        "LAINBOW_ACCEL",
        "LAINBOW_MODELS_DIR",
        "LAINBOW_LOG_LEVEL",
    ]
    token_row = next(r for r in engine_rows if r.env_var == "LAINBOW_TOKEN")
    assert token_row.value == "********"
    assert "supersecrettoken" not in token_row.value


def test_describe_is_default_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAINBOW_DATABASE_URL", "postgresql://localhost/test")
    # port is not set, music_dir is not set
    settings = load_settings(ServerSettings)
    rows_by_name = {r.env_var: r for r in describe(settings)}

    assert not rows_by_name["LAINBOW_DATABASE_URL"].is_default
    assert rows_by_name["LAINBOW_PORT"].is_default
    assert rows_by_name["LAINBOW_MUSIC_DIR"].is_default
    assert rows_by_name["LAINBOW_HOST"].is_default
    assert rows_by_name["LAINBOW_LOG_LEVEL"].is_default

    # Now explicitly set LAINBOW_PORT
    monkeypatch.setenv("LAINBOW_PORT", "9000")
    settings2 = load_settings(ServerSettings)
    rows_by_name2 = {r.env_var: r for r in describe(settings2)}
    assert not rows_by_name2["LAINBOW_PORT"].is_default
    assert rows_by_name2["LAINBOW_PORT"].value == "9000"


def test_load_settings_missing_database_url() -> None:
    with pytest.raises(SettingsError) as exc_info:
        load_settings(ServerSettings)

    message = str(exc_info.value)
    expected_line = "LAINBOW_DATABASE_URL: missing (PostgreSQL connection URL.)"
    assert expected_line in message.splitlines()


def test_load_settings_invalid_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAINBOW_DATABASE_URL", "postgresql://localhost/test")
    monkeypatch.setenv("LAINBOW_PORT", "0")

    with pytest.raises(SettingsError) as exc_info:
        load_settings(ServerSettings)

    message = str(exc_info.value)
    assert "LAINBOW_PORT" in message
    assert "Port the HTTP server listens on." in message


def test_load_settings_engine_missing_fields() -> None:
    with pytest.raises(SettingsError) as exc_info:
        load_settings(EngineSettings)

    lines = str(exc_info.value).splitlines()
    assert (
        "LAINBOW_SERVER_URL: missing (URL of the Lainbow server, e.g. http://192.168.1.10:8000.)"
        in lines
    )
    assert (
        "LAINBOW_TOKEN: missing (Engine token printed by `lainbow add-engine` on the server.)"
        in lines
    )


def test_load_settings_valid_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAINBOW_DATABASE_URL", "postgresql://localhost/test")
    monkeypatch.setenv("LAINBOW_PORT", "8080")
    monkeypatch.setenv("LAINBOW_HOST", "127.0.0.1")
    monkeypatch.setenv("LAINBOW_LOG_LEVEL", "debug")
    monkeypatch.setenv("LAINBOW_MUSIC_DIR", "/custom/music")

    settings = load_settings(ServerSettings)
    assert settings.database_url.get_secret_value() == "postgresql://localhost/test"
    assert settings.port == 8080
    assert settings.host == "127.0.0.1"
    assert settings.log_level == "debug"
    assert settings.music_dir == Path("/custom/music")


def test_load_settings_valid_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAINBOW_SERVER_URL", "http://192.168.1.50:8000")
    monkeypatch.setenv("LAINBOW_TOKEN", "token123")
    monkeypatch.setenv("LAINBOW_ACCEL", "cuda")
    monkeypatch.setenv("LAINBOW_MODELS_DIR", "/custom/models")
    monkeypatch.setenv("LAINBOW_LOG_LEVEL", "warning")

    settings = load_settings(EngineSettings)
    assert str(settings.server_url) == "http://192.168.1.50:8000/"
    assert settings.token.get_secret_value() == "token123"
    assert settings.accel == "cuda"
    assert settings.models_dir == Path("/custom/models")
    assert settings.log_level == "warning"


def test_format_settings() -> None:
    rows = [
        SettingRow(
            env_var="LAINBOW_DATABASE_URL",
            value="********",
            is_default=False,
            description="PostgreSQL connection URL.",
        ),
        SettingRow(
            env_var="LAINBOW_PORT",
            value="8000",
            is_default=True,
            description="Port the HTTP server listens on.",
        ),
    ]
    table = format_settings(rows)
    lines = table.splitlines()
    assert len(lines) == 2

    # Verify column presence
    assert "LAINBOW_DATABASE_URL" in lines[0]
    assert "********" in lines[0]
    assert "set" in lines[0]
    assert "PostgreSQL connection URL." in lines[0]

    assert "LAINBOW_PORT" in lines[1]
    assert "8000" in lines[1]
    assert "default" in lines[1]
    assert "Port the HTTP server listens on." in lines[1]

    # Empty rows returns empty string
    assert format_settings([]) == ""


def test_log_level_is_last_row(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAINBOW_DATABASE_URL", "postgresql://localhost/test")
    server_rows = describe(load_settings(ServerSettings))
    assert server_rows[-1].env_var == "LAINBOW_LOG_LEVEL"

    monkeypatch.setenv("LAINBOW_SERVER_URL", "http://127.0.0.1:8000")
    monkeypatch.setenv("LAINBOW_TOKEN", "testtoken")
    engine_rows = describe(load_settings(EngineSettings))
    assert engine_rows[-1].env_var == "LAINBOW_LOG_LEVEL"


def test_start_missing_settings(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        start(ServerSettings)
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "LAINBOW_DATABASE_URL: missing" in captured.err


def test_start_valid_settings(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LAINBOW_DATABASE_URL", "postgresql://localhost/test")
    monkeypatch.setenv("LAINBOW_LOG_LEVEL", "debug")
    settings = start(ServerSettings)
    assert isinstance(settings, ServerSettings)
    assert settings.log_level == "debug"
    captured = capsys.readouterr()
    assert "settings:\n" in captured.err
    assert "LAINBOW_DATABASE_URL" in captured.err
