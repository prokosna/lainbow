"""Tests for typed settings, describe, formatting, and validation."""

import os
from pathlib import Path

import pytest

from lainbow.core.models import ModelId
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
        "LAINBOW_MODELS",
        "LAINBOW_MUQ_MAX_SEC",
        "LAINBOW_MUQ_DROP_TAIL",
        "LAINBOW_MUQ_MULAN_MAX_SEC",
        "LAINBOW_MUQ_MULAN_DROP_TAIL",
        "LAINBOW_CLAP_MAX_SEC",
        "LAINBOW_CLAP_DROP_TAIL",
        "LAINBOW_MERT_MAX_SEC",
        "LAINBOW_MERT_DROP_TAIL",
        "LAINBOW_ACOUSTIC_MAX_SEC",
        "LAINBOW_ACOUSTIC_DROP_TAIL",
        "LAINBOW_LOG_LEVEL",
    ]
    assert env_vars == expected_order

    # List rendering in describe
    models_row = next(r for r in rows if r.env_var == "LAINBOW_MODELS")
    assert models_row.value == "muq,muq_mulan"

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


def test_models_comma_separated_env_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAINBOW_DATABASE_URL", "postgresql://localhost/test")
    monkeypatch.setenv("LAINBOW_MODELS", "muq,clap")

    settings = load_settings(ServerSettings)
    assert settings.models == [ModelId.MUQ, ModelId.CLAP]


def test_unknown_model_raises_settings_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAINBOW_DATABASE_URL", "postgresql://localhost/test")
    monkeypatch.setenv("LAINBOW_MODELS", "muq,unknown_model")

    with pytest.raises(SettingsError) as exc_info:
        load_settings(ServerSettings)

    message = str(exc_info.value)
    assert "LAINBOW_MODELS" in message


def test_method_for_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAINBOW_DATABASE_URL", "postgresql://localhost/test")
    settings = load_settings(ServerSettings)

    # Defaults: MuQ (180, False) and MuQ-MuLan (600, False)
    muq_method = settings.method_for(ModelId.MUQ)
    assert muq_method.max_sec == 180
    assert not muq_method.drop_tail

    mulan_method = settings.method_for(ModelId.MUQ_MULAN)
    assert mulan_method.max_sec == 600
    assert not mulan_method.drop_tail

    clap_method = settings.method_for(ModelId.CLAP)
    assert clap_method.max_sec == 600
    assert not clap_method.drop_tail

    mert_method = settings.method_for(ModelId.MERT)
    assert mert_method.max_sec == 600
    assert not mert_method.drop_tail

    acoustic_method = settings.method_for(ModelId.ACOUSTIC)
    assert acoustic_method.max_sec == 600
    assert not acoustic_method.drop_tail


def test_muq_max_sec_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAINBOW_DATABASE_URL", "postgresql://localhost/test")
    monkeypatch.setenv("LAINBOW_MUQ_MAX_SEC", "0")

    settings = load_settings(ServerSettings)
    method = settings.method_for(ModelId.MUQ)
    assert method.max_sec == 0
    assert not method.drop_tail


def test_muq_mulan_drop_tail_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAINBOW_DATABASE_URL", "postgresql://localhost/test")
    monkeypatch.setenv("LAINBOW_MUQ_MULAN_DROP_TAIL", "true")

    settings = load_settings(ServerSettings)
    method = settings.method_for(ModelId.MUQ_MULAN)
    assert method.drop_tail


def test_model_max_sec_negative_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAINBOW_DATABASE_URL", "postgresql://localhost/test")
    monkeypatch.setenv("LAINBOW_MUQ_MAX_SEC", "-1")

    with pytest.raises(SettingsError) as exc_info:
        load_settings(ServerSettings)

    message = str(exc_info.value)
    assert len(message.strip().splitlines()) == 1
    assert "LAINBOW_MUQ_MAX_SEC" in message
    assert "Input should be greater than or equal to 0" in message
    assert "MuQ: seconds from the start of a track to analyze" in message


def test_unknown_model_method_env_var_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAINBOW_DATABASE_URL", "postgresql://localhost/test")
    monkeypatch.setenv("LAINBOW_UNKNOWNMODEL_MAX_SEC", "100")

    settings = load_settings(ServerSettings)
    assert "unknownmodel" not in settings.methods


def test_describe_method_is_default_uses_explicit_vars_not_environ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LAINBOW_DATABASE_URL", "postgresql://localhost/test")
    monkeypatch.setenv("LAINBOW_MUQ_MAX_SEC", "120")

    settings = load_settings(ServerSettings)
    # Remove the env var to prove describe uses explicit_method_vars, not os.environ
    monkeypatch.delenv("LAINBOW_MUQ_MAX_SEC")

    rows_by_name = {r.env_var: r for r in describe(settings)}
    assert not rows_by_name["LAINBOW_MUQ_MAX_SEC"].is_default
    assert rows_by_name["LAINBOW_MUQ_MAX_SEC"].value == "120"
    assert rows_by_name["LAINBOW_MUQ_DROP_TAIL"].is_default
