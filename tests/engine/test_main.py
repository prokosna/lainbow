"""Tests for Lainbow engine main entry point."""

import os

import pytest

from lainbow.engine import __main__ as engine_main


@pytest.fixture(autouse=True)
def clean_lainbow_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure no LAINBOW_* environment variables leak into tests."""
    for key in list(os.environ.keys()):
        if key.startswith("LAINBOW_"):
            monkeypatch.delenv(key, raising=False)


def test_engine_main_missing_settings(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        engine_main.main()

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "LAINBOW_SERVER_URL: missing" in captured.err
    assert "LAINBOW_TOKEN: missing" in captured.err


def test_engine_main_valid_settings(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("LAINBOW_SERVER_URL", "http://127.0.0.1:8000")
    monkeypatch.setenv("LAINBOW_TOKEN", "super_secret_engine_token_123")

    engine_main.main()

    # Settings table logged without raw token
    captured = capsys.readouterr()
    assert "LAINBOW_SERVER_URL" in captured.err
    assert "LAINBOW_TOKEN" in captured.err
    assert "********" in captured.err
    assert "super_secret_engine_token_123" not in captured.err

    # Work loop notice logged
    assert "engine work loop is not implemented yet" in captured.err
