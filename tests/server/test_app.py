"""Tests for Lainbow server FastAPI app and main entry point."""

import os
from unittest.mock import MagicMock

import pytest
import uvicorn
from fastapi.testclient import TestClient
from pydantic import SecretStr

import lainbow
from lainbow.core.settings import ServerSettings
from lainbow.server import __main__ as server_main
from lainbow.server.app import create_app


@pytest.fixture(autouse=True)
def clean_lainbow_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure no LAINBOW_* environment variables leak into tests."""
    for key in list(os.environ.keys()):
        if key.startswith("LAINBOW_"):
            monkeypatch.delenv(key, raising=False)


def test_health_endpoint() -> None:
    settings = ServerSettings(database_url=SecretStr("postgresql://localhost/test"))
    app = create_app(settings)
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": lainbow.__version__}


def test_server_main_missing_settings(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        server_main.main()

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "LAINBOW_DATABASE_URL: missing (PostgreSQL connection URL.)" in captured.err


def test_server_main_valid_settings(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("LAINBOW_DATABASE_URL", "postgresql://user:secretpw@localhost/test")

    mock_run = MagicMock()
    mock_upgrade = MagicMock(return_value="0001_initial")
    monkeypatch.setattr(uvicorn, "run", mock_run)
    monkeypatch.setattr(server_main, "upgrade_to_head", mock_upgrade)

    server_main.main()

    assert mock_upgrade.called
    assert mock_upgrade.call_args[0][0] == "postgresql://user:secretpw@localhost/test"

    assert mock_run.called
    _args, kwargs = mock_run.call_args
    assert kwargs.get("host") == "0.0.0.0"
    assert kwargs.get("port") == 8000
    assert kwargs.get("log_level") == "info"

    # Settings table logged without raw secret, and migration revision logged
    captured = capsys.readouterr()
    assert "LAINBOW_DATABASE_URL" in captured.err
    assert "********" in captured.err
    assert "secretpw" not in captured.err
    assert "database schema is up to date (revision 0001_initial)" in captured.err
