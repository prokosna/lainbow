"""Tests for the admin command."""

from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from lainbow.server import admin, engines


@pytest.fixture
def env(clean_db: str, monkeypatch: pytest.MonkeyPatch) -> str:
    for key in list(os.environ):
        if key.startswith("LAINBOW_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("LAINBOW_DATABASE_URL", clean_db)
    return clean_db


@pytest.mark.postgres
def test_add_engine_prints_token(env: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert admin.main(["add-engine", "gpu1"]) == 0
    token = capsys.readouterr().out.strip()
    assert token.startswith("lbe_")
    eng = create_engine(env)
    with Session(eng) as s:
        assert engines.authenticate(s, token) is not None
    eng.dispose()

    assert admin.main(["list-engines"]) == 0
    assert "gpu1" in capsys.readouterr().out


@pytest.mark.postgres
def test_duplicate_name_fails(env: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert admin.main(["add-engine", "gpu1"]) == 0
    capsys.readouterr()
    assert admin.main(["add-engine", "gpu1"]) != 0
    assert "already exists" in capsys.readouterr().err


@pytest.mark.postgres
def test_remove_engine(env: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert admin.main(["add-engine", "gpu1"]) == 0
    assert admin.main(["remove-engine", "gpu1"]) == 0
    assert admin.main(["remove-engine", "gpu1"]) != 0


def test_missing_settings_exits_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("LAINBOW_DATABASE_URL", raising=False)
    assert admin.main(["list-engines"]) == 2
    assert capsys.readouterr().err
