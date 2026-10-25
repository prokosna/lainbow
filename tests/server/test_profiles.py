"""Tests for profile activation."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator

import pytest
from pydantic import SecretStr
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from lainbow.core.settings import ServerSettings
from lainbow.db.migrate import upgrade_to_head
from lainbow.db.schema import ActiveProfile
from lainbow.db.session import make_session_factory
from lainbow.db.work import Task
from lainbow.server.profiles import activate_profiles

pytestmark = pytest.mark.postgres


@pytest.fixture
def db(clean_db: str) -> Iterator[Engine]:
    upgrade_to_head(clean_db)
    eng = create_engine(clean_db)
    yield eng
    eng.dispose()


@pytest.fixture
def factory(db: Engine) -> sessionmaker[Session]:
    return make_session_factory(db)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith("LAINBOW_") and key != "LAINBOW_TEST_DATABASE_URL":
            monkeypatch.delenv(key)


def make_settings() -> ServerSettings:
    return ServerSettings(
        database_url=SecretStr("postgresql://unused/x"), models=["muq", "acoustic"]
    )


def test_activation_and_change(
    factory: sessionmaker[Session],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caplog.set_level(logging.WARNING)
    tasks = activate_profiles(factory, make_settings())
    assert tasks == (
        Task("muq", "r1-max180-keeptail"),
        Task("acoustic", "r1-max600-keeptail"),
    )
    with factory() as s:
        assert len(s.scalars(select(ActiveProfile)).all()) == 2
    assert not caplog.records

    assert activate_profiles(factory, make_settings()) == tasks
    assert not caplog.records

    monkeypatch.setenv("LAINBOW_MUQ_MAX_SEC", "300")
    new = activate_profiles(factory, make_settings())
    assert new[0] == Task("muq", "r1-max300-keeptail")
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "muq: analysis method changed (r1-max180-keeptail -> r1-max300-keeptail)" in warnings[0]
    with factory() as s:
        assert (
            s.scalar(select(ActiveProfile.profile).where(ActiveProfile.model == "muq"))
            == "r1-max300-keeptail"
        )
