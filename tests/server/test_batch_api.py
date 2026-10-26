"""Tests for the batch HTTP API."""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from lainbow.core.settings import ServerSettings
from lainbow.db.migrate import upgrade_to_head
from lainbow.db.session import make_session_factory
from lainbow.server.app import create_app

pytestmark = pytest.mark.postgres


@pytest.fixture
def engine(clean_db: str) -> Iterator[Engine]:
    upgrade_to_head(clean_db)
    eng = create_engine(clean_db)
    yield eng
    eng.dispose()


@pytest.fixture
def factory(engine: Engine) -> sessionmaker[Session]:
    return make_session_factory(engine)


def test_scan_run_and_poll(tmp_path: Path, factory: sessionmaker[Session]) -> None:
    settings = ServerSettings(
        database_url=SecretStr("postgresql://unused/x"), music_dir=tmp_path, scan_interval_min=0
    )
    with TestClient(create_app(settings, session_factory=factory)) as client:
        resp = client.post("/api/v1/batch/scan/run")
        assert resp.status_code == 200
        body = resp.json()
        assert body["kind"] == "scan"
        for _ in range(100):
            got = client.get(f"/api/v1/batch/tasks/{body['id']}").json()
            if got["state"] in ("succeeded", "failed"):
                break
            time.sleep(0.05)
        assert got["state"] == "succeeded", got
        assert got["detail"]["added"] == 0

        vac = client.post("/api/v1/batch/vacuum/run").json()
        assert vac["kind"] == "vacuum"

        assert client.get(f"/api/v1/batch/tasks/{uuid.uuid4()}").status_code == 404
