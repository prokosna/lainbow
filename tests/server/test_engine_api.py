"""End-to-end tests of the engine protocol."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import Engine, create_engine, func, select, text
from sqlalchemy.orm import Session, sessionmaker

import lainbow
from lainbow.core.settings import ServerSettings
from lainbow.db.migrate import upgrade_to_head
from lainbow.db.schema import Analysis, RegisteredEngine
from lainbow.db.session import make_session_factory
from lainbow.server import engines
from lainbow.server.app import create_app
from lainbow.server.library import Tags, scan

pytestmark = pytest.mark.postgres

MUQ = "r1-max180-keeptail"
MERT_PROFILE = "r1-max600-keeptail"


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


def fake_tags(path: Path) -> Tags:
    return Tags(title=path.stem, duration_sec=10.0)


@pytest.fixture
def client(tmp_path: Path, factory: sessionmaker[Session]) -> Iterator[TestClient]:
    (tmp_path / "a.flac").write_bytes(b"AAAA-audio-bytes")
    (tmp_path / "b.flac").write_bytes(b"BBBB-audio-bytes")
    scan(factory, tmp_path, read_tags=fake_tags)
    settings = ServerSettings(
        database_url=SecretStr("postgresql://unused/x"),
        music_dir=tmp_path,
        scan_interval_min=0,
        models=["muq", "mert"],
    )
    with TestClient(create_app(settings, session_factory=factory)) as c:
        yield c


@pytest.fixture
def auth(factory: sessionmaker[Session], client: TestClient) -> dict[str, str]:
    with factory() as s:
        token = engines.create_engine(s, "gpu1")
        s.commit()
    return {
        "Authorization": f"Bearer {token}",
        "X-Lainbow-Version": lainbow.__version__,
    }


def test_version_handshake(client: TestClient, factory: sessionmaker[Session]) -> None:
    with factory() as s:
        token = engines.create_engine(s, "test-version-engine")
        s.commit()

    # Matching header -> 200, and the version is stored
    resp = client.get(
        "/engine/v1/plan",
        headers={
            "Authorization": f"Bearer {token}",
            "X-Lainbow-Version": lainbow.__version__,
        },
    )
    assert resp.status_code == 200
    with factory() as s:
        eng = s.scalar(
            select(RegisteredEngine).where(RegisteredEngine.name == "test-version-engine")
        )
        assert eng is not None
        assert eng.version == lainbow.__version__
        initial_seen_at = eng.last_seen_at
        assert initial_seen_at is not None

    # Mismatched header -> 409 with exact message, and last_seen_at touched
    expected_msg = (
        f"server is {lainbow.__version__} but this engine is 0.0.1; "
        f"on the engine machine run: git fetch --tags && git checkout v{lainbow.__version__} "
        f"&& ./lainbow join <server url> <token> --build"
    )
    resp = client.get(
        "/engine/v1/plan",
        headers={
            "Authorization": f"Bearer {token}",
            "X-Lainbow-Version": "0.0.1",
        },
    )
    assert resp.status_code == 409
    assert resp.json() == {"detail": expected_msg}
    with factory() as s:
        eng = s.scalar(
            select(RegisteredEngine).where(RegisteredEngine.name == "test-version-engine")
        )
        assert eng is not None
        assert eng.version == "0.0.1"
        assert eng.last_seen_at is not None
        assert eng.last_seen_at >= initial_seen_at

    # Missing header -> 409 mentioning unknown, and version stored as None
    expected_unknown_msg = (
        f"server is {lainbow.__version__} but this engine is unknown; "
        f"on the engine machine run: git fetch --tags && git checkout v{lainbow.__version__} "
        f"&& ./lainbow join <server url> <token> --build"
    )
    resp = client.get(
        "/engine/v1/plan",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 409
    assert resp.json() == {"detail": expected_unknown_msg}
    with factory() as s:
        eng = s.scalar(
            select(RegisteredEngine).where(RegisteredEngine.name == "test-version-engine")
        )
        assert eng is not None
        assert eng.version is None

    # Bad token with mismatched header -> 401, not 409
    resp = client.get(
        "/engine/v1/plan",
        headers={
            "Authorization": "Bearer nope",
            "X-Lainbow-Version": "0.0.1",
        },
    )
    assert resp.status_code == 401
    assert resp.headers["WWW-Authenticate"] == "Bearer"


def test_requires_token(client: TestClient) -> None:
    for resp in (
        client.get("/engine/v1/plan"),
        client.get("/engine/v1/plan", headers={"Authorization": "Bearer nope"}),
        client.post("/engine/v1/claim", json={"models": ["muq"], "limit": 1}),
    ):
        assert resp.status_code == 401
        assert resp.headers["WWW-Authenticate"] == "Bearer"


def test_plan(client: TestClient, auth: dict[str, str]) -> None:
    body = client.get("/engine/v1/plan", headers=auth).json()
    assert [t["model"] for t in body["tasks"]] == ["muq", "mert"]
    muq = body["tasks"][0]
    assert muq["profile"] == MUQ
    assert muq["method"] == {"max_sec": 180, "drop_tail": False}
    assert (muq["sample_rate"], muq["clip_sec"], muq["dimension"]) == (24000, 180.0, 1024)


def test_full_loop(
    client: TestClient, auth: dict[str, str], factory: sessionmaker[Session]
) -> None:
    resp = client.post(
        "/engine/v1/claim", json={"models": ["muq", "mert"], "limit": 10}, headers=auth
    )
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert [i["path"] for i in items] == ["a.flac", "b.flac"]
    for i in items:
        assert {(t["model"], t["profile"]) for t in i["tasks"]} == {
            ("muq", MUQ),
            ("mert", MERT_PROFILE),
        }

    audio = client.get(f"/engine/v1/tracks/{items[0]['track_id']}/audio", headers=auth)
    assert audio.status_code == 200
    assert audio.content == b"AAAA-audio-bytes"
    ranged = client.get(
        f"/engine/v1/tracks/{items[0]['track_id']}/audio", headers={**auth, "Range": "bytes=0-3"}
    )
    assert ranged.status_code == 206
    assert ranged.content == b"AAAA"

    results = [
        {"track_id": i["track_id"], "model": "muq", "profile": MUQ, "embedding": [0.1] * 1024}
        for i in items
    ] + [
        {
            "track_id": i["track_id"],
            "model": "mert",
            "profile": MERT_PROFILE,
            "embedding": [0.2] * 1024,
            "details": {"bpm": 120},
        }
        for i in items
    ]
    resp = client.post(
        "/engine/v1/results", json={"results": results, "failures": []}, headers=auth
    )
    assert resp.json() == {"accepted": 4, "rejected": 0, "failed": 0}

    again = client.post(
        "/engine/v1/claim", json={"models": ["muq", "mert"], "limit": 10}, headers=auth
    )
    assert again.json() == {"items": []}

    with factory() as s:
        assert (
            s.scalar(select(func.count()).select_from(Analysis).where(Analysis.state == "done"))
            == 4
        )


def test_inactive_profile_rejected(client: TestClient, auth: dict[str, str]) -> None:
    items = client.post(
        "/engine/v1/claim", json={"models": ["muq"], "limit": 1}, headers=auth
    ).json()["items"]
    tid = items[0]["track_id"]
    body = {
        "results": [
            {
                "track_id": tid,
                "model": "muq",
                "profile": "r1-max1-keeptail",
                "embedding": [0.1] * 1024,
            }
        ],
        "failures": [{"track_id": tid, "model": "muq", "profile": MUQ, "error": "boom"}],
    }
    resp = client.post("/engine/v1/results", json=body, headers=auth)
    assert resp.json() == {"accepted": 0, "rejected": 1, "failed": 1}


def test_wrong_dimension_writes_nothing(
    client: TestClient, auth: dict[str, str], factory: sessionmaker[Session]
) -> None:
    items = client.post(
        "/engine/v1/claim", json={"models": ["muq"], "limit": 2}, headers=auth
    ).json()["items"]
    good, bad = items[0]["track_id"], items[1]["track_id"]
    body = {
        "results": [
            {"track_id": good, "model": "muq", "profile": MUQ, "embedding": [0.1] * 1024},
            {"track_id": bad, "model": "muq", "profile": MUQ, "embedding": [0.1] * 3},
        ],
        "failures": [],
    }
    assert client.post("/engine/v1/results", json=body, headers=auth).status_code == 422
    with factory() as s:
        assert (
            s.scalar(select(func.count()).select_from(Analysis).where(Analysis.state == "done"))
            == 0
        )


def test_release(client: TestClient, auth: dict[str, str]) -> None:
    client.post("/engine/v1/claim", json={"models": ["muq"], "limit": 2}, headers=auth)
    assert client.post("/engine/v1/release", headers=auth).json() == {"released": 2}
    items = client.post(
        "/engine/v1/claim", json={"models": ["muq"], "limit": 2}, headers=auth
    ).json()["items"]
    assert len(items) == 2


def test_claim_limit_validation(client: TestClient, auth: dict[str, str]) -> None:
    for limit in (0, 65):
        resp = client.post(
            "/engine/v1/claim", json={"models": ["muq"], "limit": limit}, headers=auth
        )
        assert resp.status_code == 422


def test_audio_not_found_and_traversal(
    client: TestClient, auth: dict[str, str], factory: sessionmaker[Session], tmp_path: Path
) -> None:
    assert client.get("/engine/v1/tracks/99999/audio", headers=auth).status_code == 404
    outside = tmp_path.parent / f"outside-{uuid.uuid4().hex}.flac"
    outside.write_bytes(b"secret")
    try:
        with factory() as s:
            for path in (f"../{outside.name}", "missing.flac"):
                s.execute(
                    text(
                        "INSERT INTO tracks (path, duration_sec, mtime, size_bytes, seen_at) "
                        "VALUES (:p, 1, now(), 1, now())"
                    ),
                    {"p": path},
                )
            s.commit()
            ids = (
                s.execute(
                    text("SELECT id FROM tracks WHERE path LIKE '../%' OR path = 'missing.flac'")
                )
                .scalars()
                .all()
            )
        for tid in ids:
            assert client.get(f"/engine/v1/tracks/{tid}/audio", headers=auth).status_code == 404
    finally:
        outside.unlink()
