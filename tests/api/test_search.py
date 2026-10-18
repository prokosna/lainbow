from collections.abc import Generator

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlmodel import Session

from api.session_manager_wrap import get_db_session
from api.v1.endpoints import songs
from domain import inference_client
from domain.embedding_models import MODEL_SPECS
from domain.schemas import EmbeddingModel, Song


@pytest.fixture
def api_client(db: Session) -> TestClient:
    app = FastAPI()
    app.include_router(songs.router, prefix="/songs")

    def override_get_db_session() -> Generator[Session, None, None]:
        yield db

    app.dependency_overrides[get_db_session] = override_get_db_session
    return TestClient(app)


def test_search_returns_stored_songs(
    api_client: TestClient,
    db: Session,
    fake_vector_store: dict[str, dict[str, list[float]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Set up a stored song in SQLite and vector store
    song = Song(
        file_path="artists/album/track.flac",
        title="Sample Track",
        artist="Sample Artist",
        album="Sample Album",
        genre="Ambient",
    )
    db.add(song)
    db.commit()

    coll_name = MODEL_SPECS[EmbeddingModel.MUQ_MULAN].collection_name
    fake_vector_store[coll_name] = {
        "artists/album/track.flac": [0.5] * 512,
    }

    def fake_run_text_embedding(model_name: str, text: str) -> list[float]:
        assert model_name == "muq_mulan"
        assert text == "ambient relaxing music"
        return [0.5] * 512

    monkeypatch.setattr(inference_client, "run_text_embedding", fake_run_text_embedding)

    response = api_client.get("/songs/search", params={"q": "ambient relaxing music"})
    assert response.status_code == 200
    data = response.json()
    assert len(data["songs"]) == 1
    assert data["songs"][0]["file_path"] == "artists/album/track.flac"
    assert data["songs"][0]["title"] == "Sample Track"
    assert data["songs"][0]["artist"] == "Sample Artist"
    assert data["songs"][0]["score"] == pytest.approx(1.0)


def test_search_returns_503_when_inference_server_unreachable(
    api_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run_text_embedding_fail(model_name: str, text: str) -> list[float]:
        req = httpx.Request("POST", "http://localhost:8001/api/v1/text-embedding/muq_mulan")
        raise httpx.ConnectError("Connection refused", request=req)

    monkeypatch.setattr(inference_client, "run_text_embedding", fake_run_text_embedding_fail)

    response = api_client.get("/songs/search", params={"q": "summer track"})
    assert response.status_code == 503
    assert (
        "Natural language search needs the inference server, which is not reachable"
        in response.json()["detail"]
    )


def test_search_returns_503_on_timeout_or_5xx_status(
    api_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Test TimeoutException -> 503
    def fake_timeout(model_name: str, text: str) -> list[float]:
        raise httpx.ReadTimeout("Request timed out")

    monkeypatch.setattr(inference_client, "run_text_embedding", fake_timeout)
    resp_timeout = api_client.get("/songs/search", params={"q": "summer track"})
    assert resp_timeout.status_code == 503
    assert (
        "Natural language search needs the inference server, which is not reachable"
        in resp_timeout.json()["detail"]
    )

    # Test 5xx HTTPStatusError -> 503
    req = httpx.Request("POST", "http://localhost:8001")
    resp_502 = httpx.Response(502, request=req)

    def fake_502(model_name: str, text: str) -> list[float]:
        raise httpx.HTTPStatusError("Bad Gateway", request=req, response=resp_502)

    monkeypatch.setattr(inference_client, "run_text_embedding", fake_502)
    resp_status = api_client.get("/songs/search", params={"q": "summer track"})
    assert resp_status.status_code == 503
    assert (
        "Natural language search needs the inference server, which is not reachable"
        in resp_status.json()["detail"]
    )
