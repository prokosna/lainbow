"""Client of the engine endpoint that turns search text into vectors."""

from __future__ import annotations

import httpx

TIMEOUT_SEC = 30.0


class TextEmbeddingUnavailableError(Exception):
    """The text engine could not be reached or refused the request."""


class TextEmbedder:
    """Embeds search text through the engine's ``/text-embedding`` endpoint."""

    def __init__(self, base_url: str) -> None:
        self._url = base_url.rstrip("/") + "/text-embedding"
        self._client = httpx.Client(timeout=TIMEOUT_SEC)

    def embed(self, model: str, text: str) -> list[float]:
        """Return the embedding of ``text`` for ``model``."""
        try:
            response = self._client.post(self._url, json={"model": model, "text": text})
            response.raise_for_status()
            embedding = response.json()["embedding"]
        except (httpx.HTTPError, ValueError, KeyError) as e:
            raise TextEmbeddingUnavailableError(str(e)) from e
        return [float(x) for x in embedding]

    def close(self) -> None:
        """Release the HTTP connections."""
        self._client.close()


__all__ = ["TextEmbedder", "TextEmbeddingUnavailableError"]
