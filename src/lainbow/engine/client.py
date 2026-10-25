"""HTTP client for the server's engine protocol."""

from collections.abc import Sequence
from pathlib import Path

import httpx

from lainbow.engine.protocol import (
    ClaimIn,
    ClaimOut,
    FailureIn,
    PlanOut,
    ReleaseOut,
    ResultIn,
    ResultsIn,
    ResultsOut,
)

UNAUTHORIZED_MESSAGE = "the server rejected the engine token; create one with lainbow add-engine"


class EngineUnauthorizedError(Exception):
    """Raised when the server rejects the engine token."""


class ServerUnreachableError(Exception):
    """Raised when the server cannot be reached."""


class ServerClient:
    """Typed wrapper around the ``/engine/v1`` endpoints."""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {token}"},
            transport=transport,
            timeout=timeout,
        )

    def close(self) -> None:
        """Close the underlying connection pool."""
        self._http.close()

    def _check(self, response: httpx.Response) -> httpx.Response:
        if response.status_code == 401:
            raise EngineUnauthorizedError(UNAUTHORIZED_MESSAGE)
        response.raise_for_status()
        return response

    def _request(self, method: str, url: str, **kwargs: object) -> httpx.Response:
        try:
            response = self._http.request(method, url, **kwargs)  # type: ignore[arg-type]  # kwargs forwarded verbatim
        except httpx.TransportError as e:
            raise ServerUnreachableError(f"cannot reach the server: {e}") from e
        return self._check(response)

    def plan(self) -> PlanOut:
        """Fetch the active tasks."""
        return PlanOut.model_validate(self._request("GET", "/engine/v1/plan").json())

    def claim(self, models: Sequence[str], limit: int) -> ClaimOut:
        """Lease tracks for the given models."""
        body = ClaimIn(models=list(models), limit=limit)
        r = self._request("POST", "/engine/v1/claim", json=body.model_dump())
        return ClaimOut.model_validate(r.json())

    def post_results(
        self, results: Sequence[ResultIn], failures: Sequence[FailureIn]
    ) -> ResultsOut:
        """Report computed embeddings and failures."""
        body = ResultsIn(results=list(results), failures=list(failures))
        r = self._request("POST", "/engine/v1/results", json=body.model_dump(mode="json"))
        return ResultsOut.model_validate(r.json())

    def release(self) -> ReleaseOut:
        """Release every lease held by this engine."""
        return ReleaseOut.model_validate(self._request("POST", "/engine/v1/release").json())

    def download(self, track_id: int, dest: Path) -> None:
        """Stream a track's audio file to ``dest``."""
        try:
            with self._http.stream("GET", f"/engine/v1/tracks/{track_id}/audio") as response:
                self._check(response)
                with dest.open("wb") as f:
                    for chunk in response.iter_bytes():
                        f.write(chunk)
        except httpx.TransportError as e:
            raise ServerUnreachableError(f"cannot reach the server: {e}") from e
