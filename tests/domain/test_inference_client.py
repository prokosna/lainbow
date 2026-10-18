import io

import httpx
import numpy as np
import numpy.typing as npt
import pytest

from domain import inference_client
from domain.inference_client import _get_client, _is_retryable_exception, run_inference


@pytest.fixture(autouse=True)
def fast_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable tenacity sleep so retries happen instantly in tests."""
    monkeypatch.setattr(run_inference.retry, "sleep", lambda x: None)


@pytest.fixture
def sample_audio() -> npt.NDArray[np.float32]:
    return np.zeros((1, 16000), dtype=np.float32)


def _encode_array(arr: npt.NDArray[np.float32]) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, arr)
    return buffer.getvalue()


def test_400_response_raises_http_status_error_after_1_request(
    monkeypatch: pytest.MonkeyPatch, sample_audio: npt.NDArray[np.float32]
) -> None:
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(400, json={"detail": "Bad Request"}, request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(inference_client, "_client", client)

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        run_inference("clap", sample_audio)

    assert exc_info.value.response.status_code == 400
    assert request_count == 1


def test_503_then_200_returns_decoded_array_after_2_requests(
    monkeypatch: pytest.MonkeyPatch, sample_audio: npt.NDArray[np.float32]
) -> None:
    request_count = 0
    expected_embedding = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        if request_count == 1:
            return httpx.Response(503, text="Service Unavailable", request=request)
        return httpx.Response(200, content=_encode_array(expected_embedding), request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(inference_client, "_client", client)

    result = run_inference("mert", sample_audio)

    assert request_count == 2
    np.testing.assert_array_equal(result, expected_embedding)


def test_five_consecutive_503_responses_raise_http_status_error_after_5_requests(
    monkeypatch: pytest.MonkeyPatch, sample_audio: npt.NDArray[np.float32]
) -> None:
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(503, text="Service Unavailable", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(inference_client, "_client", client)

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        run_inference("clap", sample_audio)

    assert exc_info.value.response.status_code == 503
    assert request_count == 5


def test_successful_response_body_round_trips_to_same_array(
    monkeypatch: pytest.MonkeyPatch, sample_audio: npt.NDArray[np.float32]
) -> None:
    request_count = 0
    expected_embedding = np.array(
        [[1.0, 2.5, -3.2], [0.0, 4.125, 9.75]],
        dtype=np.float32,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(200, content=_encode_array(expected_embedding), request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(inference_client, "_client", client)

    result = run_inference("muq", sample_audio)

    assert request_count == 1
    np.testing.assert_array_equal(result, expected_embedding)


def test_connect_error_retries_and_recovers(
    monkeypatch: pytest.MonkeyPatch, sample_audio: npt.NDArray[np.float32]
) -> None:
    request_count = 0
    expected_embedding = np.array([[1.0, 2.0]], dtype=np.float32)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        if request_count == 1:
            raise httpx.ConnectError("Connection refused", request=request)
        return httpx.Response(200, content=_encode_array(expected_embedding), request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(inference_client, "_client", client)

    result = run_inference("clap", sample_audio)

    assert request_count == 2
    np.testing.assert_array_equal(result, expected_embedding)


def test_read_timeout_retries_and_recovers(
    monkeypatch: pytest.MonkeyPatch, sample_audio: npt.NDArray[np.float32]
) -> None:
    request_count = 0
    expected_embedding = np.array([[1.0, 2.0]], dtype=np.float32)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        if request_count == 1:
            raise httpx.ReadTimeout("Read timed out", request=request)
        return httpx.Response(200, content=_encode_array(expected_embedding), request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(inference_client, "_client", client)

    result = run_inference("clap", sample_audio)

    assert request_count == 2
    np.testing.assert_array_equal(result, expected_embedding)


def test_remote_protocol_error_retries_and_recovers(
    monkeypatch: pytest.MonkeyPatch, sample_audio: npt.NDArray[np.float32]
) -> None:
    request_count = 0
    expected_embedding = np.array([[1.0, 2.0]], dtype=np.float32)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        if request_count == 1:
            raise httpx.RemoteProtocolError("Server disconnected")
        return httpx.Response(200, content=_encode_array(expected_embedding), request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(inference_client, "_client", client)

    result = run_inference("clap", sample_audio)

    assert request_count == 2
    np.testing.assert_array_equal(result, expected_embedding)


def test_500_response_does_not_retry(
    monkeypatch: pytest.MonkeyPatch, sample_audio: npt.NDArray[np.float32]
) -> None:
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(500, text="Internal Server Error", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(inference_client, "_client", client)

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        run_inference("clap", sample_audio)

    assert exc_info.value.response.status_code == 500
    assert request_count == 1


def test_retryable_exception_predicate() -> None:
    req = httpx.Request("POST", "http://example.com")
    assert _is_retryable_exception(httpx.ConnectError("fail", request=req))
    assert _is_retryable_exception(httpx.ReadTimeout("fail", request=req))
    assert _is_retryable_exception(httpx.RemoteProtocolError("fail"))
    assert _is_retryable_exception(
        httpx.HTTPStatusError("502", request=req, response=httpx.Response(502, request=req))
    )
    assert _is_retryable_exception(
        httpx.HTTPStatusError("503", request=req, response=httpx.Response(503, request=req))
    )
    assert _is_retryable_exception(
        httpx.HTTPStatusError("504", request=req, response=httpx.Response(504, request=req))
    )

    assert not _is_retryable_exception(
        httpx.HTTPStatusError("400", request=req, response=httpx.Response(400, request=req))
    )
    assert not _is_retryable_exception(
        httpx.HTTPStatusError("500", request=req, response=httpx.Response(500, request=req))
    )
    assert not _is_retryable_exception(httpx.ConnectTimeout("fail", request=req))
    assert not _is_retryable_exception(ValueError("fail"))


def test_lazy_module_level_client_reuse_and_recreation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inference_client, "_client", None)
    client1 = _get_client()
    assert isinstance(client1, httpx.Client)
    assert not client1.is_closed

    client2 = _get_client()
    assert client1 is client2

    client1.close()
    client3 = _get_client()
    assert client3 is not client1
    assert not client3.is_closed
    client3.close()
