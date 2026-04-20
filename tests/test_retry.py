"""Tests for retry behavior and connection error handling.

Covers:
- Retry on 500, 503 (retryable server errors)
- Retry with Retry-After header on 429
- No retry on 400, 401, 404
- Max retries exhausted raises the original error
- Connection error (httpx.ConnectError) triggers retry
- Total number of attempts == max_retries
- Connection errors are wrapped as Z3rnoConnectionError (sync + async)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pytest_httpx import HTTPXMock

from z3rno import (
    AsyncZ3rnoClient,
    AuthenticationError,
    NotFoundError,
    RateLimitError,
    ServerError,
    ValidationError,
    Z3rnoClient,
    Z3rnoConnectionError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MEMORY_JSON = {
    "id": "mem-1",
    "agent_id": "agent-1",
    "content": "ok",
    "memory_type": "episodic",
    "importance_score": 0.5,
    "recall_count": 0,
    "created_at": "2026-04-12T00:00:00Z",
}

_BASE = "http://test.z3rno.dev"
_URL = f"{_BASE}/v1/memories/mem-1"


# ---------------------------------------------------------------------------
# 1. Retry on 500
# ---------------------------------------------------------------------------


def test_retry_on_500_then_succeed(httpx_mock: HTTPXMock) -> None:
    """500 triggers retry; second attempt succeeds."""
    client = Z3rnoClient(base_url=_BASE, api_key="k", max_retries=3)
    httpx_mock.add_response(url=_URL, method="GET", status_code=500, json={"error": "boom"})
    httpx_mock.add_response(url=_URL, method="GET", json=_MEMORY_JSON)

    mem = client.get_memory("mem-1")
    assert mem.id == "mem-1"
    client.close()


# ---------------------------------------------------------------------------
# 2. Retry on 503
# ---------------------------------------------------------------------------


def test_retry_on_503_then_succeed(httpx_mock: HTTPXMock) -> None:
    """503 triggers retry; second attempt succeeds."""
    client = Z3rnoClient(base_url=_BASE, api_key="k", max_retries=3)
    httpx_mock.add_response(url=_URL, method="GET", status_code=503, json={"error": "unavailable"})
    httpx_mock.add_response(url=_URL, method="GET", json=_MEMORY_JSON)

    mem = client.get_memory("mem-1")
    assert mem.id == "mem-1"
    client.close()


# ---------------------------------------------------------------------------
# 3. Retry with Retry-After header on 429
# ---------------------------------------------------------------------------


def test_retry_on_429_respects_retry_after(httpx_mock: HTTPXMock) -> None:
    """429 with Retry-After header is retried, and wait time comes from header."""
    client = Z3rnoClient(base_url=_BASE, api_key="k", max_retries=3)
    httpx_mock.add_response(
        url=_URL,
        method="GET",
        status_code=429,
        json={"error": "rate_limit"},
        headers={"Retry-After": "2"},
    )
    httpx_mock.add_response(url=_URL, method="GET", json=_MEMORY_JSON)

    # Patch sleep to avoid real delay and verify Retry-After value
    with patch("tenacity.nap.time.sleep") as mock_sleep:
        mem = client.get_memory("mem-1")
        assert mem.id == "mem-1"
        # The wait function should have returned 2.0 (from Retry-After header)
        mock_sleep.assert_called()
        wait_time = mock_sleep.call_args[0][0]
        assert wait_time == pytest.approx(2.0)
    client.close()


# ---------------------------------------------------------------------------
# 4. No retry on 400, 401, 404
# ---------------------------------------------------------------------------


def test_no_retry_on_400(httpx_mock: HTTPXMock) -> None:
    """400 is not retried — raises immediately."""
    client = Z3rnoClient(base_url=_BASE, api_key="k", max_retries=3)
    httpx_mock.add_response(
        url=f"{_BASE}/v1/memories",
        method="POST",
        status_code=400,
        json={"detail": "bad request"},
    )

    with pytest.raises(ValidationError):
        client.store(agent_id="agent-1", content="test")
    # Only one request should have been made (no retries)
    assert len(httpx_mock.get_requests()) == 1
    client.close()


def test_no_retry_on_401(httpx_mock: HTTPXMock) -> None:
    """401 is not retried."""
    client = Z3rnoClient(base_url=_BASE, api_key="k", max_retries=3)
    httpx_mock.add_response(url=_URL, method="GET", status_code=401, json={"error": "unauth"})

    with pytest.raises(AuthenticationError):
        client.get_memory("mem-1")
    assert len(httpx_mock.get_requests()) == 1
    client.close()


def test_no_retry_on_404(httpx_mock: HTTPXMock) -> None:
    """404 is not retried."""
    client = Z3rnoClient(base_url=_BASE, api_key="k", max_retries=3)
    httpx_mock.add_response(url=_URL, method="GET", status_code=404, json={"detail": "not found"})

    with pytest.raises(NotFoundError):
        client.get_memory("mem-1")
    assert len(httpx_mock.get_requests()) == 1
    client.close()


# ---------------------------------------------------------------------------
# 5. Max retries exhausted raises the original error
# ---------------------------------------------------------------------------


def test_max_retries_exhausted_raises_server_error(httpx_mock: HTTPXMock) -> None:
    """After max_retries attempts, the ServerError is re-raised."""
    client = Z3rnoClient(base_url=_BASE, api_key="k", max_retries=2)
    httpx_mock.add_response(url=_URL, method="GET", status_code=500, json={"error": "down"})
    httpx_mock.add_response(url=_URL, method="GET", status_code=500, json={"error": "down"})

    with pytest.raises(ServerError, match="down"):
        client.get_memory("mem-1")
    client.close()


def test_max_retries_exhausted_raises_rate_limit_error(httpx_mock: HTTPXMock) -> None:
    """After max_retries attempts on 429, RateLimitError is re-raised."""
    client = Z3rnoClient(base_url=_BASE, api_key="k", max_retries=2)
    httpx_mock.add_response(
        url=_URL,
        method="GET",
        status_code=429,
        json={"error": "limit"},
        headers={"Retry-After": "0"},
    )
    httpx_mock.add_response(
        url=_URL,
        method="GET",
        status_code=429,
        json={"error": "limit"},
        headers={"Retry-After": "0"},
    )

    with patch("tenacity.nap.time.sleep"):
        with pytest.raises(RateLimitError):
            client.get_memory("mem-1")
    client.close()


# ---------------------------------------------------------------------------
# 6. Connection error triggers retry
# ---------------------------------------------------------------------------


def test_connect_error_triggers_retry(httpx_mock: HTTPXMock) -> None:
    """httpx.ConnectError triggers retry, then succeeds."""
    client = Z3rnoClient(base_url=_BASE, api_key="k", max_retries=3)
    httpx_mock.add_exception(httpx.ConnectError("refused"), url=_URL, method="GET")
    httpx_mock.add_response(url=_URL, method="GET", json=_MEMORY_JSON)

    with patch("tenacity.nap.time.sleep"):
        mem = client.get_memory("mem-1")
    assert mem.id == "mem-1"
    client.close()


# ---------------------------------------------------------------------------
# 7. Total number of attempts == max_retries
# ---------------------------------------------------------------------------


def test_total_attempts_equals_max_retries(httpx_mock: HTTPXMock) -> None:
    """With max_retries=3, exactly 3 requests are made before giving up."""
    client = Z3rnoClient(base_url=_BASE, api_key="k", max_retries=3)
    for _ in range(3):
        httpx_mock.add_response(url=_URL, method="GET", status_code=500, json={"error": "fail"})

    with patch("tenacity.nap.time.sleep"):
        with pytest.raises(ServerError):
            client.get_memory("mem-1")
    assert len(httpx_mock.get_requests()) == 3
    client.close()


def test_total_attempts_max_retries_one() -> None:
    """With max_retries=1, exactly 1 request is made (no retries)."""
    client = Z3rnoClient(base_url=_BASE, api_key="k", max_retries=1)
    call_count = 0

    def counting_request(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 500
        resp.headers = {"content-type": "application/json"}
        resp.json.return_value = {"error": "fail"}
        resp.text = "fail"
        return resp

    with patch.object(client._http, "request", side_effect=counting_request):
        with pytest.raises(ServerError):
            client.get_memory("mem-1")
    assert call_count == 1
    client.close()


# ===========================================================================
# Connection error tests (sync client)
# ===========================================================================


def test_connect_error_raises_z3rno_connection_error() -> None:
    """httpx.ConnectError is wrapped as Z3rnoConnectionError."""
    client = Z3rnoClient(base_url=_BASE, api_key="k", max_retries=1)
    with patch.object(client._http, "request", side_effect=httpx.ConnectError("refused")):
        with pytest.raises(Z3rnoConnectionError, match="Connection failed"):
            client.get_memory("mem-1")
    client.close()


def test_read_timeout_raises_z3rno_connection_error() -> None:
    """httpx.ReadTimeout is wrapped as Z3rnoConnectionError."""
    client = Z3rnoClient(base_url=_BASE, api_key="k", max_retries=1)
    with patch.object(client._http, "request", side_effect=httpx.ReadTimeout("timed out")):
        with pytest.raises(Z3rnoConnectionError, match="Connection failed"):
            client.get_memory("mem-1")
    client.close()


def test_write_timeout_raises_z3rno_connection_error() -> None:
    """httpx.WriteTimeout is wrapped as Z3rnoConnectionError."""
    client = Z3rnoClient(base_url=_BASE, api_key="k", max_retries=1)
    with patch.object(client._http, "request", side_effect=httpx.WriteTimeout("timed out")):
        with pytest.raises(Z3rnoConnectionError, match="Connection failed"):
            client.get_memory("mem-1")
    client.close()


def test_connect_timeout_raises_z3rno_connection_error() -> None:
    """httpx.ConnectTimeout is wrapped as Z3rnoConnectionError."""
    client = Z3rnoClient(base_url=_BASE, api_key="k", max_retries=1)
    with patch.object(client._http, "request", side_effect=httpx.ConnectTimeout("timed out")):
        with pytest.raises(Z3rnoConnectionError, match="Connection failed"):
            client.get_memory("mem-1")
    client.close()


# ===========================================================================
# Connection error tests (async client)
# ===========================================================================


async def test_async_connect_error_raises_z3rno_connection_error() -> None:
    """Async: httpx.ConnectError is wrapped as Z3rnoConnectionError."""
    async with AsyncZ3rnoClient(base_url=_BASE, api_key="k", max_retries=1) as client:
        with patch.object(
            client._http,
            "request",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("refused"),
        ):
            with pytest.raises(Z3rnoConnectionError, match="Connection failed"):
                await client.get_memory("mem-1")


async def test_async_read_timeout_raises_z3rno_connection_error() -> None:
    """Async: httpx.ReadTimeout is wrapped as Z3rnoConnectionError."""
    async with AsyncZ3rnoClient(base_url=_BASE, api_key="k", max_retries=1) as client:
        with patch.object(
            client._http,
            "request",
            new_callable=AsyncMock,
            side_effect=httpx.ReadTimeout("timed out"),
        ):
            with pytest.raises(Z3rnoConnectionError, match="Connection failed"):
                await client.get_memory("mem-1")


async def test_async_write_timeout_raises_z3rno_connection_error() -> None:
    """Async: httpx.WriteTimeout is wrapped as Z3rnoConnectionError."""
    async with AsyncZ3rnoClient(base_url=_BASE, api_key="k", max_retries=1) as client:
        with patch.object(
            client._http,
            "request",
            new_callable=AsyncMock,
            side_effect=httpx.WriteTimeout("timed out"),
        ):
            with pytest.raises(Z3rnoConnectionError, match="Connection failed"):
                await client.get_memory("mem-1")


async def test_async_connect_timeout_raises_z3rno_connection_error() -> None:
    """Async: httpx.ConnectTimeout is wrapped as Z3rnoConnectionError."""
    async with AsyncZ3rnoClient(base_url=_BASE, api_key="k", max_retries=1) as client:
        with patch.object(
            client._http,
            "request",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectTimeout("timed out"),
        ):
            with pytest.raises(Z3rnoConnectionError, match="Connection failed"):
                await client.get_memory("mem-1")
