"""Phase G slice 5 — SDK SSE streaming test.

Mocks the httpx stream() context manager to drive the parser without
a live server.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from z3rno import AsyncZ3rnoClient


class _FakeStream:
    """Minimal async-context manager mirroring httpx.AsyncClient.stream()."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    async def __aenter__(self) -> _FakeStream:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self):  # type: ignore[no-untyped-def]
        for line in self._lines:
            yield line


@pytest.mark.asyncio
async def test_recall_stream_sse_parses_step_and_done_events() -> None:
    sse_lines = [
        "event: step",
        'data: {"step": 0, "query": "seed", "results": []}',
        "",
        "event: results",
        'data: {"results": [], "total": 0, "strategy_used": "TRACE"}',
        "",
        "event: done",
        'data: {"elapsed_ms": 42.0}',
        "",
    ]

    client = AsyncZ3rnoClient(base_url="http://test", api_key="sk")
    client._http = MagicMock()  # type: ignore[assignment]
    client._http.stream = MagicMock(return_value=_FakeStream(sse_lines))

    events: list[dict[str, Any]] = [
        evt
        async for evt in client.recall_stream_sse(agent_id=str(uuid4()), query="x")
    ]

    kinds = [e["event"] for e in events]
    assert kinds == ["step", "results", "done"]
    assert events[0]["data"]["step"] == 0
    assert events[1]["data"]["strategy_used"] == "TRACE"
    assert events[-1]["data"]["elapsed_ms"] == 42.0


@pytest.mark.asyncio
async def test_recall_stream_sse_handles_error_event() -> None:
    sse_lines = [
        "event: error",
        'data: {"detail": "boom"}',
        "",
        "event: done",
        "data: {}",
        "",
    ]

    client = AsyncZ3rnoClient(base_url="http://test", api_key="sk")
    client._http = MagicMock()  # type: ignore[assignment]
    client._http.stream = MagicMock(return_value=_FakeStream(sse_lines))

    events: list[dict[str, Any]] = [
        evt async for evt in client.recall_stream_sse(agent_id=str(uuid4()))
    ]
    kinds = [e["event"] for e in events]
    assert kinds == ["error", "done"]
    assert events[0]["data"]["detail"] == "boom"


@pytest.mark.asyncio
async def test_recall_stream_sse_passes_conversation_id_in_body() -> None:
    sse_lines = ["event: done", "data: {}", ""]
    client = AsyncZ3rnoClient(base_url="http://test", api_key="sk")
    client._http = MagicMock()  # type: ignore[assignment]
    client._http.stream = MagicMock(return_value=_FakeStream(sse_lines))

    cid = str(uuid4())
    async for _ in client.recall_stream_sse(
        agent_id=str(uuid4()), query="x", conversation_id=cid
    ):
        pass

    assert client._http.stream.called
    _, kwargs = client._http.stream.call_args
    body = kwargs["json"]
    assert body["conversation_id"] == cid


@pytest.mark.asyncio
async def test_recall_stream_sse_forwards_strategy_and_rerank() -> None:
    sse_lines = ["event: done", "data: {}", ""]
    client = AsyncZ3rnoClient(base_url="http://test", api_key="sk")
    client._http = MagicMock()  # type: ignore[assignment]
    client._http.stream = MagicMock(return_value=_FakeStream(sse_lines))

    async for _ in client.recall_stream_sse(
        agent_id=str(uuid4()), query="x", strategy="TRACE", rerank=True
    ):
        pass
    _, kwargs = client._http.stream.call_args
    body = kwargs["json"]
    assert body["strategy"] == "TRACE"
    assert body["rerank"] is True
