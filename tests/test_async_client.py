"""Tests for AsyncZ3rnoClient using pytest-httpx mocks."""

from __future__ import annotations

import pytest
from pytest_httpx import HTTPXMock

from z3rno import AsyncZ3rnoClient, AuthenticationError, Memory, RecallResponse


@pytest.fixture
async def client():  # type: ignore[no-untyped-def]
    """Create a test async client."""
    async with AsyncZ3rnoClient(base_url="http://test.z3rno.dev", api_key="test-key") as c:
        yield c


async def test_async_store(client: AsyncZ3rnoClient, httpx_mock: HTTPXMock) -> None:
    """Async store returns a Memory object."""
    httpx_mock.add_response(
        url="http://test.z3rno.dev/v1/memories",
        method="POST",
        json={
            "id": "mem-async-1",
            "agent_id": "agent-1",
            "content": "async content",
            "memory_type": "semantic",
            "importance_score": 0.8,
            "recall_count": 0,
            "created_at": "2026-04-12T00:00:00Z",
        },
    )

    memory = await client.store(agent_id="agent-1", content="async content")
    assert isinstance(memory, Memory)
    assert memory.id == "mem-async-1"
    assert memory.importance_score == 0.8


async def test_async_recall(client: AsyncZ3rnoClient, httpx_mock: HTTPXMock) -> None:
    """Async recall returns RecallResponse."""
    httpx_mock.add_response(
        url="http://test.z3rno.dev/v1/memories/recall",
        method="POST",
        json={
            "results": [],
            "total": 0,
            "query": "test",
        },
    )

    result = await client.recall(agent_id="agent-1", query="test")
    assert isinstance(result, RecallResponse)
    assert result.total == 0


async def test_async_forget(client: AsyncZ3rnoClient, httpx_mock: HTTPXMock) -> None:
    """Async forget returns ForgetResult."""
    httpx_mock.add_response(
        url="http://test.z3rno.dev/v1/memories/forget",
        method="POST",
        json={
            "deleted_count": 1,
            "hard_deleted": False,
            "cascade_count": 0,
            "memory_ids": ["mem-1"],
        },
    )

    result = await client.forget(agent_id="agent-1", memory_id="mem-1")
    assert result.deleted_count == 1


async def test_async_audit(client: AsyncZ3rnoClient, httpx_mock: HTTPXMock) -> None:
    """Async audit returns AuditPage."""
    httpx_mock.add_response(
        url="http://test.z3rno.dev/v1/audit?page=1&page_size=50",
        method="GET",
        json={
            "entries": [],
            "total": 0,
            "page": 1,
            "page_size": 50,
            "has_next": False,
        },
    )

    result = await client.audit()
    assert result.total == 0
    assert result.has_next is False


async def test_async_401(client: AsyncZ3rnoClient, httpx_mock: HTTPXMock) -> None:
    """Async 401 raises AuthenticationError."""
    httpx_mock.add_response(
        url="http://test.z3rno.dev/v1/memories",
        method="POST",
        status_code=401,
        json={"error": "unauthorized"},
    )

    with pytest.raises(AuthenticationError):
        await client.store(agent_id="agent-1", content="test")


async def test_async_context_manager() -> None:
    """Async client works as context manager."""
    async with AsyncZ3rnoClient(base_url="http://test", api_key="key") as client:
        assert client is not None
