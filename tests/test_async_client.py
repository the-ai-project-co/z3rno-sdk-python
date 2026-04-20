"""Tests for AsyncZ3rnoClient using pytest-httpx mocks."""

from __future__ import annotations

import pytest
from pytest_httpx import HTTPXMock

from z3rno import (
    AsyncZ3rnoClient,
    AuthenticationError,
    BatchStoreResponse,
    Memory,
    MemoryHistoryResponse,
    NotFoundError,
    RecallResponse,
    ServerError,
    ValidationError,
)


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


async def test_async_get_memory(client: AsyncZ3rnoClient, httpx_mock: HTTPXMock) -> None:
    """Async get_memory returns a Memory object by ID."""
    httpx_mock.add_response(
        url="http://test.z3rno.dev/v1/memories/mem-async-get",
        method="GET",
        json={
            "id": "mem-async-get",
            "agent_id": "agent-1",
            "content": "async retrieved",
            "memory_type": "semantic",
            "importance_score": 0.9,
            "recall_count": 3,
            "created_at": "2026-04-12T00:00:00Z",
        },
    )

    memory = await client.get_memory("mem-async-get")
    assert isinstance(memory, Memory)
    assert memory.id == "mem-async-get"
    assert memory.content == "async retrieved"


async def test_async_get_memory_not_found(httpx_mock: HTTPXMock) -> None:
    """Async get_memory raises NotFoundError on 404."""
    async with AsyncZ3rnoClient(
        base_url="http://test.z3rno.dev", api_key="test-key", max_retries=1
    ) as no_retry_client:
        httpx_mock.add_response(
            url="http://test.z3rno.dev/v1/memories/nonexistent",
            method="GET",
            status_code=404,
            json={"detail": "Memory not found"},
        )

        with pytest.raises(NotFoundError):
            await no_retry_client.get_memory("nonexistent")


async def test_async_store_batch(client: AsyncZ3rnoClient, httpx_mock: HTTPXMock) -> None:
    """Async store_batch returns a BatchStoreResponse."""
    httpx_mock.add_response(
        url="http://test.z3rno.dev/v1/memories/batch",
        method="POST",
        json={
            "results": [
                {
                    "id": "mem-ab1",
                    "agent_id": "agent-1",
                    "content": "async batch 1",
                    "memory_type": "episodic",
                    "importance_score": 0.5,
                    "recall_count": 0,
                    "created_at": "2026-04-12T00:00:00Z",
                },
                {
                    "id": "mem-ab2",
                    "agent_id": "agent-1",
                    "content": "async batch 2",
                    "memory_type": "semantic",
                    "importance_score": 0.7,
                    "recall_count": 0,
                    "created_at": "2026-04-12T00:00:00Z",
                },
            ],
            "stored_count": 2,
        },
    )

    result = await client.store_batch(
        agent_id="agent-1",
        memories=[
            {"content": "async batch 1", "memory_type": "episodic"},
            {"content": "async batch 2", "memory_type": "semantic"},
        ],
    )
    assert isinstance(result, BatchStoreResponse)
    assert result.stored_count == 2
    assert len(result.results) == 2


async def test_async_get_memory_history(client: AsyncZ3rnoClient, httpx_mock: HTTPXMock) -> None:
    """Async get_memory_history returns a MemoryHistoryResponse."""
    httpx_mock.add_response(
        url="http://test.z3rno.dev/v1/memories/mem-hist/history",
        method="GET",
        json={
            "memory_id": "mem-hist",
            "versions": [
                {
                    "id": "v1",
                    "content": "original",
                    "memory_type": "episodic",
                    "importance_score": 0.5,
                    "valid_from": "2026-04-10T00:00:00Z",
                    "valid_to": "2026-04-12T00:00:00Z",
                    "metadata": {},
                },
            ],
            "total": 1,
        },
    )

    result = await client.get_memory_history("mem-hist")
    assert isinstance(result, MemoryHistoryResponse)
    assert result.memory_id == "mem-hist"
    assert result.total == 1
    assert result.versions[0].content == "original"


async def test_async_update_memory(client: AsyncZ3rnoClient, httpx_mock: HTTPXMock) -> None:
    """Async update_memory sends PATCH and returns updated Memory."""
    httpx_mock.add_response(
        url="http://test.z3rno.dev/v1/memories/mem-upd",
        method="PATCH",
        json={
            "id": "mem-upd",
            "agent_id": "agent-1",
            "content": "async updated",
            "memory_type": "episodic",
            "importance_score": 0.9,
            "recall_count": 1,
            "created_at": "2026-04-12T00:00:00Z",
            "metadata": {},
        },
    )

    memory = await client.update_memory("mem-upd", content="async updated", importance=0.9)
    assert isinstance(memory, Memory)
    assert memory.id == "mem-upd"
    assert memory.content == "async updated"
    assert memory.importance_score == 0.9


async def test_async_retry_on_500(httpx_mock: HTTPXMock) -> None:
    """Async retry on 500 then succeed."""
    async with AsyncZ3rnoClient(
        base_url="http://test.z3rno.dev", api_key="test-key", max_retries=3
    ) as retry_client:
        httpx_mock.add_response(
            url="http://test.z3rno.dev/v1/memories/mem-retry",
            method="GET",
            status_code=500,
            json={"error": "internal"},
        )
        httpx_mock.add_response(
            url="http://test.z3rno.dev/v1/memories/mem-retry",
            method="GET",
            json={
                "id": "mem-retry",
                "agent_id": "agent-1",
                "content": "retried",
                "memory_type": "episodic",
                "importance_score": 0.5,
                "recall_count": 0,
                "created_at": "2026-04-12T00:00:00Z",
            },
        )

        memory = await retry_client.get_memory("mem-retry")
        assert memory.id == "mem-retry"


async def test_async_no_retry_on_401(httpx_mock: HTTPXMock) -> None:
    """Async 401 errors are not retried."""
    async with AsyncZ3rnoClient(
        base_url="http://test.z3rno.dev", api_key="test-key", max_retries=3
    ) as retry_client:
        httpx_mock.add_response(
            url="http://test.z3rno.dev/v1/memories/mem-auth",
            method="GET",
            status_code=401,
            json={"error": "unauthorized"},
        )

        with pytest.raises(AuthenticationError):
            await retry_client.get_memory("mem-auth")


async def test_async_no_retry_on_422(httpx_mock: HTTPXMock) -> None:
    """Async 422 errors are not retried."""
    async with AsyncZ3rnoClient(
        base_url="http://test.z3rno.dev", api_key="test-key", max_retries=3
    ) as retry_client:
        httpx_mock.add_response(
            url="http://test.z3rno.dev/v1/memories",
            method="POST",
            status_code=422,
            json={"detail": "invalid"},
        )

        with pytest.raises(ValidationError):
            await retry_client.store(agent_id="agent-1", content="test")


async def test_async_retry_exhausted_raises(httpx_mock: HTTPXMock) -> None:
    """Async max retries exhausted raises the last error."""
    async with AsyncZ3rnoClient(
        base_url="http://test.z3rno.dev", api_key="test-key", max_retries=2
    ) as retry_client:
        httpx_mock.add_response(
            url="http://test.z3rno.dev/v1/memories/mem-fail",
            method="GET",
            status_code=500,
            json={"error": "server down"},
        )
        httpx_mock.add_response(
            url="http://test.z3rno.dev/v1/memories/mem-fail",
            method="GET",
            status_code=500,
            json={"error": "server down"},
        )

        with pytest.raises(ServerError):
            await retry_client.get_memory("mem-fail")
