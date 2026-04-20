"""Tests for Z3rnoClient (sync) using pytest-httpx mocks."""

from __future__ import annotations

import pytest
from pytest_httpx import HTTPXMock

from z3rno import (
    AuthenticationError,
    BatchStoreResponse,
    Memory,
    MemoryHistoryResponse,
    NotFoundError,
    RateLimitError,
    RecallResponse,
    ServerError,
    ValidationError,
    Z3rnoClient,
)


@pytest.fixture
def client() -> Z3rnoClient:
    """Create a test client."""
    return Z3rnoClient(base_url="http://test.z3rno.dev", api_key="test-key")


def test_client_constructor() -> None:
    """Client constructs without error."""
    client = Z3rnoClient(base_url="http://localhost:8000", api_key="sk_test")
    assert client is not None
    client.close()


def test_store_memory(client: Z3rnoClient, httpx_mock: HTTPXMock) -> None:
    """Store returns a Memory object."""
    httpx_mock.add_response(
        url="http://test.z3rno.dev/v1/memories",
        method="POST",
        json={
            "id": "mem-123",
            "agent_id": "agent-1",
            "content": "test content",
            "memory_type": "episodic",
            "importance_score": 0.5,
            "recall_count": 0,
            "created_at": "2026-04-12T00:00:00Z",
        },
    )

    memory = client.store(agent_id="agent-1", content="test content")
    assert isinstance(memory, Memory)
    assert memory.id == "mem-123"
    assert memory.content == "test content"


def test_recall_memories(client: Z3rnoClient, httpx_mock: HTTPXMock) -> None:
    """Recall returns a RecallResponse."""
    httpx_mock.add_response(
        url="http://test.z3rno.dev/v1/memories/recall",
        method="POST",
        json={
            "results": [
                {
                    "memory_id": "mem-123",
                    "content": "test",
                    "memory_type": "episodic",
                    "similarity_score": 0.9,
                    "importance_score": 0.5,
                    "relevance_score": 0.7,
                    "recall_count": 1,
                    "created_at": "2026-04-12T00:00:00Z",
                }
            ],
            "total": 1,
            "query": "test query",
        },
    )

    result = client.recall(agent_id="agent-1", query="test query")
    assert isinstance(result, RecallResponse)
    assert result.total == 1
    assert result.results[0].similarity_score == 0.9


def test_401_raises_auth_error(client: Z3rnoClient, httpx_mock: HTTPXMock) -> None:
    """401 response raises AuthenticationError."""
    httpx_mock.add_response(
        url="http://test.z3rno.dev/v1/memories",
        method="POST",
        status_code=401,
        json={"error": "unauthorized"},
    )

    with pytest.raises(AuthenticationError):
        client.store(agent_id="agent-1", content="test")


def test_422_raises_validation_error(client: Z3rnoClient, httpx_mock: HTTPXMock) -> None:
    """422 response raises ValidationError."""
    httpx_mock.add_response(
        url="http://test.z3rno.dev/v1/memories",
        method="POST",
        status_code=422,
        json={"detail": "content too short"},
    )

    with pytest.raises(ValidationError):
        client.store(agent_id="agent-1", content="test")


def test_429_raises_rate_limit_error(httpx_mock: HTTPXMock) -> None:
    """429 response raises RateLimitError with retry_after (no retries)."""
    no_retry_client = Z3rnoClient(
        base_url="http://test.z3rno.dev", api_key="test-key", max_retries=1
    )
    httpx_mock.add_response(
        url="http://test.z3rno.dev/v1/memories",
        method="POST",
        status_code=429,
        json={"error": "rate_limit_exceeded"},
        headers={"Retry-After": "30"},
    )

    with pytest.raises(RateLimitError) as exc_info:
        no_retry_client.store(agent_id="agent-1", content="test")
    assert exc_info.value.retry_after == 30
    no_retry_client.close()


def test_context_manager() -> None:
    """Client works as a context manager."""
    with Z3rnoClient(base_url="http://test", api_key="key") as client:
        assert client is not None


def test_get_memory(client: Z3rnoClient, httpx_mock: HTTPXMock) -> None:
    """get_memory returns a Memory object by ID."""
    httpx_mock.add_response(
        url="http://test.z3rno.dev/v1/memories/mem-456",
        method="GET",
        json={
            "id": "mem-456",
            "agent_id": "agent-1",
            "content": "remembered content",
            "memory_type": "semantic",
            "importance_score": 0.9,
            "recall_count": 5,
            "created_at": "2026-04-12T00:00:00Z",
        },
    )

    memory = client.get_memory("mem-456")
    assert isinstance(memory, Memory)
    assert memory.id == "mem-456"
    assert memory.content == "remembered content"
    assert memory.memory_type == "semantic"


def test_get_memory_not_found(httpx_mock: HTTPXMock) -> None:
    """get_memory raises NotFoundError on 404."""
    no_retry_client = Z3rnoClient(
        base_url="http://test.z3rno.dev", api_key="test-key", max_retries=1
    )
    httpx_mock.add_response(
        url="http://test.z3rno.dev/v1/memories/nonexistent",
        method="GET",
        status_code=404,
        json={"detail": "Memory not found"},
    )

    with pytest.raises(NotFoundError):
        no_retry_client.get_memory("nonexistent")
    no_retry_client.close()


def test_retry_on_500(httpx_mock: HTTPXMock) -> None:
    """Retry on 500 then succeed."""
    retry_client = Z3rnoClient(base_url="http://test.z3rno.dev", api_key="test-key", max_retries=3)
    # First call: 500, second call: 200
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

    memory = retry_client.get_memory("mem-retry")
    assert memory.id == "mem-retry"
    retry_client.close()


def test_no_retry_on_401(httpx_mock: HTTPXMock) -> None:
    """401 errors are not retried."""
    retry_client = Z3rnoClient(base_url="http://test.z3rno.dev", api_key="test-key", max_retries=3)
    httpx_mock.add_response(
        url="http://test.z3rno.dev/v1/memories/mem-auth",
        method="GET",
        status_code=401,
        json={"error": "unauthorized"},
    )

    with pytest.raises(AuthenticationError):
        retry_client.get_memory("mem-auth")
    retry_client.close()


def test_no_retry_on_422(httpx_mock: HTTPXMock) -> None:
    """422 errors are not retried."""
    retry_client = Z3rnoClient(base_url="http://test.z3rno.dev", api_key="test-key", max_retries=3)
    httpx_mock.add_response(
        url="http://test.z3rno.dev/v1/memories",
        method="POST",
        status_code=422,
        json={"detail": "invalid"},
    )

    with pytest.raises(ValidationError):
        retry_client.store(agent_id="agent-1", content="test")
    retry_client.close()


def test_store_batch(client: Z3rnoClient, httpx_mock: HTTPXMock) -> None:
    """store_batch returns a BatchStoreResponse."""
    httpx_mock.add_response(
        url="http://test.z3rno.dev/v1/memories/batch",
        method="POST",
        json={
            "results": [
                {
                    "id": "mem-b1",
                    "agent_id": "agent-1",
                    "content": "first memory",
                    "memory_type": "episodic",
                    "importance_score": 0.5,
                    "recall_count": 0,
                    "created_at": "2026-04-12T00:00:00Z",
                },
                {
                    "id": "mem-b2",
                    "agent_id": "agent-1",
                    "content": "second memory",
                    "memory_type": "semantic",
                    "importance_score": 0.7,
                    "recall_count": 0,
                    "created_at": "2026-04-12T00:00:00Z",
                },
            ],
            "stored_count": 2,
        },
    )

    result = client.store_batch(
        agent_id="agent-1",
        memories=[
            {"content": "first memory", "memory_type": "episodic"},
            {"content": "second memory", "memory_type": "semantic"},
        ],
    )
    assert isinstance(result, BatchStoreResponse)
    assert result.stored_count == 2
    assert len(result.results) == 2
    assert result.results[0].id == "mem-b1"
    assert result.results[1].content == "second memory"


def test_get_memory_history(client: Z3rnoClient, httpx_mock: HTTPXMock) -> None:
    """get_memory_history returns a MemoryHistoryResponse."""
    httpx_mock.add_response(
        url="http://test.z3rno.dev/v1/memories/mem-hist/history",
        method="GET",
        json={
            "memory_id": "mem-hist",
            "versions": [
                {
                    "id": "v1",
                    "content": "original content",
                    "memory_type": "episodic",
                    "importance_score": 0.5,
                    "valid_from": "2026-04-10T00:00:00Z",
                    "valid_to": "2026-04-12T00:00:00Z",
                    "metadata": {},
                },
                {
                    "id": "v2",
                    "content": "updated content",
                    "memory_type": "episodic",
                    "importance_score": 0.8,
                    "valid_from": "2026-04-12T00:00:00Z",
                    "valid_to": None,
                    "metadata": {"source": "update"},
                },
            ],
            "total": 2,
        },
    )

    result = client.get_memory_history("mem-hist")
    assert isinstance(result, MemoryHistoryResponse)
    assert result.memory_id == "mem-hist"
    assert result.total == 2
    assert len(result.versions) == 2
    assert result.versions[0].content == "original content"
    assert result.versions[1].valid_to is None


def test_update_memory(client: Z3rnoClient, httpx_mock: HTTPXMock) -> None:
    """update_memory sends PATCH and returns updated Memory."""
    httpx_mock.add_response(
        url="http://test.z3rno.dev/v1/memories/mem-upd",
        method="PATCH",
        json={
            "id": "mem-upd",
            "agent_id": "agent-1",
            "content": "updated content",
            "memory_type": "episodic",
            "importance_score": 0.9,
            "recall_count": 3,
            "created_at": "2026-04-12T00:00:00Z",
            "metadata": {"edited": True},
        },
    )

    memory = client.update_memory(
        "mem-upd",
        content="updated content",
        importance=0.9,
        metadata={"edited": True},
    )
    assert isinstance(memory, Memory)
    assert memory.id == "mem-upd"
    assert memory.content == "updated content"
    assert memory.importance_score == 0.9


def test_update_memory_partial(client: Z3rnoClient, httpx_mock: HTTPXMock) -> None:
    """update_memory only sends non-None fields."""
    httpx_mock.add_response(
        url="http://test.z3rno.dev/v1/memories/mem-partial",
        method="PATCH",
        json={
            "id": "mem-partial",
            "agent_id": "agent-1",
            "content": "new content only",
            "memory_type": "episodic",
            "importance_score": 0.5,
            "recall_count": 0,
            "created_at": "2026-04-12T00:00:00Z",
        },
    )

    memory = client.update_memory("mem-partial", content="new content only")
    assert memory.content == "new content only"


def test_retry_exhausted_raises(httpx_mock: HTTPXMock) -> None:
    """After max retries exhausted, the last error is raised."""
    retry_client = Z3rnoClient(base_url="http://test.z3rno.dev", api_key="test-key", max_retries=2)
    # Two 500s, no success
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
        retry_client.get_memory("mem-fail")
    retry_client.close()
