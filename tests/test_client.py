"""Tests for Z3rnoClient (sync) using pytest-httpx mocks."""

from __future__ import annotations

import pytest
from pytest_httpx import HTTPXMock

from z3rno import (
    AuthenticationError,
    Memory,
    RateLimitError,
    RecallResponse,
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


def test_429_raises_rate_limit_error(client: Z3rnoClient, httpx_mock: HTTPXMock) -> None:
    """429 response raises RateLimitError with retry_after."""
    httpx_mock.add_response(
        url="http://test.z3rno.dev/v1/memories",
        method="POST",
        status_code=429,
        json={"error": "rate_limit_exceeded"},
        headers={"Retry-After": "30"},
    )

    with pytest.raises(RateLimitError) as exc_info:
        client.store(agent_id="agent-1", content="test")
    assert exc_info.value.retry_after == 30


def test_context_manager() -> None:
    """Client works as a context manager."""
    with Z3rnoClient(base_url="http://test", api_key="key") as client:
        assert client is not None
