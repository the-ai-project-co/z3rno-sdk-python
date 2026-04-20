"""Integration tests against a real z3rno-server.

These tests are skipped by default. To run them, set the ``Z3RNO_TEST_URL``
and ``Z3RNO_TEST_API_KEY`` environment variables and invoke pytest with::

    pytest -m integration

Example::

    Z3RNO_TEST_URL=http://localhost:8000 \
    Z3RNO_TEST_API_KEY=z3rno_sk_test_localdev \
    uv run pytest tests/test_integration.py -v
"""

from __future__ import annotations

import os

import pytest

from z3rno import (
    Memory,
    NotFoundError,
    RecallResponse,
    ValidationError,
    Z3rnoClient,
)

TEST_URL = os.environ.get("Z3RNO_TEST_URL", "")
TEST_API_KEY = os.environ.get("Z3RNO_TEST_API_KEY", "")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not TEST_URL, reason="Z3RNO_TEST_URL not set"),
]


@pytest.fixture
def client() -> Z3rnoClient:
    """Create a client pointed at the real test server."""
    return Z3rnoClient(base_url=TEST_URL, api_key=TEST_API_KEY, max_retries=1)


def test_store_recall_forget_lifecycle(client: Z3rnoClient) -> None:
    """Full lifecycle: store a memory, recall it, then forget it."""
    # Store
    memory = client.store(
        agent_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        content="Integration test: the user prefers dark mode",
        memory_type="episodic",
        metadata={"source": "integration-test"},
    )
    assert isinstance(memory, Memory)
    assert memory.id
    assert memory.content == "Integration test: the user prefers dark mode"

    # Recall
    result = client.recall(
        agent_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        query="dark mode preference",
        top_k=5,
    )
    assert isinstance(result, RecallResponse)
    assert result.total >= 1
    memory_ids = [r.memory_id for r in result.results]
    assert memory.id in memory_ids

    # Forget (soft delete — hard_delete requires AGE graph which may not be available)
    forget_result = client.forget(
        agent_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        memory_id=memory.id,
    )
    assert forget_result.deleted_count >= 1


def test_store_validation_error(client: Z3rnoClient) -> None:
    """Storing with invalid memory_type returns 422 ValidationError."""
    with pytest.raises(ValidationError):
        client.store(
            agent_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
            content="test",
            memory_type="invalid_type_that_does_not_exist",
        )


def test_get_memory_not_found(client: Z3rnoClient) -> None:
    """Getting a non-existent memory returns 404 NotFoundError."""
    with pytest.raises(NotFoundError):
        client.get_memory("00000000-0000-0000-0000-000000000000")
