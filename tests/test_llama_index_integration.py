"""Tests for the LlamaIndex adapter (Phase G slice 3).

Skipped automatically when ``llama-index-core`` isn't installed —
the extra is opt-in (``pip install 'z3rno[llama-index]'``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

llama_index = pytest.importorskip("llama_index.core.base.llms.types")

from llama_index.core.base.llms.types import ChatMessage, MessageRole  # noqa: E402
from llama_index.core.schema import QueryBundle  # noqa: E402

from z3rno.integrations.llama_index import (  # noqa: E402
    Z3rnoBaseRetriever,
    Z3rnoChatMemoryBuffer,
)


def _fake_client_with_recall(results: list[MagicMock] | None = None) -> MagicMock:
    client = MagicMock()
    resp = MagicMock()
    resp.results = results or []
    client.recall.return_value = resp
    memory = MagicMock()
    memory.id = str(uuid4())
    client.store.return_value = memory
    client.add_turn.return_value = MagicMock(turn_index=1, needs_summary=False)
    return client


def _fake_result(content: str, role: str = "user", score: float = 0.5) -> MagicMock:
    r = MagicMock()
    r.memory_id = str(uuid4())
    r.content = content
    r.memory_type = "episodic"
    r.importance_score = 0.5
    r.relevance_score = score
    r.metadata = {"role": role}
    r.created_at = datetime.now(UTC)
    return r


# ---------------------------------------------------------------------------
# Z3rnoChatMemoryBuffer
# ---------------------------------------------------------------------------


def test_memory_get_round_trips_messages_in_chronological_order() -> None:
    # Server returns newest-first; adapter must reverse.
    client = _fake_client_with_recall(
        results=[
            _fake_result("third"),
            _fake_result("second", role="assistant"),
            _fake_result("first"),
        ]
    )
    mem = Z3rnoChatMemoryBuffer(client=client, agent_id="a-1")
    msgs = mem.get()
    assert [m.content for m in msgs] == ["first", "second", "third"]


def test_memory_put_stores_then_records_turn_when_conversation_set() -> None:
    client = _fake_client_with_recall()
    conv_id = str(uuid4())
    mem = Z3rnoChatMemoryBuffer(client=client, agent_id="a-1", conversation_id=conv_id)
    mem.put(ChatMessage(role=MessageRole.USER, content="hello"))
    client.store.assert_called_once()
    client.add_turn.assert_called_once()
    add_turn_kwargs = client.add_turn.call_args.kwargs
    assert add_turn_kwargs["turn_role"] == "user"


def test_memory_put_skips_add_turn_without_conversation_id() -> None:
    client = _fake_client_with_recall()
    mem = Z3rnoChatMemoryBuffer(client=client, agent_id="a-1")
    mem.put(ChatMessage(role=MessageRole.USER, content="hello"))
    client.store.assert_called_once()
    client.add_turn.assert_not_called()


def test_memory_recall_forwards_conversation_id() -> None:
    client = _fake_client_with_recall()
    conv_id = str(uuid4())
    mem = Z3rnoChatMemoryBuffer(client=client, agent_id="a-1", conversation_id=conv_id)
    mem.get()
    assert client.recall.call_args.kwargs["conversation_id"] == conv_id


def test_memory_reset_is_noop() -> None:
    """Phase G semantics: reset should NOT forget Z3rno rows."""
    client = _fake_client_with_recall()
    mem = Z3rnoChatMemoryBuffer(client=client, agent_id="a-1")
    mem.reset()
    client.forget.assert_not_called()


# ---------------------------------------------------------------------------
# Z3rnoBaseRetriever
# ---------------------------------------------------------------------------


def test_retriever_returns_nodes_with_scores() -> None:
    client = _fake_client_with_recall(
        results=[
            _fake_result("apple", score=0.9),
            _fake_result("banana", score=0.6),
        ]
    )
    retriever = Z3rnoBaseRetriever(client=client, agent_id="a-1", top_k=5)
    nodes = retriever._retrieve(QueryBundle(query_str="fruit"))
    assert len(nodes) == 2
    assert nodes[0].score == 0.9
    assert nodes[0].node.text == "apple"
    assert nodes[1].score == 0.6


def test_retriever_forwards_conversation_filter() -> None:
    client = _fake_client_with_recall()
    conv_id = str(uuid4())
    retriever = Z3rnoBaseRetriever(
        client=client,
        agent_id="a-1",
        conversation_id=conv_id,
        strategy="GRAPH",
    )
    retriever._retrieve(QueryBundle(query_str="x"))
    kwargs = client.recall.call_args.kwargs
    assert kwargs["conversation_id"] == conv_id
    assert kwargs["strategy"] == "GRAPH"
