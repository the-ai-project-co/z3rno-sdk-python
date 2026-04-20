"""Tests for Z3rno LangChain integration using mocked client."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from z3rno.models import ForgetResult, Memory, RecallResponse, RecallResult

# Guard: skip all tests if langchain-core is not installed.
langchain_core = pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from z3rno.integrations.langchain import (  # noqa: E402
    Z3rnoChatMessageHistory,
    Z3rnoRetriever,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_recall_result(
    memory_id: str = "mem-1",
    content: str = "User prefers dark mode",
    **kwargs: object,
) -> RecallResult:
    defaults: dict[str, object] = {
        "memory_id": memory_id,
        "content": content,
        "memory_type": "episodic",
        "similarity_score": 0.92,
        "importance_score": 0.5,
        "relevance_score": 0.88,
        "recall_count": 1,
        "created_at": datetime(2026, 4, 12, tzinfo=timezone.utc),
        "metadata": {},
    }
    defaults.update(kwargs)
    return RecallResult(**defaults)


def _make_recall_response(results: list[RecallResult] | None = None) -> RecallResponse:
    if results is None:
        results = [_make_recall_result()]
    return RecallResponse(results=results, total=len(results), query="test")


def _make_memory(memory_id: str = "mem-stored") -> Memory:
    return Memory(
        id=memory_id,
        agent_id="agent-1",
        content="stored content",
        memory_type="episodic",
        importance_score=0.5,
        recall_count=0,
        created_at=datetime(2026, 4, 12, tzinfo=timezone.utc),
    )


@pytest.fixture
def mock_client() -> MagicMock:
    client = MagicMock()
    client.recall.return_value = _make_recall_response()
    client.store.return_value = _make_memory()
    client.forget.return_value = ForgetResult(
        deleted_count=1, hard_deleted=False, cascade_count=0, memory_ids=["mem-1"]
    )
    return client


# ---------------------------------------------------------------------------
# Z3rnoChatMessageHistory tests
# ---------------------------------------------------------------------------


class TestZ3rnoChatMessageHistory:
    def test_messages_returns_human_by_default(self, mock_client: MagicMock) -> None:
        """Messages without a role in metadata default to HumanMessage."""
        history = Z3rnoChatMessageHistory(client=mock_client, agent_id="agent-1")
        msgs = history.messages

        assert len(msgs) == 1
        assert isinstance(msgs[0], HumanMessage)
        assert msgs[0].content == "User prefers dark mode"

    def test_messages_returns_ai_message(self, mock_client: MagicMock) -> None:
        """Messages with role=ai in metadata are returned as AIMessage."""
        results = [_make_recall_result(metadata={"role": "ai"})]
        mock_client.recall.return_value = _make_recall_response(results)

        history = Z3rnoChatMessageHistory(client=mock_client, agent_id="agent-1")
        msgs = history.messages

        assert len(msgs) == 1
        assert isinstance(msgs[0], AIMessage)

    def test_messages_calls_recall_with_correct_params(self, mock_client: MagicMock) -> None:
        history = Z3rnoChatMessageHistory(client=mock_client, agent_id="agent-1")
        history.messages  # noqa: B018 (access property for side effect)

        mock_client.recall.assert_called_once_with(
            agent_id="agent-1",
            top_k=50,
            filters=None,
            memory_type="episodic",
        )

    def test_messages_with_session_filter(self, mock_client: MagicMock) -> None:
        history = Z3rnoChatMessageHistory(
            client=mock_client, agent_id="agent-1", session_id="sess-42"
        )
        history.messages  # noqa: B018

        mock_client.recall.assert_called_once_with(
            agent_id="agent-1",
            top_k=50,
            filters={"session_id": "sess-42"},
            memory_type="episodic",
        )

    def test_messages_empty(self, mock_client: MagicMock) -> None:
        mock_client.recall.return_value = _make_recall_response(results=[])

        history = Z3rnoChatMessageHistory(client=mock_client, agent_id="agent-1")
        assert history.messages == []

    def test_messages_mixed_roles(self, mock_client: MagicMock) -> None:
        results = [
            _make_recall_result(memory_id="m1", content="Hello", metadata={"role": "human"}),
            _make_recall_result(memory_id="m2", content="Hi there!", metadata={"role": "ai"}),
        ]
        mock_client.recall.return_value = _make_recall_response(results)

        history = Z3rnoChatMessageHistory(client=mock_client, agent_id="agent-1")
        msgs = history.messages

        assert len(msgs) == 2
        assert isinstance(msgs[0], HumanMessage)
        assert isinstance(msgs[1], AIMessage)

    def test_add_messages_single_human(self, mock_client: MagicMock) -> None:
        history = Z3rnoChatMessageHistory(client=mock_client, agent_id="agent-1")
        history.add_messages([HumanMessage(content="What is Z3rno?")])

        mock_client.store.assert_called_once_with(
            agent_id="agent-1",
            content="What is Z3rno?",
            memory_type="episodic",
            user_id=None,
            metadata={"role": "human"},
        )

    def test_add_messages_single_ai(self, mock_client: MagicMock) -> None:
        history = Z3rnoChatMessageHistory(client=mock_client, agent_id="agent-1")
        history.add_messages([AIMessage(content="Z3rno is a memory database.")])

        mock_client.store.assert_called_once_with(
            agent_id="agent-1",
            content="Z3rno is a memory database.",
            memory_type="episodic",
            user_id=None,
            metadata={"role": "ai"},
        )

    def test_add_messages_multiple(self, mock_client: MagicMock) -> None:
        history = Z3rnoChatMessageHistory(client=mock_client, agent_id="agent-1")
        history.add_messages(
            [
                HumanMessage(content="Hi"),
                AIMessage(content="Hello!"),
            ]
        )

        assert mock_client.store.call_count == 2
        mock_client.store.assert_any_call(
            agent_id="agent-1",
            content="Hi",
            memory_type="episodic",
            user_id=None,
            metadata={"role": "human"},
        )
        mock_client.store.assert_any_call(
            agent_id="agent-1",
            content="Hello!",
            memory_type="episodic",
            user_id=None,
            metadata={"role": "ai"},
        )

    def test_add_messages_with_session_and_user(self, mock_client: MagicMock) -> None:
        history = Z3rnoChatMessageHistory(
            client=mock_client,
            agent_id="agent-1",
            session_id="sess-42",
            user_id="user-7",
        )
        history.add_messages([HumanMessage(content="test")])

        mock_client.store.assert_called_once_with(
            agent_id="agent-1",
            content="test",
            memory_type="episodic",
            user_id="user-7",
            metadata={"role": "human", "session_id": "sess-42"},
        )

    def test_add_user_message_convenience(self, mock_client: MagicMock) -> None:
        history = Z3rnoChatMessageHistory(client=mock_client, agent_id="agent-1")
        history.add_user_message("hello")

        mock_client.store.assert_called_once()
        stored_metadata = mock_client.store.call_args.kwargs["metadata"]
        assert stored_metadata["role"] == "human"

    def test_add_ai_message_convenience(self, mock_client: MagicMock) -> None:
        history = Z3rnoChatMessageHistory(client=mock_client, agent_id="agent-1")
        history.add_ai_message("hello back")

        mock_client.store.assert_called_once()
        stored_metadata = mock_client.store.call_args.kwargs["metadata"]
        assert stored_metadata["role"] == "ai"

    def test_clear(self, mock_client: MagicMock) -> None:
        history = Z3rnoChatMessageHistory(client=mock_client, agent_id="agent-1")
        history.clear()

        mock_client.forget.assert_called_once_with(agent_id="agent-1")

    def test_custom_top_k(self, mock_client: MagicMock) -> None:
        history = Z3rnoChatMessageHistory(client=mock_client, agent_id="agent-1", top_k=10)
        history.messages  # noqa: B018

        mock_client.recall.assert_called_once_with(
            agent_id="agent-1",
            top_k=10,
            filters=None,
            memory_type="episodic",
        )


# ---------------------------------------------------------------------------
# Z3rnoRetriever tests
# ---------------------------------------------------------------------------


class TestZ3rnoRetriever:
    def test_get_relevant_documents(self, mock_client: MagicMock) -> None:
        retriever = Z3rnoRetriever(client=mock_client, agent_id="agent-1")
        docs = retriever.invoke("user preferences")

        assert len(docs) == 1
        assert docs[0].page_content == "User prefers dark mode"
        assert docs[0].metadata["memory_id"] == "mem-1"
        assert docs[0].metadata["similarity_score"] == 0.92
        assert docs[0].metadata["importance_score"] == 0.5
        assert docs[0].metadata["created_at"] == "2026-04-12T00:00:00+00:00"

        mock_client.recall.assert_called_once_with(
            agent_id="agent-1",
            query="user preferences",
            top_k=10,
            memory_type=None,
            filters=None,
            similarity_threshold=0.0,
        )

    def test_get_relevant_documents_with_filters(self, mock_client: MagicMock) -> None:
        retriever = Z3rnoRetriever(
            client=mock_client,
            agent_id="agent-1",
            filters={"category": "preferences"},
            memory_type="semantic",
            similarity_threshold=0.5,
            top_k=3,
        )
        retriever.invoke("dark mode")

        mock_client.recall.assert_called_once_with(
            agent_id="agent-1",
            query="dark mode",
            top_k=3,
            memory_type="semantic",
            filters={"category": "preferences"},
            similarity_threshold=0.5,
        )

    def test_get_relevant_documents_empty_results(self, mock_client: MagicMock) -> None:
        mock_client.recall.return_value = _make_recall_response(results=[])
        retriever = Z3rnoRetriever(client=mock_client, agent_id="agent-1")
        docs = retriever.invoke("nonexistent topic")

        assert docs == []

    def test_get_relevant_documents_multiple_results(self, mock_client: MagicMock) -> None:
        results = [
            _make_recall_result(memory_id="mem-1", content="Dark mode preferred"),
            _make_recall_result(
                memory_id="mem-2",
                content="Uses vim keybindings",
                similarity_score=0.85,
                metadata={"source": "onboarding"},
            ),
        ]
        mock_client.recall.return_value = _make_recall_response(results)

        retriever = Z3rnoRetriever(client=mock_client, agent_id="agent-1")
        docs = retriever.invoke("user settings")

        assert len(docs) == 2
        assert docs[0].page_content == "Dark mode preferred"
        assert docs[1].page_content == "Uses vim keybindings"
        assert docs[1].metadata["source"] == "onboarding"

    def test_metadata_passthrough(self, mock_client: MagicMock) -> None:
        """Custom metadata from Z3rno is merged into Document metadata."""
        results = [
            _make_recall_result(
                metadata={"tag": "ui", "priority": "high"},
            ),
        ]
        mock_client.recall.return_value = _make_recall_response(results)

        retriever = Z3rnoRetriever(client=mock_client, agent_id="agent-1")
        docs = retriever.invoke("query")

        assert docs[0].metadata["tag"] == "ui"
        assert docs[0].metadata["priority"] == "high"
        # Standard Z3rno fields are still present
        assert "memory_id" in docs[0].metadata
        assert "relevance_score" in docs[0].metadata
