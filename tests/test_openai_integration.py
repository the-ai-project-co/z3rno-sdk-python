"""Tests for the OpenAI Agents SDK integration."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from z3rno.integrations.openai_agents import (
    FORGET_MEMORY_TOOL,
    RECALL_MEMORY_TOOL,
    STORE_MEMORY_TOOL,
    Z3rnoConversationMemory,
    get_memory_tools,
    handle_tool_call,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_memory(**overrides: Any) -> MagicMock:
    """Return a mock Memory object."""
    mem = MagicMock()
    mem.id = overrides.get("id", "mem-001")
    mem.agent_id = overrides.get("agent_id", "agent-1")
    mem.content = overrides.get("content", "test content")
    mem.memory_type = overrides.get("memory_type", "episodic")
    return mem


def _make_recall_result(**overrides: Any) -> MagicMock:
    """Return a mock RecallResult."""
    r = MagicMock()
    r.memory_id = overrides.get("memory_id", "mem-001")
    r.content = overrides.get("content", "recalled content")
    r.memory_type = overrides.get("memory_type", "episodic")
    r.similarity_score = overrides.get("similarity_score", 0.95)
    r.metadata = overrides.get("metadata", {"role": "user"})
    return r


def _make_forget_result(**overrides: Any) -> MagicMock:
    """Return a mock ForgetResult."""
    r = MagicMock()
    r.deleted_count = overrides.get("deleted_count", 1)
    r.hard_deleted = overrides.get("hard_deleted", False)
    return r


def _make_recall_response(results: list[Any] | None = None, total: int = 1) -> MagicMock:
    resp = MagicMock()
    resp.results = results or [_make_recall_result()]
    resp.total = total
    return resp


@pytest.fixture
def mock_client() -> MagicMock:
    """Return a mocked Z3rnoClient."""
    client = MagicMock()
    client.store.return_value = _make_memory()
    client.recall.return_value = _make_recall_response()
    client.forget.return_value = _make_forget_result()
    return client


# ---------------------------------------------------------------------------
# Tool definition tests
# ---------------------------------------------------------------------------


class TestGetMemoryTools:
    """Tests for get_memory_tools()."""

    def test_returns_three_tools(self) -> None:
        tools = get_memory_tools()
        assert len(tools) == 3

    def test_tool_names(self) -> None:
        tools = get_memory_tools()
        names = {t["function"]["name"] for t in tools}
        assert names == {"store_memory", "recall_memory", "forget_memory"}

    def test_all_tools_have_function_type(self) -> None:
        for tool in get_memory_tools():
            assert tool["type"] == "function"

    def test_store_tool_requires_content(self) -> None:
        params = STORE_MEMORY_TOOL["function"]["parameters"]
        assert "content" in params["required"]

    def test_recall_tool_requires_query(self) -> None:
        params = RECALL_MEMORY_TOOL["function"]["parameters"]
        assert "query" in params["required"]

    def test_forget_tool_requires_memory_id(self) -> None:
        params = FORGET_MEMORY_TOOL["function"]["parameters"]
        assert "memory_id" in params["required"]

    def test_store_tool_has_memory_type_enum(self) -> None:
        props = STORE_MEMORY_TOOL["function"]["parameters"]["properties"]
        assert "enum" in props["memory_type"]
        assert set(props["memory_type"]["enum"]) == {
            "working",
            "episodic",
            "semantic",
            "procedural",
        }


# ---------------------------------------------------------------------------
# handle_tool_call tests
# ---------------------------------------------------------------------------


class TestHandleToolCall:
    """Tests for handle_tool_call() dispatcher."""

    def test_store_memory(self, mock_client: MagicMock) -> None:
        result = handle_tool_call(
            mock_client,
            agent_id="agent-1",
            tool_name="store_memory",
            arguments={"content": "User likes Python"},
        )
        parsed = json.loads(result)
        assert parsed["status"] == "stored"
        assert parsed["memory_id"] == "mem-001"
        mock_client.store.assert_called_once_with(
            agent_id="agent-1",
            content="User likes Python",
            memory_type="episodic",
            metadata=None,
        )

    def test_store_memory_with_type_and_metadata(self, mock_client: MagicMock) -> None:
        handle_tool_call(
            mock_client,
            agent_id="agent-1",
            tool_name="store_memory",
            arguments={
                "content": "always use pytest",
                "memory_type": "procedural",
                "metadata": {"source": "user"},
            },
        )
        mock_client.store.assert_called_once_with(
            agent_id="agent-1",
            content="always use pytest",
            memory_type="procedural",
            metadata={"source": "user"},
        )

    def test_recall_memory(self, mock_client: MagicMock) -> None:
        result = handle_tool_call(
            mock_client,
            agent_id="agent-1",
            tool_name="recall_memory",
            arguments={"query": "user preferences"},
        )
        parsed = json.loads(result)
        assert parsed["total"] == 1
        assert len(parsed["results"]) == 1
        assert parsed["results"][0]["content"] == "recalled content"
        mock_client.recall.assert_called_once_with(
            agent_id="agent-1",
            query="user preferences",
            top_k=5,
            memory_type=None,
        )

    def test_recall_memory_with_options(self, mock_client: MagicMock) -> None:
        handle_tool_call(
            mock_client,
            agent_id="agent-1",
            tool_name="recall_memory",
            arguments={"query": "test", "top_k": 3, "memory_type": "semantic"},
        )
        mock_client.recall.assert_called_once_with(
            agent_id="agent-1",
            query="test",
            top_k=3,
            memory_type="semantic",
        )

    def test_forget_memory(self, mock_client: MagicMock) -> None:
        result = handle_tool_call(
            mock_client,
            agent_id="agent-1",
            tool_name="forget_memory",
            arguments={"memory_id": "mem-old"},
        )
        parsed = json.loads(result)
        assert parsed["status"] == "forgotten"
        assert parsed["deleted_count"] == 1
        mock_client.forget.assert_called_once_with(
            agent_id="agent-1",
            memory_id="mem-old",
            hard_delete=False,
        )

    def test_forget_memory_hard_delete(self, mock_client: MagicMock) -> None:
        handle_tool_call(
            mock_client,
            agent_id="agent-1",
            tool_name="forget_memory",
            arguments={"memory_id": "mem-old", "hard_delete": True},
        )
        mock_client.forget.assert_called_once_with(
            agent_id="agent-1",
            memory_id="mem-old",
            hard_delete=True,
        )

    def test_unknown_tool_raises(self, mock_client: MagicMock) -> None:
        with pytest.raises(ValueError, match="Unknown tool"):
            handle_tool_call(
                mock_client,
                agent_id="agent-1",
                tool_name="bad_tool",
                arguments={},
            )

    def test_arguments_as_json_string(self, mock_client: MagicMock) -> None:
        """handle_tool_call accepts a raw JSON string for arguments."""
        handle_tool_call(
            mock_client,
            agent_id="agent-1",
            tool_name="store_memory",
            arguments='{"content": "from string"}',
        )
        mock_client.store.assert_called_once()


# ---------------------------------------------------------------------------
# Z3rnoConversationMemory tests
# ---------------------------------------------------------------------------


class TestConversationMemory:
    """Tests for Z3rnoConversationMemory."""

    def test_add_message_stores_working_memory(self, mock_client: MagicMock) -> None:
        conv = Z3rnoConversationMemory(mock_client, agent_id="agent-1")
        mem_id = conv.add_message(role="user", content="Hello!")

        assert mem_id == "mem-001"
        mock_client.store.assert_called_once_with(
            agent_id="agent-1",
            content="Hello!",
            memory_type="working",
            user_id=None,
            metadata={"role": "user", "source": "conversation"},
        )

    def test_add_message_with_user_id(self, mock_client: MagicMock) -> None:
        conv = Z3rnoConversationMemory(mock_client, agent_id="agent-1", user_id="user-42")
        conv.add_message(role="user", content="Hi")
        mock_client.store.assert_called_once()
        call_kwargs = mock_client.store.call_args.kwargs
        assert call_kwargs["user_id"] == "user-42"

    def test_add_message_merges_metadata(self, mock_client: MagicMock) -> None:
        conv = Z3rnoConversationMemory(mock_client, agent_id="agent-1")
        conv.add_message(role="user", content="Hi", metadata={"turn": 1})
        call_kwargs = mock_client.store.call_args.kwargs
        assert call_kwargs["metadata"]["turn"] == 1
        assert call_kwargs["metadata"]["role"] == "user"

    def test_auto_store_disabled(self, mock_client: MagicMock) -> None:
        conv = Z3rnoConversationMemory(mock_client, agent_id="agent-1", auto_store=False)
        result = conv.add_message(role="user", content="Hi")
        assert result is None
        mock_client.store.assert_not_called()

    def test_stored_ids_tracks_messages(self, mock_client: MagicMock) -> None:
        # Return different IDs for successive calls
        mock_client.store.side_effect = [
            _make_memory(id="mem-001"),
            _make_memory(id="mem-002"),
        ]
        conv = Z3rnoConversationMemory(mock_client, agent_id="agent-1")
        conv.add_message(role="user", content="Hello")
        conv.add_message(role="assistant", content="Hi there")
        assert conv.stored_ids == ["mem-001", "mem-002"]

    def test_get_context(self, mock_client: MagicMock) -> None:
        conv = Z3rnoConversationMemory(mock_client, agent_id="agent-1")
        context = conv.get_context("preferences")

        assert len(context) == 1
        assert context[0]["role"] == "user"
        assert context[0]["content"] == "recalled content"
        assert context[0]["similarity_score"] == 0.95
        mock_client.recall.assert_called_once_with(
            agent_id="agent-1",
            query="preferences",
            top_k=10,
            memory_type="working",
        )

    def test_agent_id_property(self, mock_client: MagicMock) -> None:
        conv = Z3rnoConversationMemory(mock_client, agent_id="agent-1")
        assert conv.agent_id == "agent-1"
