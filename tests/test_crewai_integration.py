"""Tests for the CrewAI integration adapter using a mocked Z3rnoClient."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from z3rno.integrations.crewai import CREWAI_MEMORY_TYPE_MAP, Z3rnoCrewAIStorage

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_client() -> MagicMock:
    """Return a mocked Z3rnoClient."""
    return MagicMock()


@pytest.fixture
def storage(mock_client: MagicMock) -> Z3rnoCrewAIStorage:
    """Return a Z3rnoCrewAIStorage with the default episodic type."""
    return Z3rnoCrewAIStorage(mock_client, agent_id="agent-1", memory_type="episodic")


# ---------------------------------------------------------------------------
# Memory type mapping
# ---------------------------------------------------------------------------


class TestMemoryTypeMapping:
    def test_short_term_maps_to_working(self, mock_client: MagicMock) -> None:
        s = Z3rnoCrewAIStorage(mock_client, agent_id="a", memory_type="short_term")
        assert s._memory_type == "working"

    def test_long_term_maps_to_episodic(self, mock_client: MagicMock) -> None:
        s = Z3rnoCrewAIStorage(mock_client, agent_id="a", memory_type="long_term")
        assert s._memory_type == "episodic"

    def test_entity_maps_to_semantic(self, mock_client: MagicMock) -> None:
        s = Z3rnoCrewAIStorage(mock_client, agent_id="a", memory_type="entity")
        assert s._memory_type == "semantic"

    def test_native_type_passed_through(self, mock_client: MagicMock) -> None:
        s = Z3rnoCrewAIStorage(mock_client, agent_id="a", memory_type="procedural")
        assert s._memory_type == "procedural"

    def test_default_memory_type_is_episodic(self, mock_client: MagicMock) -> None:
        s = Z3rnoCrewAIStorage(mock_client, agent_id="a")
        assert s._memory_type == "episodic"

    def test_map_contains_expected_keys(self) -> None:
        assert set(CREWAI_MEMORY_TYPE_MAP) == {"short_term", "long_term", "entity"}


# ---------------------------------------------------------------------------
# save()
# ---------------------------------------------------------------------------


class TestSave:
    def test_save_calls_store(self, storage: Z3rnoCrewAIStorage, mock_client: MagicMock) -> None:
        mock_client.store.return_value = MagicMock(id="mem-1")
        result = storage.save("k1", "hello world")

        mock_client.store.assert_called_once_with(
            agent_id="agent-1",
            content="hello world",
            memory_type="episodic",
            metadata={"crewai_key": "k1"},
        )
        assert result == {"id": "mem-1", "key": "k1"}

    def test_save_merges_metadata(
        self, storage: Z3rnoCrewAIStorage, mock_client: MagicMock
    ) -> None:
        mock_client.store.return_value = MagicMock(id="mem-2")
        storage.save("k2", "content", metadata={"entity_name": "Alice"})

        call_kwargs = mock_client.store.call_args.kwargs
        assert call_kwargs["metadata"] == {
            "crewai_key": "k2",
            "entity_name": "Alice",
        }

    def test_save_with_no_metadata(
        self, storage: Z3rnoCrewAIStorage, mock_client: MagicMock
    ) -> None:
        mock_client.store.return_value = MagicMock(id="mem-3")
        result = storage.save("k3", "value")

        assert result["id"] == "mem-3"
        call_kwargs = mock_client.store.call_args.kwargs
        assert call_kwargs["metadata"] == {"crewai_key": "k3"}


# ---------------------------------------------------------------------------
# search()
# ---------------------------------------------------------------------------


class TestSearch:
    def _make_recall_result(
        self, memory_id: str, content: str, score: float, metadata: dict[str, Any] | None = None
    ) -> MagicMock:
        r = MagicMock()
        r.memory_id = memory_id
        r.content = content
        r.similarity_score = score
        r.metadata = metadata or {}
        return r

    def test_search_calls_recall(self, storage: Z3rnoCrewAIStorage, mock_client: MagicMock) -> None:
        mock_client.recall.return_value = MagicMock(results=[])
        storage.search("query")

        mock_client.recall.assert_called_once_with(
            agent_id="agent-1",
            query="query",
            memory_type="episodic",
            top_k=10,
            similarity_threshold=0.0,
        )

    def test_search_returns_formatted_results(
        self, storage: Z3rnoCrewAIStorage, mock_client: MagicMock
    ) -> None:
        r1 = self._make_recall_result("m1", "foo", 0.9, {"crewai_key": "k1"})
        r2 = self._make_recall_result("m2", "bar", 0.7)
        mock_client.recall.return_value = MagicMock(results=[r1, r2])

        results = storage.search("test query")
        assert len(results) == 2
        assert results[0] == {
            "id": "m1",
            "content": "foo",
            "score": 0.9,
            "metadata": {"crewai_key": "k1"},
        }
        assert results[1]["id"] == "m2"

    def test_search_custom_limit_and_threshold(
        self, storage: Z3rnoCrewAIStorage, mock_client: MagicMock
    ) -> None:
        mock_client.recall.return_value = MagicMock(results=[])
        storage.search("q", limit=5, score_threshold=0.5)

        call_kwargs = mock_client.recall.call_args.kwargs
        assert call_kwargs["top_k"] == 5
        assert call_kwargs["similarity_threshold"] == 0.5

    def test_search_empty_results(
        self, storage: Z3rnoCrewAIStorage, mock_client: MagicMock
    ) -> None:
        mock_client.recall.return_value = MagicMock(results=[])
        assert storage.search("nothing") == []


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_deletes_all_memories(
        self, storage: Z3rnoCrewAIStorage, mock_client: MagicMock
    ) -> None:
        r1 = MagicMock(memory_id="m1")
        r2 = MagicMock(memory_id="m2")
        mock_client.recall.return_value = MagicMock(results=[r1, r2])

        storage.reset()

        mock_client.forget.assert_called_once_with(
            agent_id="agent-1",
            memory_ids=["m1", "m2"],
        )

    def test_reset_noop_when_no_memories(
        self, storage: Z3rnoCrewAIStorage, mock_client: MagicMock
    ) -> None:
        mock_client.recall.return_value = MagicMock(results=[])
        storage.reset()
        mock_client.forget.assert_not_called()

    def test_reset_uses_correct_memory_type(self, mock_client: MagicMock) -> None:
        s = Z3rnoCrewAIStorage(mock_client, agent_id="a", memory_type="short_term")
        mock_client.recall.return_value = MagicMock(results=[])
        s.reset()

        call_kwargs = mock_client.recall.call_args.kwargs
        assert call_kwargs["memory_type"] == "working"
