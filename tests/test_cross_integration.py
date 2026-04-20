"""Cross-framework integration test: CrewAI and LangChain sharing Z3rno memory.

Demonstrates that Agent A (CrewAI style) can store memories via Z3rnoCrewAIStorage
and Agent B (LangChain style) can recall those same memories via Z3rnoRetriever,
verifying cross-framework memory sharing through a common Z3rno backend.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from z3rno.client import Z3rnoClient
from z3rno.integrations.crewai import Z3rnoCrewAIStorage
from z3rno.models import Memory, RecallResponse, RecallResult

# Guard: skip all tests if langchain-core is not installed.
pytest.importorskip("langchain_core")

from z3rno.integrations.langchain import Z3rnoRetriever


@pytest.mark.integration
class TestCrossFrameworkMemorySharing:
    """Verify that memories stored via CrewAI can be retrieved via LangChain."""

    def setup_method(self) -> None:
        """Set up a mock Z3rnoClient shared between frameworks."""
        self.client = MagicMock(spec=Z3rnoClient)
        self.agent_id = "shared-agent-001"
        self.stored_memories: list[dict] = []

        # Configure client.store to return a Memory and track what was stored
        def mock_store(**kwargs):
            memory = Memory(
                id=f"mem-{len(self.stored_memories) + 1}",
                agent_id=kwargs.get("agent_id", self.agent_id),
                content=kwargs.get("content", ""),
                memory_type=kwargs.get("memory_type", "episodic"),
                importance_score=0.8,
                recall_count=0,
                created_at=datetime.now(tz=timezone.utc),
                metadata=kwargs.get("metadata") or {},
            )
            self.stored_memories.append(
                {
                    "id": memory.id,
                    "content": memory.content,
                    "memory_type": memory.memory_type,
                    "metadata": memory.metadata,
                }
            )
            return memory

        self.client.store.side_effect = mock_store

    def _make_recall_response(self, query: str) -> RecallResponse:
        """Build a RecallResponse from stored memories (simulates server recall)."""
        results = [
            RecallResult(
                memory_id=m["id"],
                content=m["content"],
                memory_type=m["memory_type"],
                similarity_score=0.92 - i * 0.05,
                importance_score=0.8,
                relevance_score=0.85,
                recall_count=1,
                created_at=datetime.now(tz=timezone.utc),
                metadata=m["metadata"],
            )
            for i, m in enumerate(self.stored_memories)
        ]
        return RecallResponse(results=results, total=len(results), query=query)

    def test_crewai_stores_langchain_retrieves(self) -> None:
        """CrewAI stores memories, LangChain retriever finds them."""
        # --- Agent A: CrewAI stores memories ---
        crewai_storage = Z3rnoCrewAIStorage(
            self.client,
            agent_id=self.agent_id,
            memory_type="episodic",
        )

        crewai_storage.save(
            key="user-preference-1",
            value="User prefers dark mode in all applications",
            metadata={"source": "crewai-agent-a"},
        )
        crewai_storage.save(
            key="user-preference-2",
            value="User's timezone is America/New_York",
            metadata={"source": "crewai-agent-a"},
        )

        # Verify store was called twice
        assert self.client.store.call_count == 2
        assert len(self.stored_memories) == 2

        # --- Agent B: LangChain retrieves those same memories ---
        # Configure recall to return stored memories
        self.client.recall.return_value = self._make_recall_response("user preferences")

        # Use LangChain retriever with the same client and agent_id
        retriever = Z3rnoRetriever(
            client=self.client,
            agent_id=self.agent_id,
            top_k=10,
        )

        # Simulate LangChain's _get_relevant_documents call
        run_manager = MagicMock()
        docs = retriever._get_relevant_documents("user preferences", run_manager=run_manager)

        # Verify LangChain retrieved the memories that CrewAI stored
        assert len(docs) == 2
        assert "dark mode" in docs[0].page_content
        assert "timezone" in docs[1].page_content

        # Verify metadata round-trips correctly
        assert docs[0].metadata["memory_id"] == "mem-1"
        assert docs[1].metadata["memory_id"] == "mem-2"

    def test_shared_memory_preserves_metadata(self) -> None:
        """Metadata added by CrewAI is visible to LangChain retriever."""
        crewai_storage = Z3rnoCrewAIStorage(
            self.client,
            agent_id=self.agent_id,
            memory_type="semantic",
        )

        crewai_storage.save(
            key="fact-1",
            value="The project deadline is March 15, 2026",
            metadata={"category": "project-info", "priority": "high"},
        )

        # Recall returns the stored memory with its metadata
        self.client.recall.return_value = self._make_recall_response("deadline")

        retriever = Z3rnoRetriever(
            client=self.client,
            agent_id=self.agent_id,
        )

        run_manager = MagicMock()
        docs = retriever._get_relevant_documents("deadline", run_manager=run_manager)

        assert len(docs) == 1
        # CrewAI's crewai_key metadata should be present
        assert docs[0].metadata.get("crewai_key") == "fact-1"
        # Custom metadata should also be preserved
        assert docs[0].metadata.get("category") == "project-info"
        assert docs[0].metadata.get("priority") == "high"

    def test_same_client_same_agent_id_ensures_sharing(self) -> None:
        """Both frameworks using same client + agent_id share the memory store."""
        crewai_storage = Z3rnoCrewAIStorage(
            self.client,
            agent_id=self.agent_id,
            memory_type="working",
        )

        # Store via CrewAI
        crewai_storage.save(key="ctx-1", value="Current task is code review")

        # The store call used the shared agent_id
        store_call = self.client.store.call_args
        assert store_call.kwargs["agent_id"] == self.agent_id

        # When LangChain recalls, it uses the same agent_id
        self.client.recall.return_value = self._make_recall_response("current task")

        retriever = Z3rnoRetriever(
            client=self.client,
            agent_id=self.agent_id,
        )

        run_manager = MagicMock()
        retriever._get_relevant_documents("current task", run_manager=run_manager)

        # Verify recall was called with the same agent_id
        recall_call = self.client.recall.call_args
        assert recall_call.kwargs["agent_id"] == self.agent_id
