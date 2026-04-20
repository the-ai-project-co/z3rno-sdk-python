"""CrewAI integration — Z3rno as a memory storage backend for CrewAI agents.

Usage::

    from z3rno import Z3rnoClient
    from z3rno.integrations.crewai import Z3rnoCrewAIStorage

    client = Z3rnoClient(base_url="http://localhost:8000", api_key="z3rno_sk_...")

    # Use as CrewAI short-term memory storage (maps to Z3rno 'working' type)
    short_term = Z3rnoCrewAIStorage(client, agent_id="agent-1", memory_type="working")

    # Use as CrewAI long-term memory storage (maps to Z3rno 'episodic' type)
    long_term = Z3rnoCrewAIStorage(client, agent_id="agent-1", memory_type="episodic")

    # Use as CrewAI entity memory storage (maps to Z3rno 'semantic' type)
    entity = Z3rnoCrewAIStorage(client, agent_id="agent-1", memory_type="semantic")
"""

from __future__ import annotations

from typing import Any

from z3rno.client import Z3rnoClient
from z3rno.models import MemoryType

# CrewAI memory type -> Z3rno memory type mapping
CREWAI_MEMORY_TYPE_MAP: dict[str, str] = {
    "short_term": MemoryType.WORKING,
    "long_term": MemoryType.EPISODIC,
    "entity": MemoryType.SEMANTIC,
}


class Z3rnoCrewAIStorage:
    """Storage adapter that implements CrewAI's storage interface backed by Z3rno.

    This class bridges CrewAI's memory system to Z3rno's memory API. CrewAI
    expects storage backends to expose ``save``, ``search``, and ``reset``
    methods. This adapter translates those calls into the corresponding
    Z3rno client operations (``store``, ``recall``, ``forget``).

    Args:
        client: An initialized :class:`~z3rno.client.Z3rnoClient`.
        agent_id: The Z3rno agent ID to scope all memory operations to.
        memory_type: The Z3rno memory type to use. Accepts a Z3rno
            :class:`~z3rno.models.MemoryType` value (``"working"``,
            ``"episodic"``, ``"semantic"``, ``"procedural"``) **or** a CrewAI
            memory category (``"short_term"``, ``"long_term"``, ``"entity"``).
            CrewAI categories are mapped automatically via
            :data:`CREWAI_MEMORY_TYPE_MAP`.
    """

    def __init__(
        self,
        client: Z3rnoClient,
        *,
        agent_id: str,
        memory_type: str = "episodic",
    ) -> None:
        self._client = client
        self._agent_id = agent_id
        # Resolve CrewAI category names to Z3rno memory types.
        self._memory_type = CREWAI_MEMORY_TYPE_MAP.get(memory_type, memory_type)

    # ------------------------------------------------------------------
    # CrewAI storage interface
    # ------------------------------------------------------------------

    def save(
        self,
        key: str,
        value: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store a memory in Z3rno.

        Args:
            key: A unique key for the memory. Stored in metadata so it can
                be used for later retrieval.
            value: The textual content of the memory.
            metadata: Optional additional metadata.

        Returns:
            A dict with the stored memory's ``id`` and ``key``.
        """
        merged_metadata: dict[str, Any] = {"crewai_key": key}
        if metadata:
            merged_metadata.update(metadata)

        memory = self._client.store(
            agent_id=self._agent_id,
            content=value,
            memory_type=self._memory_type,
            metadata=merged_metadata,
        )
        return {"id": memory.id, "key": key}

    def search(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Search Z3rno memories by semantic similarity.

        Args:
            query: The natural-language search query.
            limit: Maximum number of results to return.
            score_threshold: Minimum similarity score (0.0 -- 1.0).

        Returns:
            A list of dicts, each containing ``id``, ``content``,
            ``score``, and ``metadata`` keys.
        """
        response = self._client.recall(
            agent_id=self._agent_id,
            query=query,
            memory_type=self._memory_type,
            top_k=limit,
            similarity_threshold=score_threshold,
        )
        return [
            {
                "id": r.memory_id,
                "content": r.content,
                "score": r.similarity_score,
                "metadata": r.metadata,
            }
            for r in response.results
        ]

    def reset(self) -> None:
        """Delete all memories for this agent and memory type.

        Issues a recall to discover memory IDs, then forgets them all.
        This is a best-effort operation: it deletes up to 1000 memories
        per call. For very large datasets, call repeatedly.
        """
        response = self._client.recall(
            agent_id=self._agent_id,
            memory_type=self._memory_type,
            top_k=1000,
        )
        if not response.results:
            return
        memory_ids = [r.memory_id for r in response.results]
        self._client.forget(
            agent_id=self._agent_id,
            memory_ids=memory_ids,
        )
