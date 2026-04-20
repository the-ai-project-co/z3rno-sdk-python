"""CrewAI memory backend using Z3rno.

This example shows how to use Z3rno as a shared memory layer for CrewAI
agents. Each agent stores observations and can recall context from the
shared memory pool.

Prerequisites:
    pip install z3rno crewai

    export Z3RNO_BASE_URL="http://localhost:8000"
    export Z3RNO_API_KEY="z3rno_sk_..."
"""

from __future__ import annotations

from z3rno import Z3rnoClient


class Z3rnoCrewMemory:
    """Shared memory backend for CrewAI agents, powered by Z3rno.

    All agents in the crew share the same Z3rno agent_id, so memories
    stored by one agent are visible to all others.  Use ``memory_type``
    to separate categories:

    - ``"episodic"`` for task-specific observations
    - ``"semantic"`` for long-lived knowledge
    - ``"procedural"`` for learned procedures
    """

    def __init__(
        self,
        crew_id: str,
        *,
        client: Z3rnoClient | None = None,
    ) -> None:
        self.crew_id = crew_id
        self._client = client or Z3rnoClient()

    # -- Store an observation ------------------------------------------------

    def store(
        self,
        agent_role: str,
        content: str,
        *,
        memory_type: str = "episodic",
        importance: float | None = None,
    ) -> str:
        """Store a memory from a specific agent role.

        Returns the memory ID.
        """
        mem = self._client.store(
            agent_id=self.crew_id,
            content=content,
            memory_type=memory_type,
            metadata={"agent_role": agent_role, "source": "crewai"},
            importance=importance,
        )
        return mem.id

    # -- Recall context for a task -------------------------------------------

    def recall(
        self,
        query: str,
        *,
        top_k: int = 5,
        memory_type: str | None = None,
    ) -> list[str]:
        """Recall relevant memories as plain-text strings.

        This is intentionally simple so it can be plugged into a
        CrewAI agent's context window.
        """
        response = self._client.recall(
            agent_id=self.crew_id,
            query=query,
            top_k=top_k,
            memory_type=memory_type,
        )
        return [r.content for r in response.results]

    # -- Forget everything ---------------------------------------------------

    def clear(self) -> int:
        """Forget all memories for this crew. Returns count deleted."""
        # Recall all, then forget
        response = self._client.recall(
            agent_id=self.crew_id,
            top_k=1000,
        )
        if not response.results:
            return 0
        ids = [r.memory_id for r in response.results]
        result = self._client.forget(
            agent_id=self.crew_id,
            memory_ids=ids,
        )
        return result.deleted_count


# ---------------------------------------------------------------------------
# Example: two CrewAI agents sharing memory
# ---------------------------------------------------------------------------


def main() -> None:
    client = Z3rnoClient()
    memory = Z3rnoCrewMemory(crew_id="research-crew", client=client)

    # Researcher agent stores findings
    memory.store(
        agent_role="researcher",
        content="Z3rno supports four memory types: working, episodic, semantic, procedural.",
        memory_type="semantic",
        importance=0.9,
    )
    memory.store(
        agent_role="researcher",
        content="The Z3rno API uses Bearer token authentication.",
        memory_type="semantic",
    )
    memory.store(
        agent_role="researcher",
        content="Searched documentation site and found quickstart guide.",
        memory_type="episodic",
    )

    # Writer agent recalls context for its task
    context = memory.recall("What memory types does Z3rno support?", top_k=3)
    print("Writer agent received context:")
    for i, c in enumerate(context, 1):
        print(f"  {i}. {c}")

    # Only recall semantic memories
    semantic_context = memory.recall(
        "authentication",
        top_k=3,
        memory_type="semantic",
    )
    print("\nSemantic-only recall:")
    for i, c in enumerate(semantic_context, 1):
        print(f"  {i}. {c}")

    # Clean up
    deleted = memory.clear()
    print(f"\nCleared {deleted} memories")
    client.close()


if __name__ == "__main__":
    main()
