"""Basic Z3rno SDK usage: store, recall, and forget workflow.

Prerequisites:
    pip install z3rno

    export Z3RNO_BASE_URL="http://localhost:8000"
    export Z3RNO_API_KEY="z3rno_sk_..."
"""

from z3rno import Z3rnoClient


def main() -> None:
    # The client reads Z3RNO_BASE_URL and Z3RNO_API_KEY from environment
    # variables automatically, or you can pass them explicitly.
    with Z3rnoClient() as client:
        # --- Store memories ---
        mem1 = client.store(
            agent_id="agent-1",
            content="User prefers dark mode and compact layout",
            memory_type="semantic",
            metadata={"source": "settings_page"},
        )
        print(f"Stored memory: {mem1.id}")

        mem2 = client.store(
            agent_id="agent-1",
            content="User asked about pricing plans on 2026-04-10",
            memory_type="episodic",
            importance=0.8,
        )
        print(f"Stored memory: {mem2.id}")

        # --- Recall memories ---
        results = client.recall(
            agent_id="agent-1",
            query="What are the user's UI preferences?",
            top_k=5,
        )
        print(f"\nRecalled {results.total} memories:")
        for r in results.results:
            print(f"  [{r.similarity_score:.2f}] {r.content}")

        # --- Update a memory ---
        updated = client.update_memory(
            mem1.id,
            content="User prefers dark mode, compact layout, and large fonts",
        )
        print(f"\nUpdated memory content: {updated.content}")

        # --- View memory history ---
        history = client.get_memory_history(mem1.id)
        print(f"\nMemory {mem1.id} has {history.total} version(s)")

        # --- Forget a memory ---
        forget_result = client.forget(agent_id="agent-1", memory_id=mem2.id)
        print(f"\nForgot {forget_result.deleted_count} memory/memories")

        # --- Batch store ---
        batch = client.store_batch(
            agent_id="agent-1",
            memories=[
                {"content": "Batch memory 1", "memory_type": "episodic"},
                {"content": "Batch memory 2", "memory_type": "semantic", "importance": 0.9},
            ],
        )
        print(f"\nBatch stored {batch.stored_count} memories")


if __name__ == "__main__":
    main()
