"""LangChain memory provider backed by Z3rno.

This example shows how to create a custom ChatMessageHistory that stores
and recalls conversation messages using Z3rno as the persistent backend.

Prerequisites:
    pip install z3rno langchain-core

    export Z3RNO_BASE_URL="http://localhost:8000"
    export Z3RNO_API_KEY="z3rno_sk_..."
"""

from __future__ import annotations

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from z3rno import Z3rnoClient


class Z3rnoChatMessageHistory(BaseChatMessageHistory):
    """A LangChain ChatMessageHistory backed by Z3rno memory storage.

    Each message is stored as an episodic memory with metadata indicating
    the role (human / ai) and the session id. Recall uses semantic search
    so the most relevant messages are returned first.
    """

    def __init__(
        self,
        agent_id: str,
        session_id: str,
        *,
        client: Z3rnoClient | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.session_id = session_id
        self._client = client or Z3rnoClient()
        self._memory_ids: list[str] = []

    # -- Required interface --------------------------------------------------

    @property
    def messages(self) -> list[BaseMessage]:  # type: ignore[override]
        """Retrieve all messages for this session from Z3rno."""
        response = self._client.recall(
            agent_id=self.agent_id,
            query=f"session:{self.session_id}",
            top_k=100,
        )
        msgs: list[BaseMessage] = []
        for r in response.results:
            role = r.metadata.get("role", "human")
            if role == "ai":
                msgs.append(AIMessage(content=r.content))
            else:
                msgs.append(HumanMessage(content=r.content))
        return msgs

    def add_message(self, message: BaseMessage) -> None:
        """Store a single message in Z3rno."""
        role = "ai" if isinstance(message, AIMessage) else "human"
        mem = self._client.store(
            agent_id=self.agent_id,
            content=message.content if isinstance(message.content, str) else str(message.content),
            memory_type="episodic",
            metadata={
                "role": role,
                "session_id": self.session_id,
                "source": "langchain",
            },
        )
        self._memory_ids.append(mem.id)

    def clear(self) -> None:
        """Delete all messages for this session."""
        if self._memory_ids:
            self._client.forget(
                agent_id=self.agent_id,
                memory_ids=self._memory_ids,
            )
            self._memory_ids.clear()


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------


def main() -> None:
    client = Z3rnoClient()
    history = Z3rnoChatMessageHistory(
        agent_id="langchain-agent",
        session_id="session-001",
        client=client,
    )

    # Simulate a conversation
    history.add_message(HumanMessage(content="What is Z3rno?"))
    history.add_message(AIMessage(content="Z3rno is an AI agent memory database."))
    history.add_message(HumanMessage(content="How do I store memories?"))
    history.add_message(
        AIMessage(content="Use client.store(agent_id=..., content=...) to store memories.")
    )

    # Retrieve the conversation
    print("Conversation history:")
    for msg in history.messages:
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"  {role}: {msg.content}")

    # Clean up
    history.clear()
    client.close()


if __name__ == "__main__":
    main()
