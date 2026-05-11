"""LangChain integration for Z3rno.

Provides :class:`Z3rnoChatMessageHistory` (conversation memory) and
:class:`Z3rnoRetriever` (RAG retrieval) that wrap the Z3rno client as
thin adapters.

Requires the ``langchain`` extra::

    pip install z3rno[langchain]
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import ConfigDict, SkipValidation

try:
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.documents import Document
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
    from langchain_core.retrievers import BaseRetriever
except ImportError as _err:
    raise ImportError(
        "langchain-core is required for Z3rno LangChain integration. "
        "Install it with: pip install z3rno[langchain]"
    ) from _err

from z3rno.client import Z3rnoClient


class Z3rnoChatMessageHistory(BaseChatMessageHistory):
    """LangChain chat message history backed by Z3rno.

    Stores conversation messages via ``client.store()`` and retrieves them
    via ``client.recall()``.

    Usage::

        from z3rno import Z3rnoClient
        from z3rno.integrations.langchain import Z3rnoChatMessageHistory

        client = Z3rnoClient(base_url="...", api_key="...")
        history = Z3rnoChatMessageHistory(client=client, agent_id="agent-1")

        # Use with RunnableWithMessageHistory or directly
        history.add_user_message("What is Z3rno?")
        history.add_ai_message("Z3rno is a memory database for AI agents.")
        print(history.messages)
    """

    def __init__(
        self,
        client: Z3rnoClient,
        agent_id: str,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        top_k: int = 50,
        # Phase G slice 2 — when set, recall is scoped to this Z3rno
        # Conversation and new messages are recorded as turns. Falls
        # back to the legacy ``session_id`` metadata path when None.
        conversation_id: str | None = None,
    ) -> None:
        self.client = client
        self.agent_id = agent_id
        self.session_id = session_id
        self.user_id = user_id
        self.top_k = top_k
        self.conversation_id = conversation_id

    @property
    def messages(self) -> list[BaseMessage]:  # type: ignore[override]
        """Retrieve stored messages from Z3rno."""
        filters: dict[str, Any] | None = None
        if self.session_id:
            filters = {"session_id": self.session_id}

        response = self.client.recall(
            agent_id=self.agent_id,
            top_k=self.top_k,
            filters=filters,
            memory_type="episodic",
            conversation_id=self.conversation_id,
        )

        msgs: list[BaseMessage] = []
        for result in response.results:
            role = result.metadata.get("role", "human")
            if role == "ai":
                msgs.append(AIMessage(content=result.content))
            else:
                msgs.append(HumanMessage(content=result.content))
        return msgs

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Store messages in Z3rno."""
        for message in messages:
            role = "ai" if isinstance(message, AIMessage) else "human"
            metadata: dict[str, Any] = {"role": role}
            if self.session_id:
                metadata["session_id"] = self.session_id

            content = message.content if isinstance(message.content, str) else str(message.content)

            memory = self.client.store(
                agent_id=self.agent_id,
                content=content,
                memory_type="episodic",
                user_id=self.user_id,
                metadata=metadata,
            )
            if self.conversation_id:
                self.client.add_turn(
                    self.conversation_id,
                    memory_id=memory.id,
                    turn_role="assistant" if role == "ai" else "user",
                )

    def clear(self) -> None:
        """Forget all memories for this agent."""
        self.client.forget(agent_id=self.agent_id)


class Z3rnoRetriever(BaseRetriever):
    """LangChain retriever backed by Z3rno vector search.

    Returns :class:`~langchain_core.documents.Document` objects from Z3rno
    recall results, suitable for use in RAG chains.

    Usage::

        from z3rno import Z3rnoClient
        from z3rno.integrations.langchain import Z3rnoRetriever

        client = Z3rnoClient(base_url="...", api_key="...")
        retriever = Z3rnoRetriever(client=client, agent_id="agent-1")

        docs = retriever.invoke("user preferences")
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: SkipValidation[Z3rnoClient]
    agent_id: str
    top_k: int = 10
    memory_type: str | None = None
    filters: dict[str, Any] | None = None
    similarity_threshold: float = 0.0

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Retrieve documents from Z3rno matching the query."""
        response = self.client.recall(
            agent_id=self.agent_id,
            query=query,
            top_k=self.top_k,
            memory_type=self.memory_type,
            filters=self.filters,
            similarity_threshold=self.similarity_threshold,
        )

        return [
            Document(
                page_content=result.content,
                metadata={
                    "memory_id": result.memory_id,
                    "memory_type": result.memory_type,
                    "similarity_score": result.similarity_score,
                    "importance_score": result.importance_score,
                    "relevance_score": result.relevance_score,
                    "recall_count": result.recall_count,
                    "created_at": result.created_at.isoformat(),
                    **result.metadata,
                },
            )
            for result in response.results
        ]


__all__ = ["Z3rnoChatMessageHistory", "Z3rnoRetriever"]
