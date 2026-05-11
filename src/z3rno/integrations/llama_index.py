"""LlamaIndex integration for Z3rno.

Provides:

  * :class:`Z3rnoChatMemoryBuffer` — a ``BaseMemory`` impl that
    stores chat history in Z3rno (one Memo per message, optionally
    scoped to a :class:`Conversation` via ``conversation_id``).
  * :class:`Z3rnoBaseRetriever` — a ``BaseRetriever`` impl that
    surfaces Z3rno recall results as ``NodeWithScore`` objects for
    plug-in to a ``RetrieverQueryEngine``.

Requires the ``llama-index`` extra::

    pip install z3rno[llama-index]
"""

from __future__ import annotations

from typing import Any

try:
    from llama_index.core.base.llms.types import ChatMessage, MessageRole
    from llama_index.core.bridge.pydantic import Field, PrivateAttr
    from llama_index.core.memory.types import BaseMemory
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
except ImportError as _err:  # pragma: no cover — soft dep
    raise ImportError(
        "llama-index-core is required for Z3rno LlamaIndex integration. "
        "Install it with: pip install 'z3rno[llama-index]'"
    ) from _err

from z3rno.client import Z3rnoClient


class Z3rnoChatMemoryBuffer(BaseMemory):
    """LlamaIndex chat memory backed by Z3rno conversations.

    When ``conversation_id`` is supplied, messages are persisted as
    Z3rno Conversation turns (preserving order via ``turn_index``).
    Without it, the adapter falls back to plain Memo metadata tagged
    with ``role`` — useful for stateless agents.

    Usage::

        from llama_index.core import Settings
        from z3rno import Z3rnoClient
        from z3rno.integrations.llama_index import Z3rnoChatMemoryBuffer

        client = Z3rnoClient(...)
        conv = client.create_conversation(agent_id="agent-1")
        memory = Z3rnoChatMemoryBuffer(
            client=client, agent_id="agent-1", conversation_id=conv.id
        )
        Settings.memory = memory
    """

    agent_id: str = Field(description="Z3rno agent id")
    conversation_id: str | None = Field(default=None)
    top_k: int = Field(default=50)

    _client: Z3rnoClient = PrivateAttr()

    def __init__(self, *, client: Z3rnoClient, **data: Any) -> None:
        super().__init__(**data)
        self._client = client

    @classmethod
    def class_name(cls) -> str:
        return "Z3rnoChatMemoryBuffer"

    @classmethod
    def from_defaults(
        cls,
        *,
        client: Z3rnoClient,
        agent_id: str,
        conversation_id: str | None = None,
        top_k: int = 50,
    ) -> Z3rnoChatMemoryBuffer:
        return cls(
            client=client,
            agent_id=agent_id,
            conversation_id=conversation_id,
            top_k=top_k,
        )

    # --- read ---------------------------------------------------------

    def get(self, input: str | None = None, **_: Any) -> list[ChatMessage]:  # noqa: A002
        resp = self._client.recall(
            agent_id=self.agent_id,
            top_k=self.top_k,
            memory_type="episodic",
            conversation_id=self.conversation_id,
        )
        # Newest-first from the server — LlamaIndex wants oldest-first.
        results = list(reversed(resp.results))
        return [self._to_message(r) for r in results]

    def get_all(self) -> list[ChatMessage]:
        return self.get()

    # --- write --------------------------------------------------------

    def put(self, message: ChatMessage) -> None:
        role = self._role_str(message.role)
        memory = self._client.store(
            agent_id=self.agent_id,
            content=self._content_str(message.content),
            memory_type="episodic",
            metadata={"role": role},
        )
        if self.conversation_id:
            self._client.add_turn(
                self.conversation_id,
                memory_id=memory.id,
                turn_role=role if role in {"user", "assistant", "system", "tool"} else "user",
            )

    def set(self, messages: list[ChatMessage]) -> None:
        for msg in messages:
            self.put(msg)

    def reset(self) -> None:
        # No-op: forgetting all messages is destructive and not what most
        # callers expect. ``reset`` in LlamaIndex usually means "drop the
        # working buffer" — Z3rno already keeps full history, and recall
        # is scoped per-conversation. Leave the rows alone.
        return None

    # --- helpers ------------------------------------------------------

    @staticmethod
    def _content_str(content: Any) -> str:
        if isinstance(content, str):
            return content
        return str(content)

    @staticmethod
    def _role_str(role: MessageRole | str) -> str:
        if isinstance(role, MessageRole):
            return role.value.lower()
        return str(role).lower()

    @staticmethod
    def _to_message(result: Any) -> ChatMessage:
        role_str = (result.metadata or {}).get("role", "user")
        role = MessageRole.ASSISTANT if role_str == "assistant" else MessageRole.USER
        return ChatMessage(role=role, content=result.content)


class Z3rnoBaseRetriever(BaseRetriever):
    """LlamaIndex retriever backed by Z3rno recall.

    Usage::

        from llama_index.core.query_engine import RetrieverQueryEngine
        from z3rno.integrations.llama_index import Z3rnoBaseRetriever

        retriever = Z3rnoBaseRetriever(client=client, agent_id="agent-1")
        engine = RetrieverQueryEngine(retriever=retriever)
        response = engine.query("what does the user prefer?")
    """

    def __init__(
        self,
        *,
        client: Z3rnoClient,
        agent_id: str,
        top_k: int = 10,
        memory_type: str | None = None,
        conversation_id: str | None = None,
        similarity_threshold: float = 0.0,
        strategy: str = "AUTO",
    ) -> None:
        super().__init__()
        self._client = client
        self._agent_id = agent_id
        self._top_k = top_k
        self._memory_type = memory_type
        self._conversation_id = conversation_id
        self._similarity_threshold = similarity_threshold
        self._strategy = strategy

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        resp = self._client.recall(
            agent_id=self._agent_id,
            query=query_bundle.query_str,
            top_k=self._top_k,
            memory_type=self._memory_type,
            conversation_id=self._conversation_id,
            similarity_threshold=self._similarity_threshold,
            strategy=self._strategy,
        )
        out: list[NodeWithScore] = []
        for r in resp.results:
            node = TextNode(
                id_=r.memory_id,
                text=r.content,
                metadata={
                    **(r.metadata or {}),
                    "memory_type": r.memory_type,
                    "importance_score": r.importance_score,
                },
            )
            out.append(NodeWithScore(node=node, score=r.relevance_score))
        return out
