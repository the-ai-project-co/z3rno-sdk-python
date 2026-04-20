"""OpenAI Agents SDK integration for Z3rno.

Provides memory as OpenAI function tools so that agents can store, recall,
and forget memories via standard tool-calling.

Usage::

    from z3rno import Z3rnoClient
    from z3rno.integrations.openai_agents import get_memory_tools, handle_tool_call

    client = Z3rnoClient(base_url="...", api_key="...")
    tools = get_memory_tools()

    # Pass *tools* to your OpenAI agent definition, then dispatch calls:
    result = handle_tool_call(client, agent_id="agent-1", tool_name="store_memory",
                              arguments={"content": "User prefers dark mode"})
"""

from __future__ import annotations

import json
from typing import Any

from z3rno.client import Z3rnoClient

# ---------------------------------------------------------------------------
# Tool definitions (plain dicts — no openai dependency required)
# ---------------------------------------------------------------------------

STORE_MEMORY_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "store_memory",
        "description": (
            "Persist a piece of information to long-term memory so it can be "
            "recalled later. Use this whenever the conversation contains facts, "
            "preferences, or context worth remembering."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The information to store.",
                },
                "memory_type": {
                    "type": "string",
                    "enum": ["working", "episodic", "semantic", "procedural"],
                    "description": ("Category of memory. Defaults to 'episodic' if omitted."),
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional key-value metadata to attach.",
                    "additionalProperties": True,
                },
            },
            "required": ["content"],
            "additionalProperties": False,
        },
    },
}

RECALL_MEMORY_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "recall_memory",
        "description": (
            "Retrieve relevant memories by semantic search. Use this to look up "
            "previously stored facts, preferences, or context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language search query.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Defaults to 5.",
                },
                "memory_type": {
                    "type": "string",
                    "enum": ["working", "episodic", "semantic", "procedural"],
                    "description": "Filter results to this memory type.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

FORGET_MEMORY_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "forget_memory",
        "description": ("Remove a memory that is outdated, incorrect, or no longer relevant."),
        "parameters": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "ID of the memory to forget.",
                },
                "hard_delete": {
                    "type": "boolean",
                    "description": (
                        "If true the memory is permanently deleted; otherwise it is "
                        "soft-deleted and can be audited. Defaults to false."
                    ),
                },
            },
            "required": ["memory_id"],
            "additionalProperties": False,
        },
    },
}


def get_memory_tools() -> list[dict[str, Any]]:
    """Return OpenAI function-tool definitions for store, recall, and forget.

    The returned list can be passed directly as the ``tools`` parameter when
    creating an OpenAI chat completion or agent.
    """
    return [STORE_MEMORY_TOOL, RECALL_MEMORY_TOOL, FORGET_MEMORY_TOOL]


# ---------------------------------------------------------------------------
# Tool-call dispatcher
# ---------------------------------------------------------------------------


def handle_tool_call(
    client: Z3rnoClient,
    *,
    agent_id: str,
    tool_name: str,
    arguments: dict[str, Any] | str,
) -> str:
    """Dispatch an OpenAI tool call to the corresponding Z3rno client method.

    Args:
        client: An initialised :class:`Z3rnoClient`.
        agent_id: The agent whose memories are being accessed.
        tool_name: One of ``"store_memory"``, ``"recall_memory"``, or
            ``"forget_memory"``.
        arguments: The parsed arguments dict (or raw JSON string) from the
            tool call.

    Returns:
        A JSON string suitable for returning as the tool-call result content.

    Raises:
        ValueError: If *tool_name* is not recognised.
    """
    if isinstance(arguments, str):
        arguments = json.loads(arguments)

    if tool_name == "store_memory":
        return _handle_store(client, agent_id, arguments)
    if tool_name == "recall_memory":
        return _handle_recall(client, agent_id, arguments)
    if tool_name == "forget_memory":
        return _handle_forget(client, agent_id, arguments)

    raise ValueError(f"Unknown tool: {tool_name}")


def _handle_store(client: Z3rnoClient, agent_id: str, args: dict[str, Any]) -> str:
    memory = client.store(
        agent_id=agent_id,
        content=args["content"],
        memory_type=args.get("memory_type", "episodic"),
        metadata=args.get("metadata"),
    )
    return json.dumps(
        {"memory_id": memory.id, "status": "stored"},
        default=str,
    )


def _handle_recall(client: Z3rnoClient, agent_id: str, args: dict[str, Any]) -> str:
    response = client.recall(
        agent_id=agent_id,
        query=args["query"],
        top_k=args.get("top_k", 5),
        memory_type=args.get("memory_type"),
    )
    results = [
        {
            "memory_id": r.memory_id,
            "content": r.content,
            "memory_type": r.memory_type,
            "similarity_score": r.similarity_score,
        }
        for r in response.results
    ]
    return json.dumps({"results": results, "total": response.total}, default=str)


def _handle_forget(client: Z3rnoClient, agent_id: str, args: dict[str, Any]) -> str:
    result = client.forget(
        agent_id=agent_id,
        memory_id=args["memory_id"],
        hard_delete=args.get("hard_delete", False),
    )
    return json.dumps(
        {"deleted_count": result.deleted_count, "status": "forgotten"},
        default=str,
    )


# ---------------------------------------------------------------------------
# Automatic conversation memory
# ---------------------------------------------------------------------------


class Z3rnoConversationMemory:
    """Intercepts conversation messages and auto-stores them as working memories.

    Usage::

        conv = Z3rnoConversationMemory(client, agent_id="agent-1")
        conv.add_message(role="user", content="I like Python")
        conv.add_message(role="assistant", content="Great choice!")
        # Both messages are stored as working memories automatically.

    Retrieve recent context with :meth:`get_context` before sending the next
    request to the model.
    """

    def __init__(
        self,
        client: Z3rnoClient,
        *,
        agent_id: str,
        user_id: str | None = None,
        auto_store: bool = True,
    ) -> None:
        self._client = client
        self._agent_id = agent_id
        self._user_id = user_id
        self._auto_store = auto_store
        self._message_ids: list[str] = []

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def stored_ids(self) -> list[str]:
        """Memory IDs of all messages stored so far."""
        return list(self._message_ids)

    def add_message(
        self,
        *,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Record a conversation message.

        If *auto_store* is enabled (the default), the message is persisted as
        a ``working`` memory and the new memory ID is returned.
        """
        if not self._auto_store:
            return None

        msg_metadata: dict[str, Any] = {"role": role, "source": "conversation"}
        if metadata:
            msg_metadata.update(metadata)

        memory = self._client.store(
            agent_id=self._agent_id,
            content=content,
            memory_type="working",
            user_id=self._user_id,
            metadata=msg_metadata,
        )
        self._message_ids.append(memory.id)
        return memory.id

    def get_context(self, query: str, *, top_k: int = 10) -> list[dict[str, Any]]:
        """Recall relevant conversation context for a query.

        Returns a list of dicts with ``role``, ``content``, and
        ``similarity_score`` keys, ordered by relevance.
        """
        response = self._client.recall(
            agent_id=self._agent_id,
            query=query,
            top_k=top_k,
            memory_type="working",
        )
        return [
            {
                "role": r.metadata.get("role", "unknown"),
                "content": r.content,
                "similarity_score": r.similarity_score,
            }
            for r in response.results
        ]
