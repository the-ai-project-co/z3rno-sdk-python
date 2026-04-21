"""Anthropic Claude tool_use definitions for Z3rno memory operations.

Provides pre-built tool definitions compatible with Anthropic's tool_use API.
For full MCP server integration, see the z3rno-mcp package.

Usage::

    from z3rno.integrations.anthropic import get_memory_tools

    tools = get_memory_tools()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        tools=tools,
        messages=[...],
    )
"""

from __future__ import annotations

from typing import Any

from z3rno import Z3rnoClient

# ---------------------------------------------------------------------------
# Tool definitions (Anthropic tool_use format)
# ---------------------------------------------------------------------------

STORE_MEMORY_TOOL: dict[str, Any] = {
    "name": "store_memory",
    "description": (
        "Store a piece of information in the agent's persistent memory. "
        "Use this to save facts, preferences, decisions, or context that "
        "should be remembered across conversations."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The information to remember.",
            },
            "memory_type": {
                "type": "string",
                "enum": ["working", "episodic", "semantic", "procedural"],
                "description": "Type of memory: working (active context), episodic (events), semantic (facts), procedural (how-to).",
                "default": "semantic",
            },
            "metadata": {
                "type": "object",
                "description": "Optional key-value metadata tags.",
            },
        },
        "required": ["content"],
    },
}

RECALL_MEMORY_TOOL: dict[str, Any] = {
    "name": "recall_memory",
    "description": (
        "Search the agent's memory for relevant information. "
        "Use this when you need to remember something from past conversations "
        "or retrieve stored knowledge."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query.",
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of results to return.",
                "default": 5,
            },
            "memory_type": {
                "type": "string",
                "enum": ["working", "episodic", "semantic", "procedural"],
                "description": "Filter by memory type (optional).",
            },
        },
        "required": ["query"],
    },
}

FORGET_MEMORY_TOOL: dict[str, Any] = {
    "name": "forget_memory",
    "description": (
        "Remove a specific memory that is outdated, incorrect, or no longer needed. "
        "Use this to keep memory clean and accurate."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "The UUID of the memory to forget.",
            },
            "hard_delete": {
                "type": "boolean",
                "description": "If true, permanently delete (GDPR). If false, soft delete (recoverable).",
                "default": False,
            },
            "reason": {
                "type": "string",
                "description": "Why this memory is being forgotten.",
            },
        },
        "required": ["memory_id"],
    },
}


def get_memory_tools() -> list[dict[str, Any]]:
    """Return all Z3rno memory tool definitions for Anthropic's tool_use API.

    Usage::

        import anthropic
        from z3rno.integrations.anthropic import get_memory_tools

        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            tools=get_memory_tools(),
            messages=[{"role": "user", "content": "What do you remember about me?"}],
        )
    """
    return [STORE_MEMORY_TOOL, RECALL_MEMORY_TOOL, FORGET_MEMORY_TOOL]


def handle_tool_use(
    client: Z3rnoClient,
    agent_id: str,
    tool_name: str,
    tool_input: dict[str, Any],
) -> str:
    """Handle a tool_use block from an Anthropic API response.

    Args:
        client: An initialized Z3rnoClient.
        agent_id: The agent ID for memory operations.
        tool_name: The tool name from the tool_use block.
        tool_input: The input dict from the tool_use block.

    Returns:
        A string result to pass back as tool_result content.
    """
    import json  # noqa: PLC0415

    if tool_name == "store_memory":
        store_result = client.store(
            agent_id=agent_id,
            content=tool_input["content"],
            memory_type=tool_input.get("memory_type", "semantic"),
            metadata=tool_input.get("metadata"),
        )
        return json.dumps({"stored": True, "memory_id": str(store_result.id)})

    elif tool_name == "recall_memory":
        recall_result = client.recall(
            agent_id=agent_id,
            query=tool_input["query"],
            top_k=tool_input.get("top_k", 5),
            memory_type=tool_input.get("memory_type"),
        )
        memories = [
            {"content": r.content, "score": r.similarity_score, "memory_type": r.memory_type}
            for r in recall_result.results
        ]
        return json.dumps({"results": memories, "total": len(memories)})

    elif tool_name == "forget_memory":
        forget_result = client.forget(
            agent_id=agent_id,
            memory_id=tool_input["memory_id"],
            hard_delete=tool_input.get("hard_delete", False),
            reason=tool_input.get("reason"),
        )
        return json.dumps({"forgotten": True, "deleted_count": forget_result.deleted_count})

    else:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
