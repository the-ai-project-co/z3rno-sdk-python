# z3rno-sdk-python

> The official Python SDK for Z3rno. A thin, typed HTTP client over `httpx` + `pydantic` that talks to `z3rno-server`. No database drivers, no embedding providers, no connection pools — all business logic is server-side.

**License:** Apache 2.0
**Status:** Early development — not yet on PyPI
**Part of:** [Z3rno](https://github.com/the-ai-project-co) — the database for AI agent memory

## Installation

```bash
pip install z3rno  # (when published)
```

## Quickstart

```python
from z3rno import Z3rnoClient

client = Z3rnoClient(
    base_url="https://api.z3rno.dev",  # or your self-hosted z3rno-server
    api_key="z3rno_sk_...",
)

# Store a memory
memory = client.store(
    agent_id="agent-1",
    content="User prefers dark mode and uses Python.",
    memory_type="semantic",
)

# Recall memories
results = client.recall(
    agent_id="agent-1",
    query="What does the user prefer?",
    top_k=5,
)
```

An `AsyncZ3rnoClient` is also exported for async codebases.

## Design

- **Thin HTTP client.** The only runtime dependencies are `httpx`, `pydantic`, and `tenacity`. No `psycopg`, no database driver.
- **Generated from OpenAPI.** The Pydantic models mirror the `z3rno-server` OpenAPI 3.1 spec exactly.
- **Typed errors.** `Z3rnoRateLimitError`, `Z3rnoAuthenticationError`, `Z3rnoValidationError`, etc. — each wraps a specific HTTP status.
- **Retry policy.** Exponential backoff on connection errors and 5xx. `Retry-After` honoured on 429.

## Framework integrations

Each is a separate package that depends on `z3rno`:

- `z3rno-langchain` — LangChain `BaseMemory` + `BaseRetriever` adapter
- `z3rno-crewai` — CrewAI memory provider
- `z3rno-openai` — OpenAI Agents SDK function tools

For Anthropic Claude, see the dedicated `z3rno-mcp` Model Context Protocol server.
