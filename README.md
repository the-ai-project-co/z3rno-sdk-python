# z3rno (Python SDK)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/the-ai-project-co/z3rno-sdk-python/actions/workflows/ci.yml/badge.svg)](https://github.com/the-ai-project-co/z3rno-sdk-python/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/z3rno)](https://pypi.org/project/z3rno/)

Python SDK for Z3rno -- thin HTTP client for the Z3rno memory API.

## Installation

```bash
pip install z3rno
```

## Quickstart

```python
from z3rno import Z3rnoClient

client = Z3rnoClient(base_url="https://api.z3rno.dev", api_key="z3rno_sk_...")
memory = client.store(agent_id="agent-1", content="User prefers dark mode", memory_type="semantic")
results = client.recall(agent_id="agent-1", query="What does the user prefer?", top_k=5)
client.forget(memory_id=memory.id)
```

## Async Usage

```python
from z3rno import AsyncZ3rnoClient

async def main():
    async with AsyncZ3rnoClient(base_url="https://api.z3rno.dev", api_key="z3rno_sk_...") as client:
        memory = await client.store(agent_id="agent-1", content="User prefers dark mode")
        results = await client.recall(agent_id="agent-1", query="preferences")
        await client.forget(memory_id=memory.id)
```

## Methods

| Method | Description |
|--------|-------------|
| `store(...)` | Store a new memory with optional type, metadata, relationships, TTL, and importance |
| `recall(...)` | Recall memories by semantic similarity query |
| `forget(...)` | Soft-delete a memory by ID |
| `audit(...)` | Query the audit trail with optional filters and pagination |

All methods are available on both `Z3rnoClient` (sync) and `AsyncZ3rnoClient` (async).

## Features

- **Thin HTTP client** -- only `httpx`, `pydantic`, and `tenacity` at runtime. No database drivers.
- **Typed errors** -- `Z3rnoRateLimitError`, `Z3rnoAuthenticationError`, `Z3rnoValidationError`, each mapping to a specific HTTP status.
- **Automatic retries** -- exponential backoff on connection errors and 5xx responses. `Retry-After` honored on 429.
- **Context manager support** -- use `with` / `async with` for automatic cleanup.

## Framework Integrations

Each is a separate package that depends on `z3rno`:

- `z3rno-langchain` -- LangChain `BaseMemory` + `BaseRetriever` adapter
- `z3rno-crewai` -- CrewAI memory provider
- `z3rno-openai` -- OpenAI Agents SDK function tools

For Anthropic Claude, see the `z3rno-mcp` Model Context Protocol server.

For a detailed step-by-step setup, see [QUICKSTART.md](QUICKSTART.md).

## API Documentation

Full API reference: [astron-bb4261fd.mintlify.app/sdk/python](https://astron-bb4261fd.mintlify.app/sdk/python)

## Development

```bash
uv sync --dev
uv run ruff check .
uv run mypy .
uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

## License

Apache 2.0 -- see [LICENSE](LICENSE).
