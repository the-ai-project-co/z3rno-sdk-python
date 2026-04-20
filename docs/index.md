# Z3rno Python SDK

The official Python SDK for [Z3rno](https://z3rno.dev) -- the AI agent memory database.

## Installation

```bash
pip install z3rno
```

## Quick Start

```python
from z3rno import Z3rnoClient

client = Z3rnoClient(
    base_url="http://localhost:8000",
    api_key="z3rno_sk_...",
)

# Store a memory
memory = client.store(
    agent_id="agent-1",
    content="User prefers dark mode",
    memory_type="semantic",
)

# Recall memories
results = client.recall(agent_id="agent-1", query="user preferences")
for r in results.results:
    print(r.content, r.similarity_score)

# Forget a memory
client.forget(agent_id="agent-1", memory_id=memory.id)
```

## Async Support

```python
from z3rno import AsyncZ3rnoClient

async with AsyncZ3rnoClient(base_url="...", api_key="...") as client:
    memory = await client.store(agent_id="agent-1", content="async memory")
    results = await client.recall(agent_id="agent-1", query="async")
```

## Features

- **Sync and async clients** -- choose what fits your stack.
- **Automatic retries** with exponential backoff for 5xx and 429 errors.
- **Pydantic models** for all request/response types.
- **Typed exception hierarchy** mapping HTTP status codes.
- **Environment variable configuration** -- set `Z3RNO_BASE_URL` and `Z3RNO_API_KEY`.

## API Reference

- [Z3rnoClient](client.md) -- synchronous client
- [AsyncZ3rnoClient](async_client.md) -- asynchronous client
- [Models](models.md) -- Pydantic data models
- [Exceptions](exceptions.md) -- error hierarchy
