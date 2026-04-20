# Quickstart: z3rno (Python SDK)

A detailed getting-started guide for the Z3rno Python SDK.

## Prerequisites

- Python 3.10+
- A running Z3rno server (local or hosted)
- A Z3rno API key

If you do not have a Z3rno server running, see the [z3rno-server quickstart](https://github.com/the-ai-project-co/z3rno-server/blob/main/QUICKSTART.md) to set one up locally with Docker Compose.

## Step-by-step Installation

### 1. Install the SDK

```bash
pip install z3rno
```

Or with uv:

```bash
uv add z3rno
```

### 2. Set up your credentials

```python
# Option A: pass directly
client = Z3rnoClient(base_url="http://localhost:8000", api_key="z3rno_sk_test_localdev")

# Option B: use environment variables
# export Z3RNO_BASE_URL=http://localhost:8000
# export Z3RNO_API_KEY=z3rno_sk_test_localdev
client = Z3rnoClient()  # reads from env
```

## First Working Example

```python
from z3rno import Z3rnoClient

client = Z3rnoClient(base_url="http://localhost:8000", api_key="z3rno_sk_test_localdev")

# Store a memory
memory = client.store(agent_id="agent-1", content="User prefers dark mode", memory_type="semantic")
print(f"Stored: {memory.id}")

# Recall memories
results = client.recall(agent_id="agent-1", query="What does the user prefer?", top_k=5)
for r in results:
    print(f"  - {r.content} (score: {r.score:.3f})")

# Forget
client.forget(memory_id=memory.id)
print("Memory forgotten")
```

Save as `example.py` and run:

```bash
python example.py
```

## Async Usage

```python
import asyncio
from z3rno import AsyncZ3rnoClient

async def main():
    async with AsyncZ3rnoClient(base_url="http://localhost:8000", api_key="z3rno_sk_test_localdev") as client:
        memory = await client.store(agent_id="agent-1", content="User prefers dark mode")
        results = await client.recall(agent_id="agent-1", query="preferences")
        print(f"Found {len(results)} memories")
        await client.forget(memory_id=memory.id)

asyncio.run(main())
```

## Running Locally (Development)

To work on the SDK itself:

```bash
git clone https://github.com/the-ai-project-co/z3rno-sdk-python.git
cd z3rno-sdk-python
uv sync --dev
uv run pytest
```

## Common Issues / Troubleshooting

### 1. "Connection refused" errors

The Z3rno server is not running at the configured `base_url`. Start it with:

```bash
# In the z3rno-server repo
docker compose -f docker-compose.dev.yml up
```

### 2. "401 Unauthorized"

Your API key is incorrect or missing. For local development, use `z3rno_sk_test_localdev`.

### 3. "429 Too Many Requests"

You are being rate-limited. The SDK automatically retries with exponential backoff and respects the `Retry-After` header. If you see this persistently, reduce your request rate.

### 4. Import errors: "No module named z3rno"

Ensure the package is installed in your active Python environment:

```bash
pip list | grep z3rno
# Should show: z3rno  0.x.x
```

### 5. SSL/TLS errors when connecting to production

If you are behind a corporate proxy, you may need to configure httpx transport settings or set `SSL_CERT_FILE` environment variable.
