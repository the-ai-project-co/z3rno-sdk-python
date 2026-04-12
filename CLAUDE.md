# CLAUDE.md

## Project

z3rno-sdk-python is the Python SDK for Z3rno. It is a thin HTTP client (httpx) that talks to z3rno-server. It has zero database dependencies, zero embedding logic, and zero business rules. All intelligence is server-side.

## Quick Reference

```bash
uv sync --dev                    # Install dependencies
uv run ruff check .              # Lint
uv run ruff format .             # Format
uv run mypy .                    # Type check
uv run pytest                    # Run tests (38 tests, mocked HTTP)
```

## Architecture

- `src/z3rno/__init__.py` — Public exports: Z3rnoClient, AsyncZ3rnoClient, models, exceptions
- `src/z3rno/client.py` — Sync client (httpx.Client), all operations
- `src/z3rno/async_client.py` — Async client (httpx.AsyncClient), all operations
- `src/z3rno/models.py` — Pydantic models mirroring server schemas
- `src/z3rno/exceptions.py` — Z3rnoError hierarchy (401, 404, 429, 5xx)
- `src/z3rno/config.py` — Immutable config dataclass

## SDK Methods (same on both sync and async clients)

- `store(agent_id, content, memory_type, ...)` -> Memory
- `recall(agent_id, query, top_k, ...)` -> RecallResponse
- `forget(agent_id, memory_id, ...)` -> ForgetResult
- `audit(agent_id, page, ...)` -> AuditPage
- `start_session(agent_id, ...)` -> Session
- `end_session(session_id)` -> dict

## Key Conventions

- Python 3.10+ (StrEnum backport for 3.10 compatibility)
- Minimal dependencies: httpx[http2], pydantic, tenacity
- No psycopg, no asyncpg, no sqlalchemy, no litellm
- API key sent as Authorization: Bearer on every request
- All responses validated via Pydantic models
- Exceptions map HTTP status codes: AuthenticationError (401), RateLimitError (429), ValidationError (400/422), NotFoundError (404), ServerError (5xx)
- Tests use pytest-httpx for mocking (no real server needed)
- Package name on PyPI: `z3rno`
- Conventional commits

## Publishing

- PyPI publish on `v*.*.*` tag push (GitHub Actions)
- Requires `PYPI_API_TOKEN` repository secret
