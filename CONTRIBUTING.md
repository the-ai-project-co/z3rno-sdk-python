# Contributing to z3rno-sdk-python

Thank you for your interest in contributing to Z3rno. This guide covers the development workflow for the Python SDK.

## Getting Started

1. Fork and clone the repository.
2. Install Python 3.10+ and [uv](https://docs.astral.sh/uv/).
3. Install dependencies:

```bash
uv sync --dev
```

4. Run the checks:

```bash
uv run ruff check .
uv run mypy .
uv run pytest
```

## Development Workflow

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feat/your-feature
   ```
2. Write your code with tests.
3. Run linting, type checking, and tests (see above).
4. Commit using conventional commit messages.
5. Open a pull request against `main`.

## Code Style

This project uses **ruff** for linting and formatting. Configuration lives in `pyproject.toml`.

```bash
# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run mypy .
```

## Testing

Tests use `pytest` with `pytest-asyncio` and `pytest-httpx` for mocking HTTP responses. No running server is required for unit tests.

```bash
# All tests
uv run pytest

# With coverage
uv run pytest --cov=src/z3rno
```

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation only
- `test:` adding or updating tests
- `refactor:` code change that neither fixes a bug nor adds a feature
- `chore:` maintenance (deps, CI, tooling)

Examples:
- `feat: add list_memories method to client`
- `fix: retry logic not honoring Retry-After header`

## Pull Request Process

1. Ensure all checks pass (`ruff check`, `mypy`, `pytest`).
2. Keep PRs focused -- one logical change per PR.
3. Update or add tests for any changed behavior.
4. Maintain parity with the TypeScript SDK where applicable.
5. Fill out the PR template description.
6. A maintainer will review and merge.

## Questions?

Open a [GitHub Discussion](https://github.com/the-ai-project-co/z3rno-sdk-python/discussions) or reach out at engineering@z3rno.dev.
