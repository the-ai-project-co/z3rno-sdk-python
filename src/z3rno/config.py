"""SDK configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

_DEFAULT_BASE_URL = "https://api.z3rno.dev"


@dataclass(frozen=True)
class Z3rnoConfig:
    """Configuration for the Z3rno client.

    If *base_url* or *api_key* are empty / default, the corresponding
    environment variables (``Z3RNO_BASE_URL``, ``Z3RNO_API_KEY``) are
    read automatically — matching the convention used by Stripe, OpenAI,
    and similar SDKs.

    Args:
        base_url: API server URL. Falls back to ``Z3RNO_BASE_URL`` env var.
        api_key: API key. Falls back to ``Z3RNO_API_KEY`` env var.
        timeout: Default request timeout in seconds.
        max_retries: Maximum number of retry attempts for transient failures.
        user_agent: User-Agent header value.
        headers: Additional HTTP headers to send with every request.
        proxy: HTTP/HTTPS proxy URL (e.g. ``http://proxy.corp:8080``).
            Passed directly to httpx as the ``proxy`` kwarg.
    """

    base_url: str = _DEFAULT_BASE_URL
    api_key: str = ""
    timeout: float = 30.0
    max_retries: int = 3
    user_agent: str = "z3rno-python/0.0.1"
    headers: dict[str, str] = field(default_factory=dict)
    proxy: str | None = None

    def __post_init__(self) -> None:
        # frozen=True prevents normal assignment — use object.__setattr__
        if self.base_url == _DEFAULT_BASE_URL:
            env_url = os.environ.get("Z3RNO_BASE_URL", "")
            if env_url:
                object.__setattr__(self, "base_url", env_url)

        if not self.api_key:
            env_key = os.environ.get("Z3RNO_API_KEY", "")
            if env_key:
                object.__setattr__(self, "api_key", env_key)

    @property
    def auth_headers(self) -> dict[str, str]:
        """Build headers with auth."""
        h = {
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": self.user_agent,
            "Content-Type": "application/json",
        }
        h.update(self.headers)
        return h
