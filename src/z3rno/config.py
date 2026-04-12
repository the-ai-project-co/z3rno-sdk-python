"""SDK configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Z3rnoConfig:
    """Configuration for the Z3rno client."""

    base_url: str = "https://api.z3rno.dev"
    api_key: str = ""
    timeout: float = 30.0
    max_retries: int = 3
    user_agent: str = "z3rno-python/0.0.1"
    headers: dict[str, str] = field(default_factory=dict)

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
