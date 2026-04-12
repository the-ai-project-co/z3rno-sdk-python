"""Tests for SDK configuration."""

from __future__ import annotations

import pytest

from z3rno.config import Z3rnoConfig


def test_default_config() -> None:
    """Default config values."""
    cfg = Z3rnoConfig()
    assert cfg.base_url == "https://api.z3rno.dev"
    assert cfg.api_key == ""
    assert cfg.timeout == 30.0
    assert cfg.max_retries == 3


def test_custom_config() -> None:
    """Custom config values."""
    cfg = Z3rnoConfig(
        base_url="http://localhost:8000",
        api_key="sk_test",
        timeout=10.0,
        max_retries=5,
    )
    assert cfg.base_url == "http://localhost:8000"
    assert cfg.api_key == "sk_test"
    assert cfg.timeout == 10.0
    assert cfg.max_retries == 5


def test_auth_headers() -> None:
    """Auth headers include Bearer token."""
    cfg = Z3rnoConfig(api_key="my_key")
    headers = cfg.auth_headers
    assert headers["Authorization"] == "Bearer my_key"
    assert headers["Content-Type"] == "application/json"
    assert "User-Agent" in headers


def test_auth_headers_custom() -> None:
    """Custom headers are merged."""
    cfg = Z3rnoConfig(api_key="key", headers={"X-Custom": "value"})
    headers = cfg.auth_headers
    assert headers["X-Custom"] == "value"
    assert headers["Authorization"] == "Bearer key"


def test_config_frozen() -> None:
    """Config is frozen (immutable)."""
    cfg = Z3rnoConfig()
    with pytest.raises(AttributeError):
        cfg.base_url = "http://other"  # type: ignore[misc]
