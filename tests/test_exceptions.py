"""Tests for SDK exception hierarchy."""

from __future__ import annotations

from z3rno.exceptions import (
    AuthenticationError,
    NotFoundError,
    RateLimitError,
    ServerError,
    ValidationError,
    Z3rnoError,
)


def test_z3rno_error_base() -> None:
    """Base error has message and optional status_code."""
    err = Z3rnoError("test error")
    assert str(err) == "test error"
    assert err.status_code is None


def test_z3rno_error_with_status() -> None:
    """Base error accepts status_code."""
    err = Z3rnoError("test", status_code=500)
    assert err.status_code == 500


def test_authentication_error() -> None:
    """AuthenticationError is a Z3rnoError with 401."""
    err = AuthenticationError("bad key")
    assert isinstance(err, Z3rnoError)
    assert err.status_code == 401


def test_rate_limit_error() -> None:
    """RateLimitError has retry_after."""
    err = RateLimitError("too fast", retry_after=30)
    assert isinstance(err, Z3rnoError)
    assert err.status_code == 429
    assert err.retry_after == 30


def test_rate_limit_error_default_retry() -> None:
    """RateLimitError defaults retry_after to None."""
    err = RateLimitError("too fast")
    assert err.retry_after is None


def test_validation_error() -> None:
    """ValidationError is a Z3rnoError."""
    err = ValidationError("bad input", status_code=422)
    assert isinstance(err, Z3rnoError)
    assert err.status_code == 422


def test_not_found_error() -> None:
    """NotFoundError has 404."""
    err = NotFoundError("missing")
    assert isinstance(err, Z3rnoError)
    assert err.status_code == 404


def test_server_error() -> None:
    """ServerError has 500+."""
    err = ServerError("boom", status_code=503)
    assert isinstance(err, Z3rnoError)
    assert err.status_code == 503


def test_exception_hierarchy() -> None:
    """All errors inherit from Z3rnoError."""
    for cls in (AuthenticationError, RateLimitError, ValidationError, NotFoundError, ServerError):
        assert issubclass(cls, Z3rnoError)
        assert issubclass(cls, Exception)
