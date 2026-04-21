"""Structured logging for Z3rno SDK requests."""

from __future__ import annotations

import os
import time

import structlog

_configured = False


def _ensure_configured() -> None:
    """Configure structlog once (JSON output to stderr)."""
    global _configured  # noqa: PLW0603
    if _configured:
        return
    log_level = structlog.get_level_from_env("Z3RNO_LOG_LEVEL", default="INFO")  # type: ignore[operator]
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def _get_logger() -> structlog.stdlib.BoundLogger:
    _ensure_configured()
    return structlog.get_logger("z3rno.sdk")  # type: ignore[no-any-return]


def log_request(
    *,
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    request_id: str | None,
) -> None:
    """Emit a single structured log line per HTTP request."""
    _get_logger().info(
        "http_request",
        method=method,
        path=path,
        status_code=status_code,
        duration_ms=round(duration_ms, 1),
        request_id=request_id,
    )


def logging_enabled(flag: bool) -> bool:
    """Return True if SDK logging should be active."""
    if flag:
        return True
    return os.environ.get("Z3RNO_LOG_LEVEL", "") != ""


def start_timer() -> float:
    """Return a monotonic timestamp for duration measurement."""
    return time.monotonic()


def elapsed_ms(start: float) -> float:
    """Return elapsed milliseconds since *start*."""
    return (time.monotonic() - start) * 1000
