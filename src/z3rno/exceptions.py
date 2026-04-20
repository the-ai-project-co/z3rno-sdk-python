"""Z3rno SDK exception hierarchy."""

from __future__ import annotations


class Z3rnoError(Exception):
    """Base exception for all Z3rno SDK errors."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class AuthenticationError(Z3rnoError):
    """401 - Invalid or missing API key."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=401)


class RateLimitError(Z3rnoError):
    """429 - Rate limit exceeded."""

    def __init__(self, message: str, retry_after: int | None = None) -> None:
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class ValidationError(Z3rnoError):
    """400/422 - Invalid request parameters."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message, status_code=status_code)


class NotFoundError(Z3rnoError):
    """404 - Resource not found."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=404)


class ServerError(Z3rnoError):
    """500+ - Server-side error."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message, status_code=status_code)


class Z3rnoConnectionError(Z3rnoError):
    """Network failure — DNS, timeout, connection refused."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=None)
