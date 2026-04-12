"""Z3rno Python SDK - AI agent memory database client.

Usage::

    from z3rno import Z3rnoClient

    client = Z3rnoClient(base_url="http://localhost:8000", api_key="z3rno_sk_...")
    memory = client.store(agent_id="agent-1", content="User prefers dark mode")
    results = client.recall(agent_id="agent-1", query="user preferences")
"""

from z3rno.async_client import AsyncZ3rnoClient
from z3rno.client import Z3rnoClient
from z3rno.exceptions import (
    AuthenticationError,
    NotFoundError,
    RateLimitError,
    ServerError,
    ValidationError,
    Z3rnoError,
)
from z3rno.models import (
    AuditEntry,
    AuditPage,
    ForgetResult,
    Memory,
    MemoryType,
    RecallResponse,
    RecallResult,
    Relationship,
    RelationshipType,
    Session,
)

__all__ = [
    "AsyncZ3rnoClient",
    "AuditEntry",
    "AuditPage",
    "AuthenticationError",
    "ForgetResult",
    "Memory",
    "MemoryType",
    "NotFoundError",
    "RateLimitError",
    "RecallResponse",
    "RecallResult",
    "Relationship",
    "RelationshipType",
    "ServerError",
    "Session",
    "ValidationError",
    "Z3rnoClient",
    "Z3rnoError",
]

__version__ = "0.0.1"
