"""Pydantic models mirroring the z3rno-server API schemas."""

from __future__ import annotations

import sys
from datetime import datetime

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        """Backport of StrEnum for Python 3.10."""


from typing import Any

from pydantic import BaseModel, Field


class MemoryType(StrEnum):
    """Memory type classification."""

    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class RelationshipType(StrEnum):
    """Memory relationship types."""

    DERIVED_FROM = "derived_from"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    SUPERSEDES = "supersedes"
    RELATED_TO = "related_to"
    CAUSED_BY = "caused_by"


class Relationship(BaseModel):
    """Input for creating a memory relationship."""

    target_memory_id: str
    relationship_type: RelationshipType
    weight: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class Memory(BaseModel):
    """A stored memory."""

    id: str
    agent_id: str
    content: str
    memory_type: str
    importance_score: float
    recall_count: int
    embedding_model: str | None = None
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class RecallResult(BaseModel):
    """A single recall result with scoring."""

    memory_id: str
    content: str
    summary: str | None = None
    memory_type: str
    similarity_score: float
    importance_score: float
    relevance_score: float
    recall_count: int
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class RecallResponse(BaseModel):
    """Response from a recall query."""

    results: list[RecallResult]
    total: int
    query: str | None = None


class ForgetResult(BaseModel):
    """Response from a forget operation."""

    deleted_count: int
    hard_deleted: bool
    cascade_count: int
    memory_ids: list[str]


class AuditEntry(BaseModel):
    """A single audit log entry."""

    id: int
    agent_id: str | None = None
    user_id: str | None = None
    operation: str
    memory_id: str | None = None
    memory_type: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    ip_address: str | None = None
    created_at: datetime


class AuditPage(BaseModel):
    """Paginated audit log response."""

    entries: list[AuditEntry]
    total: int
    page: int
    page_size: int
    has_next: bool


class Session(BaseModel):
    """A session state."""

    session_id: str
    agent_id: str
    session_type: str
    started_at: datetime
    metadata: dict[str, str] = Field(default_factory=dict)


class BatchStoreResponse(BaseModel):
    """Response from a batch store operation."""

    results: list[Memory]
    stored_count: int


class MemoryVersion(BaseModel):
    """A single version of a memory in its history."""

    id: str
    content: str
    memory_type: str
    importance_score: float
    valid_from: datetime
    valid_to: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryHistoryResponse(BaseModel):
    """Response from a memory history query."""

    memory_id: str
    versions: list[MemoryVersion]
    total: int
