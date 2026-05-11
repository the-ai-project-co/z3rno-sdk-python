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


class RetrievalStrategy(StrEnum):
    """Retrieval strategy for ``recall(strategy=...)`` (Phase C).

    AUTO is the default; the server's LLM router picks one of the others
    when configured. Use the explicit enum value to bypass routing.
    """

    AUTO = "AUTO"        # LLM router picks the best strategy (default)
    VECTOR = "VECTOR"    # Cosine similarity over pgvector HNSW
    LEXICAL = "LEXICAL"  # Postgres tsvector + ts_rank
    GRAPH = "GRAPH"      # Vector seeds → AGE subgraph → optional LLM synthesis
    TRIPLET = "TRIPLET"  # LLM-parsed (S, P, O) → AGE traversal → slot fill
    TRACE = "TRACE"      # Chain-of-thought multi-step retrieval
    TEMPORAL = "TEMPORAL"  # SCD-2 time-travel (with optional LLM time parsing)
    ASK = "ASK"          # NL → Cypher → execute (read-only)
    CYPHER = "CYPHER"    # Raw Cypher passthrough (gated by ALLOW_CYPHER_QUERY)


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
    # Phase C: per-source signals (vector, lexical, graph_richness,
    # reranker, …). Optional — older SDKs ignoring this field keep working.
    score_components: dict[str, float] = Field(default_factory=dict)


class RecallResponse(BaseModel):
    """Response from a recall query."""

    results: list[RecallResult]
    total: int
    query: str | None = None
    # Phase C: provenance for the dispatched strategy.
    strategy_used: str = "VECTOR"
    strategies_considered: list[str] = Field(default_factory=list)
    reranked: bool = False
    elapsed_ms: float = 0.0


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


class Conversation(BaseModel):
    """A conversation session (Phase G slice 2)."""

    id: str
    agent_id: str
    user_id: str | None = None
    title: str | None = None
    summary_cadence: int
    turn_count: int
    last_summary_turn: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class TurnAddResponse(BaseModel):
    """Result of appending a turn — includes the cadence flag."""

    turn_index: int
    needs_summary: bool


class Turn(BaseModel):
    """One turn within a conversation."""

    memory_id: str
    turn_index: int
    turn_role: str
    content: str
    created_at: datetime


class TurnListResponse(BaseModel):
    """List response for ``GET /v1/conversations/{id}/turns``."""

    turns: list[Turn]
    total: int
    conversation_id: str


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


# ---------------------------------------------------------------------------
# Forge verbs (Phase A / B / D) — ingest / distill / refine
# ---------------------------------------------------------------------------


class IngestJob(BaseModel):
    """Response from POST /v1/ingest — the enqueue ack."""

    job_id: str
    kind: str  # "text" | "url" | "file"
    status: str
    dataset_id: str | None = None
    enqueued_at: datetime


class IngestJobStatus(BaseModel):
    """Response from GET /v1/ingest/{job_id} — full job state."""

    job_id: str
    agent_id: str
    dataset_id: str | None = None
    kind: str
    status: str
    source_uri: str | None = None
    content_type: str | None = None
    filename: str | None = None
    file_size: int | None = None
    memory_ids: list[str] = Field(default_factory=list)
    memos_written: int = 0
    distill_job_id: str | None = None
    codegraph_memos_written: int = 0
    codegraph_edges_written: int = 0
    error: str | None = None
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class DistillJob(BaseModel):
    """Response from POST /v1/distill — the enqueue ack."""

    job_id: str
    status: str
    memory_ids: list[str]
    enqueued_at: datetime


class DistillJobStatus(BaseModel):
    """Response from GET /v1/distill/{job_id} — full job state."""

    job_id: str
    agent_id: str
    status: str
    model: str
    memory_ids: list[str]
    chunk_size: int
    chunk_overlap: int
    max_concurrency: int
    chunks_total: int
    chunks_failed: int
    entities_extracted: int
    relationships_extracted: int
    memos_written: int
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class RefineJob(BaseModel):
    """Response from POST /v1/refine — the enqueue ack."""

    job_id: str
    status: str
    dataset_id: str | None = None
    enqueued_at: datetime


class RefineJobStatus(BaseModel):
    """Response from GET /v1/refine/{job_id} — full job state."""

    job_id: str
    status: str
    dataset_id: str | None = None
    trigger: str
    memos_scanned: int = 0
    memos_deduped: int = 0
    edges_reweighted: int = 0
    edges_pruned: int = 0
    feedback_drained: int = 0
    job_metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
