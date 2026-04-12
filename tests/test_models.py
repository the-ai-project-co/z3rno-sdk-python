"""Tests for SDK Pydantic models."""

from __future__ import annotations

from datetime import datetime, timezone

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


def test_memory_type_values() -> None:
    """MemoryType enum has correct values."""
    assert MemoryType.WORKING.value == "working"
    assert MemoryType.EPISODIC.value == "episodic"
    assert MemoryType.SEMANTIC.value == "semantic"
    assert MemoryType.PROCEDURAL.value == "procedural"


def test_relationship_type_values() -> None:
    """RelationshipType enum has correct values."""
    assert len(RelationshipType) == 6
    assert RelationshipType.DERIVED_FROM.value == "derived_from"
    assert RelationshipType.CONTRADICTS.value == "contradicts"


def test_memory_model() -> None:
    """Memory model parses correctly."""
    m = Memory(
        id="mem-1",
        agent_id="agent-1",
        content="test",
        memory_type="episodic",
        importance_score=0.5,
        recall_count=0,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert m.id == "mem-1"
    assert m.metadata == {}


def test_memory_with_metadata() -> None:
    """Memory model with metadata."""
    m = Memory(
        id="mem-1",
        agent_id="agent-1",
        content="test",
        memory_type="semantic",
        importance_score=0.9,
        recall_count=5,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        metadata={"tag": "important"},
    )
    assert m.metadata["tag"] == "important"


def test_recall_result() -> None:
    """RecallResult model construction."""
    r = RecallResult(
        memory_id="mem-1",
        content="test",
        memory_type="episodic",
        similarity_score=0.95,
        importance_score=0.5,
        relevance_score=0.8,
        recall_count=1,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert r.similarity_score == 0.95
    assert r.summary is None


def test_recall_response() -> None:
    """RecallResponse with results."""
    resp = RecallResponse(results=[], total=0, query="test")
    assert resp.total == 0
    assert resp.query == "test"


def test_forget_result() -> None:
    """ForgetResult model."""
    f = ForgetResult(
        deleted_count=3,
        hard_deleted=True,
        cascade_count=1,
        memory_ids=["a", "b", "c"],
    )
    assert f.deleted_count == 3
    assert f.hard_deleted is True


def test_audit_entry() -> None:
    """AuditEntry model."""
    e = AuditEntry(
        id=1,
        operation="store",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert e.operation == "store"
    assert e.agent_id is None


def test_audit_page() -> None:
    """AuditPage model."""
    p = AuditPage(entries=[], total=0, page=1, page_size=50, has_next=False)
    assert p.has_next is False


def test_relationship() -> None:
    """Relationship model."""
    r = Relationship(
        target_memory_id="mem-2",
        relationship_type=RelationshipType.SUPPORTS,
    )
    assert r.weight == 1.0
    assert r.metadata == {}


def test_session_model() -> None:
    """Session model."""
    s = Session(
        session_id="sess-1",
        agent_id="agent-1",
        session_type="conversation",
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert s.session_type == "conversation"
