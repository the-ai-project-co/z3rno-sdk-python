"""Tests for the Forge SDK methods — ingest / distill / refine.

Mocks the HTTP layer via pytest-httpx; verifies the request shape +
the Pydantic response parse. No live server.
"""

from __future__ import annotations

import pytest
from pytest_httpx import HTTPXMock

from z3rno import (
    AsyncZ3rnoClient,
    DistillJob,
    DistillJobStatus,
    IngestJob,
    IngestJobStatus,
    RefineJob,
    RefineJobStatus,
    Z3rnoClient,
)


@pytest.fixture
def client() -> Z3rnoClient:
    return Z3rnoClient(base_url="http://test.z3rno.dev", api_key="test-key")


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


def test_ingest_text(client: Z3rnoClient, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="http://test.z3rno.dev/v1/ingest",
        method="POST",
        match_json={
            "kind": "text",
            "agent_id": "agent-1",
            "text": "hello world",
        },
        json={
            "job_id": "ij-1",
            "kind": "text",
            "status": "queued",
            "dataset_id": None,
            "enqueued_at": "2026-05-11T00:00:00Z",
        },
    )
    job = client.ingest_text(agent_id="agent-1", text="hello world")
    assert isinstance(job, IngestJob)
    assert job.job_id == "ij-1"
    assert job.kind == "text"


def test_ingest_text_threads_dataset_id(client: Z3rnoClient, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="http://test.z3rno.dev/v1/ingest",
        method="POST",
        match_json={
            "kind": "text",
            "agent_id": "a",
            "text": "x",
            "dataset_id": "ds-1",
        },
        json={
            "job_id": "j",
            "kind": "text",
            "status": "queued",
            "dataset_id": "ds-1",
            "enqueued_at": "2026-05-11T00:00:00Z",
        },
    )
    job = client.ingest_text(agent_id="a", text="x", dataset_id="ds-1")
    assert job.dataset_id == "ds-1"


def test_ingest_url(client: Z3rnoClient, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="http://test.z3rno.dev/v1/ingest",
        method="POST",
        match_json={
            "kind": "url",
            "agent_id": "agent-1",
            "url": "https://example.com",
        },
        json={
            "job_id": "ij-url",
            "kind": "url",
            "status": "queued",
            "dataset_id": None,
            "enqueued_at": "2026-05-11T00:00:00Z",
        },
    )
    job = client.ingest_url(agent_id="agent-1", url="https://example.com")
    assert job.kind == "url"


def test_get_ingest_status(client: Z3rnoClient, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="http://test.z3rno.dev/v1/ingest/ij-1",
        method="GET",
        json={
            "job_id": "ij-1",
            "agent_id": "agent-1",
            "dataset_id": None,
            "kind": "text",
            "status": "completed",
            "source_uri": None,
            "content_type": None,
            "filename": None,
            "file_size": 11,
            "memory_ids": ["mem-1"],
            "memos_written": 1,
            "distill_job_id": None,
            "codegraph_memos_written": 0,
            "codegraph_edges_written": 0,
            "error": None,
            "warnings": [],
            "started_at": "2026-05-11T00:00:00Z",
            "completed_at": "2026-05-11T00:00:01Z",
            "created_at": "2026-05-11T00:00:00Z",
            "updated_at": "2026-05-11T00:00:01Z",
        },
    )
    status = client.get_ingest_status("ij-1")
    assert isinstance(status, IngestJobStatus)
    assert status.status == "completed"
    assert status.memory_ids == ["mem-1"]


# ---------------------------------------------------------------------------
# distill
# ---------------------------------------------------------------------------


def test_distill(client: Z3rnoClient, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="http://test.z3rno.dev/v1/distill",
        method="POST",
        match_json={
            "agent_id": "agent-1",
            "memory_ids": ["m-1", "m-2"],
            "include_summary": True,
        },
        json={
            "job_id": "dj-1",
            "status": "queued",
            "memory_ids": ["m-1", "m-2"],
            "enqueued_at": "2026-05-11T00:00:00Z",
        },
    )
    job = client.distill(agent_id="agent-1", memory_ids=["m-1", "m-2"])
    assert isinstance(job, DistillJob)
    assert job.job_id == "dj-1"


def test_distill_with_tuning(client: Z3rnoClient, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="http://test.z3rno.dev/v1/distill",
        method="POST",
        match_json={
            "agent_id": "a",
            "memory_ids": ["m"],
            "include_summary": False,
            "chunk_size": 512,
            "chunk_overlap": 64,
            "max_concurrency": 2,
            "summary_style": "bullet",
        },
        json={
            "job_id": "dj",
            "status": "queued",
            "memory_ids": ["m"],
            "enqueued_at": "2026-05-11T00:00:00Z",
        },
    )
    client.distill(
        agent_id="a",
        memory_ids=["m"],
        chunk_size=512,
        chunk_overlap=64,
        max_concurrency=2,
        summary_style="bullet",
        include_summary=False,
    )


def test_get_distill_status(client: Z3rnoClient, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="http://test.z3rno.dev/v1/distill/dj-1",
        method="GET",
        json={
            "job_id": "dj-1",
            "agent_id": "agent-1",
            "status": "completed",
            "model": "openai/gpt-4o-mini",
            "memory_ids": ["m-1"],
            "chunk_size": 1024,
            "chunk_overlap": 128,
            "max_concurrency": 4,
            "chunks_total": 3,
            "chunks_failed": 0,
            "entities_extracted": 12,
            "relationships_extracted": 5,
            "memos_written": 17,
            "error": None,
        },
    )
    status = client.get_distill_status("dj-1")
    assert isinstance(status, DistillJobStatus)
    assert status.memos_written == 17


# ---------------------------------------------------------------------------
# refine
# ---------------------------------------------------------------------------


def test_refine_without_dataset(client: Z3rnoClient, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="http://test.z3rno.dev/v1/refine",
        method="POST",
        match_json={},
        json={
            "job_id": "rj-1",
            "status": "queued",
            "dataset_id": None,
            "enqueued_at": "2026-05-11T00:00:00Z",
        },
    )
    job = client.refine()
    assert isinstance(job, RefineJob)
    assert job.dataset_id is None


def test_refine_with_dataset(client: Z3rnoClient, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="http://test.z3rno.dev/v1/refine",
        method="POST",
        match_json={"dataset_id": "ds-1"},
        json={
            "job_id": "rj-2",
            "status": "queued",
            "dataset_id": "ds-1",
            "enqueued_at": "2026-05-11T00:00:00Z",
        },
    )
    job = client.refine(dataset_id="ds-1")
    assert job.dataset_id == "ds-1"


def test_get_refine_status(client: Z3rnoClient, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="http://test.z3rno.dev/v1/refine/rj-1",
        method="GET",
        json={
            "job_id": "rj-1",
            "status": "completed",
            "dataset_id": None,
            "trigger": "api",
            "memos_scanned": 100,
            "memos_deduped": 12,
            "edges_reweighted": 5,
            "edges_pruned": 2,
            "feedback_drained": 18,
            "job_metadata": {},
            "error": None,
        },
    )
    status = client.get_refine_status("rj-1")
    assert isinstance(status, RefineJobStatus)
    assert status.memos_deduped == 12
    assert status.feedback_drained == 18


# ---------------------------------------------------------------------------
# Async smoke — verifies the async client mirrors the sync surface.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_ingest_text(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="http://test.z3rno.dev/v1/ingest",
        method="POST",
        json={
            "job_id": "ij-async",
            "kind": "text",
            "status": "queued",
            "dataset_id": None,
            "enqueued_at": "2026-05-11T00:00:00Z",
        },
    )
    client = AsyncZ3rnoClient(base_url="http://test.z3rno.dev", api_key="k")
    try:
        job = await client.ingest_text(agent_id="a", text="x")
        assert job.job_id == "ij-async"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_async_refine(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="http://test.z3rno.dev/v1/refine",
        method="POST",
        json={
            "job_id": "rj-async",
            "status": "queued",
            "dataset_id": None,
            "enqueued_at": "2026-05-11T00:00:00Z",
        },
    )
    client = AsyncZ3rnoClient(base_url="http://test.z3rno.dev", api_key="k")
    try:
        job = await client.refine()
        assert job.job_id == "rj-async"
    finally:
        await client.close()
