"""Phase G slice 2 — conversation SDK methods.

Mocks the HTTP layer with pytest-httpx; pins the path + body + parsed
response model for every new method.
"""

from __future__ import annotations

import json as _json
from datetime import UTC, datetime
from uuid import uuid4

from pytest_httpx import HTTPXMock

from z3rno import Z3rnoClient


def _client(httpx_mock: HTTPXMock) -> Z3rnoClient:
    return Z3rnoClient(base_url="http://test", api_key="sk-test")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def test_create_conversation_posts_and_parses(httpx_mock: HTTPXMock) -> None:
    cid = str(uuid4())
    agent = str(uuid4())
    httpx_mock.add_response(
        method="POST",
        url="http://test/v1/conversations",
        json={
            "id": cid,
            "agent_id": agent,
            "user_id": None,
            "title": "demo",
            "summary_cadence": 5,
            "turn_count": 0,
            "last_summary_turn": 0,
            "metadata": {},
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        },
    )
    client = _client(httpx_mock)
    conv = client.create_conversation(
        agent_id=agent, title="demo", summary_cadence=5
    )
    assert conv.id == cid
    assert conv.summary_cadence == 5
    assert conv.title == "demo"


def test_get_conversation(httpx_mock: HTTPXMock) -> None:
    cid = str(uuid4())
    agent = str(uuid4())
    httpx_mock.add_response(
        method="GET",
        url=f"http://test/v1/conversations/{cid}",
        json={
            "id": cid,
            "agent_id": agent,
            "user_id": None,
            "title": None,
            "summary_cadence": 10,
            "turn_count": 3,
            "last_summary_turn": 0,
            "metadata": {},
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        },
    )
    client = _client(httpx_mock)
    conv = client.get_conversation(cid)
    assert conv.turn_count == 3


def test_add_turn_returns_index_and_needs_summary(httpx_mock: HTTPXMock) -> None:
    cid = str(uuid4())
    mid = str(uuid4())
    httpx_mock.add_response(
        method="POST",
        url=f"http://test/v1/conversations/{cid}/turns",
        json={"turn_index": 11, "needs_summary": True},
    )
    client = _client(httpx_mock)
    resp = client.add_turn(cid, memory_id=mid, turn_role="assistant")
    assert resp.turn_index == 11
    assert resp.needs_summary is True


def test_list_turns_with_pagination(httpx_mock: HTTPXMock) -> None:
    cid = str(uuid4())
    mid = str(uuid4())
    now = _now_iso()
    httpx_mock.add_response(
        method="GET",
        url=f"http://test/v1/conversations/{cid}/turns?limit=10&after_turn=5",
        json={
            "turns": [
                {
                    "memory_id": mid,
                    "turn_index": 6,
                    "turn_role": "user",
                    "content": "hello",
                    "created_at": now,
                }
            ],
            "total": 1,
            "conversation_id": cid,
        },
    )
    client = _client(httpx_mock)
    out = client.list_turns(cid, after_turn=5, limit=10)
    assert out.total == 1
    assert out.turns[0].turn_index == 6
    assert out.turns[0].turn_role == "user"


def test_recall_forwards_conversation_id(httpx_mock: HTTPXMock) -> None:
    agent = str(uuid4())
    cid = str(uuid4())
    httpx_mock.add_response(
        method="POST",
        url="http://test/v1/memories/recall",
        json={
            "results": [],
            "total": 0,
            "query": "q",
            "strategy_used": "AUTO",
            "strategies_considered": [],
            "reranked": False,
            "elapsed_ms": 1.0,
        },
    )
    client = _client(httpx_mock)
    client.recall(agent_id=agent, query="q", conversation_id=cid)
    request = httpx_mock.get_request()
    assert request is not None
    body = _json.loads(request.content)
    assert body["conversation_id"] == cid
