"""v0.19.6 — Mastra Python adapter tests.

Mirrors the TS adapter contract: ``get_messages`` returns ordered
history, ``add_message`` persists, ``clear`` is a deliberate no-op.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import uuid4

from z3rno.integrations.mastra import MastraMessage, Z3rnoMastraMemory


def _fake_result(content: str, role: str = "user") -> MagicMock:
    r = MagicMock()
    r.memory_id = str(uuid4())
    r.content = content
    r.memory_type = "episodic"
    r.metadata = {"role": role}
    r.relevance_score = 0.5
    r.created_at = datetime.now(UTC)
    return r


def _fake_client() -> MagicMock:
    client = MagicMock()
    resp = MagicMock()
    resp.results = []
    client.recall.return_value = resp
    memory = MagicMock()
    memory.id = str(uuid4())
    client.store.return_value = memory
    client.add_turn.return_value = MagicMock(turn_index=1, needs_summary=False)
    client.list_turns.return_value = MagicMock(turns=[])
    return client


def test_get_messages_uses_list_turns_when_conversation_id_set() -> None:
    client = _fake_client()
    conv = str(uuid4())
    turn = MagicMock()
    turn.memory_id = str(uuid4())
    turn.turn_index = 1
    turn.turn_role = "assistant"
    turn.content = "hello"
    turn.created_at = datetime.now(UTC)
    client.list_turns.return_value = MagicMock(turns=[turn])

    mem = Z3rnoMastraMemory(client=client, agent_id="a-1", conversation_id=conv)
    msgs = mem.get_messages()

    assert len(msgs) == 1
    assert msgs[0].role == "assistant"
    assert msgs[0].thread_id == conv
    client.list_turns.assert_called_once_with(conv, limit=50)


def test_get_messages_falls_back_to_recall_without_conversation() -> None:
    client = _fake_client()
    resp = MagicMock()
    resp.results = [_fake_result("a"), _fake_result("b", role="ai")]
    client.recall.return_value = resp

    mem = Z3rnoMastraMemory(client=client, agent_id="a-1")
    msgs = mem.get_messages()
    # recall is newest-first; adapter reverses to chronological.
    assert [m.content for m in msgs] == ["b", "a"]
    # "ai" → "assistant" normalisation.
    assert msgs[0].role == "assistant"


def test_add_message_stores_and_pushes_turn_when_scoped() -> None:
    client = _fake_client()
    conv = str(uuid4())
    mem = Z3rnoMastraMemory(client=client, agent_id="a-1", conversation_id=conv)
    mem.add_message(MastraMessage(role="user", content="hi"))
    client.store.assert_called_once()
    client.add_turn.assert_called_once()
    kwargs = client.add_turn.call_args.kwargs
    assert kwargs["turn_role"] == "user"


def test_add_message_skips_add_turn_without_conversation() -> None:
    client = _fake_client()
    mem = Z3rnoMastraMemory(client=client, agent_id="a-1")
    mem.add_message(MastraMessage(role="user", content="hi"))
    client.store.assert_called_once()
    client.add_turn.assert_not_called()


def test_clear_is_noop() -> None:
    client = _fake_client()
    mem = Z3rnoMastraMemory(client=client, agent_id="a-1")
    mem.clear()
    client.forget.assert_not_called()


def test_unknown_role_normalises_to_user() -> None:
    client = _fake_client()
    resp = MagicMock()
    r = _fake_result("x", role="bogus_role")
    resp.results = [r]
    client.recall.return_value = resp
    mem = Z3rnoMastraMemory(client=client, agent_id="a-1")
    msgs = mem.get_messages()
    assert msgs[0].role == "user"
