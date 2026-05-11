"""Mastra (Python) adapter for Z3rno (v0.19.6).

Mastra's official runtime is TypeScript, but the community Python
port (``mastra-py``) accepts the same memory contract:
``get_messages`` / ``add_message`` / ``clear``. This adapter mirrors
the Z3rno TS adapter shape so users porting an agent between
runtimes don't have to relearn the memory wiring.

Conversation-scoped when ``conversation_id`` is supplied; otherwise
falls back to recall-based history. ``clear()`` is a deliberate
no-op — Z3rno keeps audit-grade history and recall is already
session-scoped.

No hard dep on ``mastra``; the typed shape is duck-compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from z3rno.client import Z3rnoClient

MastraRole = Literal["user", "assistant", "system", "tool"]


@dataclass(frozen=True)
class MastraMessage:
    role: MastraRole
    content: str
    thread_id: str | None = None
    resource_id: str | None = None


@dataclass
class Z3rnoMastraMemory:
    """Mastra-compatible memory backed by Z3rno conversations.

    Drop into a Mastra ``Agent``::

        memory = Z3rnoMastraMemory(
            client=client, agent_id="support", conversation_id=conv.id
        )
        agent = Agent(name="support", memory=memory)
    """

    client: Z3rnoClient
    agent_id: str
    conversation_id: str | None = None
    top_k: int = 50
    _last_role: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def get_messages(self, *, limit: int | None = None) -> list[MastraMessage]:
        """Mastra contract: return ordered prior messages."""
        n = limit or self.top_k
        if self.conversation_id:
            page = self.client.list_turns(self.conversation_id, limit=n)
            return [
                MastraMessage(
                    role=self._normalise_role(t.turn_role),
                    content=t.content,
                    thread_id=self.conversation_id,
                )
                for t in page.turns
            ]
        # No conversation — recall-based history (newest first → reverse).
        resp = self.client.recall(agent_id=self.agent_id, top_k=n, memory_type="episodic")
        out: list[MastraMessage] = []
        for r in reversed(resp.results):
            md: dict[str, Any] = dict(r.metadata or {})
            out.append(
                MastraMessage(
                    role=self._normalise_role(md.get("role")),
                    content=r.content,
                )
            )
        return out

    def add_message(self, message: MastraMessage) -> None:
        """Mastra contract: persist one message."""
        memory = self.client.store(
            agent_id=self.agent_id,
            content=message.content,
            memory_type="episodic",
            metadata={"role": message.role},
        )
        if self.conversation_id:
            self.client.add_turn(
                self.conversation_id,
                memory_id=memory.id,
                turn_role=message.role,
            )

    def clear(self) -> None:
        """Deliberate no-op. Z3rno keeps audit-grade history and recall
        is already conversation-scoped — flushing here would risk
        losing lineage."""
        return None

    @staticmethod
    def _normalise_role(raw: Any) -> MastraRole:
        s = str(raw).lower() if raw is not None else "user"
        if s in ("assistant", "ai"):
            return "assistant"
        if s == "system":
            return "system"
        if s == "tool":
            return "tool"
        return "user"
