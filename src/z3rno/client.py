"""Synchronous Z3rno client using httpx."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx

from z3rno.config import Z3rnoConfig
from z3rno.exceptions import (
    AuthenticationError,
    NotFoundError,
    RateLimitError,
    ServerError,
    ValidationError,
    Z3rnoError,
)
from z3rno.models import (
    AuditPage,
    ForgetResult,
    Memory,
    MemoryType,
    RecallResponse,
    Relationship,
    Session,
)


class Z3rnoClient:
    """Synchronous Z3rno SDK client.

    Usage::

        client = Z3rnoClient(base_url="http://localhost:8000", api_key="z3rno_sk_...")
        memory = client.store(agent_id="agent-1", content="User prefers dark mode")
        results = client.recall(agent_id="agent-1", query="user preferences")
        client.forget(memory_id=memory.id)
    """

    def __init__(
        self,
        base_url: str = "https://api.z3rno.dev",
        api_key: str = "",
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self._config = Z3rnoConfig(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._http = httpx.Client(
            base_url=self._config.base_url,
            headers=self._config.auth_headers,
            timeout=timeout,
        )

    def close(self) -> None:
        """Close the HTTP client."""
        self._http.close()

    def __enter__(self) -> Z3rnoClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # --- Store ---

    def store(
        self,
        *,
        agent_id: str,
        content: str,
        memory_type: str | MemoryType = "episodic",
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        relationships: list[Relationship] | None = None,
        ttl_seconds: int | None = None,
        importance: float | None = None,
    ) -> Memory:
        """Store a new memory."""
        body: dict[str, Any] = {
            "agent_id": agent_id,
            "content": content,
            "memory_type": str(memory_type),
        }
        if user_id:
            body["user_id"] = user_id
        if metadata:
            body["metadata"] = metadata
        if relationships:
            body["relationships"] = [r.model_dump() for r in relationships]
        if ttl_seconds is not None:
            body["ttl_seconds"] = ttl_seconds
        if importance is not None:
            body["importance"] = importance

        resp = self._request("POST", "/v1/memories", json=body)
        return Memory.model_validate(resp)

    # --- Recall ---

    def recall(
        self,
        *,
        agent_id: str,
        query: str | None = None,
        memory_type: str | None = None,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        similarity_threshold: float = 0.0,
        time_range: tuple[datetime, datetime] | None = None,
        as_of: datetime | None = None,
    ) -> RecallResponse:
        """Recall memories by query."""
        body: dict[str, Any] = {"agent_id": agent_id, "top_k": top_k}
        if query:
            body["query"] = query
        if memory_type:
            body["memory_type"] = memory_type
        if filters:
            body["filters"] = filters
        if similarity_threshold > 0:
            body["similarity_threshold"] = similarity_threshold
        if time_range:
            body["time_range"] = [t.isoformat() for t in time_range]
        if as_of:
            body["as_of"] = as_of.isoformat()

        resp = self._request("POST", "/v1/memories/recall", json=body)
        return RecallResponse.model_validate(resp)

    # --- Forget ---

    def forget(
        self,
        *,
        agent_id: str,
        memory_id: str | None = None,
        memory_ids: list[str] | None = None,
        hard_delete: bool = False,
        cascade: bool = False,
        reason: str | None = None,
    ) -> ForgetResult:
        """Forget (delete) memories."""
        body: dict[str, Any] = {"agent_id": agent_id}
        if memory_id:
            body["memory_id"] = memory_id
        if memory_ids:
            body["memory_ids"] = memory_ids
        if hard_delete:
            body["hard_delete"] = True
        if cascade:
            body["cascade"] = True
        if reason:
            body["reason"] = reason

        resp = self._request("POST", "/v1/memories/forget", json=body)
        return ForgetResult.model_validate(resp)

    # --- Audit ---

    def audit(
        self,
        *,
        agent_id: str | None = None,
        operation: str | None = None,
        memory_id: str | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> AuditPage:
        """Query the audit log."""
        params: dict[str, Any] = {"page": page, "page_size": page_size}
        if agent_id:
            params["agent_id"] = agent_id
        if operation:
            params["operation"] = operation
        if memory_id:
            params["memory_id"] = memory_id

        resp = self._request("GET", "/v1/audit", params=params)
        return AuditPage.model_validate(resp)

    # --- Sessions ---

    def start_session(
        self,
        *,
        agent_id: str,
        session_type: str = "conversation",
    ) -> Session:
        """Start a new session."""
        resp = self._request(
            "POST",
            "/v1/sessions",
            json={"agent_id": agent_id, "session_type": session_type},
        )
        return Session.model_validate(resp)

    def end_session(self, session_id: str) -> dict[str, Any]:
        """End a session."""
        return self._request("POST", f"/v1/sessions/{session_id}/end")

    # --- HTTP layer ---

    def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make an HTTP request and handle errors."""
        resp = self._http.request(method, path, **kwargs)
        return self._handle_response(resp)

    @staticmethod
    def _handle_response(resp: httpx.Response) -> dict[str, Any]:
        """Parse response and raise appropriate exceptions."""
        if resp.status_code in {200, 201}:
            return resp.json()  # type: ignore[no-any-return]

        body = (
            resp.json()
            if resp.headers.get("content-type", "").startswith("application/json")
            else {}
        )
        detail = body.get("detail", body.get("error", resp.text))

        if resp.status_code == 401:
            raise AuthenticationError(f"Authentication failed: {detail}", status_code=401)
        if resp.status_code == 404:
            raise NotFoundError(f"Not found: {detail}", status_code=404)
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", "60"))
            raise RateLimitError(f"Rate limit exceeded: {detail}", retry_after=retry_after)
        if resp.status_code in (400, 422):
            raise ValidationError(f"Validation error: {detail}", status_code=resp.status_code)
        if resp.status_code >= 500:
            raise ServerError(f"Server error: {detail}", status_code=resp.status_code)
        raise Z3rnoError(
            f"Unexpected error ({resp.status_code}): {detail}", status_code=resp.status_code
        )
