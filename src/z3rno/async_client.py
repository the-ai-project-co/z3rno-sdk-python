"""Async Z3rno client using httpx.AsyncClient."""

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


class AsyncZ3rnoClient:
    """Async Z3rno SDK client.

    Usage::

        async with AsyncZ3rnoClient(base_url="...", api_key="...") as client:
            memory = await client.store(agent_id="agent-1", content="...")
            results = await client.recall(agent_id="agent-1", query="...")
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
        self._http = httpx.AsyncClient(
            base_url=self._config.base_url,
            headers=self._config.auth_headers,
            timeout=timeout,
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._http.aclose()

    async def __aenter__(self) -> AsyncZ3rnoClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    # --- Store ---

    async def store(
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

        resp = await self._request("POST", "/v1/memories", json=body)
        return Memory.model_validate(resp)

    # --- Recall ---

    async def recall(
        self,
        *,
        agent_id: str,
        query: str | None = None,
        memory_type: str | None = None,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        similarity_threshold: float = 0.0,
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
        if as_of:
            body["as_of"] = as_of.isoformat()

        resp = await self._request("POST", "/v1/memories/recall", json=body)
        return RecallResponse.model_validate(resp)

    # --- Forget ---

    async def forget(
        self,
        *,
        agent_id: str,
        memory_id: str | None = None,
        memory_ids: list[str] | None = None,
        hard_delete: bool = False,
    ) -> ForgetResult:
        """Forget (delete) memories."""
        body: dict[str, Any] = {"agent_id": agent_id}
        if memory_id:
            body["memory_id"] = memory_id
        if memory_ids:
            body["memory_ids"] = memory_ids
        if hard_delete:
            body["hard_delete"] = True

        resp = await self._request("POST", "/v1/memories/forget", json=body)
        return ForgetResult.model_validate(resp)

    # --- Audit ---

    async def audit(
        self,
        *,
        agent_id: str | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> AuditPage:
        """Query the audit log."""
        params: dict[str, Any] = {"page": page, "page_size": page_size}
        if agent_id:
            params["agent_id"] = agent_id

        resp = await self._request("GET", "/v1/audit", params=params)
        return AuditPage.model_validate(resp)

    # --- Sessions ---

    async def start_session(self, *, agent_id: str) -> Session:
        """Start a new session."""
        resp = await self._request("POST", "/v1/sessions", json={"agent_id": agent_id})
        return Session.model_validate(resp)

    # --- HTTP layer ---

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        """Make an async HTTP request."""
        resp = await self._http.request(method, path, **kwargs)
        return _handle_response(resp)


def _handle_response(resp: httpx.Response) -> dict[str, Any]:
    """Parse response and raise exceptions."""
    if resp.status_code in (200, 201):
        return resp.json()  # type: ignore[no-any-return]

    body = resp.json() if "application/json" in resp.headers.get("content-type", "") else {}
    detail = body.get("detail", body.get("error", resp.text))

    if resp.status_code == 401:
        raise AuthenticationError(f"Authentication failed: {detail}", status_code=401)
    if resp.status_code == 404:
        raise NotFoundError(f"Not found: {detail}", status_code=404)
    if resp.status_code == 429:
        retry = int(resp.headers.get("Retry-After", "60"))
        raise RateLimitError(f"Rate limit exceeded: {detail}", retry_after=retry)
    if resp.status_code in (400, 422):
        raise ValidationError(f"Validation error: {detail}", status_code=resp.status_code)
    if resp.status_code >= 500:
        raise ServerError(f"Server error: {detail}", status_code=resp.status_code)
    raise Z3rnoError(f"Unexpected ({resp.status_code}): {detail}", status_code=resp.status_code)
