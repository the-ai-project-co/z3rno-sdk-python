"""Async Z3rno client using httpx.AsyncClient."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import httpx
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
)

from z3rno.config import Z3rnoConfig
from z3rno.exceptions import (
    AuthenticationError,
    NotFoundError,
    RateLimitError,
    ServerError,
    ValidationError,
    Z3rnoConnectionError,
    Z3rnoError,
)
from z3rno.models import (
    AuditPage,
    BatchStoreResponse,
    ForgetResult,
    Memory,
    MemoryHistoryResponse,
    MemoryType,
    RecallResponse,
    RecallResult,
    Relationship,
    Session,
)


def _should_retry_async(exc: BaseException) -> bool:
    """Return True for retryable exceptions (5xx, connection errors, 429)."""
    if isinstance(exc, ServerError):
        return True
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, Z3rnoConnectionError):
        return True
    return isinstance(exc, httpx.ConnectError | httpx.ReadTimeout | httpx.WriteTimeout)


def _wait_with_retry_after_async(retry_state: RetryCallState) -> float:
    """Wait function that honors Retry-After header on 429 responses."""
    exc = retry_state.outcome and retry_state.outcome.exception()
    if isinstance(exc, RateLimitError) and exc.retry_after:
        return float(exc.retry_after)
    # Exponential backoff: 1s, 2s, 4s, ...
    attempt = retry_state.attempt_number
    return min(2 ** (attempt - 1), 8)


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
        proxy: str | None = None,
    ) -> None:
        self._config = Z3rnoConfig(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            proxy=proxy,
        )
        client_kwargs: dict[str, Any] = {
            "base_url": self._config.base_url,
            "headers": self._config.auth_headers,
            "timeout": timeout,
        }
        if self._config.proxy:
            client_kwargs["proxy"] = self._config.proxy
        self._http = httpx.AsyncClient(**client_kwargs)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._http.aclose()

    async def __aenter__(self) -> AsyncZ3rnoClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    # --- Get Memory ---

    async def get_memory(self, memory_id: str, *, timeout: float | None = None) -> Memory:
        """Retrieve a single memory by ID."""
        resp = await self._request("GET", f"/v1/memories/{memory_id}", timeout=timeout)
        return Memory.model_validate(resp)

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
        timeout: float | None = None,
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

        resp = await self._request("POST", "/v1/memories", json=body, timeout=timeout)
        return Memory.model_validate(resp)

    # --- Store Batch ---

    async def store_batch(
        self,
        *,
        agent_id: str,
        memories: list[dict[str, Any]],
        timeout: float | None = None,
    ) -> BatchStoreResponse:
        """Store multiple memories in a single request."""
        items = []
        for mem in memories:
            item: dict[str, Any] = {
                "agent_id": agent_id,
                "content": mem["content"],
                "memory_type": mem.get("memory_type", "episodic"),
            }
            if "metadata" in mem:
                item["metadata"] = mem["metadata"]
            if "importance" in mem:
                item["importance"] = mem["importance"]
            items.append(item)

        resp = await self._request(
            "POST", "/v1/memories/batch", json={"memories": items}, timeout=timeout
        )
        return BatchStoreResponse.model_validate(resp)

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
        time_range: tuple[datetime, datetime] | None = None,
        as_of: datetime | None = None,
        timeout: float | None = None,
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

        resp = await self._request("POST", "/v1/memories/recall", json=body, timeout=timeout)
        return RecallResponse.model_validate(resp)

    async def recall_stream(
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
        timeout: float | None = None,
    ) -> AsyncIterator[RecallResult]:
        """Recall memories, yielding results one at a time.

        This is a client-side streaming wrapper around :meth:`recall` that
        reduces peak memory usage for large result sets.
        """
        response = await self.recall(
            agent_id=agent_id,
            query=query,
            memory_type=memory_type,
            filters=filters,
            time_range=time_range,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            as_of=as_of,
            timeout=timeout,
        )
        for result in response.results:
            yield result

    # --- Forget ---

    async def forget(
        self,
        *,
        agent_id: str,
        memory_id: str | None = None,
        memory_ids: list[str] | None = None,
        hard_delete: bool = False,
        timeout: float | None = None,
    ) -> ForgetResult:
        """Forget (delete) memories."""
        body: dict[str, Any] = {"agent_id": agent_id}
        if memory_id:
            body["memory_id"] = memory_id
        if memory_ids:
            body["memory_ids"] = memory_ids
        if hard_delete:
            body["hard_delete"] = True

        resp = await self._request("POST", "/v1/memories/forget", json=body, timeout=timeout)
        return ForgetResult.model_validate(resp)

    # --- Audit ---

    async def audit(
        self,
        *,
        agent_id: str | None = None,
        page: int = 1,
        page_size: int = 50,
        timeout: float | None = None,
    ) -> AuditPage:
        """Query the audit log."""
        params: dict[str, Any] = {"page": page, "page_size": page_size}
        if agent_id:
            params["agent_id"] = agent_id

        resp = await self._request("GET", "/v1/audit", params=params, timeout=timeout)
        return AuditPage.model_validate(resp)

    # --- Memory History ---

    async def get_memory_history(
        self, memory_id: str, *, timeout: float | None = None
    ) -> MemoryHistoryResponse:
        """Retrieve the version history of a memory."""
        resp = await self._request("GET", f"/v1/memories/{memory_id}/history", timeout=timeout)
        return MemoryHistoryResponse.model_validate(resp)

    # --- Update Memory ---

    async def update_memory(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
        importance: float | None = None,
        timeout: float | None = None,
    ) -> Memory:
        """Update a memory's content, metadata, or importance."""
        body: dict[str, Any] = {}
        if content is not None:
            body["content"] = content
        if metadata is not None:
            body["metadata"] = metadata
        if importance is not None:
            body["importance"] = importance

        resp = await self._request("PATCH", f"/v1/memories/{memory_id}", json=body, timeout=timeout)
        return Memory.model_validate(resp)

    # --- Sessions ---

    async def start_session(
        self,
        *,
        agent_id: str,
        session_type: str = "conversation",
        timeout: float | None = None,
    ) -> Session:
        """Start a new session."""
        resp = await self._request(
            "POST",
            "/v1/sessions",
            json={"agent_id": agent_id, "session_type": session_type},
            timeout=timeout,
        )
        return Session.model_validate(resp)

    async def end_session(self, session_id: str, *, timeout: float | None = None) -> dict[str, Any]:
        """End a session."""
        return await self._request("POST", f"/v1/sessions/{session_id}/end", timeout=timeout)

    @asynccontextmanager
    async def session(
        self, *, agent_id: str, session_type: str = "conversation"
    ) -> AsyncIterator[Session]:
        """Async context manager that starts a session and ensures it is ended.

        Usage::

            async with client.session(agent_id="agent-1") as sess:
                await client.store(agent_id="agent-1", content="...",
                                   metadata={"session_id": str(sess.session_id)})
        """
        sess = await self.start_session(agent_id=agent_id, session_type=session_type)
        try:
            yield sess
        finally:
            await self.end_session(str(sess.session_id))

    # --- HTTP layer ---

    async def _request(
        self, method: str, path: str, *, timeout: float | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Make an async HTTP request with automatic retry on transient failures.

        Args:
            timeout: Per-request timeout override. When provided, overrides the
                default client timeout for this specific request.
        """
        if timeout is not None:
            kwargs["timeout"] = timeout
        retrying = retry(
            retry=retry_if_exception(_should_retry_async),
            wait=_wait_with_retry_after_async,
            stop=stop_after_attempt(self._config.max_retries),
            reraise=True,
        )
        return await retrying(self._do_request)(method, path, **kwargs)

    async def _do_request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        """Execute a single async HTTP request and handle the response."""
        try:
            resp = await self._http.request(method, path, **kwargs)
        except (
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.ConnectTimeout,
        ) as exc:
            raise Z3rnoConnectionError(f"Connection failed: {exc}") from exc
        return _handle_response(resp)


def _handle_response(resp: httpx.Response) -> dict[str, Any]:
    """Parse response and raise exceptions."""
    if resp.status_code in (200, 201):
        return resp.json()  # type: ignore[no-any-return]

    body = resp.json() if "application/json" in resp.headers.get("content-type", "") else {}
    detail = body.get("detail", body.get("error", resp.text))

    if resp.status_code == 401:
        raise AuthenticationError(f"Authentication failed: {detail}")
    if resp.status_code == 404:
        raise NotFoundError(f"Not found: {detail}")
    if resp.status_code == 429:
        retry = int(resp.headers.get("Retry-After", "60"))
        raise RateLimitError(f"Rate limit exceeded: {detail}", retry_after=retry)
    if resp.status_code in (400, 422):
        raise ValidationError(f"Validation error: {detail}", status_code=resp.status_code)
    if resp.status_code >= 500:
        raise ServerError(f"Server error: {detail}", status_code=resp.status_code)
    raise Z3rnoError(f"Unexpected ({resp.status_code}): {detail}", status_code=resp.status_code)
