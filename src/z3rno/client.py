"""Synchronous Z3rno client using httpx."""

from __future__ import annotations

from collections.abc import Generator, Iterator
from contextlib import contextmanager
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
from z3rno.logging import elapsed_ms, log_request, logging_enabled, start_timer
from z3rno.models import (
    AuditPage,
    BatchStoreResponse,
    Conversation,
    DistillJob,
    DistillJobStatus,
    ForgetResult,
    IngestJob,
    IngestJobStatus,
    Memory,
    MemoryHistoryResponse,
    MemoryType,
    RecallResponse,
    RecallResult,
    RefineJob,
    RefineJobStatus,
    Relationship,
    Session,
    TenantBudgets,
    TenantBudgetsView,
    TurnAddResponse,
    TurnListResponse,
)


def _should_retry(exc: BaseException) -> bool:
    """Return True for retryable exceptions (5xx, connection errors, 429)."""
    if isinstance(exc, ServerError):
        return True
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, Z3rnoConnectionError):
        return True
    return isinstance(exc, httpx.ConnectError | httpx.ReadTimeout | httpx.WriteTimeout)


def _wait_with_retry_after(retry_state: RetryCallState) -> float:
    """Wait function that honors Retry-After header on 429 responses."""
    exc = retry_state.outcome and retry_state.outcome.exception()
    if isinstance(exc, RateLimitError) and exc.retry_after:
        return float(exc.retry_after)
    # Exponential backoff: 1s, 2s, 4s, ...
    attempt = retry_state.attempt_number
    return float(min(2 ** (attempt - 1), 8))


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
        proxy: str | None = None,
        enable_logging: bool = False,
    ) -> None:
        self._config = Z3rnoConfig(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            proxy=proxy,
        )
        self._logging = logging_enabled(enable_logging)
        client_kwargs: dict[str, Any] = {
            "base_url": self._config.base_url,
            "headers": self._config.auth_headers,
            "timeout": timeout,
        }
        if self._config.proxy:
            client_kwargs["proxy"] = self._config.proxy
        self._http = httpx.Client(**client_kwargs)

    def close(self) -> None:
        """Close the HTTP client."""
        self._http.close()

    def __enter__(self) -> Z3rnoClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # --- Get Memory ---

    def get_memory(self, memory_id: str, *, timeout: float | None = None) -> Memory:
        """Retrieve a single memory by ID."""
        resp = self._request("GET", f"/v1/memories/{memory_id}", timeout=timeout)
        return Memory.model_validate(resp)

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

        resp = self._request("POST", "/v1/memories", json=body, timeout=timeout)
        return Memory.model_validate(resp)

    # --- Store Batch ---

    def store_batch(
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

        resp = self._request(
            "POST", "/v1/memories/batch", json={"memories": items}, timeout=timeout
        )
        return BatchStoreResponse.model_validate(resp)

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
        # Phase C: strategy selection + opt-in re-ranking.
        strategy: str = "AUTO",
        rerank: bool = False,
        # Phase G slice 2 — scope to a single conversation.
        conversation_id: str | None = None,
        timeout: float | None = None,
    ) -> RecallResponse:
        """Recall memories by query.

        Args:
            strategy: One of ``AUTO | VECTOR | LEXICAL | GRAPH | TRIPLET |
                TRACE | TEMPORAL | ASK | CYPHER`` (see
                :class:`RetrievalStrategy`). Case-insensitive. ``AUTO``
                (default) lets the server's LLM router pick the best
                fit per query.
            rerank: When ``True``, apply a cross-encoder re-ranker over
                the strategy's top-K results. Requires the server to
                have ``sentence-transformers`` installed.
        """
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
        # Strategy + rerank always sent. Older servers silently ignore
        # unknown fields.
        body["strategy"] = strategy
        body["rerank"] = rerank
        if conversation_id:
            body["conversation_id"] = conversation_id

        resp = self._request("POST", "/v1/memories/recall", json=body, timeout=timeout)
        return RecallResponse.model_validate(resp)

    def recall_stream(
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
    ) -> Iterator[RecallResult]:
        """Recall memories, yielding results one at a time.

        This is a client-side streaming wrapper around :meth:`recall` that
        reduces peak memory usage for large result sets. The full response is
        fetched from the server, but results are yielded individually instead
        of being returned in a single list.
        """
        response = self.recall(
            agent_id=agent_id,
            query=query,
            memory_type=memory_type,
            filters=filters,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            time_range=time_range,
            as_of=as_of,
            timeout=timeout,
        )
        yield from response.results

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
        if cascade:
            body["cascade"] = True
        if reason:
            body["reason"] = reason

        resp = self._request("POST", "/v1/memories/forget", json=body, timeout=timeout)
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
        timeout: float | None = None,
    ) -> AuditPage:
        """Query the audit log."""
        params: dict[str, Any] = {"page": page, "page_size": page_size}
        if agent_id:
            params["agent_id"] = agent_id
        if operation:
            params["operation"] = operation
        if memory_id:
            params["memory_id"] = memory_id

        resp = self._request("GET", "/v1/audit", params=params, timeout=timeout)
        return AuditPage.model_validate(resp)

    # --- Memory History ---

    def get_memory_history(
        self, memory_id: str, *, timeout: float | None = None
    ) -> MemoryHistoryResponse:
        """Retrieve the version history of a memory."""
        resp = self._request("GET", f"/v1/memories/{memory_id}/history", timeout=timeout)
        return MemoryHistoryResponse.model_validate(resp)

    # --- Update Memory ---

    def update_memory(
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

        resp = self._request("PATCH", f"/v1/memories/{memory_id}", json=body, timeout=timeout)
        return Memory.model_validate(resp)

    # --- Tenant budgets (v0.20.3) ---

    def get_my_budgets(
        self, *, timeout: float | None = None
    ) -> TenantBudgetsView:
        """Return this org's stored budget overrides + resolved effective caps."""
        resp = self._request("GET", "/v1/tenants/me/budgets", timeout=timeout)
        return TenantBudgetsView.model_validate(resp)

    def set_my_budgets(
        self,
        budgets: TenantBudgets | dict[str, int],
        *,
        timeout: float | None = None,
    ) -> TenantBudgetsView:
        """Replace this org's budget overrides. Zero / missing fields
        inherit the server default."""
        body = (
            budgets.model_dump()
            if isinstance(budgets, TenantBudgets)
            else dict(budgets)
        )
        resp = self._request(
            "PUT", "/v1/tenants/me/budgets", json=body, timeout=timeout
        )
        return TenantBudgetsView.model_validate(resp)

    # --- Conversations (Phase G slice 2) ---

    def create_conversation(
        self,
        *,
        agent_id: str,
        user_id: str | None = None,
        title: str | None = None,
        summary_cadence: int = 10,
        metadata: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Conversation:
        """Create a new conversation. Returns the freshly-created row."""
        body: dict[str, Any] = {
            "agent_id": agent_id,
            "summary_cadence": summary_cadence,
        }
        if user_id:
            body["user_id"] = user_id
        if title:
            body["title"] = title
        if metadata:
            body["metadata"] = metadata
        resp = self._request("POST", "/v1/conversations", json=body, timeout=timeout)
        return Conversation.model_validate(resp)

    def get_conversation(
        self, conversation_id: str, *, timeout: float | None = None
    ) -> Conversation:
        """Fetch conversation metadata."""
        resp = self._request(
            "GET", f"/v1/conversations/{conversation_id}", timeout=timeout
        )
        return Conversation.model_validate(resp)

    def add_turn(
        self,
        conversation_id: str,
        *,
        memory_id: str,
        turn_role: str,
        timeout: float | None = None,
    ) -> TurnAddResponse:
        """Stamp an existing Memo as the next turn of ``conversation_id``.

        ``turn_role`` ∈ {user, assistant, system, tool, summary}.
        Returns the assigned ``turn_index`` plus ``needs_summary`` —
        when ``True``, the conversation has crossed its summary
        cadence threshold and the caller should enqueue a Forge
        summarization.
        """
        resp = self._request(
            "POST",
            f"/v1/conversations/{conversation_id}/turns",
            json={"memory_id": memory_id, "turn_role": turn_role},
            timeout=timeout,
        )
        return TurnAddResponse.model_validate(resp)

    def delete_conversation(
        self,
        conversation_id: str,
        *,
        timeout: float | None = None,
    ) -> None:
        """v0.19.3 — soft-delete a conversation. Existing turn Memos
        stay queryable through standard recall; the conversation
        itself stops accepting turns and its endpoints 404."""
        self._request(
            "DELETE", f"/v1/conversations/{conversation_id}", timeout=timeout
        )

    def list_turns(
        self,
        conversation_id: str,
        *,
        after_turn: int | None = None,
        limit: int = 50,
        timeout: float | None = None,
    ) -> TurnListResponse:
        """Fetch turns in order, optionally paginated forward."""
        params: dict[str, Any] = {"limit": limit}
        if after_turn is not None:
            params["after_turn"] = after_turn
        resp = self._request(
            "GET",
            f"/v1/conversations/{conversation_id}/turns",
            params=params,
            timeout=timeout,
        )
        return TurnListResponse.model_validate(resp)

    # --- Sessions ---

    def start_session(
        self,
        *,
        agent_id: str,
        session_type: str = "conversation",
        timeout: float | None = None,
    ) -> Session:
        """Start a new session."""
        resp = self._request(
            "POST",
            "/v1/sessions",
            json={"agent_id": agent_id, "session_type": session_type},
            timeout=timeout,
        )
        return Session.model_validate(resp)

    def end_session(self, session_id: str, *, timeout: float | None = None) -> dict[str, Any]:
        """End a session."""
        return self._request("POST", f"/v1/sessions/{session_id}/end", timeout=timeout)

    @contextmanager
    def session(
        self, *, agent_id: str, session_type: str = "conversation"
    ) -> Generator[Session, None, None]:
        """Context manager that starts a session and ensures it is ended.

        Usage::

            with client.session(agent_id="agent-1") as sess:
                client.store(agent_id="agent-1", content="...", metadata={"session_id": str(sess.session_id)})
        """
        sess = self.start_session(agent_id=agent_id, session_type=session_type)
        try:
            yield sess
        finally:
            self.end_session(str(sess.session_id))

    # --- Forge: ingest / distill / refine -------------------------------
    #
    # These wrap the Phase A / B / D server endpoints. The server gates
    # them behind operator flags (INGEST_ENABLED / DISTILL_ENABLED /
    # REFINE_ENABLED); when off, the routes are not registered and
    # these methods raise NotFoundError. Callers can either catch that
    # or check OpenAPI before invoking.

    def ingest_text(
        self,
        *,
        agent_id: str,
        text: str,
        dataset_id: str | None = None,
        timeout: float | None = None,
    ) -> IngestJob:
        """Enqueue a text-ingest job. Returns the job ack — poll
        ``get_ingest_status(job_id)`` for the final state."""
        body: dict[str, Any] = {
            "kind": "text",
            "agent_id": agent_id,
            "text": text,
        }
        if dataset_id:
            body["dataset_id"] = dataset_id
        resp = self._request("POST", "/v1/ingest", json=body, timeout=timeout)
        return IngestJob.model_validate(resp)

    def ingest_url(
        self,
        *,
        agent_id: str,
        url: str,
        dataset_id: str | None = None,
        timeout: float | None = None,
    ) -> IngestJob:
        """Enqueue a URL-ingest job. Server fetches the URL with the
        configured timeout + scheme allowlist."""
        body: dict[str, Any] = {
            "kind": "url",
            "agent_id": agent_id,
            "url": url,
        }
        if dataset_id:
            body["dataset_id"] = dataset_id
        resp = self._request("POST", "/v1/ingest", json=body, timeout=timeout)
        return IngestJob.model_validate(resp)

    def get_ingest_status(
        self, job_id: str, *, timeout: float | None = None
    ) -> IngestJobStatus:
        """Fetch the full state of an ingest job."""
        resp = self._request("GET", f"/v1/ingest/{job_id}", timeout=timeout)
        return IngestJobStatus.model_validate(resp)

    def distill(
        self,
        *,
        agent_id: str,
        memory_ids: list[str],
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        max_concurrency: int | None = None,
        summary_style: str | None = None,
        include_summary: bool = True,
        timeout: float | None = None,
    ) -> DistillJob:
        """Enqueue a Forge distillation job over ``memory_ids``.

        Returns the enqueue ack. Idempotent — re-running over already-
        distilled memories is a no-op on the server.
        """
        body: dict[str, Any] = {
            "agent_id": agent_id,
            "memory_ids": memory_ids,
            "include_summary": include_summary,
        }
        if chunk_size is not None:
            body["chunk_size"] = chunk_size
        if chunk_overlap is not None:
            body["chunk_overlap"] = chunk_overlap
        if max_concurrency is not None:
            body["max_concurrency"] = max_concurrency
        if summary_style is not None:
            body["summary_style"] = summary_style
        resp = self._request("POST", "/v1/distill", json=body, timeout=timeout)
        return DistillJob.model_validate(resp)

    def get_distill_status(
        self, job_id: str, *, timeout: float | None = None
    ) -> DistillJobStatus:
        """Fetch the full state of a distill job."""
        resp = self._request("GET", f"/v1/distill/{job_id}", timeout=timeout)
        return DistillJobStatus.model_validate(resp)

    def refine(
        self,
        *,
        dataset_id: str | None = None,
        timeout: float | None = None,
    ) -> RefineJob:
        """Enqueue a refine pass (dedupe → reweight → prune; opt-in
        infer + summarize stages controlled server-side). Admin-only
        on the server.
        """
        body: dict[str, Any] = {}
        if dataset_id:
            body["dataset_id"] = dataset_id
        resp = self._request("POST", "/v1/refine", json=body, timeout=timeout)
        return RefineJob.model_validate(resp)

    def get_refine_status(
        self, job_id: str, *, timeout: float | None = None
    ) -> RefineJobStatus:
        """Fetch the full state of a refine job."""
        resp = self._request("GET", f"/v1/refine/{job_id}", timeout=timeout)
        return RefineJobStatus.model_validate(resp)

    # --- HTTP layer ---

    def _request(
        self,
        method: str,
        path: str,
        *,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make an HTTP request with automatic retry on transient failures.

        Args:
            timeout: Per-request timeout override. When provided, overrides the
                default client timeout for this specific request.
        """
        if timeout is not None:
            kwargs["timeout"] = timeout
        retrying = retry(
            retry=retry_if_exception(_should_retry),
            wait=_wait_with_retry_after,
            stop=stop_after_attempt(self._config.max_retries),
            reraise=True,
        )
        return retrying(self._do_request)(method, path, **kwargs)

    def _do_request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute a single HTTP request and handle the response."""
        t0 = start_timer() if self._logging else 0.0
        try:
            resp = self._http.request(method, path, **kwargs)
        except (
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.ConnectTimeout,
        ) as exc:
            raise Z3rnoConnectionError(f"Connection failed: {exc}") from exc
        if self._logging:
            log_request(
                method=method,
                path=path,
                status_code=resp.status_code,
                duration_ms=elapsed_ms(t0),
                request_id=resp.headers.get("x-request-id"),
            )
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
            raise AuthenticationError(f"Authentication failed: {detail}")
        if resp.status_code == 404:
            raise NotFoundError(f"Not found: {detail}")
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
