"""v0.22.1 — SDK tests for client.admin.{get,set}_budgets (slice 21.3)."""

from __future__ import annotations

import json as _json

import pytest
from pytest_httpx import HTTPXMock

from z3rno import AsyncZ3rnoClient, TenantBudgets, Z3rnoClient

ORG_ID = "11111111-1111-1111-1111-111111111111"


def _view_payload(tokens: int = 1000) -> dict[str, object]:
    return {
        "overrides": {
            "daily_tokens": tokens,
            "daily_llm_calls": 0,
            "daily_embeddings": 0,
            "monthly_tokens": 0,
            "monthly_llm_calls": 0,
            "monthly_embeddings": 0,
        },
        "effective": {
            "daily_tokens": tokens,
            "daily_llm_calls": 0,
            "daily_embeddings": 0,
            "monthly_tokens": 0,
            "monthly_llm_calls": 0,
            "monthly_embeddings": 0,
        },
    }


# ---- Sync ----


def test_admin_get_budgets_hits_path(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"http://test/v1/tenants/{ORG_ID}/budgets",
        json=_view_payload(tokens=7000),
    )
    client = Z3rnoClient(base_url="http://test", api_key="sa-key")
    view = client.admin.get_budgets(ORG_ID)
    assert view.overrides.daily_tokens == 7000


def test_admin_set_budgets_sends_body(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="PUT",
        url=f"http://test/v1/tenants/{ORG_ID}/budgets",
        json=_view_payload(tokens=9000),
    )
    client = Z3rnoClient(base_url="http://test", api_key="sa-key")
    view = client.admin.set_budgets(
        ORG_ID, TenantBudgets(daily_tokens=9000)
    )
    assert view.overrides.daily_tokens == 9000

    req = httpx_mock.get_request()
    assert req is not None
    sent = _json.loads(req.content)
    assert sent["daily_tokens"] == 9000


def test_admin_set_budgets_accepts_dict(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="PUT",
        url=f"http://test/v1/tenants/{ORG_ID}/budgets",
        json=_view_payload(tokens=4000),
    )
    client = Z3rnoClient(base_url="http://test", api_key="sa-key")
    view = client.admin.set_budgets(ORG_ID, {"daily_tokens": 4000})
    assert view.overrides.daily_tokens == 4000


# ---- Async ----


@pytest.mark.asyncio
async def test_async_admin_get_budgets(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"http://test/v1/tenants/{ORG_ID}/budgets",
        json=_view_payload(tokens=3000),
    )
    async with AsyncZ3rnoClient(base_url="http://test", api_key="sa-key") as c:
        view = await c.admin.get_budgets(ORG_ID)
    assert view.overrides.daily_tokens == 3000
