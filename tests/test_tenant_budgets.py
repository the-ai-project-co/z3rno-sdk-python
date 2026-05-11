"""v0.20.3 — SDK tests for /v1/tenants/me/budgets."""

from __future__ import annotations

import json as _json

from pytest_httpx import HTTPXMock

from z3rno import TenantBudgets, Z3rnoClient


def _client() -> Z3rnoClient:
    return Z3rnoClient(base_url="http://test", api_key="sk-test")


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


def test_get_my_budgets_parses_view(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="GET",
        url="http://test/v1/tenants/me/budgets",
        json=_view_payload(tokens=5000),
    )
    view = _client().get_my_budgets()
    assert view.overrides.daily_tokens == 5000
    assert view.effective.daily_tokens == 5000


def test_set_my_budgets_round_trips_dataclass(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="PUT",
        url="http://test/v1/tenants/me/budgets",
        json=_view_payload(tokens=2500),
    )
    view = _client().set_my_budgets(TenantBudgets(daily_tokens=2500))
    assert view.overrides.daily_tokens == 2500


def test_set_my_budgets_accepts_dict(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="PUT",
        url="http://test/v1/tenants/me/budgets",
        json=_view_payload(tokens=750),
    )
    client = _client()
    _ = client.set_my_budgets({"daily_tokens": 750, "monthly_tokens": 0})
    request = httpx_mock.get_request()
    assert request is not None
    body = _json.loads(request.content)
    assert body["daily_tokens"] == 750


def test_set_my_budgets_forwards_zero_inheritance(httpx_mock: HTTPXMock) -> None:
    """A request with all zeros should still go through — that's how
    you reset overrides to "inherit server defaults"."""
    httpx_mock.add_response(
        method="PUT",
        url="http://test/v1/tenants/me/budgets",
        json=_view_payload(tokens=0),
    )
    _client().set_my_budgets(TenantBudgets())
    request = httpx_mock.get_request()
    assert request is not None
    body = _json.loads(request.content)
    assert body == {
        "daily_tokens": 0,
        "daily_llm_calls": 0,
        "daily_embeddings": 0,
        "monthly_tokens": 0,
        "monthly_llm_calls": 0,
        "monthly_embeddings": 0,
    }
