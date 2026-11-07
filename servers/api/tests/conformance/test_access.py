"""ZONOS3_ACCESS_TOKEN: bearer, cookie and the one-time query flow (runs only with --api-token)."""

from __future__ import annotations

import httpx
import pytest


@pytest.fixture
def token(api_token):
    if not api_token:
        pytest.skip("pass --api-token (the server's ZONOS3_ACCESS_TOKEN)")
    return api_token


def test_denied_without_token(base_url, token):
    r = httpx.get(f"{base_url}/v2/capabilities", timeout=60)
    assert r.status_code == 403 and r.text == "Access denied"


def test_bearer(base_url, token):
    r = httpx.get(f"{base_url}/v2/capabilities", headers={"Authorization": f"Bearer {token}"}, timeout=60)
    assert r.status_code == 200
    assert r.headers["cache-control"] == "no-store"
    assert r.headers["referrer-policy"] == "no-referrer"


def test_cookie(base_url, token):
    r = httpx.get(f"{base_url}/v2/capabilities", cookies={"zonos3_access": token}, timeout=60)
    assert r.status_code == 200


def test_query_sets_the_cookie(base_url, token):
    r = httpx.get(f"{base_url}/", params={"access_token": token}, timeout=60, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/"
    cookie = r.headers["set-cookie"]
    assert cookie.startswith(f"zonos3_access={token}") and "httponly" in cookie.lower()
    assert httpx.get(f"{base_url}/v2/health", cookies={"zonos3_access": token}, timeout=60).status_code == 200
    assert httpx.get(f"{base_url}/", params={"access_token": "wrong"}, timeout=60).status_code == 403
