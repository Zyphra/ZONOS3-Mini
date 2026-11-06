"""Meta routes, CORS, the studio serving rule and the removed v1 paths."""

from __future__ import annotations

import re

import pytest

from .support import check

EXPOSED = {"x-audio-sample-rate", "x-audio-channels", "x-audio-format", "x-control-span-follow", "x-seed"}


def test_health(client):
    r = client.get("/v2/health")
    assert r.status_code == 200
    check(r.json(), "#/components/schemas/Health")


def test_models(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    check(r.json(), "#/components/schemas/ModelList")


def test_cors_preflight_and_exposed_headers(client):
    r = client.options(
        "/v2/generate",
        headers={"Origin": "https://studio.example", "Access-Control-Request-Method": "POST",
                 "Access-Control-Request-Headers": "content-type"},
    )
    assert r.status_code in (200, 204), r.text
    assert r.headers.get("access-control-allow-origin") in ("*", "https://studio.example")
    r = client.get("/v2/health", headers={"Origin": "https://studio.example"})
    assert r.headers.get("access-control-allow-origin") in ("*", "https://studio.example")
    exposed = {h.strip().lower() for h in r.headers.get("access-control-expose-headers", "").split(",")}
    assert EXPOSED <= exposed, exposed


@pytest.fixture(scope="module")
def index(client):
    r = client.get("/")
    if r.status_code == 404:
        pytest.skip("the studio is not built on this server")
    assert r.status_code == 200, r.text[:300]
    return r


@pytest.fixture(scope="module", params=["/studio", "/lite"])
def page(client, index, request):
    """A studio page by its extensionless path (the serving rule maps /<page> to <page>.html)."""
    r = client.get(request.param)
    assert r.status_code == 200, f"{request.param}: {r.text[:300]}"
    return r


def _assert_page(page, api_token):
    """A studio page is HTML that the browser never stores (a rebuild changes its hashed asset names)."""
    assert page.headers["content-type"].startswith("text/html")
    if not api_token:  # the access middleware forces no-store on every non-media response
        assert "no-store" in page.headers.get("cache-control", "")


def _assert_assets_immutable(client, page, api_token):
    assets = re.findall(r'(?:src|href)="(/assets/[^"]+)"', page.text)
    assert assets
    for path in assets:
        r = client.get(path)
        assert r.status_code == 200, path
        if not api_token:
            assert "immutable" in r.headers.get("cache-control", ""), path


def test_root_serves_the_chooser(client, index, api_token):
    _assert_page(index, api_token)
    assert re.search(r'href="/studio"', index.text) and re.search(r'href="/lite"', index.text)
    _assert_assets_immutable(client, index, api_token)


def test_pages_are_served_by_name_and_not_stored(client, page, api_token):
    _assert_page(page, api_token)
    _assert_assets_immutable(client, page, api_token)
    # the same page under its file name
    named = client.get(page.url.path + ".html")
    assert named.status_code == 200 and named.text == page.text


def test_studio_page_holds_the_app(client, index):
    assert '<div id="app">' in client.get("/studio").text


@pytest.mark.parametrize(
    ("method", "path"),
    [("GET", "/studio/"), ("GET", "/v1"), ("POST", "/tts/generate"), ("GET", "/tts/capabilities"),
     ("GET", "/v2/nope"), ("GET", "/v2/"), ("POST", "/v2/nope")],
)
def test_unknown_paths_are_404(client, method, path):
    r = client.request(method, path, json={"text": "x"} if method == "POST" else None)
    assert r.status_code == 404, f"{method} {path}: {r.status_code}"
    check(r.json(), "#/components/schemas/Error")
