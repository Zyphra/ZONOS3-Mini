"""The sgl server's plugin seam (zonos3.server.plugins): LACI takes on an extended speaking-rate scale, and plugin
routes mounted ahead of the studio catch-all."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

api_server = pytest.importorskip("zonos3_sgl.server.api_server")
TestClient = pytest.importorskip("fastapi.testclient").TestClient
import torch  # noqa: E402
from fastapi import APIRouter  # noqa: E402
from zonos3.server import host, plugins  # noqa: E402
from zonos3.server.schemas import GenerateRequest  # noqa: E402

ROW_EDGES = (10.117, 12.123, 13.286, 14.25, 15.14, 16.079, 17.094, 18.489, 21.208)
# fr_fr's extended scale: bucket -1 below the row (slower), 10 and 11 above it (faster)
EDGES = (9.2, *ROW_EDGES, 23.3, 25.5)
DOSES = {-1: -0.99, 10: 1.44, 11: 1.93}


def _take(bucket: int, scaled: bool = True) -> plugins.LaciTakeRequest:
    request = GenerateRequest(text="Oui.", language="fr_fr", seed=1, max_tokens=90, stream=False, laci=True)
    if not scaled:
        return plugins.LaciTakeRequest(request=request, target_frames=30.0)
    return plugins.LaciTakeRequest(request=request, target_frames=30.0, rate_edges=EDGES, rate_first=-1,
                                   rate_bucket=bucket, rate_doses=DOSES)


def test_take_on_an_extended_scale_carries_its_bucket_and_deltas(monkeypatch):
    """A take in a steered bucket: the bucket's tempo dose on top of its own steering delta, and the scale with its
    bucket on it (pinned takes too: LACI reports the bucket from it, not the unknown row id)."""
    bundle = SimpleNamespace(rate_extension=SimpleNamespace(direction="d"))
    monkeypatch.setattr(host, "resolve_controls", lambda bundle, controls, spans, **kw: SimpleNamespace(
        hidden_delta=torch.full((4,), controls["d"])))
    base = torch.ones(4)
    scale = host.rate_scale(_take(-1), bundle, text="Oui.", base_delta=base, sources=(0,), hidden_size=4)
    assert (scale.first, scale.bucket, scale.edges) == (-1, -1, EDGES)
    assert torch.allclose(scale.delta, torch.full((4,), 1 - 0.99))
    assert len(scale.deltas) == 13 and scale.deltas[1 + 5] is base  # row buckets: own delta only
    assert torch.allclose(scale.deltas[1 + 11], torch.full((4,), 1 + 1.93))
    assert host.rate_scale(_take(5, scaled=False), bundle, text="Oui.", base_delta=None, sources=(0,),
                           hidden_size=4) is None


def test_take_doses_need_a_rate_extension():
    with pytest.raises(ValueError):
        host.rate_scale(_take(-1), None, text="Oui.", base_delta=None, sources=(), hidden_size=4)


class _FakePlugin:
    name = "fake"
    host_api = plugins.HOST_API
    keepalive_paths = ("/v2/fake/x",)

    def __init__(self):
        self.host = None

    def router(self, host):
        self.host = host
        router = APIRouter()

        @router.get("/v2/fake/x")
        async def x():
            return {"plugin": "fake"}

        @router.get("/fake")
        async def page():
            return {"page": "fake"}

        return router

    def close(self):
        pass


def test_plugin_routes_are_reached_before_the_studio_catch_all(monkeypatch, tmp_path):
    app = api_server.app
    monkeypatch.setattr(app.router, "routes", list(app.router.routes))
    keepalive = list(api_server._KEEPALIVE)
    monkeypatch.setattr(api_server, "_KEEPALIVE", keepalive)
    monkeypatch.setattr(api_server, "_studio_dir", lambda: tmp_path)
    monkeypatch.delenv("ZONOS3_ACCESS_TOKEN", raising=False)
    plugin = _FakePlugin()
    plugins.mount(app, [plugin], api_server._SglHost(), keepalive)
    assert plugin.host.asr is api_server._ASR and plugin.host.aligner is api_server._FUSEALIGN
    assert app.router.routes[-1].path == "/{path:path}"
    assert keepalive == ["/v2/edit", "/v2/fake/x"]
    client = TestClient(app)
    assert client.get("/v2/fake/x").json() == {"plugin": "fake"}
    assert client.get("/fake").json() == {"page": "fake"}
    assert client.get("/v2/fake/nope").status_code == 404
