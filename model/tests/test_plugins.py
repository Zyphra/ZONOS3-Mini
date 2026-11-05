"""zonos3.server.plugins: loading by name and mounting next to the studio catch-all."""

import sys
import types

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from zonos3.server import plugins
from zonos3.server.keepalive import KeepaliveMiddleware
from zonos3.server.plugins import HOST_API, HostFacts, LaciTakeRequest
from zonos3.server.schemas import GenerateRequest
from zonos3.tts.steering import RateExtension


class FakePlugin:
    def __init__(self, name="fake", host_api=HOST_API, paths=("/v2/fake/run", "/fake")):
        self.name, self.host_api, self.paths = name, host_api, paths
        self.keepalive_paths = ("/v2/fake/run",)
        self.host = None
        self.closed = False

    def router(self, host):
        self.host = host
        router = APIRouter()
        for path in self.paths:
            router.add_api_route(path, lambda: {"plugin": self.name}, methods=["GET", "POST"])
        return router

    def close(self):
        self.closed = True


@pytest.fixture
def fake_module(monkeypatch):
    module = types.ModuleType("zonos3_fake")
    module.plugin = FakePlugin
    monkeypatch.setitem(sys.modules, "zonos3_fake", module)
    return module


def test_load(fake_module):
    (plugin,) = plugins.load(["fake"])
    assert plugin.name == "fake" and plugins.load([]) == []


@pytest.mark.parametrize("name", ["Fake", "1fake", "fa-ke", "", "../x"])
def test_load_rejects_bad_names(name):
    with pytest.raises(ValueError, match="plugin name"):
        plugins.load([name])


def test_load_missing_package():
    with pytest.raises(ValueError, match="zonos3_nosuchplugin is not installed in this interpreter"):
        plugins.load(["nosuchplugin"])


def test_load_name_and_host_api_mismatch(fake_module):
    fake_module.plugin = lambda: FakePlugin(name="other")
    with pytest.raises(ValueError, match="named 'other'"):
        plugins.load(["fake"])
    fake_module.plugin = lambda: FakePlugin(host_api=HOST_API + 1)
    with pytest.raises(ValueError, match="host API"):
        plugins.load(["fake"])


def studio_app() -> FastAPI:
    app = FastAPI()

    @app.get("/v2/capabilities")
    def capabilities():
        return {"plugins": []}

    @app.get("/{path:path}")
    def serve_studio(path: str):
        return {"studio": path}

    return app


@pytest.mark.parametrize("path", ["/v2/other/run", "/v2/fakery", "/fakes", "/v2/edit"])
def test_mount_rejects_routes_outside_the_plugin(path):
    with pytest.raises(ValueError, match="outside /v2/fake and /fake"):
        plugins.mount(studio_app(), [FakePlugin(paths=("/v2/fake/run", path))], object(), [])


def test_mount_keeps_the_catch_all_last_and_extends_keepalive():
    app = studio_app()
    keepalive = ["/v2/edit"]
    app.add_middleware(KeepaliveMiddleware, paths=keepalive, interval=0.01)
    plugin, host = FakePlugin(paths=("/v2/fake/run", "/fake", "/fake/{rest:path}")), object()
    plugins.mount(app, [plugin], host, keepalive)
    assert plugin.host is host
    assert app.router.routes[-1].path == "/{path:path}"
    assert keepalive == ["/v2/edit", "/v2/fake/run"]
    client = TestClient(app)
    assert client.get("/v2/fake/run").json() == {"plugin": "fake"}
    assert client.get("/fake").json() == {"plugin": "fake"}
    assert client.get("/fake/assets/x.js").json() == {"plugin": "fake"}
    assert client.get("/elsewhere").json() == {"studio": "elsewhere"}
    # the middleware reads the extended list
    reply = client.post("/v2/fake/run", headers={"X-Keepalive": "1"})
    assert reply.headers["x-keepalive"] == "1" and reply.json() == {"plugin": "fake"}


def test_host_facts_json_roundtrip():
    facts = HostFacts(model_name="ckpt", rate_edges=(8.0, 12.0), rate_buckets=3, laci_languages=frozenset({"en_us"}),
                      speaker_modes=("reference", "embedding"), raw_controls=frozenset({"voice.deeper"}),
                      phoneme_tokens=True,
                      rate_extension=RateExtension(direction="pace.tempo", row_edges=(8.0, 12.0), below=(6.0,),
                                                   above=(20.0,), doses={"en_us": {-1: 0.5, 3: -0.5}},
                                                   reference_sides=("slower",)),
                      worker_device="cpu")
    data = facts.to_json()
    assert data["rate_extension"]["doses"] == {"en_us": {"-1": 0.5, "3": -0.5}}
    assert HostFacts.from_json(data) == facts
    bare = HostFacts(model_name="m", rate_edges=(), rate_buckets=0, laci_languages=frozenset(), speaker_modes=(),
                     raw_controls=frozenset(), phoneme_tokens=False, rate_extension=None)
    assert HostFacts.from_json(bare.to_json()) == bare


def test_laci_take_request_json():
    request = GenerateRequest(text="bonjour", language="fr_fr", laci=True)
    take = LaciTakeRequest(request=request, target_frames=45.0)
    assert take.to_json("/tmp/a.wav") == {"request": {"text": "bonjour", "language": "fr_fr", "laci": True},
                                          "target_frames": 45.0, "adjust_rate": True, "scale": None,
                                          "out": "/tmp/a.wav"}
    scaled = LaciTakeRequest(request=request, target_frames=45.0, adjust_rate=False, rate_edges=(6.0, 8.0, 12.0),
                             rate_first=-1, rate_bucket=-1, rate_doses={-1: 0.5})
    assert scaled.to_json("o")["scale"] == {"edges": [6.0, 8.0, 12.0], "first": -1, "bucket": -1, "doses": {"-1": 0.5}}
