"""zonos3/workers/pyhost.py as the ggml server runs it: a subprocess whose engine calls a Python stand-in for ggml
answers. Edits use explicit spans and ``min_fill_ratio: 0`` (no FuseAlign), ZONOS3_ASR_PYTHON points the host's
AsrWorker at a fake that only answers ``ping``, and a fake plugin ``zonos3_fake`` (written to a temp dir on PYTHONPATH)
exercises ``http`` and ``laci_takes``."""

import base64
import json
import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

MODEL_SRC = Path(__file__).resolve().parents[1] / "src"
WORKER = MODEL_SRC / "zonos3" / "workers" / "pyhost.py"
K = 4
FRAMES = 30
EDIT_MODEL = {"frame_rate": 10.0, "n_codebooks": K, "audio_pad_id": 1024, "codebook_sizes": [1024] * K,
              "max_spans": 4, "speaker_row": True}
HOST = {"model_name": "test", "rate_edges": [8.0, 12.0, 16.0], "rate_buckets": 4, "laci_languages": ["en_us"],
        "speaker_modes": ["reference", "embedding"], "raw_controls": ["voice.deeper"], "phoneme_tokens": False,
        "rate_extension": {"direction": "pace.tempo", "row_edges": [8.0, 12.0, 16.0], "below": [6.0],
                           "above": [20.0], "doses": {"en_us": {"-1": 0.5, "4": -0.5}},
                           "reference_sides": ["slower"]},
        "worker_device": None}
UPLOAD = b"an upload"
EDIT = {"audio_base64": base64.b64encode(UPLOAD).decode("ascii"), "target_text": "hello brave new world",
        "spans": [[1.0, 1.5]], "min_fill_ratio": 0.0, "seed": 3}
# The worker's libraries printing on stdout (and writing fd 1) in the middle of a request.
NOISY = textwrap.dedent("""
    import os, runpy, sys
    from zonos3.tts import edit_runtime

    edit = edit_runtime.EditRuntime.edit

    async def noisy(self, req):
        print("stray print")
        sys.stdout.flush()
        os.write(1, b"stray fd write\\n")
        return await edit(self, req)

    edit_runtime.EditRuntime.edit = noisy
    sys.argv = sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name="__main__")
""")
FAKE_ASR = """\
import json, os, sys
for line in sys.stdin:
    request = json.loads(line)
    assert request["op"] == "ping", request
    print(json.dumps({"id": request["id"], "status": "ok", "model": "fake", "languages": ["English"]}), flush=True)
open(os.environ["FAKE_ASR_CLOSED"], "w").close()
"""
FAKE_PLUGIN = """\
import os

import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse

from zonos3.server.errors import NotFound, call
from zonos3.server.plugins import HOST_API, LaciTakeRequest
from zonos3.server.schemas import GenerateRequest


class Fake:
    name = "fake"
    host_api = HOST_API
    keepalive_paths = ("/v2/fake/takes",)

    def router(self, host):
        router = APIRouter()

        @router.get("/v2/fake/facts")
        async def facts(q: str = ""):
            f = host.facts()
            return {"q": q, "model_name": f.model_name, "doses": f.rate_extension.doses["en_us"],
                    "asr": type(host.asr).__name__}

        @router.post("/v2/fake/upload")
        async def upload(request: Request):
            body = await request.body()
            return {"bytes": len(body), "type": request.headers.get("content-type")}

        @router.post("/v2/fake/takes")
        async def takes(request: Request):
            session = request.headers.get("x-tts-session-id")
            req = GenerateRequest(text="bonjour", language="fr_fr", laci=True, stream=False)
            out = await call(host.synthesize_laci([
                LaciTakeRequest(req, 30.0),
                LaciTakeRequest(req, 45.0, adjust_rate=False, rate_edges=(6.0, 8.0), rate_first=-1, rate_bucket=-1,
                                rate_doses={-1: 0.5})], session))
            return [{"samples": None if a is None else int(np.asarray(a).size), "record": r} for a, r in out]

        @router.get("/v2/fake/gone")
        async def gone():
            async def missing():
                raise NotFound("no such thing")
            return await call(missing())

        @router.get("/fake")
        async def page():
            return PlainTextResponse("fake app", headers={"Cache-Control": "no-store"})

        return router

    def close(self):
        open(os.environ["FAKE_PLUGIN_CLOSED"], "w").close()


def plugin():
    return Fake()
"""


class EngineError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


class Ggml:
    """The ggml side of the pipe: requests out, replies collected by id, every engine call answered on its own
    thread by ``engine[op](call)``."""

    def __init__(self, tmp: Path, engine: dict):
        asr = tmp / "fake_asr"
        asr.write_text(f"#!{sys.executable}\n{FAKE_ASR}")
        asr.chmod(0o755)
        self.asr_closed = tmp / "asr_closed"
        self.plugin_closed = tmp / "plugin_closed"
        plugins = tmp / "plugins"
        plugins.mkdir(exist_ok=True)
        (plugins / "zonos3_fake.py").write_text(FAKE_PLUGIN)
        env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        env.update(PYTHONPATH=os.pathsep.join([str(MODEL_SRC), str(plugins)]), ZONOS3_ASR_PYTHON=str(asr),
                   FAKE_ASR_CLOSED=str(self.asr_closed), FAKE_PLUGIN_CLOSED=str(self.plugin_closed),
                   HF_HUB_OFFLINE="1")
        self.stderr_path = tmp / "worker.stderr"
        self._stderr = open(self.stderr_path, "w")
        self.proc = subprocess.Popen([sys.executable, "-c", NOISY, str(WORKER)], env=env, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=self._stderr, text=True, encoding="utf-8")
        self.engine = engine
        self.calls: list[dict] = []
        self.stray: list[str] = []  # stdout lines that are not protocol JSON
        self.replies: dict[int, dict] = {}
        self._cond = threading.Condition()
        self._write_lock = threading.Lock()
        threading.Thread(target=self._read, daemon=True).start()

    def _write(self, message: dict) -> None:
        with self._write_lock:
            self.proc.stdin.write(json.dumps(message) + "\n")
            self.proc.stdin.flush()

    def _read(self) -> None:
        for line in self.proc.stdout:
            try:
                message = json.loads(line)
            except ValueError:
                self.stray.append(line)
                continue
            if "call" in message:
                self.calls.append(message)
                threading.Thread(target=self._answer, args=(message,), daemon=True).start()
            else:
                with self._cond:
                    self.replies[message["id"]] = message
                    self._cond.notify_all()

    def _answer(self, call: dict) -> None:
        try:
            reply = {"call": call["call"], "status": "ok", **self.engine[call["op"]](call)}
        except EngineError as exc:
            reply = {"call": call["call"], "status": "error", "code": exc.code, "message": str(exc)}
        self._write(reply)

    def send(self, rid: int, op: str, **payload) -> None:
        self._write({"id": rid, "op": op, **payload})

    def reply(self, rid: int, timeout: float = 60.0) -> dict:
        with self._cond:
            assert self._cond.wait_for(lambda: rid in self.replies, timeout), f"no reply to {rid}"
            return self.replies[rid]

    def request(self, rid: int, op: str, **payload) -> dict:
        self.send(rid, op, **payload)
        return self.reply(rid)

    def close(self) -> int:
        self.proc.stdin.close()
        code = self.proc.wait(timeout=60)
        self._stderr.close()
        return code


def wav(path: str, frames: int, sample_rate: int) -> None:
    sf.write(path, np.zeros(frames, dtype=np.float32), sample_rate, subtype="FLOAT")


def engine(**override) -> dict:
    def decode_audio(call):
        assert Path(call["audio"]).read_bytes() == UPLOAD
        wav(call["out"], 3 * 16000, 16000)
        return {"sample_rate": 16000}

    def fill(call):
        n = 7  # the fill's frames plus its delay tail, ended on span-end at frame n
        return {"rows": [[900] * K] * (n + K - 1), "eos_frame": n}

    def decode(call):
        wav(call["out"], len(call["codes"]) * 4410, 44100)
        return {}

    return {"decode_audio": decode_audio, "embed": lambda call: {"embedding": [0.25] * 8},
            "encode": lambda call: {"codes": np.arange(FRAMES * K).reshape(FRAMES, K).tolist()},
            "check": lambda call: {"raw": []}, "fill": fill, "decode": decode, **override}


@pytest.fixture
def make_ggml(tmp_path):
    started = []

    def make(**override) -> Ggml:
        ggml = Ggml(tmp_path, engine(**override))
        started.append(ggml)
        assert ggml.request(0, "init", edit=EDIT_MODEL, host=HOST, plugins=["fake"]) == {
            "id": 0, "status": "ok", "result": {"plugins": [{"name": "fake", "keepalive_paths": ["/v2/fake/takes"]}]}}
        return ggml

    yield make
    for ggml in started:
        if ggml.proc.poll() is None:
            ggml.proc.kill()
            ggml.proc.wait()
        ggml._stderr.close()


def test_edit_over_the_pipe(make_ggml):
    ggml = make_ggml()
    reply = ggml.request(1, "edit", request=EDIT)
    assert reply["status"] == "ok", reply
    (take,) = reply["result"]["takes"]
    assert take["seed"] == 3 and take["edited_regions"] == [[1.0, 1.7]] and take["stopped"] == [True]
    (fill,) = [c for c in ggml.calls if c["op"] == "fill"]
    assert fill["spans"] == [[10, 5]] and fill["speaker_embedding"] == [0.25] * 8 and fill["fills"] == []
    assert np.asarray(fill["window_codes"]).shape == (FRAMES, K)
    (decode,) = [c for c in ggml.calls if c["op"] == "decode"]
    spliced = np.asarray(decode["codes"])
    assert spliced.shape == (FRAMES + 2, K) and (spliced[10:17] == 900).all()


def test_stdout_pollution_stays_off_the_pipe(make_ggml):
    ggml = make_ggml()
    assert ggml.request(1, "edit", request=EDIT)["status"] == "ok"
    assert ggml.close() == 0
    assert ggml.stray == []
    assert "stray print" in ggml.stderr_path.read_text() and "stray fd write" in ggml.stderr_path.read_text()


def test_concurrent_requests_interleave(make_ggml):
    both = threading.Barrier(2, timeout=30)  # each edit's fill waits for the other's: both must be in flight
    fill = engine()["fill"]

    def paired_fill(call):
        try:
            both.wait()
        except threading.BrokenBarrierError:
            raise EngineError(500, "the requests ran one after the other") from None
        return fill(call)

    ggml = make_ggml(fill=paired_fill)
    ggml.send(1, "edit", request={**EDIT, "seed": 1})
    ggml.send(2, "edit", request={**EDIT, "seed": 2})
    for rid in (1, 2):
        reply = ggml.reply(rid)
        assert reply["status"] == "ok", reply
        assert reply["result"]["takes"][0]["seed"] == rid


def test_errors(make_ggml):
    def check(call):
        if call["doses"]:
            raise EngineError(400, "This server has no steering bundle; controls need one.")
        return {"raw": []}

    ggml = make_ggml(check=check)
    assert ggml.request(1, "edit", request={**EDIT, "controls": {"voice.deeper": 1.0}}) == {
        "id": 1, "status": "error", "code": 400, "message": "This server has no steering bundle; controls need one."}
    assert ggml.request(2, "nope") == {"id": 2, "status": "error", "code": 400, "message": "unknown op 'nope'"}
    invalid = ggml.request(3, "edit", request={"target_text": "hi"})
    assert invalid["code"] == 422 and invalid["message"][0]["loc"] == ["body", "audio_base64"]
    unloaded = ggml.request(4, "http", plugin="nope", method="GET", path="/v2/nope", query="", headers=[], body=None,
                            out="unused")
    assert unloaded == {"id": 4, "status": "error", "code": 400, "message": "plugin 'nope' is not loaded"}
    assert ggml.request(5, "edit", request=EDIT)["status"] == "ok"  # the worker lives on


def test_eof_closes_the_workers(make_ggml):
    ggml = make_ggml()
    assert ggml.request(1, "edit", request=EDIT)["status"] == "ok"  # starts the ASR worker (its ping)
    assert not ggml.asr_closed.exists()
    assert ggml.close() == 0
    assert ggml.asr_closed.exists() and ggml.plugin_closed.exists()


def http(ggml, rid, tmp_path, method, path, query="", headers=(), body: bytes | None = None):
    body_path = None
    if body is not None:
        body_path = tmp_path / f"body{rid}"
        body_path.write_bytes(body)
    out = tmp_path / f"out{rid}"
    reply = ggml.request(rid, "http", plugin="fake", method=method, path=path, query=query,
                         headers=[list(h) for h in headers], body=None if body_path is None else str(body_path),
                         out=str(out))
    assert reply["status"] == "ok", reply
    return reply["result"]["status"], dict(reply["result"]["headers"]), out.read_bytes()


def test_plugin_http(make_ggml, tmp_path):
    ggml = make_ggml()
    status, _, body = http(ggml, 1, tmp_path, "GET", "/v2/fake/facts", query="q=a%20b")
    assert status == 200 and json.loads(body) == {"q": "a b", "model_name": "test", "doses": {"-1": 0.5, "4": -0.5},
                                                  "asr": "AsrWorker"}
    big = os.urandom(3 * (1 << 20) + 5)  # several body chunks
    status, _, body = http(ggml, 2, tmp_path, "POST", "/v2/fake/upload",
                           headers=[("Content-Type", "application/octet-stream")], body=big)
    assert status == 200 and json.loads(body) == {"bytes": len(big), "type": "application/octet-stream"}
    status, headers, body = http(ggml, 3, tmp_path, "GET", "/fake")
    assert status == 200 and body == b"fake app" and headers["cache-control"] == "no-store"
    status, _, body = http(ggml, 4, tmp_path, "GET", "/v2/fake/gone")
    assert status == 404 and json.loads(body) == {"detail": "no such thing"}
    status, _, _ = http(ggml, 5, tmp_path, "GET", "/v2/elsewhere")
    assert status == 404


def test_plugin_laci_takes(make_ggml, tmp_path):
    def laci_takes(call):
        for k, take in enumerate(call["takes"]):
            if k == 0:
                wav(take["out"], 4410, 44100)
        return {"takes": [{"audio": k == 0, "record": {"k": k}} for k in range(len(call["takes"]))]}

    ggml = make_ggml(laci_takes=laci_takes)
    status, _, body = http(ggml, 1, tmp_path, "POST", "/v2/fake/takes", headers=[("X-TTS-Session-ID", "s1")])
    assert status == 200, body
    assert json.loads(body) == [{"samples": 4410, "record": {"k": 0}}, {"samples": None, "record": {"k": 1}}]
    (call,) = [c for c in ggml.calls if c["op"] == "laci_takes"]
    assert call["session"] == "s1"
    first, second = call["takes"]
    assert first["request"] == {"text": "bonjour", "language": "fr_fr", "laci": True, "stream": False}
    assert first["target_frames"] == 30.0 and first["adjust_rate"] is True and first["scale"] is None
    assert second["adjust_rate"] is False
    assert second["scale"] == {"edges": [6.0, 8.0], "first": -1, "bucket": -1, "doses": {"-1": 0.5}}


def test_plugin_engine_errors_map_to_http(make_ggml, tmp_path):
    def laci_takes(call):
        raise EngineError(400, "LACI needs heads for fr_fr")

    ggml = make_ggml(laci_takes=laci_takes)
    status, _, body = http(ggml, 1, tmp_path, "POST", "/v2/fake/takes")
    assert status == 400 and json.loads(body) == {"detail": "LACI needs heads for fr_fr"}
