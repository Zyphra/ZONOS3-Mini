"""zonos3/workers/edit_dub.py as the ggml server runs it: a subprocess whose engine calls a Python stand-in for ggml
answers. Edits use explicit spans and ``min_fill_ratio: 0`` (no FuseAlign), and ZONOS3_ASR_PYTHON points the worker's
AsrWorker at a fake that only answers ``ping``."""

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
from zonos3.dub.plan import RateScale
from zonos3.workers import edit_dub

MODEL_SRC = Path(__file__).resolve().parents[1] / "src"
WORKER = MODEL_SRC / "zonos3" / "workers" / "edit_dub.py"
K = 4
FRAMES = 30
EDIT_MODEL = {"frame_rate": 10.0, "n_codebooks": K, "audio_pad_id": 1024, "codebook_sizes": [1024] * K,
              "max_spans": 4, "speaker_row": True}
DUB_MODEL = {"name": "test", "rate_edges": [8.0, 12.0, 16.0], "n_buckets": 4, "laci_languages": ["en_us"],
             "studio_voices": True, "source_voice": True, "raw_controls": ["voice.deeper"], "phoneme_groups": False,
             "rate_extension": {"direction": "pace.tempo", "row_edges": [8.0, 12.0, 16.0], "below": [6.0],
                                "above": [20.0], "doses": {"en_us": {"-1": 0.5, "4": -0.5}},
                                "reference_sides": ["slower"]}}
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
        env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        env.update(PYTHONPATH=str(MODEL_SRC), ZONOS3_ASR_PYTHON=str(asr), FAKE_ASR_CLOSED=str(self.asr_closed),
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
        assert ggml.request(0, "init", edit=EDIT_MODEL, dub=DUB_MODEL) == {"id": 0, "status": "ok", "result": {}}
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
    assert ggml.request(4, "dub_transcribe", ref_id="gone", ref_lang=None) == {
        "id": 4, "status": "error", "code": 404, "message": "unknown ref_id/dub_id; upload or generate again"}
    assert ggml.request(5, "edit", request=EDIT)["status"] == "ok"  # the worker lives on


def test_eof_closes_the_workers(make_ggml):
    ggml = make_ggml()
    assert ggml.request(1, "edit", request=EDIT)["status"] == "ok"  # starts the ASR worker (its ping)
    assert not ggml.asr_closed.exists()
    assert ggml.close() == 0
    assert ggml.asr_closed.exists()


def test_dub_init_and_take_jobs():
    model = edit_dub.dub_model(DUB_MODEL)
    assert model.raw_controls == frozenset({"voice.deeper"}) and model.laci_languages == frozenset({"en_us"})
    assert model.rate_scales["studio"]["en_us"] == RateScale((6.0, 8.0, 12.0, 16.0, 20.0), -1, {-1: 0.5, 4: -0.5})
    assert model.rate_scales["source"]["en_us"] == RateScale((6.0, 8.0, 12.0, 16.0), -1, {-1: 0.5})
    job = {"text": "bonjour", "bucket": 4, "rate_scale": model.rate_scales["studio"]["en_us"], "ref_codes": None}
    assert edit_dub.take_job(job) == {"text": "bonjour", "bucket": 4, "ref_codes": None, "rate_scale": {
        "edges": [6.0, 8.0, 12.0, 16.0, 20.0], "first": -1, "doses": {"-1": 0.5, "4": -0.5}}}
    assert edit_dub.take_job({**job, "rate_scale": None})["rate_scale"] is None
