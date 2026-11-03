"""Dubbing in the Python studio servers (``/v2/dub/*`` of zonos3_sgl and zonos3_mlx).

``DubRuntime`` keeps the uploaded references and the generated dubs, runs the dubbing worker
(zonos3/workers/dub_align.py in .venv-asr: SimAlign, AV-sync), transcribes with the server's ASR worker
(zonos3.tts.edit_align.AsrWorker: Qwen3-ASR), aligns words with the server's FuseAlign worker
(zonos3.tts.edit_align.FuseAlignWorker; zonos3.tts.edit_align.align_words, with Qwen3-ASR anchors past 30 s) and plans,
picks and places the takes (zonos3.dub.plan). The takes are synthesized by the server's scheduler through the ``synthesize``
callable the server hands in: the /v2/generate request path with laci=true and each take's LACI duration target
(zonos3.tts.laci.DurationTarget), so a dub take can use any studio voice and any steering control. The worker starts
on first use and again after it died; its calls run in threads, so nothing here blocks the server's event loop.
"""

from __future__ import annotations

import asyncio
import base64
import collections
import copy
import io
import json
import math
import os
import re
import subprocess
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Mapping

import numpy as np
import soundfile as sf

import zonos3
import zonos3.workers
from zonos3.dub import plan as P
from zonos3.tts.align_text import align_tokens
from zonos3.tts.edit_align import AsrWorker, FuseAlignWorker, WorkerError, align_words, asr_python, transcribe_audio

DUB_ROOT = Path(__file__).resolve().parents[4]  # the repo root
WORD_ALIGNER_MODEL = "google-bert/bert-base-multilingual-cased"
STORE_CAPACITY = 8
VOICE_MODES = ("source", "studio")
MAX_REF_WORDS = 20000
# Pronunciation groups (U+E000 IPA U+E001, phoneme-token checkpoints) in a target text: the TTS reads them, the forced
# aligner gets their IPA, the rate counts their symbols in letter-equivalents (zonos3.tts.edit_align.token_chars).
_GROUP_RE = re.compile("(\ue000[^\ue000\ue001]*\ue001)")


class AlignWorkerError(RuntimeError):
    pass


class UnknownId(LookupError):
    """A ref_id / dub_id the store no longer holds (HTTP 404)."""


class AlignWorker:
    """One zonos3/workers/dub_align.py process (the .venv-asr interpreter); ``call`` is thread-safe."""

    def __init__(self, python: str, device: str, word_aligner_model: str, avsync_home: str | None):
        self.script = Path(zonos3.workers.__file__).parent / "dub_align.py"
        if not Path(python).is_file():
            raise AlignWorkerError(f"alignment interpreter not found: {python} (run tools/setup_asr_env.sh)")
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV")}
        env["PYTHONPATH"] = str(Path(zonos3.__file__).resolve().parents[1])  # model/src
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env.setdefault("HF_HUB_OFFLINE", "1")
        cmd = [
            python,
            str(self.script),
            "--device",
            device,
            "--word-aligner-model",
            word_aligner_model,
        ]
        if avsync_home:
            cmd += ["--avsync-home", avsync_home]
        self._process = subprocess.Popen(cmd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self._lock = threading.Lock()
        self._next_id = 0

    @property
    def alive(self) -> bool:
        return self._process.poll() is None

    def call(self, op: str, **payload) -> dict:
        with self._lock:
            self._next_id += 1
            request = {"id": self._next_id, "op": op, **payload}
            try:
                self._process.stdin.write(json.dumps(request) + "\n")
                self._process.stdin.flush()
            except (BrokenPipeError, OSError):
                raise AlignWorkerError(f"alignment worker exited with code {self._process.wait()}") from None
            line = self._process.stdout.readline()
            if not line:
                raise AlignWorkerError(f"alignment worker exited with code {self._process.wait()}")
            resp = json.loads(line)
        if resp.get("status") != "ok":
            if resp.get("invalid"):  # the input is at fault (e.g. no face in the video): HTTP 400
                raise ValueError(resp.get("message") or resp.get("error"))
            raise AlignWorkerError(resp.get("error", "unknown error"))
        return resp

    def close(self) -> None:
        if self.alive:
            self._process.stdin.close()
        self._process.wait()


@dataclass(frozen=True)
class WorkerSettings:
    """Where the dubbing worker and AV-sync live, and its word aligner. Defaults: the ASR worker's interpreter
    (``ZONOS3_ASR_PYTHON``, else the repo's .venv-asr) and the repo's .avsync (tools/setup_asr_env.sh);
    ZONOS3_DUB_AVSYNC_HOME / ZONOS3_ASR_DEVICE override the AV-sync checkout and the device."""

    python: str = str(DUB_ROOT / ".venv-asr" / "bin" / "python")
    avsync_home: str = str(DUB_ROOT / ".avsync")
    device: str = "cuda"
    word_aligner_model: str = WORD_ALIGNER_MODEL

    @classmethod
    def from_env(cls) -> "WorkerSettings":
        d = cls()
        return cls(
            python=asr_python(),
            avsync_home=os.environ.get("ZONOS3_DUB_AVSYNC_HOME") or d.avsync_home,
            device=os.environ.get("ZONOS3_ASR_DEVICE") or d.device,
        )

    @property
    def installed(self) -> bool:
        return Path(self.python).is_file()

    @property
    def avsync_ready(self) -> bool:
        data = Path(self.avsync_home) / "data"  # zonos3/workers/dub_avsync.models_present
        return (data / "syncnet_v2.model").is_file() and (data / "sfd_face.pth").is_file()


@dataclass(frozen=True)
class DubModel:
    """What the dub needs to know of the served checkpoint and server."""

    name: str
    rate_edges: tuple[float, ...]  # speaking-rate condition-row bucket edges (UTF-8 bytes / s)
    n_buckets: int
    laci_languages: frozenset[str]  # TTS language codes with LACI alignment heads (--laci-heads)
    studio_voices: bool  # speaker-embedding checkpoint: studio voices (speaker_mode=embedding)
    source_voice: bool  # raw-reference checkpoint: the reference speaker is cloned from its audio
    raw_controls: frozenset[str] = field(default_factory=frozenset)  # steering ids that move the speaker embedding
    phoneme_groups: bool = False
    # voice mode ("source" / "studio") -> TTS language -> its speaking-rate scale when it has tempo-steered buckets past
    # the row (the steering bundle's rate_extension; source voices only on the sides it lists); otherwise the row's
    rate_scales: Mapping[str, Mapping[str, P.RateScale]] = field(default_factory=dict)


# Take synthesis (the server's): jobs -> per job (float32 audio at P.SAMPLE_RATE, or None for no audio; LACI record).
Synthesize = Callable[[list[dict], "str | None"], Awaitable[list[tuple["np.ndarray | None", dict]]]]
# Reference codes of a voice prompt (the server's codec): (mono float32 audio, sample rate) -> [[code] * C] * frames.
EncodeVoice = Callable[[np.ndarray, int], Awaitable[list[list[int]]]]


class _Store:
    """Small LRU of uploaded references / generated dubs."""

    def __init__(self, capacity: int = STORE_CAPACITY):
        self._items: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()
        self._capacity = capacity

    def put(self, value: dict) -> str:
        key = uuid.uuid4().hex
        with self._lock:
            self._items[key] = value
            while len(self._items) > self._capacity:
                self._items.popitem(last=False)
        return key

    def get(self, key: str) -> dict:
        with self._lock:
            value = self._items.get(key)
            if value is None:
                raise UnknownId(key)
            self._items.move_to_end(key)
            return value


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


# ---------------------------------------------------------------------------------------------------- request shapes


def merge_controls(*maps: Mapping | None, where: str = "controls") -> dict[str, float] | None:
    """Steering doses summed per control id (the voice's, then the segment's: as /v2/edit's region_controls), zero
    doses dropped; None when nothing is left."""
    out: dict[str, float] = {}
    for m in maps:
        if m is None:
            continue
        if not isinstance(m, Mapping):
            raise ValueError(f"{where}: controls must map control ids to doses")
        for did, dose in m.items():
            try:
                value = float(dose)
            except (TypeError, ValueError):
                raise ValueError(f"{where}: dose of {did!r} must be a number") from None
            if not math.isfinite(value):
                raise ValueError(f"{where}: dose of {did!r} must be finite")
            out[str(did)] = out.get(str(did), 0.0) + value
    out = {did: v for did, v in out.items() if v != 0.0}
    return out or None


def normalize_voice(raw: Mapping | None, where: str) -> dict | None:
    """A request's voice: None (inherit), ``{"mode": "source", "controls"?}`` (clone the reference speaker from its
    own audio) or ``{"mode": "studio", "speaker_embedding_id", "controls"?}`` (a studio library / saved voice in
    embedding mode with Cast-bench steering)."""
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where}: voice must be an object")
    unknown = sorted(set(raw) - {"mode", "speaker_embedding_id", "controls"})
    if unknown:
        raise ValueError(f"{where}: unknown voice field(s) {unknown}")
    mode = raw.get("mode") or "source"
    controls = merge_controls(raw.get("controls"), where=f"{where}.controls")
    if mode == "source":
        if raw.get("speaker_embedding_id") is not None:
            raise ValueError(f"{where}: a source voice clones the reference; speaker_embedding_id needs mode 'studio'")
        return {"mode": "source", "controls": controls}
    if mode == "studio":
        sid = raw.get("speaker_embedding_id")
        if not isinstance(sid, str) or not sid.strip():
            raise ValueError(f"{where}: a studio voice needs speaker_embedding_id")
        return {"mode": "studio", "speaker_embedding_id": sid.strip(), "controls": controls}
    raise ValueError(f"{where}: voice mode must be one of {list(VOICE_MODES)}")


def normalize_ref_words(raw, duration: float) -> list[dict]:
    """Word timings the caller already has (``/v2/dub/analyze`` ``ref_words``): ``[{word, start, end}]`` in time order,
    inside the reference."""
    if not isinstance(raw, list) or not raw:
        raise ValueError("ref_words must be a non-empty list of {word, start, end}")
    if len(raw) > MAX_REF_WORDS:
        raise ValueError(f"ref_words: at most {MAX_REF_WORDS} words")
    words, prev = [], 0.0
    for k, w in enumerate(raw):
        if not isinstance(w, Mapping):
            raise ValueError(f"ref_words[{k}] must be an object {{word, start, end}}")
        text = str(w.get("word") or "").strip()
        try:
            start, end = float(w["start"]), float(w["end"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"ref_words[{k}]: start and end must be numbers (seconds)") from None
        if not text:
            raise ValueError(f"ref_words[{k}]: empty word")
        if not (math.isfinite(start) and math.isfinite(end) and 0.0 <= start <= end <= duration + 0.05):
            raise ValueError(f"ref_words[{k}]: need 0 <= start <= end <= {duration:.2f}")
        if start < prev - 1e-6:
            raise ValueError(f"ref_words[{k}]: words must be in time order")
        prev = start
        words.append({"word": text, "start": start, "end": end})
    return words


def spoken_text(text: str) -> str:
    """The text the forced aligner reads: nonverbal markup removed (its enclosed words kept), each pronunciation
    group by its IPA."""
    from zonos3.tts.markup import strip_nonverbal_markup

    return _GROUP_RE.sub(lambda m: m.group(0)[1:-1], strip_nonverbal_markup(text))


def text_bytes(text: str) -> int:
    """UTF-8 bytes of a target text for its speaking rate; a pronunciation group counts its IPA symbols in
    letter-equivalents (zonos3.tts.edit_align.token_chars), nonverbal markers count nothing (their enclosed words
    do)."""
    from zonos3.tts.edit_align import token_chars
    from zonos3.tts.markup import strip_nonverbal_markup

    return int(round(sum(token_chars(p) if p.startswith("\ue000") else len(p.encode())
                         for p in _GROUP_RE.split(strip_nonverbal_markup(text)) if p)))


def decode_upload(path: str) -> tuple[np.ndarray, int]:
    """Mono float32 audio of an uploaded file: a sound file, or the first audio stream of a video (ffmpeg)."""
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception:  # noqa: BLE001 - not a sound file: try it as a video
        with tempfile.TemporaryDirectory(prefix="dub-") as tmp:
            wav = Path(tmp) / "audio.wav"
            proc = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", path, "-vn", "-map", "0:a:0",
                 "-ac", "1", "-ar", str(P.SAMPLE_RATE), "-c:a", "pcm_f32le", str(wav)],
                capture_output=True, text=True,
            )
            if proc.returncode != 0:
                raise ValueError(f"upload an audio file or a video with an audio track ({proc.stderr.strip()[-200:]})")
            data, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    return np.ascontiguousarray(data.mean(axis=1), dtype=np.float32), int(sr)


class DubRuntime:
    def __init__(self, model: DubModel, synthesize: Synthesize, encode_voice: EncodeVoice, asr: AsrWorker,
                 aligner: FuseAlignWorker, worker: WorkerSettings | None = None):
        self.model = model
        self.synthesize = synthesize
        self.encode_voice = encode_voice
        # the server's ASR and FuseAlign workers (shared with speech editing; the server closes them)
        self.asr = asr
        self.aligner = aligner
        self.settings = worker or WorkerSettings.from_env()
        self.rate_edges = [float(x) for x in model.rate_edges]
        self.n_buckets = int(model.n_buckets)
        self.row_scale = P.RateScale(tuple(self.rate_edges))
        self._worker: AlignWorker | None = None
        self._worker_lock = threading.Lock()  # the worker is called from several request threads
        self.refs, self.dubs = _Store(), _Store()

    # ---- alignment worker -------------------------------------------------------------------------------------

    def _ensure_worker(self) -> AlignWorker:
        with self._worker_lock:
            if self._worker is not None and self._worker.alive:
                return self._worker
            s = self.settings
            self._worker = AlignWorker(s.python, s.device, s.word_aligner_model,
                                       s.avsync_home if s.avsync_ready else None)
            return self._worker

    def _call_sync(self, op: str, **payload) -> dict:
        return self._ensure_worker().call(op, **payload)

    async def _call(self, op: str, **payload) -> dict:
        return await asyncio.to_thread(self._call_sync, op, **payload)

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()

    def rate_scale(self, target_lang: str, voice_mode: str) -> P.RateScale:
        """The speaking-rate buckets of a dub target language for a voice mode: the row's, extended by its steered
        buckets."""
        return self.model.rate_scales.get(voice_mode, {}).get(P.TARGET_LANGUAGES[target_lang][1], self.row_scale)

    def voice_modes(self) -> list[str]:
        m = self.model
        return [v for v, ok in (("source", m.source_voice), ("studio", m.studio_voices)) if ok]

    def buckets(self, laci: list[str]) -> list[dict]:
        """Every bucket of the target languages' scales: index, label, nominal rate (UTF-8 bytes / s: its midpoint)
        and edges ``low`` / ``high`` on the scale that has it furthest from its ends (the scales share their edges;
        None: open below / above on every scale; a language's own lowest / highest bucket is open below / above),
        steered, the target languages and voice modes that have it and, for a steered one, its tempo dose per
        language."""
        if not self.rate_edges:
            return []
        scales = {(v, c): self.rate_scale(c, v) for v in self.voice_modes() for c in laci} or {("", ""): self.row_scale}
        out = []
        for b in range(min(s.first for s in scales.values()), max(s.last for s in scales.values()) + 1):
            having = {vc: s for vc, s in scales.items() if s.first <= b <= s.last}
            inner = max(having.values(), key=lambda s: min(b - s.first, s.last - b))
            i = b - inner.first
            steered = not 0 <= b < self.n_buckets
            entry = {"index": b, "label": str(b), "rate": inner.rate(b),
                     "low": inner.edges[i - 1] if i > 0 else None, "high": inner.edges[i] if i < len(inner.edges) else None,
                     "steered": steered, "languages": sorted({c for _, c in having if c}),
                     "voice_modes": sorted({v for v, _ in having if v})}
            if steered:
                entry["doses"] = {c: s.doses[b] for (_, c), s in sorted(having.items())}
            out.append(entry)
        return out

    def capabilities(self) -> dict:
        """The dub's capabilities; blocks on the ASR and FuseAlign workers' pings (they start the workers), so run it
        in a thread. A worker that is unavailable reports its model as "" and the dub as unavailable."""
        m = self.model
        laci = sorted(c for c, (_, tts) in P.TARGET_LANGUAGES.items() if tts in m.laci_languages)
        try:
            asr_model = self.asr.model()
        except WorkerError:
            asr_model = ""
        try:
            aligner_model = self.aligner.model()
        except WorkerError:
            aligner_model = ""
        return {
            "available": (self.settings.installed and bool(asr_model) and bool(aligner_model) and bool(laci)
                          and bool(self.rate_edges)),
            "ref_languages": [{"code": c, "name": n} for c, n in P.REF_LANGUAGES.items()],
            "target_languages": [{"code": c, "label": v[0], "tts_language": v[1]} for c, v in P.TARGET_LANGUAGES.items()
                                 if c in laci],
            "max_ref_seconds": P.MAX_REF_SECONDS,
            "max_video_bytes": P.MAX_VIDEO_BYTES,
            "max_segments": P.MAX_SEGMENTS,
            "default_min_pause_s": P.DEFAULT_MIN_PAUSE_S,
            "min_pause_range": list(P.MIN_PAUSE_RANGE),
            "sample_rate": P.SAMPLE_RATE,
            "frame_rate": P.FRAME_RATE,
            "rate_edges": self.rate_edges,
            "n_buckets": self.n_buckets,
            "buckets": self.buckets(laci),
            "candidate_bucket_offsets": list(P.CANDIDATE_BUCKET_OFFSETS),
            "candidate_seeds": P.CANDIDATE_SEEDS,
            "model": m.name,
            "asr_model": asr_model,
            "aligner_model": aligner_model,
            "word_aligner": self.settings.word_aligner_model,
            "align_worker": {"installed": self.settings.installed,
                             "running": self._worker is not None and self._worker.alive},
            "lipsync": self.settings.installed and self.settings.avsync_ready,
            "voice_modes": self.voice_modes(),
            "segment_controls": True,
            "phoneme_groups": m.phoneme_groups,
            "ref_words": True,
        }

    # ---- references -------------------------------------------------------------------------------------------

    def add_ref(self, path: str) -> dict:
        """A reference from an uploaded sound file or a video's audio track (runs in a request thread)."""
        audio, sr = decode_upload(path)
        duration = len(audio) / sr
        if not P.MIN_REF_SECONDS <= duration <= P.MAX_REF_SECONDS:
            raise ValueError(f"reference must be {P.MIN_REF_SECONDS}–{P.MAX_REF_SECONDS:g} s, got {duration:.1f} s")
        ref_id = self.refs.put({"audio": audio, "sr": sr, "duration": duration})
        return {"ref_id": ref_id, "duration": duration, "sample_rate": sr}

    @staticmethod
    def _write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
        sf.write(str(path), audio, sr, subtype="FLOAT")

    def _check_ref_lang(self, ref_lang: str | None) -> None:
        if ref_lang is not None and ref_lang not in P.REF_LANGUAGES:
            raise ValueError(f"unsupported reference language: {ref_lang}")

    async def _align(self, items: list[dict]) -> list[list[dict]]:
        """FuseAlign forced alignment of ``[{audio, text, language, utterances?}]`` (``language`` a REF_LANGUAGES
        code; ``utterances`` the long-form anchors of a transcript of the audio) -> per item one ``{word, start, end,
        presence, placed}`` per ``align_tokens`` token of the text (an unplaced word keeps FuseAlign's fallback times).
        Long-form items get Qwen3-ASR anchors (``align_words``)."""
        reqs = [{"audio": it["audio"], "tokens": align_tokens(it["text"], it["language"]),
                 "language": P.REF_LANGUAGES.get(it["language"]), "utterances": it.get("utterances")} for it in items]
        return await asyncio.to_thread(align_words, self.asr, self.aligner, reqs)

    async def transcribe(self, ref_id: str, ref_lang: str | None) -> dict:
        """Qwen3-ASR transcript of the reference (``transcribe_audio``: chunk-wise past 30 s) and its words (FuseAlign,
        anchored by the transcript's chunks) in the given language, else the detected one: Qwen3-ASR's (past 30 s the
        first spoken chunk's), its first name when it reports several."""
        ref = self.refs.get(ref_id)
        self._check_ref_lang(ref_lang)
        with tempfile.TemporaryDirectory(prefix="dub-") as tmp:
            wav = Path(tmp) / "ref.wav"
            self._write_wav(wav, ref["audio"], ref["sr"])
            out = await asyncio.to_thread(transcribe_audio, self.asr, self.aligner, str(wav),
                                          P.REF_LANGUAGES[ref_lang] if ref_lang else None)
            name = out.language.split(",")[0].strip()
            by_name = {n: c for c, n in P.REF_LANGUAGES.items()}
            lang, text = ref_lang or by_name.get(name), out.text.strip()
            words = []
            if lang and text:
                items = (await self._align([{"audio": str(wav), "text": text, "language": lang,
                                             "utterances": out.utterances}]))[0]
                words = [{"word": it["word"], "start": it["start"], "end": it["end"]} for it in items]
        return {"text": text, "language": lang, "language_name": name, "words": words}

    def prepare_lips(self, ref_id: str, video_path: str) -> dict:
        """Face track and lip embeddings of the reference's video for the lip-sync step (AV-sync), kept with the
        reference (runs in a request thread)."""
        if not (self.settings.installed and self.settings.avsync_ready):
            raise ValueError("lip-sync (AV-sync) is not set up on this server; run tools/setup_asr_env.sh")
        ref = self.refs.get(ref_id)
        lip_dir = tempfile.TemporaryDirectory(prefix="dub-lips-")  # removed with the last reference / dub using it
        out = Path(lip_dir.name) / "lips.pt"
        r = self._call_sync("lips_prepare", video=video_path, out=str(out))
        ref["lips"] = {"dir": lip_dir, "path": str(out), "start": r["start"], "end": r["end"]}
        return {k: r[k] for k in ("start", "end", "track_frames", "video_frames", "tracks")}

    def _lips_of(self, ref_id: str, lipsync: bool) -> dict | None:
        if not lipsync:
            return None
        try:
            lips = self.refs.get(ref_id).get("lips")
        except UnknownId:
            lips = None
        if lips is None:
            raise ValueError("lip-sync needs the video's face track: POST /v2/dub/lips/{ref_id} with the video first")
        return lips

    # ---- analysis ---------------------------------------------------------------------------------------------

    async def analyze(self, req: dict) -> dict:
        """Segments of the reference at its pauses, each with the span of the target transcript SimAlign projects onto
        it. ``ref_words`` (timings the caller has) replace the forced alignment of ``ref_text``."""
        ref = self.refs.get(req["ref_id"])
        ref_lang, target_lang = req["ref_lang"], req["target_lang"]
        self._check_ref_lang(ref_lang)
        if target_lang not in P.TARGET_LANGUAGES:
            raise ValueError(f"unsupported target language: {target_lang}")
        target_text = (req.get("target_text") or "").strip()
        if not target_text:
            raise ValueError("the target transcript is required")
        min_pause = float(req["min_pause_s"])
        if not P.MIN_PAUSE_RANGE[0] <= min_pause <= P.MIN_PAUSE_RANGE[1]:
            raise ValueError(f"min_pause_s must be within {P.MIN_PAUSE_RANGE[0]}–{P.MIN_PAUSE_RANGE[1]} s")

        if req.get("ref_words") is not None:
            words = normalize_ref_words(req["ref_words"], ref["duration"])
        else:
            ref_text = (req.get("ref_text") or "").strip()
            if not ref_text:
                raise ValueError("send the reference transcript (ref_text) or its word timings (ref_words)")
            with tempfile.TemporaryDirectory(prefix="dub-") as tmp:
                wav = Path(tmp) / "ref.wav"
                self._write_wav(wav, ref["audio"], ref["sr"])
                aligned = (await self._align([{"audio": str(wav), "text": ref_text, "language": ref_lang}]))[0]
            if not aligned:
                raise ValueError("the forced aligner returned no words; check the transcript and its language")
            words = [{"word": it["word"], "start": it["start"], "end": it["end"]} for it in aligned]

        units = P.text_units(target_text, target_lang)
        if not units:
            raise ValueError("the target transcript has no words")
        links = (await self._call("word_align", src=[w["word"] for w in words],
                                  tgt=[spoken_text(u["text"]) for u in units]))["links"]

        ranges = P.ref_segments(words, min_pause)
        if len(ranges) > P.MAX_SEGMENTS:
            raise ValueError("too many pauses; raise the pause threshold")
        spans = [
            (min(w["start"] for w in words[a:b]), max(w["end"] for w in words[a:b])) for a, b in ranges
        ]
        ubytes = P.unit_bytes(target_text, units)
        breaks = P.break_after(target_text, units)
        projected = P.project_segments(ranges, spans, [tuple(x) for x in links], ubytes, breaks)
        segments = []
        for k, ((wa, wb), (ua, ub), (s, e)) in enumerate(zip(ranges, projected, spans)):
            text = P.span_text(target_text, units, ua, ub)
            segments.append(
                {
                    "index": k,
                    "ref_start": s,
                    "ref_end": e,
                    "ref_word_range": [wa, wb],
                    "target_unit_range": [ua, ub],
                    "ref_text": P.join_words([w["word"] for w in words[wa:wb]], ref_lang),
                    "target_text": text,
                    # on the default voice's (source) scale; the studio's voices may have more steered buckets
                    "bucket_auto": (self.rate_scale(target_lang, "source" if self.model.source_voice else "studio")
                                    .bucket(text_bytes(text) / (e - s)) if text and e > s else None),
                }
            )
        warnings = []
        if not links:
            warnings.append("The word aligner found no links; segments are split by length")
        return {
            "duration": ref["duration"],
            "ref_words": words,
            "target_units": units,
            "links": links,
            "segments": segments,
            "warnings": warnings,
        }

    # ---- generation -------------------------------------------------------------------------------------------

    def _segment_voice(self, k: int, voice: dict | None, own: Mapping | None) -> tuple[dict, dict | None]:
        """A segment's effective voice and its own controls (checked against the voice: speaker-embedding controls
        need a studio voice)."""
        voice = voice or {"mode": "source", "controls": None}
        if voice["mode"] == "source" and not self.model.source_voice:
            raise ValueError(f"segment {k}: this checkpoint has no reference-clone path; choose a studio voice")
        if voice["mode"] == "studio" and not self.model.studio_voices:
            raise ValueError(f"segment {k}: this checkpoint has no speaker-embedding row for studio voices")
        controls = merge_controls(own, where=f"segments[{k}].controls")
        raw = sorted(d for d in (merge_controls(voice["controls"], controls) or {}) if d in self.model.raw_controls)
        if raw and voice["mode"] != "studio":
            raise ValueError(f"segment {k}: controls {', '.join(raw)} move the speaker embedding and need a studio voice")
        return voice, controls

    async def generate(self, req: dict, session_id: str | None) -> dict:
        ref = self.refs.get(req["ref_id"])
        target_lang, seed = req["target_lang"], int(req["seed"])
        if target_lang not in P.TARGET_LANGUAGES:
            raise ValueError(f"unsupported target language: {target_lang}")
        if seed < 0:
            raise ValueError("seed must be >= 0")
        raw = req["segments"]
        if not 1 <= len(raw) <= P.MAX_SEGMENTS:
            raise ValueError(f"need 1–{P.MAX_SEGMENTS} segments")
        default_voice = normalize_voice(req.get("voice"), "voice")
        duration, audio, sr = ref["duration"], ref["audio"], ref["sr"]
        segs, voices, prev_end = [], [], 0.0
        for k, s in enumerate(raw):
            a, b = float(s["ref_start"]), float(s["ref_end"])
            if not (0 <= a < b <= duration + 0.05):
                raise ValueError(f"segment {k}: need 0 <= ref_start < ref_end <= {duration:.2f}")
            if a < prev_end - 1e-6:
                raise ValueError(f"segment {k}: segments must be sorted and non-overlapping")
            prev_end = b
            bucket = s.get("bucket")
            segs.append(P.RefSegment(a, b, str(s["target_text"]).strip(), None if bucket is None else int(bucket)))
            voices.append(self._segment_voice(k, normalize_voice(s.get("voice"), f"segments[{k}].voice") or default_voice,
                                              s.get("controls")))
        if not any(seg.text for seg in segs):
            raise ValueError("every segment is silent; enter target text")

        lipsync = bool(req.get("lipsync"))
        dub = {"duration": duration, "target_lang": target_lang, "ref_codes": None, "takes": {},
               "ref_id": req["ref_id"], "lips": self._lips_of(req["ref_id"], lipsync),
               "segments": [self._segment_entry(k, seg, seed, *voices[k], self.rate_scale(target_lang, voices[k][0]["mode"]))
                            for k, seg in enumerate(segs)]}
        began = time.time()
        await self._ensure_voice_prompt(dub, audio, sr, [seg for seg, (v, _) in zip(segs, voices) if v["mode"] == "source"])
        stats = await self._render(dub, [k for k, seg in enumerate(segs) if seg.text], session_id)
        return await asyncio.to_thread(self._assemble, dub, stats, time.time() - began)

    async def regenerate(self, req: dict, session_id: str | None) -> dict:
        """New takes for one segment of a dub (optionally with a new text, bucket, voice or controls); the others
        keep theirs."""
        old = self.dubs.get(req["dub_id"])
        seed, k = int(req["seed"]), int(req["segment"])
        if seed < 0:
            raise ValueError("seed must be >= 0")
        if not 0 <= k < len(old["segments"]):
            raise ValueError(f"segment must be in [0, {len(old['segments'])})")
        seg = old["segments"][k]
        text = str(req["target_text"]).strip()
        bucket = req.get("bucket")
        voice = normalize_voice(req.get("voice"), "voice") or seg["voice"]
        own = req["controls"] if req.get("controls") is not None else seg["controls"]
        voice, controls = self._segment_voice(k, voice, own)
        lipsync = bool(req.get("lipsync"))
        lips = (old["lips"] or self._lips_of(old["ref_id"], True)) if lipsync else None
        dub = {**old, "segments": copy.deepcopy(old["segments"]), "takes": dict(old["takes"]), "lips": lips}
        entry = self._segment_entry(
            k, P.RefSegment(seg["ref_start"], seg["ref_end"], text, None if bucket is None else int(bucket)), seed,
            voice, controls, self.rate_scale(old["target_lang"], voice["mode"]),
        )
        dub["segments"][k] = entry
        dub["takes"].pop(k, None)
        if not text and not dub["takes"]:
            raise ValueError("every segment would be silent; enter target text")
        began = time.time()
        if voice["mode"] == "source" and dub["ref_codes"] is None:
            ref = self.refs.get(old["ref_id"])  # UnknownId (404): the reference left the store
            await self._ensure_voice_prompt(dub, ref["audio"], ref["sr"], [P.RefSegment(entry["ref_start"], entry["ref_end"])])
        stats = await self._render(dub, [k] if text else [], session_id)
        return await asyncio.to_thread(self._assemble, dub, stats, time.time() - began)

    async def _ensure_voice_prompt(self, dub: dict, audio: np.ndarray, sr: int, source_segs: list[P.RefSegment]) -> None:
        """Reference codes of the source voice: the clip from the first source-voice segment, up to
        P.VOICE_PROMPT_MAX_S (plan.voice_prompt_span), as the branch's offline dub."""
        if dub["ref_codes"] is not None or not source_segs:
            return
        a, b = P.voice_prompt_span(source_segs, dub["duration"])
        clip = np.ascontiguousarray(audio[int(a * sr) : int(b * sr)])
        dub["ref_codes"] = await self.encode_voice(clip, sr)

    def _segment_entry(self, k: int, seg: P.RefSegment, seed: int, voice: dict, controls: dict | None,
                       scale: P.RateScale) -> dict:
        """The response entry of one segment, its candidate plan included (placement fields filled later). A set
        ``seg.bucket`` pins every candidate to it; otherwise the plan spreads around the text's rate bucket. Buckets
        are on the target language's ``scale`` (row buckets and its tempo-steered ones)."""
        if seg.bucket is not None and not scale.first <= seg.bucket <= scale.last:
            raise ValueError(f"segment {k}: bucket must be in [{scale.first}, {scale.last}] for this target language")
        d = seg.ref_end - seg.ref_start
        nbytes = text_bytes(seg.text)
        entry = {
            "index": k,
            "ref_start": seg.ref_start,
            "ref_end": seg.ref_end,
            "target_text": seg.text,
            "status": "ok" if seg.text else "silent",
            "seed": seed,
            "bytes": nbytes,
            "rate": nbytes / d if seg.text else None,
            "bucket": None,
            "pinned": seg.bucket is not None,
            "voice": voice,  # the segment's effective voice (its own or the request's)
            "controls": controls,  # the segment's own direction, added to the voice's controls
            "candidates": [],
            "chosen": None,
            "final_bucket": None,
            "retries": None,
            "fit": None,
            "speech_start": None,
            "speech_end": None,
            "shift": None,
            "start_error": None,
            "end_error": None,
            "duration_error": None,
            "lip_start": None,  # speech start / end after the lip-sync warp
            "lip_end": None,
            # how much the lip-sync warp time-scales the segment's speech (plan.lip_warp)
            "lip_scale": None,
            "lip_stretch": None,
            "lip_squeeze": None,
            "lip_warp": None,
        }
        if seg.text:
            if seg.bucket is None:
                entry["bucket"] = scale.bucket(entry["rate"])
                plan = P.candidate_plan(entry["bucket"], scale, seed, k)
            else:
                entry["bucket"] = seg.bucket
                plan = P.pinned_candidate_plan(seg.bucket, seed, k)
            entry["candidates"] = [
                {**c, "speech_s": None, "attempts": None, "retries": None, "final_bucket": None,
                 "end": None, "capped": None, "score": None}
                for c in plan
            ]
        return entry

    async def _synthesize(self, dub: dict, jobs: list[dict], session_id: str | None) -> list[dict]:
        """Every job's take under LACI on the server's scheduler, each with its segment's duration target: a miss is
        retaken in the speaking-rate bucket the measured length asks for, or only recorded for a set bucket."""
        from zonos3.tts.laci import PACE_KINDS

        tts_lang = P.TARGET_LANGUAGES[dub["target_lang"]][1]
        requests = []
        for j in jobs:
            e = dub["segments"][j["segment"]]
            voice = e["voice"]
            scale = self.rate_scale(dub["target_lang"], voice["mode"])
            requests.append({
                "text": j["text"], "language": tts_lang, "seed": j["seed"], "max_tokens": j["max_tokens"],
                "bucket": j["bucket"], "target_frames": j["target_frames"], "adjust_rate": not j["pinned"],
                "voice": voice, "controls": merge_controls(voice["controls"], e["controls"]),
                "ref_codes": dub["ref_codes"] if voice["mode"] == "source" else None,
                # the segment's scale when it has steered buckets (a take's bucket and its retakes move on it)
                "rate_scale": scale if scale.doses else None,
            })
        out = []
        for audio, rec in await self.synthesize(requests, session_id):
            out.append(
                {
                    "audio": audio,
                    "attempts": rec["attempts"],
                    "retries": dict(collections.Counter(f["kind"] for f in rec["failures"] if not f["ignored"])),
                    "exhausted": rec["capped"],  # used LACI's whole retry budget
                    "end": rec["end"],
                    # a content failure LACI could not repair; an unrepaired duration miss shows in the length
                    "capped": any(f["ignored"] and f["kind"] not in PACE_KINDS for f in rec["failures"]),
                    "failures": rec["failures"],
                    "final_bucket": rec["rate_bucket"],
                }
            )
        return out

    async def _render(self, dub: dict, ks: list[int], session_id: str | None) -> dict:
        """Synthesize, align and choose the takes of segments ``ks`` into ``dub["takes"]``. Returns this request's
        synthesis counts: candidates, LACI retries by kind over all of them, takes that used the whole budget, and
        the synthesis / alignment wall times."""
        stats = {"candidates": 0, "retries": collections.Counter(), "exhausted": 0, "synthesis_s": 0.0, "align_s": 0.0}
        if not ks:
            return stats
        info, target_lang = dub["segments"], dub["target_lang"]
        jobs = []
        for k in ks:
            e = info[k]
            d = e["ref_end"] - e["ref_start"]
            for slot, c in enumerate(e["candidates"]):
                jobs.append({"bucket": c["bucket"], "seed": c["seed"], "segment": k, "slot": slot, "text": e["target_text"],
                             "max_tokens": P.max_tokens_for(d), "target_frames": d * P.FRAME_RATE,
                             "pinned": e["pinned"]})
        t0 = time.time()
        takes = await self._synthesize(dub, jobs, session_id)
        stats["synthesis_s"] = time.time() - t0
        stats["candidates"] = len(jobs)
        for t in takes:
            stats["retries"].update(t["retries"])
            stats["exhausted"] += t["exhausted"]

        t0 = time.time()
        valid = [i for i, t in enumerate(takes) if t["audio"] is not None and len(t["audio"])]
        aligned: dict[int, list[dict]] = {}
        with tempfile.TemporaryDirectory(prefix="dub-") as tmp:
            reqs = []
            for i in valid:
                path = Path(tmp) / f"take{i}.wav"
                self._write_wav(path, takes[i]["audio"], P.SAMPLE_RATE)
                reqs.append({"audio": str(path), "text": spoken_text(jobs[i]["text"]), "language": target_lang})
            if reqs:
                for i, items in zip(valid, await self._align(reqs)):
                    aligned[i] = items
        stats["align_s"] = time.time() - t0
        spans: dict[int, tuple[float, float]] = {}
        for i in valid:
            sp = P.speech_span(aligned[i], takes[i]["audio"], P.SAMPLE_RATE)
            if sp is not None and sp[1] > sp[0]:
                spans[i] = sp

        nonempty = [j for j, e in enumerate(info) if e["status"] == "ok"]
        for k in ks:
            e = info[k]
            d = e["ref_end"] - e["ref_start"]
            later = [j for j in nonempty if j > k]
            room = info[later[0]]["ref_start"] - P.MIN_GAP_S - e["ref_start"] if later else math.inf
            mine = [i for i, j in enumerate(jobs) if j["segment"] == k]
            cands = []
            for i in mine:
                if i not in spans:
                    cands.append(None)
                    continue
                t, (s0, s1) = takes[i], spans[i]
                c = {"speech_s": s1 - s0, "capped": t["capped"], "end": t["end"]}
                cands.append(c)
                e["candidates"][jobs[i]["slot"]].update(
                    speech_s=s1 - s0, attempts=t["attempts"], retries=t["retries"],
                    final_bucket=t["final_bucket"], end=t["end"], capped=t["capped"], score=P.candidate_score(c, d, room),
                )
            pick = P.choose_candidate(cands, d, room)
            if pick is None:
                raise RuntimeError(f"segment {k}: no usable take")
            e["chosen"] = jobs[mine[pick]]["slot"]
            t = takes[mine[pick]]
            s0, s1 = spans[mine[pick]]
            # the kept take: its LACI retries by kind, final speaking-rate bucket, and speech length over the segment's
            e.update(final_bucket=t["final_bucket"], retries=t["retries"], fit=(s1 - s0) / d)
            dub["takes"][k] = {"audio": t["audio"], "s0": s0, "s1": s1,
                               "items": aligned[mine[pick]], "attempts": t["attempts"], "retries": t["retries"]}
        return stats

    def _assemble(self, dub: dict, stats: dict, generation_s: float) -> dict:
        """Place every chosen take on the timeline, lip-sync it to the video (``dub["lips"]``, AV-sync) when asked, store
        the dub and build the response (runs in a thread: the lip-sync call blocks on the worker)."""
        info, takes = dub["segments"], dub["takes"]
        nonempty = sorted(takes)
        items = [
            {"ref_start": info[k]["ref_start"], "ref_end": info[k]["ref_end"],
             "take_len": len(takes[k]["audio"]) / P.SAMPLE_RATE, "s0": takes[k]["s0"], "s1": takes[k]["s1"]}
            for k in nonempty
        ]
        placed, timeline_len = P.place_segments(items, dub["duration"])

        sr_out = P.SAMPLE_RATE
        fade = max(1, int(P.FADE_S * sr_out))
        chunks = []
        for k, p in zip(nonempty, placed):
            take = takes[k]["audio"]
            piece = take[int(round(p["crop_start"] * sr_out)) : int(round(p["crop_end"] * sr_out))].astype(np.float32)
            nf = min(fade, len(piece) // 2)
            if nf:
                ramp = np.linspace(0.0, 1.0, nf, dtype=np.float32)
                piece[:nf] *= ramp
                piece[-nf:] *= ramp[::-1]
            chunks.append((int(round(p["place_start"] * sr_out)), piece))
        total = max([int(round(timeline_len * sr_out))] + [pos + len(pc) for pos, pc in chunks])
        mix = np.zeros(total, dtype=np.float32)
        for pos, piece in chunks:
            mix[pos : pos + len(piece)] += piece
        lip_map, lipsync = None, None
        lipsync_s = 0.0
        if dub["lips"] is not None:
            t0 = time.time()
            mix, lip_map, lipsync = self._lip_sync(dub["lips"], mix)
            lipsync_s = time.time() - t0
        mix = np.clip(mix, -1.0, 1.0)
        total = len(mix)
        buf = io.BytesIO()
        sf.write(buf, mix, sr_out, format="WAV", subtype="PCM_16")
        wav_bytes = buf.getvalue()

        def warp(t: float) -> float:
            return float(lip_map.out_time(t)) if lip_map is not None else t

        words = []
        for k, p in zip(nonempty, placed):
            s0 = takes[k]["s0"]
            info[k].update(
                start=p["speech_start"],
                end=p["speech_end"],
                speech_start=p["speech_start"],
                speech_end=p["speech_end"],
                shift=p["shift"],
                start_error=p["start_error"],
                end_error=p["end_error"],
                duration_error=p["duration_error"],
                # skipped warp (no sync gain): the speech plays where it was placed, untouched
                lip_start=warp(p["speech_start"]) if lipsync is not None else None,
                lip_end=warp(p["speech_end"]) if lipsync is not None else None,
            )
            if lip_map is not None:
                info[k].update(P.lip_warp(lip_map, p["speech_start"], p["speech_end"]))
            elif lipsync is not None:
                info[k].update(lip_scale=1.0, lip_stretch=1.0, lip_squeeze=1.0, lip_warp=1.0)
            for it in takes[k]["items"]:
                words.append(
                    {
                        "word": it["word"],
                        "start": warp(p["speech_start"] + (it["start"] - s0)),
                        "end": warp(p["speech_start"] + (it["end"] - s0)),
                        "segment": k,
                    }
                )
        metrics = {
            "mean_abs_start_error_ms": 1000 * _mean([abs(p["start_error"]) for p in placed]),
            "mean_abs_end_error_ms": 1000 * _mean([abs(p["end_error"]) for p in placed]),
            "mean_abs_duration_error_ms": 1000 * _mean([abs(p["duration_error"]) for p in placed]),
            "max_shift_ms": 1000 * max(p["shift"] for p in placed),
            # LACI retries (by kind: skip, hallucination, truncation, pace_fast, pace_slow) of the dub's kept takes, and
            # of every candidate this request synthesized (most retries happen on candidates that are not kept)
            "laci_retries": sum(takes[k]["attempts"] for k in nonempty),
            "retries_by_kind": dict(sum((collections.Counter(takes[k]["retries"]) for k in nonempty), collections.Counter())),
            "all_retries": sum(stats["retries"].values()),
            "all_retries_by_kind": dict(stats["retries"]),
            "exhausted_takes": stats["exhausted"],  # candidates that used LACI's whole retry budget
            "candidates": stats["candidates"],  # synthesized by this request
            "generation_s": generation_s,
            "synthesis_s": stats["synthesis_s"],
            "align_s": stats["align_s"],
            "lipsync_s": lipsync_s,
        }
        dub_id = self.dubs.put({**dub, "wav": wav_bytes})
        return {
            "dub_id": dub_id,
            "audio_base64": base64.b64encode(wav_bytes).decode("ascii"),
            "sample_rate": sr_out,
            "duration": total / sr_out,
            "segments": [{"k": e["index"], **e} for e in info],
            "words": words,
            "metrics": metrics,
            "lipsync": lipsync,
        }

    def _lip_sync(self, lips: dict, mix: np.ndarray) -> tuple[np.ndarray, P.LipMap | None, dict]:
        """AV-sync's warp of the dub timeline onto the lips of the video (alignment worker), bounded by the guard and
        kept only when it lowers the mean SyncNet distance of lips and dub over the track by more than
        P.LIP_MIN_SYNC_GAIN; otherwise the unwarped dub comes back (``applied`` false, no lip map)."""
        sr = P.SAMPLE_RATE
        with tempfile.TemporaryDirectory(prefix="dub-") as tmp:
            src, out = Path(tmp) / "dub.wav", Path(tmp) / "synced.wav"
            pad = max(0, int(math.ceil(lips["end"] * sr)) - len(mix))  # the warp reads the whole face track
            self._write_wav(src, np.pad(mix, (0, pad)), sr)
            r = self._call_sync("lips_sync", lips=lips["path"], audio=str(src), out=str(out),
                                max_rate=P.LIP_GUARD_MAX_RATE, max_shift_s=P.LIP_GUARD_MAX_SHIFT_S)
            synced, _ = sf.read(str(out), dtype="float32")
        lip_map = P.LipMap.from_frames(r["start"], r["rate"], r["src_frames"])
        shifts = np.abs(lip_map.shifts())
        track = P.lip_warp(lip_map, float(lip_map.src_t[0]), float(lip_map.src_t[-1]))
        info = {
            "start": r["start"],
            "end": r["end"],
            "max_shift_ms": 1000 * float(shifts.max()),
            "mean_abs_shift_ms": 1000 * float(shifts.mean()),
            "max_stretch": track["lip_stretch"],
            "max_squeeze": track["lip_squeeze"],
            "mean_warp": track["lip_warp"],
            "sync_distance_before": r["sync_distance_before"],
            "sync_distance_after": r["sync_distance_after"],
            "guard": {"max_rate": P.LIP_GUARD_MAX_RATE, "max_shift_s": P.LIP_GUARD_MAX_SHIFT_S},
        }
        gain = r["sync_distance_before"] - r["sync_distance_after"]
        info["applied"] = gain > P.LIP_MIN_SYNC_GAIN
        if not info["applied"]:
            info["reason"] = (f"no sync gain: SyncNet distance {r['sync_distance_before']:.3f} unwarped, "
                              f"{r['sync_distance_after']:.3f} warped")
            return mix, None, info
        return synced, lip_map, info

    # ---- video ------------------------------------------------------------------------------------------------

    def mux(self, dub_id: str, video_path: str, out_path: str) -> None:
        """Replace the video's audio with the dub (video stream copied; re-encoded if the copy fails)."""
        dub = self.dubs.get(dub_id)
        dub_path = Path(video_path).with_name("dub.wav")
        dub_path.write_bytes(dub["wav"])
        base = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", video_path, "-i", str(dub_path)]
        tail = ["-map", "0:v:0", "-map", "1:a:0"]
        audio = ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out_path]
        copy = subprocess.run(base + tail + ["-c:v", "copy"] + audio, check=False, capture_output=True)
        if copy.returncode == 0:
            return
        enc = subprocess.run(
            base + tail + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"] + audio,
            check=False,
            capture_output=True,
        )
        if enc.returncode != 0:
            err = enc.stderr.decode(errors="replace")[-400:]
            raise ValueError(f"ffmpeg could not mux this video: {err}")
