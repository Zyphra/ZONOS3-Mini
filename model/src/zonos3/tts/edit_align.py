"""Speech-editing helpers: the speech-recognition (Qwen3-ASR) and forced-alignment (FuseAlign) worker clients shared by
speech editing and plugins, their long-form orchestration, and transcript-diff edit regions.

Both workers are JSON-lines processes, one of each per server, started on first use and again after they exited:
``AsrWorker`` runs zonos3/workers/asr.py in .venv-asr (``ZONOS3_ASR_PYTHON``; tools/setup_asr_env.sh) and
``FuseAlignWorker`` runs zonos3/workers/fusealign.py in the server's interpreter. Their failures are ``WorkerError``s
(``AsrError`` / ``FuseAlignError``: the worker is unavailable, HTTP 503) or ``ValueError``s (the input is at fault,
HTTP 400), each with its final message.

Long-form alignment: a FuseAlign item whose audio is longer than ``LONG_FORM_SECONDS`` or whose tokens (joined with
spaces) exceed ``MAX_TEXT_BYTES`` UTF-8 bytes is routed by approximate word times. They come from Qwen3-ASR on the
recording's VAD chunks (the FuseAlign worker's ``segments``): every chunk transcribed (``start`` / ``end``) gives one
utterance ``{text, start, end}`` (empty texts dropped), sent as the item's ``utterances``; FuseAlign's built-in
Whisper anchoring never runs. A transcript Qwen3-ASR makes of a recording over ``LONG_FORM_SECONDS``
(``transcribe_audio``) is made the same way, chunk by chunk, and its utterances anchor the alignments that follow.
"""
from dataclasses import dataclass
import difflib
import importlib.util
import json
import math
import os
import re
from pathlib import Path
import subprocess
import sys
import threading

from zonos3 import languages

from .align_text import align_tokens
from .markup import _NONVERBAL_MARKUP, has_nonverbal_markup
from . import textmatch

MODEL_SRC = Path(__file__).resolve().parents[2]  # model/src: PYTHONPATH of both workers
ASR_WORKER = MODEL_SRC / 'zonos3' / 'workers' / 'asr.py'
FUSEALIGN_WORKER = MODEL_SRC / 'zonos3' / 'workers' / 'fusealign.py'
LONG_FORM_SECONDS = 30.0  # fusealign_hub.aligner.MAX_SECONDS: one FuseAlign forward
MAX_TEXT_BYTES = 1022  # fusealign_hub.longform.MAX_TEXT_BYTES
# Qwen3-ASR names of the unspaced languages (zonos3.tts.align_text.UNSPACED: ja, zh, yue): their chunk texts join
# with "".
UNSPACED_LANGUAGES = frozenset(lang.name for lang in languages.SPOKEN if lang.unspaced)


# Server language codes (normalized, as /v2/generate) -> (the Whisper-style code the edit's words are tokenized by,
# zonos3.tts.align_text.align_tokens; the Qwen3-ASR language name its recognition is forced to). Speech editing serves
# the entries whose Qwen3-ASR language the ASR worker supports (its ping).
EDIT_LANGUAGES = {lang.code: (lang.speech, languages.SPOKEN_BY_CODE[lang.speech].name) for lang in languages.SERVER}


def edit_language(language: str, supported) -> tuple[str, str]:
    """``(code, Qwen3-ASR name)`` of a normalized server language code, given the ASR worker's ``supported`` language
    names; ValueError for languages speech editing does not serve."""
    served = [k for k, (_, name) in EDIT_LANGUAGES.items() if name in supported]
    if language not in served:
        raise ValueError(f"Speech editing does not serve language {language!r} (supported: {', '.join(served)}).")
    return EDIT_LANGUAGES[language]


class WorkerError(RuntimeError):
    """A worker is unavailable: it could not start, exited twice in a row, or failed without the input being at fault
    (e.g. its model could not be loaded). The message is final (HTTP 503)."""


class AsrError(WorkerError):
    """The ASR worker is unavailable (``ASR is unavailable: ...``)."""


class FuseAlignError(WorkerError):
    """The FuseAlign worker is unavailable (``FuseAlign alignment is unavailable: ...``)."""


def fusealign_installed() -> bool:
    """Whether this interpreter, the FuseAlign worker's, has ``fusealign_hub`` (third_party/fusealign-multilingual)."""
    return importlib.util.find_spec('fusealign_hub') is not None


def asr_python() -> str:
    """The interpreter of the ASR worker and of plugin helper workers: ``ZONOS3_ASR_PYTHON``, else the repo's
    .venv-asr (the checkout's root, three levels above model/src; a package installed elsewhere has none)."""
    return os.environ.get('ZONOS3_ASR_PYTHON') or str(MODEL_SRC.parents[1] / '.venv-asr' / 'bin' / 'python')


class _JsonLinesWorker:
    """A persistent worker process (``PYTHONPATH`` = model/src). It starts on the first call and again after it exited;
    a call that meets an exited worker (EOF) is retried once on a fresh one. Thread-safe: calls are serialized.

    ``call`` raises ValueError (``<failed>: <message>``) when the worker reports the input at fault and ``error``
    (``<unavailable>: <detail>``) otherwise."""

    error: type[WorkerError] = WorkerError
    name = 'worker'
    unavailable = 'worker is unavailable'
    failed = 'worker failed'

    def __init__(self, command: list[str]):
        self._command = command
        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._next_id = 0

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _start(self) -> subprocess.Popen:
        env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'VIRTUAL_ENV')}
        env['PYTHONPATH'] = str(MODEL_SRC)
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        env.setdefault('HF_HUB_OFFLINE', '1')
        try:
            return subprocess.Popen(self._command, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                    encoding='utf-8')
        except OSError as exc:
            raise self.error(f'{self.unavailable}: cannot start the {self.name}: {exc}') from exc

    def call(self, op: str, **payload) -> dict:
        with self._lock:
            self._next_id += 1
            line = json.dumps({'id': self._next_id, 'op': op, **payload}, ensure_ascii=False) + '\n'
            for _ in range(2):
                if not self.alive:
                    self._process = self._start()
                try:
                    self._process.stdin.write(line)
                    self._process.stdin.flush()
                    reply = self._process.stdout.readline()
                except (BrokenPipeError, OSError):
                    reply = ''
                if reply:
                    break
                code = self._process.wait()
            else:
                raise self.error(f'{self.unavailable}: the {self.name} exited with code {code}')
        response = json.loads(reply)
        if response.get('status') != 'ok':
            if response.get('invalid'):
                raise ValueError(f"{self.failed}: {response.get('message') or response.get('error')}")
            raise self.error(f"{self.unavailable}: {response.get('error', 'unknown error')}")
        return response

    def close(self) -> None:
        with self._lock:  # after any in-flight call
            if self.alive:
                self._process.stdin.close()
            if self._process is not None:
                self._process.wait()
                self._process = None


class AsrWorker(_JsonLinesWorker):
    """The workers/asr.py process (Qwen3-ASR) in .venv-asr: one per server, shared by speech editing and plugins.
    ``device`` defaults to ``ZONOS3_ASR_DEVICE`` (else the worker's own default), ``model`` to the worker's."""

    error = AsrError
    name = 'ASR worker'
    unavailable = 'ASR is unavailable'
    failed = 'ASR failed'

    def __init__(self, python: str | None = None, device: str | None = None, model: str | None = None):
        self.python = python or asr_python()
        device = device or os.environ.get('ZONOS3_ASR_DEVICE')
        super().__init__([self.python, str(ASR_WORKER)] + (['--device', device] if device else [])
                         + (['--model', model] if model else []))
        self._ping: dict | None = None

    @property
    def installed(self) -> bool:
        """Whether the worker's interpreter exists (tools/setup_asr_env.sh)."""
        return Path(self.python).is_file()

    def _start(self) -> subprocess.Popen:
        if not self.installed:
            raise AsrError(f'{self.unavailable}: interpreter not found: {self.python} (run tools/setup_asr_env.sh)')
        return super()._start()

    def ping(self) -> dict:
        """``{"model", "languages"}`` of the worker (cached)."""
        if self._ping is None:
            response = self.call('ping')
            self._ping = {'model': response['model'], 'languages': list(response['languages'])}
        return self._ping

    def model(self) -> str:
        return self.ping()['model']

    def languages(self) -> list[str]:
        """The Qwen3-ASR language names the worker transcribes."""
        return self.ping()['languages']

    def transcribe(self, items: list[dict]) -> list[dict]:
        """``[{audio, start?, end?, language}]`` -> per item ``{text, language}`` (workers/asr.py)."""
        return self.call('transcribe', items=items)['results']


class FuseAlignWorker(_JsonLinesWorker):
    """The workers/fusealign.py process (this interpreter): one per server, shared by speech editing and plugins."""

    error = FuseAlignError
    name = 'FuseAlign worker'
    unavailable = 'FuseAlign alignment is unavailable'
    failed = 'FuseAlign alignment failed'

    def __init__(self, device: str | None = None, checkpoint: str | None = None, python: str = sys.executable):
        super().__init__([python, str(FUSEALIGN_WORKER)] + (['--device', device] if device else [])
                         + (['--checkpoint', checkpoint] if checkpoint else []))
        self._model: str | None = None

    def model(self) -> str:
        """The checkpoint id the worker aligns with (its ``ping``)."""
        if self._model is None:
            self._model = self.call('ping')['model']
        return self._model

    def speech(self, audios: list[str]) -> list[list[list[float]]]:
        """Per audio path its Silero VAD speech regions ``[[start, end], ...]`` (seconds, in time order)."""
        return [r['regions'] for r in self.call('speech', items=[{'audio': a} for a in audios])['results']]

    def segments(self, audios: list[str]) -> list[list[list[float]]]:
        """Per audio path its speech chunks ``[[start, end], ...]`` (seconds, each at most 25 s) for recognition."""
        return [r['chunks'] for r in self.call('segments', items=[{'audio': a} for a in audios])['results']]

    def align(self, items: list[dict]) -> list[list[dict]]:
        """``[{audio, tokens, utterances?}]`` -> per item one ``{word, start, end, presence, placed}`` per token."""
        return [r['words'] for r in self.call('align', items=items)['results']]


def audio_seconds(path: str) -> float:
    """Duration of a sound file (frames / sample rate)."""
    import soundfile

    info = soundfile.info(path)
    return info.frames / info.samplerate


def needs_anchors(seconds: float, tokens: list[str]) -> bool:
    """Whether FuseAlign aligns an item long-form: audio over ``LONG_FORM_SECONDS`` or tokens over ``MAX_TEXT_BYTES``."""
    return seconds > LONG_FORM_SECONDS or len(' '.join(tokens).encode('utf-8')) > MAX_TEXT_BYTES


def chunk_utterances(asr: AsrWorker, aligner: FuseAlignWorker, audios: list[str],
                     languages: list[str | None]) -> list[list[dict]]:
    """Per audio the Qwen3-ASR transcripts of its ``segments`` chunks (forced to its language, a Qwen3-ASR name, or
    detected for None) as utterances ``{text, start, end, language}`` in time order, empty texts dropped. One
    ``segments`` and one ``transcribe`` call for all of them."""
    if not audios:
        return []
    chunks = aligner.segments(audios)
    items = [{'audio': audio, 'start': start, 'end': end, 'language': language}
             for audio, language, cs in zip(audios, languages, chunks, strict=True) for start, end in cs]
    results = iter(asr.transcribe(items) if items else [])
    out = []
    for cs in chunks:
        utterances = []
        for start, end in cs:
            r = next(results)
            if r['text']:
                utterances.append({'text': r['text'], 'start': start, 'end': end, 'language': r['language']})
        out.append(utterances)
    return out


@dataclass
class Transcript:
    text: str
    language: str  # as Qwen3-ASR reports it: the forced language, or the detected one(s), comma-joined
    utterances: list[dict] | None  # the chunk transcripts {text, start, end} of audio over LONG_FORM_SECONDS


def transcribe_audio(asr: AsrWorker, aligner: FuseAlignWorker, audio: str, language: str | None) -> Transcript:
    """Qwen3-ASR transcript of a sound file, forced to ``language`` (a Qwen3-ASR name) or detected (None).

    Up to ``LONG_FORM_SECONDS`` one recognition of the whole file. Longer audio is transcribed chunk by chunk
    (``chunk_utterances``): the text is the chunk texts joined with " " (with "" when the language, the forced one or
    else the first spoken chunk's first detected one, is unspaced: ``UNSPACED_LANGUAGES``), the reported language is
    the first spoken chunk's ("" without speech), and the utterances are returned to anchor later alignments of the
    file (``align_words``)."""
    if audio_seconds(audio) <= LONG_FORM_SECONDS:
        r = asr.transcribe([{'audio': audio, 'language': language}])[0]
        return Transcript(r['text'], r['language'], None)
    utterances = chunk_utterances(asr, aligner, [audio], [language])[0]
    reported = utterances[0]['language'] if utterances else ''
    name = language or reported.split(',')[0].strip()
    joiner = '' if name in UNSPACED_LANGUAGES else ' '
    return Transcript(joiner.join(u['text'] for u in utterances), reported,
                      [{k: u[k] for k in ('text', 'start', 'end')} for u in utterances])


def align_words(asr: AsrWorker, aligner: FuseAlignWorker, items: list[dict]) -> list[list[dict]]:
    """FuseAlign words of ``[{audio, tokens, language, utterances?}]`` (``language``: the Qwen3-ASR name anchors are
    recognized in, None to detect; ``utterances``: anchors already made for the audio, e.g. ``Transcript.utterances``)
    -> per item one ``{word, start, end, presence, placed}`` per token.

    An item that ``needs_anchors`` carries ``utterances``: its own, else ``chunk_utterances`` of its audio (one batch
    for every such item); the other items carry none."""
    long = [k for k, it in enumerate(items) if it['tokens'] and needs_anchors(audio_seconds(it['audio']), it['tokens'])]
    missing = [k for k in long if items[k].get('utterances') is None]
    made = dict(zip(missing, chunk_utterances(asr, aligner, [items[k]['audio'] for k in missing],
                                              [items[k].get('language') for k in missing])))
    requests = []
    for k, it in enumerate(items):
        request = {'audio': it['audio'], 'tokens': list(it['tokens'])}
        if k in long:
            utterances = made[k] if k in made else it['utterances']
            request['utterances'] = [{'text': u['text'], 'start': u['start'], 'end': u['end']} for u in utterances]
        requests.append(request)
    return aligner.align(requests)


# Pronunciation groups (U+E000 IPA U+E001, phoneme-token checkpoints) in a target text. The edit tokens' normalize
# (zonos3.tts.textmatch) mangles them (IPA upper-cased, the private-use markers and tie bars become word breaks), so a
# group is never normalized: it is one target token of its own, the group string itself, which no recognized token of
# the original can equal (a changed word in the diff). Its length for the expected fill counts its IPA
# symbols in letter-equivalents: GROUP_LETTERS_PER_SYMBOL, fitted on edits that replace a word by a group
# and by the spelled word (z3m_mect_edit_phonemes, 4 words x 8 takes each: the letter count that gives a
# group fill the spelled fill's fill/expected; Siobhan 1.40, Niamh 1.67, route 1.39, data 1.17, median
# 1.39). The gold lexicon's English entries average 1.07 letters per symbol (1.17 over running text), which
# left group fills at 1.1-1.9x their expected length: spoken duration follows the word, and the short
# words and names people respell carry more letters per phone than the average entry.
GROUP_OPEN, GROUP_CLOSE = "\ue000", "\ue001"
GROUP_LETTERS_PER_SYMBOL = 1.4
_GROUP_RE = re.compile("(\ue000[^\ue000\ue001]*\ue001)")
# a whitespace word's pieces: pronunciation groups, nonverbal markers (zonos3.tts.prompt) and the text around them
_PIECE_RE = re.compile(rf"({_GROUP_RE.pattern[1:-1]}|{_NONVERBAL_MARKUP.pattern})")


def is_group_token(token: str) -> bool:
    return token.startswith(GROUP_OPEN) and token.endswith(GROUP_CLOSE)


def token_chars(token: str) -> float:
    """Text length of a target token: characters, or a group's IPA symbols in letter-equivalents."""
    return GROUP_LETTERS_PER_SYMBOL * (len(token) - 2) if is_group_token(token) else len(token)


def is_marker_token(token: str) -> bool:
    return _NONVERBAL_MARKUP.fullmatch(token) is not None


def canonical_markup(text: str) -> str:
    """``text`` with the whitespace inside its nonverbal markers removed (``<laugh />`` -> ``<laugh/>``), so every
    marker is one whitespace word piece."""
    return _NONVERBAL_MARKUP.sub(lambda m: "".join(m.group(0).split()), text) if has_nonverbal_markup(text) else text


def target_tokens(target_text: str) -> list[list[str]]:
    """``[target tokens, tokens of target word 0, of word 1, ...]`` over the whitespace words of ``target_text``
    (``zonos3.tts.textmatch.tokens``). Without groups or markup these are the tokens of the target and of its words;
    with them only the plain pieces between them are normalized, each group is a token and each nonverbal marker
    (``canonical_markup`` text) none: markup is text for the model only, outside the word diff, the alignment and
    verify."""
    words = target_text.split()
    if GROUP_OPEN not in target_text and not has_nonverbal_markup(target_text):
        return [textmatch.tokens(target_text)] + [textmatch.tokens(word) for word in words]
    pieces = [[piece for piece in _PIECE_RE.split(word) if piece] for word in words]
    word_tokens = [[t for piece in word for t in (
        [piece] if is_group_token(piece) else [] if is_marker_token(piece) else textmatch.tokens(piece))]
        for word in pieces]
    return [[t for ts in word_tokens for t in ts]] + word_tokens


@dataclass
class AlignedWord:
    word: str
    start: float
    end: float
    confidence: float | None
    tokens: list[str]


def aligned_words(words: list[dict], word_tokens: list[list[str]]) -> list[AlignedWord]:
    """The edit's words of an alignment: FuseAlign's words (``FuseAlignWorker.align``, one per ``align_tokens``
    token) with each word's edit tokens (``textmatch.tokens`` of the word) and FuseAlign's presence as the
    confidence. A word without tokens (punctuation only) carries nothing the edit reads and is dropped; an
    unplaced word keeps FuseAlign's fallback times. FuseAlign does not repair order, so neighbours may overlap by a
    few ms: an overlap is split at its midpoint, which keeps the words valid ``original_words`` (``edit_words``)."""
    if len(words) != len(word_tokens):
        raise ValueError("One token list is needed per aligned word.")
    out = [AlignedWord(w['word'], float(w['start']), float(w['end']), float(w['presence']), list(tokens))
           for w, tokens in zip(words, word_tokens) if tokens]
    for previous, word in zip(out, out[1:]):
        if word.start < previous.end:
            previous.end = word.start = (previous.end + word.start) / 2
    return out


def edit_aligned_words(asr: AsrWorker, aligner: FuseAlignWorker, audio: str, text: str, language: tuple[str, str],
                       utterances: list[dict] | None = None) -> list[AlignedWord]:
    """The words of ``text`` in a sound file with their edit tokens: FuseAlign over ``align_tokens(text, code)``
    (``align_words``; ``language`` is the ``edit_language`` pair, ``utterances`` anchors already made for the file)."""
    code, name = language
    words = align_tokens(text, code)
    aligned = align_words(asr, aligner, [{'audio': audio, 'tokens': words, 'language': name,
                                          'utterances': utterances}])[0] if words else []
    return aligned_words(aligned, [textmatch.tokens(w) for w in words])


@dataclass
class EditDiff:
    spans: list[tuple[int, int]]
    new_target_tokens: list[int]


def seconds_to_spans(pairs, total_frames: int, frame_rate: float = 15.0) -> list[tuple[int, int]]:
    """``[start, end]`` seconds -> ``(start_frame, length_frames)``, outward-rounded and clipped to the audio."""
    spans = []
    for start, end in pairs:
        first = max(0, math.floor(start * frame_rate + 1e-6))
        spans.append((first, min(total_frames, math.ceil(end * frame_rate - 1e-6)) - first))
    return spans


def diff_spans(words, target_tokens, *, total_frames, frame_rate=15.0, margin_seconds=0.08, min_gap_frames=5, max_spans=3):
    expanded = [(token, w) for w in words for token in w.tokens]
    original = [token for token, _ in expanded]
    spans, new_tokens = [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, original, target_tokens, autojunk=False).get_opcodes():
        if op == 'equal':
            continue
        if i1 != i2:
            t0, t1 = expanded[i1][1].start, expanded[i2 - 1][1].end
        else:
            t0 = expanded[i1 - 1][1].end if i1 else 0.0
            t1 = expanded[i1][1].start if i1 < len(expanded) else total_frames / frame_rate
        start = min(total_frames - 1, max(0, math.floor((t0 - margin_seconds) * frame_rate)))
        end = max(start + 1, min(total_frames, math.ceil((t1 + margin_seconds) * frame_rate)))
        if spans and start - (spans[-1][0] + spans[-1][1]) < min_gap_frames:
            previous_start, previous_length = spans[-1]
            spans[-1] = (previous_start, max(end, previous_start + previous_length) - previous_start)
        else:
            spans.append((start, end - start))
        new_tokens.extend(range(j1, j2))
    if not spans:
        raise ValueError('target text equals the original transcript')
    if len(spans) > max_spans:
        raise ValueError(f'edit touches {len(spans)} regions; at most {max_spans} are supported')
    return EditDiff(spans, new_tokens)


def edit_words(words, word_tokens, original_tokens, *, duration) -> list[AlignedWord]:
    """Caller-supplied word timings of the original (``{word, start, end, confidence?}`` in
    order, e.g. corrected aligned words) as AlignedWords with ``word_tokens`` (edit tokens per word).
    Raises ValueError unless the times are finite, each word has start < end, the words are
    ascending and non-overlapping inside [0, duration] and their tokens spell ``original_tokens``
    (the edit tokens of the original text)."""
    if not words:
        raise ValueError("original_words is empty.")
    previous_end = 0.0
    if float(words[0]["start"]) < 0:
        raise ValueError(f"original_words[0] ({words[0]['word']!r}) starts before the clip.")
    for k, w in enumerate(words):
        start, end = float(w["start"]), float(w["end"])
        label = f"original_words[{k}] ({w['word']!r})"
        if not (math.isfinite(start) and math.isfinite(end)):
            raise ValueError(f"{label} has a non-finite time.")
        if not start < end:
            raise ValueError(f"{label} must start before it ends ({start:.3f} >= {end:.3f}).")
        if start < previous_end - 1e-6:
            raise ValueError(f"{label} starts at {start:.3f} s, before the previous word ends ({previous_end:.3f} s).")
        if end > duration + 1e-3:
            raise ValueError(f"{label} ends at {end:.3f} s, after the clip ({duration:.3f} s).")
        previous_end = end
    spelled = [t for tokens in word_tokens for t in tokens]
    if spelled != list(original_tokens):
        k = next((i for i, (a, b) in enumerate(zip(spelled, original_tokens)) if a != b), min(len(spelled), len(original_tokens)))
        raise ValueError(
            "original_words do not spell original_text: they read "
            f"{' '.join(spelled[max(0, k - 2):k + 3]) or '(nothing)'!r} where the text reads "
            f"{' '.join(original_tokens[max(0, k - 2):k + 3]) or '(nothing)'!r} (normalized)."
        )
    return [AlignedWord(w["word"], float(w["start"]), float(w["end"]), w.get("confidence"), list(tokens))
            for w, tokens in zip(words, word_tokens)]


def new_word_confidence(words, token_indices) -> float | None:
    indices = set(token_indices)
    selected, cursor = [], 0
    for word in words:
        if any(i in indices for i in range(cursor, cursor + len(word.tokens))):
            selected.append(word.confidence)
        cursor += len(word.tokens)
    return sum(selected) / len(selected) if selected else None


@dataclass
class EditWindow:
    """The audio frames ``[start, end)`` of the clip the model sees for some regions (indices into
    the edit's spans) and the target text of that stretch."""
    start: int
    end: int
    regions: list[int]
    text: str

    def spans(self, spans) -> list[tuple[int, int]]:
        """This window's regions as window-relative ``(start_frame, length_frames)``."""
        return [(spans[k][0] - self.start, spans[k][1]) for k in self.regions]


def whole_clip_window(spans, total_frames: int, target_text: str) -> EditWindow:
    return EditWindow(0, total_frames, list(range(len(spans))), target_text)


class WordDiff:
    """The original's aligned words (with edit ``tokens``) against the target's whitespace words and
    their edit tokens, diffed over the tokens: maps original word ranges to target word ranges."""

    def __init__(self, words, target_words, target_word_tokens):
        if len(target_word_tokens) != len(target_words):
            raise ValueError("One token list is needed per target word.")
        self.words, self.target_words, self.target_word_tokens = words, target_words, target_word_tokens
        self.orig_start, self.tgt_start = [0], [0]
        for w in words:
            self.orig_start.append(self.orig_start[-1] + len(w.tokens))
        for tokens in target_word_tokens:
            self.tgt_start.append(self.tgt_start[-1] + len(tokens))
        self.opcodes = difflib.SequenceMatcher(
            None, [t for w in words for t in w.tokens], [t for ts in target_word_tokens for t in ts], autojunk=False
        ).get_opcodes()

    def _target_cut(self, i, left):
        """Target token position of a cut before original token ``i`` (an insertion at the cut
        joins the range), or None inside a changed block."""
        for tag, i1, i2, j1, j2 in (self.opcodes if left else reversed(self.opcodes)):
            if i1 < i < i2:
                return j1 + i - i1 if tag == "equal" else None
            if (i1 if left else i2) == i:
                return j1 if left else j2
        return 0 if i == 0 else self.tgt_start[-1]

    def target_word(self, k, left):
        """Target word index of a cut before original word ``k`` (as the ``left`` or right edge
        of a range), or None when the cut falls inside a changed block or a target word."""
        j = self._target_cut(self.orig_start[k], left)
        at = [t for t, start in enumerate(self.tgt_start) if start == j] if j is not None else []
        return (at[0] if left else at[-1]) if at else None

    def clean(self, a, b):
        """Original word range [a, b) widened until both edges map onto target word boundaries."""
        while a > 0 and self.target_word(a, True) is None:
            a -= 1
        while b < len(self.words) and self.target_word(b, False) is None:
            b += 1
        return a, b

    def target_range(self, a, b):
        """Target words aligned with the clean original word range [a, b)."""
        return self.target_word(a, True), self.target_word(b, False)


def plan_edit_windows(spans, words, target_words, target_word_tokens, *, total_frames, frame_rate=15.0,
                      context_seconds=4.0, target_text=None) -> list[EditWindow]:
    """Word-snapped context windows around edit regions of a long clip.

    ``words`` are the aligned words of the original (with edit ``tokens``); ``target_words`` the
    whitespace words of the target text and ``target_word_tokens`` their edit tokens. Each region
    takes the words within ``context_seconds`` of it; an edge inside a changed block of the
    original -> target token diff (or inside a target word) moves outward to the block edge, and
    overlapping windows merge. A window's audio is cut halfway between its edge words and their
    neighbours; its text is the target words aligned with its original words. A window over the
    whole clip takes ``target_text`` itself.
    """
    n = len(words)
    if not n:
        raise ValueError("The aligner found no words in the audio, so the edit cannot be windowed.")
    diff = WordDiff(words, target_words, target_word_tokens)

    def boundary(k):
        if k == 0:
            return 0
        if k == n:
            return total_frames
        return min(total_frames, round((words[k - 1].end + words[k].start) / 2 * frame_rate))

    windows = []
    for index, (start, length) in enumerate(spans):
        t0, t1 = start / frame_rate - context_seconds, (start + length) / frame_rate + context_seconds
        a = next((k for k in range(n) if words[k].end > t0), n)
        b = max(a, next((k + 1 for k in reversed(range(n)) if words[k].start < t1), 0))
        windows.append([*diff.clean(a, b), [index]])

    def frames(window):
        a, b, regions = window
        first = min(spans[k][0] for k in regions)
        last = max(spans[k][0] + spans[k][1] for k in regions)
        return min(boundary(a), first), max(boundary(b), last)

    windows.sort(key=lambda w: (w[0], w[1]))
    merged = [windows[0]]
    for window in windows[1:]:
        current = merged[-1]
        if window[0] < current[1] or frames(window)[0] < frames(current)[1]:
            current[0], current[1] = min(current[0], window[0]), max(current[1], window[1])
            current[2] = sorted(current[2] + window[2])
        else:
            merged.append(window)
    out = []
    for a, b, regions in merged:
        start, end = frames([a, b, regions])
        if a == 0 and b == n and target_text is not None:
            text = target_text
        else:
            ja, jb = diff.target_range(a, b)
            text = " ".join(target_words[ja:jb])
        out.append(EditWindow(start, end, regions, text))
    return out


def expected_fill_seconds(spans, words, target_words, target_word_tokens, *, frame_rate=15.0) -> list[float]:
    """How long each region's fill should take: the region's seconds scaled by the target over
    the original text length (edit token characters) of the words in it (word midpoints inside
    the region; the range widened to clean diff edges, so insertions at its edges count). A
    region holding no original word (a pure insertion) takes the clip's characters per second."""
    diff = WordDiff(words, target_words, target_word_tokens)

    def chars(token_lists):
        return sum(token_chars(t) for tokens in token_lists for t in tokens)

    spoken = sum(w.end - w.start for w in words)
    rate = chars(w.tokens for w in words) / spoken if spoken > 0 else 0.0
    mids = [(w.start + w.end) / 2 for w in words]
    out = []
    for start, length in spans:
        r0, r1 = start / frame_rate, (start + length) / frame_rate
        inside = [k for k, mid in enumerate(mids) if r0 <= mid < r1]
        a = inside[0] if inside else next((k for k, mid in enumerate(mids) if mid >= r0), len(words))
        a, b = diff.clean(a, inside[-1] + 1 if inside else a)
        ja, jb = diff.target_range(a, b)
        original, target = chars(w.tokens for w in words[a:b]), chars(target_word_tokens[ja:jb])
        if original:
            out.append((r1 - r0) * target / original)
        else:
            out.append(target / rate if rate > 0 else r1 - r0)
    return out


def _levenshtein(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def wildcard_verify(reference_tokens, hypothesis_tokens) -> dict:
    """WER / CER of a transcript against reference tokens in which each pronunciation group is a
    wildcard word: it matches any one recognized word at no cost (a missing word still costs one).
    Ties between word alignments go to the one with the fewest character edits on the plain words,
    so a group takes the word recognized where it was spoken. Returns ``word_errors``,
    ``ref_words``, ``cer`` (characters of the reference with each group replaced by the word heard
    there, against the hypothesis, as textmatch.char_error_rate) and ``groups``: each group's IPA and
    the recognized word aligned with it (None if none)."""
    ref, hyp = list(reference_tokens), list(hypothesis_tokens)
    n, m = len(ref), len(hyp)
    wild = [is_group_token(t) for t in ref]

    def sub(i, j):   # (word errors, character edits) of aligning ref[i] with hyp[j]
        if wild[i]:
            return (0, 0)
        return (int(ref[i] != hyp[j]), _levenshtein(ref[i], hyp[j]))

    def add(a, b):
        return (a[0] + b[0], a[1] + b[1])

    cost = [[(0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = add(cost[i - 1][0], (1, 0 if wild[i - 1] else len(ref[i - 1])))
    for j in range(1, m + 1):
        cost[0][j] = add(cost[0][j - 1], (1, len(hyp[j - 1])))
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i][j] = min(add(cost[i - 1][j - 1], sub(i - 1, j - 1)),
                             add(cost[i - 1][j], (1, 0 if wild[i - 1] else len(ref[i - 1]))),
                             add(cost[i][j - 1], (1, len(hyp[j - 1]))))
    heard, i, j = {}, n, m
    while i > 0:
        if j > 0 and cost[i][j] == add(cost[i - 1][j - 1], sub(i - 1, j - 1)):
            if wild[i - 1]:
                heard[i - 1] = hyp[j - 1]
            i, j = i - 1, j - 1
        elif j > 0 and cost[i][j] == add(cost[i][j - 1], (1, len(hyp[j - 1]))):
            j -= 1
        else:
            i -= 1
    filled = "".join(heard.get(k, "") if wild[k] else t for k, t in enumerate(ref))
    said = "".join(hyp)
    cer = 0.0 if not filled else 1.0 if not said else _levenshtein(filled, said) / len(filled)
    return {"word_errors": cost[n][m][0], "ref_words": n, "cer": cer,
            "groups": [{"ipa": t[1:-1], "heard": heard.get(k)} for k, t in enumerate(ref) if wild[k]]}


def fill_groups(text: str, heard) -> str:
    """``text`` with its pronunciation groups replaced, in order, by the words in ``heard``
    (lower-cased; a group with nothing heard is dropped): plain text the aligner can place."""
    heard = iter(heard)
    return " ".join(_GROUP_RE.sub(lambda _m: (next(heard) or "").lower(), text).split())
