"""MMS subprocess isolation and transcript-diff edit regions."""
from dataclasses import dataclass
import difflib
import json
import math
import os
import re
from pathlib import Path
import subprocess
import threading

from .markup import _NONVERBAL_MARKUP, has_nonverbal_markup

ZSA_ROOT = os.environ.get('ZSA_ROOT')
ZSA_PYTHON = Path(os.environ.get('ZSA_PYTHON') or (Path(ZSA_ROOT) / '.venv/bin/python' if ZSA_ROOT else ''))
WORKER = Path(__file__).resolve().parents[1] / 'workers' / 'zsa_mms.py'


# Server language codes (normalized, as /v2/generate) -> the Whisper-style codes zsa maps to MMS-1B-all adapters
# (WHISPER_TO_MMS); the dub target languages.
MMS_LANGUAGES = {'en_us': 'en', 'fr_fr': 'fr', 'de': 'de', 'es': 'es', 'it': 'it', 'pt_br': 'pt', 'ja': 'ja',
                 'cmn': 'zh', 'ko': 'ko'}


def mms_language(language: str) -> str:
    """The MMS worker language of a normalized server language code; ValueError for languages MMS alignment does
    not serve."""
    code = MMS_LANGUAGES.get(language)
    if code is None:
        raise ValueError(f"Speech editing has no MMS alignment for language {language!r} "
                         f"(supported: {', '.join(MMS_LANGUAGES)}).")
    return code


class MMSWorker:
    """One MMS-1B-all process in the zsa interpreter with one language adapter; requests are answered in order.

    The model loads once per worker, so a long-lived worker (the server's) pays
    the load only on its first request. ``run`` is thread-safe.
    """

    def __init__(self, device: str | None = None, language: str = 'en'):
        if not ZSA_ROOT:
            raise RuntimeError("Speech editing needs ZSA_ROOT: a z-speech-alignment checkout "
                               "(git@github.com:Zyphra/z-speech-alignment.git) with its uv environment (run `uv sync` "
                               "in it); ZSA_PYTHON overrides the interpreter.")
        env = dict(os.environ, PYTHONPATH=str(ZSA_ROOT), HF_HUB_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
        env.pop('VIRTUAL_ENV', None)
        command = [str(ZSA_PYTHON), str(WORKER), '--lang', language] + (['--device', device] if device else [])
        self._process = subprocess.Popen(command, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self._lock = threading.Lock()

    @property
    def alive(self) -> bool:
        return self._process.poll() is None

    def run(self, requests: list[dict]) -> list[dict]:
        responses = []
        with self._lock:
            for request in requests:
                self._process.stdin.write(json.dumps(request) + '\n')
                self._process.stdin.flush()
                line = self._process.stdout.readline()
                if not line:
                    raise RuntimeError(f'MMS worker exited with code {self._process.wait()}')
                responses.append(json.loads(line))
        return responses

    def close(self) -> None:
        with self._lock:  # after any in-flight run
            if self.alive:
                self._process.stdin.close()
            self._process.wait()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def run_mms(requests: list[dict], device: str | None = None, language: str = 'en') -> list[dict]:
    with MMSWorker(device, language) as worker:
        return worker.run(requests)


def expect_ok(resp):
    if 'error' in resp:
        raise RuntimeError(f"zsa {resp['id']}: {resp['error']}")
    return resp


# Pronunciation groups (U+E000 IPA U+E001, phoneme-token checkpoints) in a target text. zsa's normalize
# mangles them (IPA upper-cased, the private-use markers and tie bars become word breaks), so a group
# never goes to MMS: it is one target token of its own, the group string itself, which no MMS token of
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


def target_normalize_request(target_text: str):
    """What to send to the MMS worker's normalize for a target text, and how to read the answer.

    Returns ``(texts, assemble)``: ``assemble(normalized)`` (the worker's lists for ``texts``, in
    order) gives ``[target tokens, tokens of target word 0, of word 1, ...]`` over the whitespace
    words of ``target_text``. Without groups or markup the texts are the target and its words, as
    before; with them only the plain pieces between them are normalized, each group is a token and
    each nonverbal marker (``canonical_markup`` text) none: markup is text for the model only, outside
    the word diff, the alignment and verify."""
    words = target_text.split()
    if GROUP_OPEN not in target_text and not has_nonverbal_markup(target_text):
        return [target_text] + words, lambda normalized: list(normalized)
    pieces = [[piece for piece in _PIECE_RE.split(word) if piece] for word in words]
    plain = [piece for word in pieces for piece in word if not is_group_token(piece) and not is_marker_token(piece)]

    def assemble(normalized):
        normalized = iter(normalized)
        word_tokens = [[t for piece in word for t in (
            [piece] if is_group_token(piece) else [] if is_marker_token(piece) else next(normalized))]
            for word in pieces]
        return [[t for tokens in word_tokens for t in tokens]] + word_tokens

    return plain, assemble


@dataclass
class AlignedWord:
    word: str
    start: float
    end: float
    confidence: float
    tokens: list[str]


def aligned_words(resp) -> list[AlignedWord]:
    return [AlignedWord(**w) for w in expect_ok(resp)['words']]


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
    order, e.g. corrected MMS words) as AlignedWords with ``word_tokens`` (zsa tokens per word).
    Raises ValueError unless the times are finite, each word has start < end, the words are
    ascending and non-overlapping inside [0, duration] and their tokens spell ``original_tokens``
    (the zsa tokens of the original text)."""
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
    """The original's MMS words (with zsa ``tokens``) against the target's whitespace words and
    their zsa tokens, diffed over the tokens: maps original word ranges to target word ranges."""

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

    ``words`` are the MMS words of the original (with zsa ``tokens``); ``target_words`` the
    whitespace words of the target text and ``target_word_tokens`` their zsa tokens. Each region
    takes the words within ``context_seconds`` of it; an edge inside a changed block of the
    original -> target token diff (or inside a target word) moves outward to the block edge, and
    overlapping windows merge. A window's audio is cut halfway between its edge words and their
    neighbours; its text is the target words aligned with its original words. A window over the
    whole clip takes ``target_text`` itself.
    """
    n = len(words)
    if not n:
        raise ValueError("MMS found no words in the audio, so the edit cannot be windowed.")
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
    the original text length (zsa token characters) of the words in it (word midpoints inside
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
    there, against the hypothesis, as zsa's char_error_rate) and ``groups``: each group's IPA and
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
    (lower-cased; a group with nothing heard is dropped): plain text MMS can align."""
    heard = iter(heard)
    return " ".join(_GROUP_RE.sub(lambda _m: (next(heard) or "").lower(), text).split())
