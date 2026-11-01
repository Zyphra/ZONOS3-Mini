/**
 * Client-side stand-in for the dubbing endpoints (`/dub/*`), emulating the dub server's runtime (zonos3/dub/runtime.py
 * and plan.py): the same validation messages, pause segmentation, target projection, candidate plans, placement and
 * metrics. Synthesis is the offline babbler; LACI's retries, the unsynthesized candidates' lengths and AV-sync's lip
 * warp are simulated, deterministic per seed. Only a segment's chosen candidate is synthesized, steered onto its length
 * with the tempo control (the babbler has no speaking-rate condition).
 */
import { ApiError, type ControlsSpec, type Dose, type DubAnalysis, type DubAnalyzeRequest, type DubCandidate, type DubCaps, type DubGenerateRequest, type DubLips, type DubLipsync, type DubMetrics, type DubPlanSegment, type DubRef, type DubRegenerateRequest, type DubResult, type DubSegmentOut, type DubTranscript, type DubVoice, type DubWord } from '../api';
import { resample } from '../audio/mix';
import { bytesToBase64, encodeWav, parseWav } from '../lib/files';
import { clamp, gaussian, hash32, hashHex, rng, uid } from '../lib/math';
import { ipaGloss } from '../lib/pron';
import { energyAlign, knownClip } from './edit';
import { syntheticEmbedding } from './embeddings';
import { synthesizeWords, type SynthWord } from './synth';
import { stripTags } from '../lib/tags';

const SR = 44100;
const REF_LANGUAGES: Record<string, string> = {
  en: 'English',
  fr: 'French',
  de: 'German',
  es: 'Spanish',
  it: 'Italian',
  pt: 'Portuguese',
  ja: 'Japanese',
  zh: 'Chinese',
  ko: 'Korean',
  ru: 'Russian',
  yue: 'Cantonese',
};
const TARGET_LANGUAGES: Record<string, string> = {
  en: 'English',
  fr: 'French',
  de: 'German',
  es: 'Spanish',
  it: 'Italian',
  pt: 'Portuguese',
  ja: 'Japanese',
  zh: 'Chinese',
  ko: 'Korean',
};
const UNSPACED: Record<string, true> = { ja: true, zh: true, yue: true };
const MIN_REF_SECONDS = 0.5;
const MAX_SEGMENTS = 120;
const PRE_PAD_S = 0.04;
const POST_PAD_S = 0.06;
const MIN_GAP_S = 0.05;
const FADE_S = 0.01;
const BREAK_PUNCT = ',.;:!?…、。，；：！？';
const SENTENCE_END = '.!?…。！？';
/** LACI's retry budget (LaciConfig.max_retries) and its duration check (PACE_TOLERANCE, PACE_SLACK_FRAMES at 15 fps). */
const MAX_RETRIES = 8;
const PACE_TOLERANCE = 0.1;
const PACE_SLACK_S = 3 / 15;
const PACE_MAX_RETRIES = 2;
const UNKNOWN = 'unknown ref_id/dub_id; upload or generate again';
const LIPS_MISSING = "lip-sync needs the video's face track: wait for the tracking to finish, or drop the video again";
const TEMPO = 'delivery.tempo';
const SOURCE: DubVoice = { mode: 'source' };

export const CAPS: Omit<DubCaps, 'fixture'> = {
  ref_languages: Object.entries(REF_LANGUAGES).map(([code, name]) => ({ code, name })),
  target_languages: Object.entries(TARGET_LANGUAGES).map(([code, label]) => ({ code, label })),
  max_ref_seconds: 180,
  max_video_bytes: 2 ** 30,
  default_min_pause_s: 0.3,
  min_pause_range: [0.1, 2],
  sample_rate: SR,
  frame_rate: 15,
  // The served checkpoint's speaking_rate edges (UTF-8 bytes / s): 10 buckets.
  rate_edges: [10.117, 12.123, 13.286, 14.25, 15.14, 16.079, 17.094, 18.489, 21.208],
  candidate_bucket_offsets: [0, -1, 1, -2, 2],
  candidate_seeds: 2,
  model: 'fixture',
  lipsync: true,
  voice_modes: ['source', 'studio'],
};
const N_BUCKETS = CAPS.rate_edges.length + 1;

/**
 * Extended speaking-rate scale: below bucket 0 and above bucket 9 the rate is reached by pace steering (rate row
 * unknown + a calibrated `delivery.tempo` dose), per target language only as far as it measured safe (plausible
 * fixture values: English −5…12, French −3…11, the rest row buckets only).
 */
const STEERED_RANGE: Record<string, [number, number]> = { en: [-5, 12], fr: [-3, 11] };
type Mode = 'source' | 'studio';
/** A language's bucket range; the slower steered buckets serve studio voices only (a cloned source voice lost words). */
const scaleOf = (lang: string, mode: Mode = 'studio'): [number, number] => {
  const [lo, hi] = STEERED_RANGE[lang] ?? [0, N_BUCKETS - 1];
  return [mode === 'source' ? Math.max(0, lo) : lo, hi];
};
const modeOfVoice = (v: DubVoice | undefined): Mode => (v?.mode === 'studio' ? 'studio' : 'source');

CAPS.buckets = (() => {
  const lo = Math.min(...Object.values(STEERED_RANGE).map((r) => r[0]));
  const hi = Math.max(...Object.values(STEERED_RANGE).map((r) => r[1]));
  const out: NonNullable<DubCaps['buckets']> = [];
  for (let b = lo; b <= hi; b++) {
    const steered = b < 0 || b >= N_BUCKETS;
    const languages = steered ? Object.keys(STEERED_RANGE).filter((l) => b >= STEERED_RANGE[l][0] && b <= STEERED_RANGE[l][1]) : Object.keys(TARGET_LANGUAGES);
    // French needs a little more tempo for the same rate than English (doses differ per language).
    const dose = (l: string) => Math.round((b < 0 ? -0.72 + 0.22 * b : 1.04 + 0.3 * (b - N_BUCKETS + 1)) * (l === 'fr' ? 1.12 : 1) * 100) / 100;
    out.push({ voice_modes: b < 0 ? ['studio'] : ['source', 'studio'], index: b, label: String(b), rate: Math.round(bucketRate(b) * 100) / 100, low: b === lo ? null : edgeBelow(b), high: b === hi ? null : edgeBelow(b + 1), steered, languages, ...(steered ? { doses: Object.fromEntries(languages.map((l) => [l, dose(l)])) } : {}) });
  }
  return out;
})();

/** The nominal edge between bucket b−1 and b: a row edge inside the row, the geometric middle of nominal rates beyond. */
function edgeBelow(b: number): number {
  if (b >= 1 && b <= CAPS.rate_edges.length) return CAPS.rate_edges[b - 1];
  return Math.round(Math.sqrt(bucketRate(b - 1) * bucketRate(b)) * 1000) / 1000;
}

/** What the fixture engine needs from the app: the controls spec, studio voices, the pronunciation-group check. */
export interface Context {
  spec: ControlsSpec;
  embeddingOf: (speakerId: string) => Promise<Float32Array>;
  validateText: (text: string) => void;
}

// ── Stores ────────────────────────────────────────────────────────────────────────────────────────

interface Ref {
  pcm: Float32Array;
  sr: number;
  duration: number;
  /** The reference speaker's synthetic voice, keyed by the audio's content (a re-upload keeps its voice). */
  voiceKey: string;
  lips: DubLips | null;
}

interface Voice {
  embedding: Float32Array;
  controls: Dose;
}

/** A segment's kept take: its audio, speech span (s) and word timings within it, LACI's record. */
interface Take {
  pcm: Float32Array;
  s0: number;
  s1: number;
  words: SynthWord[];
  attempts: number;
  retries: Record<string, number>;
}

interface Dub {
  refId: string;
  duration: number;
  targetLang: string;
  lips: DubLips | null;
  /** The reference speaker's voice, resolved once so a regenerate never needs the (evictable) reference. */
  source: Voice;
  segments: DubSegmentOut[];
  takes: Map<number, Take>;
}

/** Small LRU of uploaded references / generated dubs, as the server's (8 entries); a miss is its 404. */
class Store<T> {
  private items = new Map<string, T>();

  put(value: T): string {
    const key = uid() + uid();
    this.items.set(key, value);
    while (this.items.size > 8) {
      const first = this.items.keys().next();
      if (first.done) break;
      this.items.delete(first.value);
    }
    return key;
  }

  peek(key: string): T | undefined {
    return this.items.get(key);
  }

  get(key: string): T {
    const value = this.items.get(key);
    if (value === undefined) throw new ApiError(404, UNKNOWN);
    this.items.delete(key);
    this.items.set(key, value);
    return value;
  }
}

const refs = new Store<Ref>();
const dubs = new Store<Dub>();

// ── Helpers ───────────────────────────────────────────────────────────────────────────────────────

function wait(ms: number, signal?: AbortSignal) {
  const { promise, resolve, reject } = Promise.withResolvers<void>();
  if (signal?.aborted) reject(new DOMException('Aborted', 'AbortError'));
  const t = setTimeout(resolve, ms);
  signal?.addEventListener('abort', () => {
    clearTimeout(t);
    reject(new DOMException('Aborted', 'AbortError'));
  });
  return promise;
}

/** An upload's progress: a few ticks over ~0.3–0.8 s by size. */
async function upload(bytes: number, onProgress?: (f: number) => void) {
  const ms = 300 + Math.min(500, bytes / 40_000);
  for (let i = 1; i <= 5; i++) {
    await wait(ms / 5);
    onProgress?.(i / 5);
  }
}

const round3 = (x: number) => Math.round(x * 1000) / 1000;
const encoder = new TextEncoder();
const utf8 = (s: string) => encoder.encode(s).length;
const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);
const known = (table: Record<string, string>, code: string) => Object.hasOwn(table, code);

/** plan.rate_bucket: bisect_right over the edges, continued onto the language's steered buckets by nominal rate. */
function rateBucket(rate: number, lang: string, mode: Mode): number {
  // The server's rule: bisect the nominal edges of the language's scale, its lowest bucket open below, its highest open above.
  const [lo, hi] = scaleOf(lang, mode);
  let i = lo;
  while (i < hi && rate >= edgeBelow(i + 1)) i++;
  return i;
}

/** Typical speaking rate (bytes / s) of a bucket: its middle; the open end buckets reach a little past their edge. */
function bucketRate(b: number): number {
  const e = CAPS.rate_edges;
  // Steered buckets continue the row geometrically (×0.88 per step below, ×1.12 above).
  if (b <= 0) return (e[0] - 1.2) * Math.pow(0.88, -b);
  if (b >= e.length) return (e[e.length - 1] + 2) * Math.pow(1.12, b - e.length);
  return (e[b - 1] + e[b]) / 2;
}

function sumDoses(...ds: (Dose | null | undefined)[]): Dose {
  const out: Dose = {};
  for (const d of ds) for (const [k, v] of Object.entries(d ?? {})) out[k] = (out[k] ?? 0) + v;
  return out;
}

// A pronunciation group as sent; the aligner writes some spelling of it, never the IPA.
const SENT_GROUP = /\ue000([^\ue000\ue001]*)\ue001/g;
const heard = (t: string) => t.replace(SENT_GROUP, (_, ipa: string) => ipaGloss(ipa)).replace(/[\ue000\ue001]/g, '');

// ── Text (plan.text_units, unit_bytes, break_after, ref_segments, project_segments) ──────────────────────

interface Unit {
  text: string;
  start: number;
  end: number;
}

const wordy = (ch: string) => /[\p{L}\p{N}']/u.test(ch);
const CJK_BLOCKS: [number, number][] = [
  [0x3040, 0x30ff],
  [0x31f0, 0x31ff],
  [0x3400, 0x4dbf],
  [0x4e00, 0x9fff],
  [0xf900, 0xfaff],
  [0xff66, 0xff9f],
];
const cjk = (ch: string) => {
  const o = ch.codePointAt(0) ?? 0;
  return CJK_BLOCKS.some(([lo, hi]) => lo <= o && o <= hi);
};

/** Word units of a text (UTF-16 offsets): whitespace words, or CJK characters in unspaced languages; punctuation
 * sticks to the unit before it (leading punctuation to the first). */
function textUnits(text: string, lang: string): Unit[] {
  const spans: [number, number][] = [];
  let lead = -1; // start of the punctuation before the first unit
  const attach = (s: number, e: number) => {
    if (spans.length) spans[spans.length - 1][1] = e;
    else if (lead < 0) lead = s;
  };
  if (UNSPACED[lang]) {
    let runEnd = -1;
    let i = 0;
    for (const ch of text) {
      const j = i + ch.length;
      if (/\s/u.test(ch)) {
        // skip
      } else if (wordy(ch)) {
        if (cjk(ch)) {
          spans.push([i, j]);
          runEnd = -1;
        } else if (runEnd === i) {
          spans[spans.length - 1][1] = j;
          runEnd = j;
        } else {
          spans.push([i, j]);
          runEnd = j;
        }
      } else attach(i, j);
      i = j;
    }
  } else {
    for (const m of text.matchAll(/\S+/g)) {
      const s = m.index ?? 0;
      if ([...m[0]].some(wordy)) spans.push([s, s + m[0].length]);
      else attach(s, s + m[0].length);
    }
  }
  if (spans.length && lead >= 0) spans[0][0] = lead;
  return spans.map(([a, b]) => ({ text: text.slice(a, b), start: a, end: b }));
}

const unitBytes = (text: string, units: Unit[]) => units.map((u, k) => utf8(text.slice(u.start, k + 1 < units.length ? units[k + 1].start : u.end)));

/** Half-open word ranges split at pauses >= minPause; zero-width words never start a segment. */
function refSegments(words: DubWord[], minPause: number): [number, number][] {
  if (!words.length) return [];
  const starts = [0];
  let prevEnd: number | null = null;
  for (let j = 0; j < words.length; j++) {
    const w = words[j];
    if (w.end - w.start < 0.001) continue;
    if (prevEnd !== null && j > 0 && w.start - prevEnd >= minPause) starts.push(j);
    prevEnd = w.end;
  }
  return starts.map((a, i): [number, number] => [a, i + 1 < starts.length ? starts[i + 1] : words.length]);
}

/** plan.project_segments: one contiguous target unit range per reference segment, by dynamic programming over the
 * links that would cross segments, the byte share against the duration share, and boundaries off punctuation. */
function projectSegments(ranges: [number, number][], spans: [number, number][], links: [number, number][], ubytes: number[], breaks: boolean[]): [number, number][] {
  const n = ranges.length;
  const m = ubytes.length;
  if (!n) return [];
  if (n === 1) return [[0, m]];
  const segOf = new Map<number, number>();
  ranges.forEach(([a, b], k) => {
    for (let i = a; i < b; i++) segOf.set(i, k);
  });
  // prefix[k][j]: links into units < j whose reference word lies outside segment k
  const prefix = Array.from({ length: n }, () => new Float64Array(m + 1));
  const perUnit = Array.from({ length: n }, () => new Float64Array(m));
  for (const [i, j] of links) {
    const k = segOf.get(i);
    if (j < 0 || j >= m || k === undefined) continue;
    for (let q = 0; q < n; q++) perUnit[q][j] += 1;
    perUnit[k][j] -= 1;
  }
  for (let k = 0; k < n; k++) for (let j = 0; j < m; j++) prefix[k][j + 1] = prefix[k][j] + perUnit[k][j];
  const cum = new Float64Array(m + 1);
  for (let j = 0; j < m; j++) cum[j + 1] = cum[j] + ubytes[j];
  const totalBytes = cum[m];
  const durations = spans.map(([s, e]) => Math.max(0, e - s));
  const totalDur = durations.reduce((a, b) => a + b, 0);
  const pen = new Float64Array(m + 1);
  for (let b = 1; b < m; b++) if (!breaks[b - 1]) pen[b] = 0.5;

  let f = new Float64Array(m + 1).fill(Infinity);
  f[0] = 0;
  const back: Int32Array[] = [];
  for (let k = 0; k < n; k++) {
    const g = new Float64Array(m + 1);
    const arg = new Int32Array(m + 1);
    const share = totalBytes > 0 && totalDur > 0 ? durations[k] / totalDur : null;
    for (let b = 0; b <= m; b++) {
      let best = Infinity;
      let at = 0;
      for (let a = 0; a <= b; a++) {
        if (f[a] === Infinity) continue;
        let c = f[a] + prefix[k][b] - prefix[k][a];
        if (share !== null) c += Math.abs((cum[b] - cum[a]) / totalBytes - share);
        if (c < best) {
          best = c;
          at = a;
        }
      }
      g[b] = best + (k < n - 1 ? pen[b] : 0);
      arg[b] = at;
    }
    back.push(arg);
    f = g;
  }
  const bounds = [m];
  let b = m;
  for (let k = n - 1; k >= 0; k--) {
    b = back[k][b];
    bounds.push(b);
  }
  bounds.reverse();
  return ranges.map((_, k): [number, number] => [bounds[k], bounds[k + 1]]);
}

const CLOSERS = /["'”’)\]»」』]+$/u;

/** The target's sentences (or lines) as unit ranges when there is exactly one per segment: the translator's own
 * segmentation beats any projection. */
function sentenceRanges(text: string, units: Unit[], n: number): [number, number][] | null {
  if (n < 2) return null;
  const cuts: number[] = [];
  for (let j = 0; j + 1 < units.length; j++) {
    const last = units[j].text.replace(CLOSERS, '').slice(-1);
    if ((last && SENTENCE_END.includes(last)) || text.slice(units[j].end, units[j + 1].start).includes('\n')) cuts.push(j + 1);
  }
  if (cuts.length !== n - 1) return null;
  const bounds = [0, ...cuts, units.length];
  return cuts.concat(units.length).map((b, k): [number, number] => [bounds[k], b]);
}

// ── References ────────────────────────────────────────────────────────────────────────────────────

function fingerprint(pcm: Float32Array): string {
  let s = `${pcm.length}`;
  const step = Math.max(1, Math.floor(pcm.length / 64));
  for (let i = 0; i < pcm.length; i += step) s += `,${Math.round(pcm[i] * 1e4)}`;
  return `dub:${hashHex(s)}`;
}

export async function addRef(wav: ArrayBuffer, onProgress?: (f: number) => void): Promise<DubRef> {
  await upload(wav.byteLength, onProgress);
  let parsed: { pcm: Float32Array; sr: number };
  try {
    parsed = parseWav(wav);
  } catch {
    throw new ApiError(400, 'upload a WAV file');
  }
  const duration = parsed.pcm.length / parsed.sr;
  if (!(duration >= MIN_REF_SECONDS && duration <= CAPS.max_ref_seconds)) {
    throw new ApiError(400, `reference must be ${MIN_REF_SECONDS}–${CAPS.max_ref_seconds} s, got ${duration.toFixed(1)} s`);
  }
  const ref_id = refs.put({ ...parsed, duration, voiceKey: fingerprint(parsed.pcm), lips: null });
  return { ref_id, duration, sample_rate: parsed.sr };
}

/** The video is the reference's own clip: its face track covers most of it, at 25 fps. */
export async function lips(refId: string, video: Blob, onProgress?: (f: number) => void): Promise<DubLips> {
  await upload(video.size, onProgress);
  const ref = refs.get(refId);
  await wait(1000 + Math.min(1000, ref.duration * 15));
  const fps = 25;
  const start = 0.2;
  const end = Math.max(start + 0.1, ref.duration - 0.3);
  ref.lips = { start, end, track_frames: Math.round((end - start) * fps), video_frames: Math.round(ref.duration * fps), tracks: 1 };
  return { ...ref.lips };
}

const STOCK =
  'Welcome back, everyone. Today I want to show you something a little different from the usual, so stay with me for a few minutes. ' +
  'We started this project last spring with a simple question: how far can a small team get with the right tools? ' +
  'It turned out to be much further than we expected, and the results surprised all of us. Let me walk you through it.';

/** Reference words: the studio clip's exact timings when the transcript has its word count, else energy-aligned. */
function refWords(ref: Ref, text: string, lang: string): DubWord[] {
  const spaced = UNSPACED[lang] ? textUnits(text, lang).map((u) => u.text).join(' ') : text;
  const raw = spaced.match(/\S+/g) ?? [];
  const k = knownClip(ref.duration);
  if (k && k.words.length === raw.length) return k.words.map((w, i) => ({ word: raw[i], start: round3(w.start), end: round3(w.end) }));
  return energyAlign(ref.pcm, ref.sr, spaced).map(({ word, start, end }) => ({ word, start, end }));
}

/** ASR: the text of a clip the studio synthesized, else a stock English monologue as long as the speech. */
export async function transcribe(refId: string, refLang: string | null): Promise<DubTranscript> {
  const ref = refs.get(refId);
  if (refLang !== null && !known(REF_LANGUAGES, refLang)) throw new ApiError(400, `unsupported reference language: ${refLang}`);
  await wait(600 + Math.min(2400, ref.duration * 40));
  const clip = knownClip(ref.duration);
  let text: string;
  if (clip) text = clip.text;
  else {
    const stock = STOCK.split(' ');
    const count = Math.max(2, Math.round(ref.duration * 2.6));
    text = energyAlign(ref.pcm, ref.sr, Array.from({ length: count }, (_, i) => stock[i % stock.length]).join(' '))
      .map((w) => w.word)
      .join(' ');
  }
  const code = refLang ?? 'en';
  return { text: text.trim(), language: code, language_name: REF_LANGUAGES[code] };
}

// ── Analysis ──────────────────────────────────────────────────────────────────────────────────────

export async function analyze(req: DubAnalyzeRequest): Promise<DubAnalysis> {
  const ref = refs.get(req.ref_id);
  if (!known(REF_LANGUAGES, req.ref_lang)) throw new ApiError(400, `unsupported reference language: ${req.ref_lang}`);
  if (!known(TARGET_LANGUAGES, req.target_lang)) throw new ApiError(400, `unsupported target language: ${req.target_lang}`);
  const refText = req.ref_text.trim();
  const targetText = req.target_text.trim();
  if (!refText || !targetText) throw new ApiError(400, 'both the reference and the target transcript are required');
  const minPause = req.min_pause_s ?? CAPS.default_min_pause_s;
  const [lo, hi] = CAPS.min_pause_range;
  if (!(minPause >= lo && minPause <= hi)) throw new ApiError(400, `min_pause_s must be within ${lo.toFixed(1)}–${hi.toFixed(1)} s`);

  // Studio-known timings skip the "aligner"; alignment time grows with the clip.
  const words = req.ref_words ? req.ref_words.map(({ word, start, end }) => ({ word, start, end })) : refWords(ref, refText, req.ref_lang);
  await wait(400 + Math.min(600, words.length * 3 + (req.ref_words ? 0 : ref.duration * 8)));
  if (!words.length) throw new ApiError(400, 'the forced aligner returned no words; check the transcript and its language');
  const units = textUnits(targetText, req.target_lang);
  if (!units.length) throw new ApiError(400, 'the target transcript has no words');
  const ranges = refSegments(words, minPause);
  if (ranges.length > MAX_SEGMENTS) throw new ApiError(400, 'too many pauses; raise the pause threshold');
  const spans = ranges.map(([a, b]): [number, number] => [Math.min(...words.slice(a, b).map((w) => w.start)), Math.max(...words.slice(a, b).map((w) => w.end))]);
  // The word aligner's links, emulated as the monotone diagonal: target units shared out by reference word count.
  const m = units.length;
  const links = words.map((_, i): [number, number] => [i, Math.min(m - 1, Math.floor(((i + 0.5) * m) / words.length))]);
  const projected =
    sentenceRanges(targetText, units, ranges.length) ??
    projectSegments(ranges, spans, links, unitBytes(targetText, units), units.map((u) => BREAK_PUNCT.includes(targetText[u.end - 1])));

  const warnings: string[] = [];
  const segments = ranges.map(([wa, wb], k): DubPlanSegment => {
    const [ua, ub] = projected[k];
    const [s, e] = spans[k];
    const target = ua === ub ? '' : targetText.slice(units[ua].start, units[ub - 1].end);
    if (!target) warnings.push(`segment ${k}: no target words (it will stay silent)`);
    return {
      index: k,
      ref_start: s,
      ref_end: e,
      ref_word_range: [wa, wb],
      target_unit_range: [ua, ub],
      ref_text: words.slice(wa, wb).map((w) => w.word).join(UNSPACED[req.ref_lang] ? '' : ' '),
      target_text: target,
      bucket_auto: target && e > s ? rateBucket(utf8(target) / (e - s), req.target_lang, 'source') : null,
    };
  });
  return { duration: ref.duration, ref_words: words, segments, warnings };
}

// ── Generation ────────────────────────────────────────────────────────────────────────────────────

interface Planned {
  bucket: number;
  offset: number;
  seed: number;
}

function candidatePlan(bucket: number, seed: number, k: number, lang: string, mode: Mode): Planned[] {
  const out: Planned[] = [];
  const used = new Set<number>();
  const [lo, hi] = scaleOf(lang, mode);
  CAPS.candidate_bucket_offsets.forEach((offset, oi) => {
    const b = bucket + offset;
    if (b < lo || b > hi || used.has(b)) return;
    used.add(b);
    for (let r = 0; r < CAPS.candidate_seeds; r++) out.push({ bucket: b, offset, seed: seed + 100 * k + 10 * oi + r });
  });
  return out;
}

/** A set bucket: the full count of an automatic plan, all in that bucket, told apart by seed only. */
function pinnedCandidatePlan(bucket: number, seed: number, k: number): Planned[] {
  const n = CAPS.candidate_bucket_offsets.length * CAPS.candidate_seeds;
  return Array.from({ length: n }, (_, r) => ({ bucket, offset: 0, seed: seed + 100 * k + r }));
}

interface SegmentSpec {
  ref_start: number;
  ref_end: number;
  target_text: string;
  bucket: number | null;
}

/** runtime._segment_entry: the response entry of a segment with its candidate plan (placement filled later). */
function segmentEntry(k: number, s: SegmentSpec, seed: number, voice: DubVoice, controls: Dose | undefined, lang: string): DubSegmentOut {
  const mode = modeOfVoice(voice);
  const [lo, hi] = scaleOf(lang, mode);
  if (s.bucket !== null && !(Number.isInteger(s.bucket) && s.bucket >= lo && s.bucket <= hi)) {
    throw new ApiError(400, `segment ${k}: bucket must be in [${lo}, ${hi}] for this target language${mode === 'source' ? ' and a source voice' : ''}`);
  }
  const d = s.ref_end - s.ref_start;
  const text = s.target_text;
  const nbytes = utf8(text);
  const pinned = s.bucket !== null;
  const bucket = text ? (s.bucket ?? rateBucket(nbytes / d, lang, mode)) : null;
  const plan = bucket === null ? [] : pinned ? pinnedCandidatePlan(bucket, seed, k) : candidatePlan(bucket, seed, k, lang, mode);
  return {
    index: k,
    ref_start: s.ref_start,
    ref_end: s.ref_end,
    target_text: text,
    status: text ? 'ok' : 'silent',
    seed,
    bytes: nbytes,
    rate: text ? nbytes / d : null,
    bucket,
    pinned,
    candidates: plan.map((c) => ({ ...c, speech_s: null, attempts: null, retries: null, final_bucket: null, end: null, capped: null, score: null })),
    chosen: null,
    ...UNPLACED,
    voice,
    ...(controls ? { controls } : {}),
  };
}

const UNPLACED = {
  speech_start: null,
  speech_end: null,
  shift: null,
  start_error: null,
  end_error: null,
  duration_error: null,
  lip_start: null,
  lip_end: null,
  lip_scale: null,
  lip_stretch: null,
  lip_squeeze: null,
  lip_warp: null,
} as const;

const CONTENT_KINDS: [string, number][] = [
  ['truncation', 0.4],
  ['skip', 0.35],
  ['hallucination', 0.25],
];

function contentKind(u: number): string {
  for (const [kind, p] of CONTENT_KINDS) {
    if (u < p) return kind;
    u -= p;
  }
  return CONTENT_KINDS[0][0];
}

type Outcome = Pick<DubCandidate, 'attempts' | 'retries' | 'final_bucket' | 'end' | 'capped'> & { speech_s: number };

/** One candidate under LACI, simulated: its length from the bucket's typical rate (±9 %), content retries (most takes
 * none or one), pace retries that move an automatic plan's bucket toward the target length (a pinned bucket only
 * records the miss), now and then a take that used the whole budget or ran to max_tokens. */
function simulateCandidate(c: Planned, text: string, d: number, pinned: boolean, lang: string, mode: Mode): Outcome {
  const [lo, hi] = scaleOf(lang, mode);
  const r = rng(hash32(`laci:${c.seed}:${c.bucket}:${text}`));
  let speech = (utf8(text) / bucketRate(c.bucket)) * Math.exp(0.09 * gaussian(r));
  const retries: Record<string, number> = {};
  const bump = (kind: string) => (retries[kind] = (retries[kind] ?? 0) + 1);
  const u = r();
  const capped = u > 0.985;
  const content = capped ? MAX_RETRIES : u < 0.55 ? 0 : u < 0.85 ? 1 : u < 0.97 ? 2 : 3;
  for (let i = 0; i < content; i++) bump(contentKind(r()));
  let finalBucket = c.bucket;
  let pace = 0;
  if (!pinned && !capped) {
    const tolerance = PACE_TOLERANCE * d + PACE_SLACK_S;
    while (pace < PACE_MAX_RETRIES && Math.abs(speech - d) > tolerance) {
      const fast = speech < d;
      bump(fast ? 'pace_fast' : 'pace_slow');
      finalBucket = clamp(finalBucket + (fast ? -1 : 1), lo, hi);
      speech = d * Math.pow(speech / d, 0.3 + 0.25 * r());
      pace++;
    }
  }
  const end = (capped ? r() < 0.5 : r() < 0.01) ? 'max_tokens' : 'eos';
  // Ran to max_tokens (15 fps × (2d + 3) s): the take babbles on well past its text.
  if (end === 'max_tokens') speech = Math.max(speech, 2 * d + 3 - r());
  return { speech_s: speech, attempts: content + pace, retries, final_bucket: finalBucket, end, capped };
}

/** plan.candidate_score: log length miss, plus penalties for overrunning the room, a capped take, max_tokens. */
function candidateScore(c: Pick<DubCandidate, 'speech_s' | 'capped' | 'end'>, d: number, room: number): number | null {
  if (c.speech_s === null || c.speech_s <= 0) return null;
  let score = Math.abs(Math.log(c.speech_s / d));
  if (c.speech_s > room) score += 0.5;
  if (c.capped) score += 1;
  if (c.end === 'max_tokens') score += 1;
  return score;
}

const rateOf = (q: number) => (q >= 0 ? 1 + 0.3 * q : 1 / (1 + 0.45 * -q));
const tempoOf = (rate: number) => (rate >= 1 ? (rate - 1) / 0.3 : -(1 / rate - 1) / 0.45);

interface Synth {
  pcm: Float32Array;
  s0: number;
  s1: number;
  words: SynthWord[];
}

/** The chosen candidate's audio: babbled at the voice's controls, the tempo dose corrected (up to 3 passes) until the
 * speech lasts about `target` seconds. Unspaced languages babble a syllable per character. */
function synthTake(spec: ControlsSpec, text: string, lang: string, voice: Voice, seed: number, target: number): Synth | null {
  // Nonverbal markup is allowed in dub targets (scale 1); the babble does not voice it.
  const plain = stripTags(text);
  const spoken = UNSPACED[lang] ? textUnits(plain, lang).map((u) => u.text).join(' ') : plain;
  let q = voice.controls[TEMPO] ?? 0;
  let best: Synth | null = null;
  let bestMiss = Infinity;
  for (let pass = 0; pass < 3; pass++) {
    const { pcm, words } = synthesizeWords(spec, { text: spoken, embedding: voice.embedding, controls: { ...voice.controls, [TEMPO]: q }, seed });
    if (!words.length) return null;
    const s0 = words[0].start;
    const s1 = words[words.length - 1].end;
    const miss = Math.log((s1 - s0) / target);
    if (Math.abs(miss) < bestMiss) {
      best = { pcm, s0, s1, words };
      bestMiss = Math.abs(miss);
    }
    if (Math.abs(miss) < 0.04) break;
    const next = clamp(tempoOf(rateOf(q) * Math.exp(miss)), -4, 6);
    if (next === q) break;
    q = next;
  }
  return best && { ...best, words: best.words.map((w) => ({ ...w, word: heard(w.word) })).filter((w) => w.word) };
}

interface Stats {
  candidates: number;
  retries: Record<string, number>;
  exhausted: number;
}

/** The reference speaker: the studio clip's voice (with its raw steering) when the reference is one, else a synthetic
 * voice of the audio. */
function sourceVoice(spec: ControlsSpec, ref: Ref): Voice {
  const clip = knownClip(ref.duration);
  if (clip) return { embedding: clip.embedding, controls: Object.fromEntries(Object.entries(clip.controls ?? {}).filter(([id]) => spec.directions[id]?.space === 'raw')) };
  return { embedding: syntheticEmbedding(spec, ref.voiceKey, ''), controls: {} };
}

/** runtime._render: plan, "synthesize" and choose the takes of segments `ks` into `dub.takes`. */
async function render(ctx: Context, dub: Dub, ks: number[], signal?: AbortSignal): Promise<Stats> {
  const stats: Stats = { candidates: 0, retries: {}, exhausted: 0 };
  const studio = new Map<string, Promise<Float32Array>>();
  for (const k of ks) {
    await wait(0, signal); // yield between segments: the page stays responsive and an abort lands
    const e = dub.segments[k];
    const d = e.ref_end - e.ref_start;
    const next = dub.segments.find((x) => x.index > k && x.status === 'ok');
    const room = next ? next.ref_start - MIN_GAP_S - e.ref_start : Infinity;
    for (const c of e.candidates) {
      Object.assign(c, simulateCandidate(c, e.target_text, d, e.pinned, dub.targetLang, modeOfVoice(e.voice)));
      c.score = candidateScore(c, d, room);
      for (const [kind, n] of Object.entries(c.retries ?? {})) stats.retries[kind] = (stats.retries[kind] ?? 0) + n;
      if ((c.attempts ?? 0) >= MAX_RETRIES) stats.exhausted++;
    }
    stats.candidates += e.candidates.length;
    let pick = 0;
    e.candidates.forEach((c, i) => {
      if (c.score !== null && c.score < (e.candidates[pick].score ?? Infinity)) pick = i;
    });
    const chosen = e.candidates[pick];

    const v = e.voice ?? SOURCE;
    let voice: Voice;
    if (v.mode === 'studio') {
      let emb = studio.get(v.speaker_embedding_id);
      if (!emb) studio.set(v.speaker_embedding_id, (emb = ctx.embeddingOf(v.speaker_embedding_id)));
      voice = { embedding: await emb, controls: sumDoses(v.controls, e.controls) };
    } else voice = { embedding: dub.source.embedding, controls: sumDoses(dub.source.controls, e.controls) };
    const take = synthTake(ctx.spec, e.target_text, dub.targetLang, voice, chosen.seed, chosen.speech_s ?? d);
    if (!take) throw new ApiError(500, `segment ${k}: no usable take`);
    chosen.speech_s = take.s1 - take.s0;
    const best = candidateScore(chosen, d, room) ?? 0;
    chosen.score = best;
    // The measured length replaces the estimate; estimates that would now beat it move away from the target, so the
    // choice stays the best score as the server reports it.
    const r = rng(hash32(`rescore:${chosen.seed}:${k}`));
    for (const c of e.candidates) {
      if (c === chosen || c.score === null || c.speech_s === null || c.score > best) continue;
      const side = c.speech_s < d ? -1 : 1;
      c.speech_s = d * Math.exp(side * (best + 0.03 + 0.1 * r()));
      c.score = candidateScore(c, d, room);
    }
    e.chosen = pick;
    dub.takes.set(k, { ...take, attempts: chosen.attempts ?? 0, retries: chosen.retries ?? {} });
  }
  return stats;
}

interface Placed {
  cropStart: number;
  cropEnd: number;
  placeStart: number;
  speech_start: number;
  speech_end: number;
  shift: number;
  start_error: number;
  end_error: number;
  duration_error: number;
}

/** plan.place_segments: speech starts at its reference start unless the previous one ran over (then MIN_GAP_S after
 * it); takes are cropped to their speech plus pads. */
function placeSegments(items: { ref_start: number; ref_end: number; take_len: number; s0: number; s1: number }[], refDuration: number) {
  const placed: Placed[] = [];
  let prevEnd = -Infinity;
  let timeline = refDuration;
  for (const it of items) {
    let cropStart = Math.max(0, it.s0 - PRE_PAD_S);
    const cropEnd = Math.min(it.take_len, it.s1 + POST_PAD_S);
    const speechStart = Math.max(it.ref_start, prevEnd + MIN_GAP_S);
    const speechEnd = speechStart + (it.s1 - it.s0);
    let placeStart = speechStart - (it.s0 - cropStart);
    if (placeStart < 0) {
      cropStart += -placeStart;
      placeStart = 0;
    }
    prevEnd = speechEnd;
    timeline = Math.max(timeline, placeStart + cropEnd - cropStart);
    placed.push({
      cropStart,
      cropEnd,
      placeStart,
      speech_start: speechStart,
      speech_end: speechEnd,
      shift: speechStart - it.ref_start,
      start_error: speechStart - it.ref_start,
      end_error: speechEnd - it.ref_end,
      duration_error: it.s1 - it.s0 - (it.ref_end - it.ref_start),
    });
  }
  return { placed, timeline };
}

interface LipSeg {
  start: number;
  end: number;
  scale: number;
  stretch: number;
  squeeze: number;
  warp: number;
}

/** AV-sync's warp of each placed segment, simulated per (seed, segment, text): its speech moves by up to ±80 ms and
 * scales by a few %; the largest local stretch / squeeze is mostly < 1.1 (green), sometimes < 1.3 (amber), now and
 * then up to the guard's 1.5 (red). A dub of 3+ segments shows at least one amber. */
function simulateLips(segs: { index: number; seed: number; text: string; start: number; end: number }[]): LipSeg[] {
  const draws = segs.map((s) => {
    const r = rng(hash32(`lips:${s.seed}:${s.index}:${s.text}`));
    return { r, tier: r() };
  });
  if (draws.length >= 3 && draws.every((x) => x.tier >= 0.3)) draws.reduce((a, b) => (b.tier < a.tier ? b : a)).tier = 0.2;
  return segs.map((s, i) => {
    const { r, tier } = draws[i];
    // The server's guard holds every local rate within 1/1.5…1.5 (LIP_GUARD_MAX_RATE).
    const peak = tier < 0.08 ? 1.3 + 0.2 * r() : tier < 0.3 ? 1.1 + 0.19 * r() : 1.01 + 0.08 * r();
    const stretches = r() < 0.5;
    const other = 1 + (peak - 1) * 0.6 * r();
    const scale = 1 + (stretches ? 1 : -1) * (peak - 1) * 0.15 * r();
    const start = Math.max(0, s.start + (r() * 2 - 1) * 0.08);
    return {
      start,
      end: start + (s.end - s.start) * scale,
      scale,
      stretch: stretches ? peak : other,
      squeeze: stretches ? other : peak,
      warp: 1 + (peak - 1) * (0.25 + 0.2 * r()),
    };
  });
}

/** runtime._assemble: place the kept takes, mix them (10 ms fades), lip-sync when asked, store the dub. With lip-sync
 * each segment's audio is resampled onto its warped span (a pitch shift of a few %), so words and audio agree. */
function assemble(dub: Dub, stats: Stats, generationS: number): DubResult {
  const info = dub.segments;
  const entries = [...dub.takes].sort((a, b) => a[0] - b[0]);
  const kept = entries.map(([k]) => k);
  const takes = entries.map(([, t]) => t);
  const { placed, timeline } = placeSegments(
    kept.map((k, i) => ({ ref_start: info[k].ref_start, ref_end: info[k].ref_end, take_len: takes[i].pcm.length / SR, s0: takes[i].s0, s1: takes[i].s1 })),
    dub.duration,
  );
  const proposed = dub.lips ? simulateLips(kept.map((k, i) => ({ index: k, seed: info[k].seed, text: info[k].target_text, start: placed[i].speech_start, end: placed[i].speech_end }))) : null;
  // The server keeps the guarded warp only when it lowers the SyncNet distance; about a third of fixture dubs don't.
  const sr = rng(hash32(`sync:${dub.refId}:${kept.map((k) => `${info[k].seed}:${info[k].target_text}`).join('|')}`));
  const before = Math.round((0.15 + 0.03 * sr()) * 1000) / 1000;
  const after = Math.round((before + (sr() - 0.65) * 0.05) * 1000) / 1000;
  const applied = after < before;
  const lipSegs = applied ? proposed : null;

  const fade = Math.max(1, Math.round(FADE_S * SR));
  const chunks = kept.map((_, i) => {
    const p = placed[i];
    let piece: Float32Array = takes[i].pcm.slice(Math.round(p.cropStart * SR), Math.round(p.cropEnd * SR));
    let at = p.placeStart;
    const lip = lipSegs?.[i];
    if (lip) {
      piece = resample(piece, SR, SR * lip.scale);
      at = Math.max(0, lip.start - (p.speech_start - p.placeStart) * lip.scale);
    }
    const nf = Math.min(fade, piece.length >> 1);
    for (let j = 0; j < nf; j++) {
      piece[j] *= j / nf;
      piece[piece.length - 1 - j] *= j / nf;
    }
    return { pos: Math.round(at * SR), piece };
  });
  const total = Math.max(Math.round(timeline * SR), ...chunks.map((c) => c.pos + c.piece.length));
  const mix = new Float32Array(total);
  for (const { pos, piece } of chunks) for (let j = 0; j < piece.length; j++) mix[pos + j] += piece[j];
  for (let j = 0; j < total; j++) mix[j] = clamp(mix[j], -1, 1);

  for (const e of info) Object.assign(e, UNPLACED);
  const words: DubWord[] = [];
  kept.forEach((k, i) => {
    const p = placed[i];
    const lip = lipSegs?.[i];
    const unwarped = proposed && !lipSegs ? { lip_start: p.speech_start, lip_end: p.speech_end, lip_scale: 1, lip_stretch: 1, lip_squeeze: 1, lip_warp: 1 } : {};
    Object.assign(info[k], {
      speech_start: p.speech_start,
      speech_end: p.speech_end,
      shift: p.shift,
      start_error: p.start_error,
      end_error: p.end_error,
      duration_error: p.duration_error,
      ...(lip ? { lip_start: lip.start, lip_end: lip.end, lip_scale: round3(lip.scale), lip_stretch: round3(lip.stretch), lip_squeeze: round3(lip.squeeze), lip_warp: round3(lip.warp) } : unwarped),
    });
    const warp = (t: number) => (lip ? lip.start + (t - p.speech_start) * lip.scale : t);
    for (const w of takes[i].words) {
      words.push({ word: w.word, start: round3(warp(p.speech_start + w.start - takes[i].s0)), end: round3(warp(p.speech_start + w.end - takes[i].s0)), segment: k });
    }
  });

  let lipsync: DubLipsync | null = null;
  if (dub.lips && proposed) {
    const shifts = proposed.flatMap((l, i) => [Math.abs(l.start - placed[i].speech_start), Math.abs(l.end - placed[i].speech_end)]);
    lipsync = {
      start: dub.lips.start,
      end: dub.lips.end,
      max_shift_ms: 1000 * Math.max(0, ...shifts),
      mean_abs_shift_ms: 1000 * mean(shifts) * 0.8,
      max_stretch: round3(Math.max(1, ...proposed.map((l) => l.stretch))),
      max_squeeze: round3(Math.max(1, ...proposed.map((l) => l.squeeze))),
      mean_warp: round3(proposed.length ? 1 + (mean(proposed.map((l) => l.warp)) - 1) * 0.8 : 1),
      // At the scale the live server reports (a guarded cross-language dub measured 0.173 → 0.151).
      sync_distance_before: before,
      sync_distance_after: after,
      guard: { max_rate: 1.5, max_shift_s: 0.4 },
      applied,
      ...(applied ? {} : { reason: `no sync gain: SyncNet distance ${before.toFixed(3)} unwarped, ${after.toFixed(3)} warped` }),
    };
  }

  const keptRetries: Record<string, number> = {};
  for (const t of takes) for (const [kind, n] of Object.entries(t.retries)) keptRetries[kind] = (keptRetries[kind] ?? 0) + n;
  const metrics: DubMetrics = {
    mean_abs_start_error_ms: 1000 * mean(placed.map((p) => Math.abs(p.start_error))),
    mean_abs_end_error_ms: 1000 * mean(placed.map((p) => Math.abs(p.end_error))),
    mean_abs_duration_error_ms: 1000 * mean(placed.map((p) => Math.abs(p.duration_error))),
    max_shift_ms: 1000 * Math.max(0, ...placed.map((p) => p.shift)),
    laci_retries: takes.reduce((a, t) => a + t.attempts, 0),
    retries_by_kind: keptRetries,
    all_retries: Object.values(stats.retries).reduce((a, b) => a + b, 0),
    all_retries_by_kind: stats.retries,
    exhausted_takes: stats.exhausted,
    candidates: stats.candidates,
    generation_s: Math.round(generationS * 100) / 100,
  };
  const dub_id = dubs.put(dub);
  return {
    dub_id,
    audio_base64: bytesToBase64(encodeWav(mix, SR)),
    sample_rate: SR,
    duration: total / SR,
    segments: structuredClone(info),
    words,
    metrics,
    lipsync,
  };
}

/** The server's latency (~0.6 s + 0.25 s per synthesized segment), counting the time the babbler took. */
async function finish(dub: Dub, stats: Stats, began: number, synthesized: number, signal?: AbortSignal): Promise<DubResult> {
  const target = 600 + 250 * synthesized;
  const elapsed = performance.now() - began;
  await wait(Math.max(0, target - elapsed), signal);
  return assemble(dub, stats, Math.max(target, elapsed) / 1000);
}

export async function generate(ctx: Context, req: DubGenerateRequest, signal?: AbortSignal): Promise<DubResult> {
  const began = performance.now();
  const ref = refs.get(req.ref_id);
  const seed = req.seed;
  if (!known(TARGET_LANGUAGES, req.target_lang)) throw new ApiError(400, `unsupported target language: ${req.target_lang}`);
  if (!(seed >= 0)) throw new ApiError(400, 'seed must be >= 0');
  if (!(req.segments.length >= 1 && req.segments.length <= MAX_SEGMENTS)) throw new ApiError(400, `need 1–${MAX_SEGMENTS} segments`);
  let prevEnd = 0;
  const segs = req.segments.map((s, k) => {
    const a = s.ref_start;
    const b = s.ref_end;
    if (!(a >= 0 && a < b && b <= ref.duration + 0.05)) throw new ApiError(400, `segment ${k}: need 0 <= ref_start < ref_end <= ${ref.duration.toFixed(2)}`);
    if (a < prevEnd - 1e-6) throw new ApiError(400, `segment ${k}: segments must be sorted and non-overlapping`);
    prevEnd = b;
    return { ...s, target_text: s.target_text.trim(), bucket: s.bucket ?? null };
  });
  if (!segs.some((s) => s.target_text)) throw new ApiError(400, 'every segment is silent; enter target text');
  for (const s of segs) if (s.target_text) ctx.validateText(s.target_text);
  if (req.lipsync && !ref.lips) throw new ApiError(400, LIPS_MISSING);
  const dub: Dub = {
    refId: req.ref_id,
    duration: ref.duration,
    targetLang: req.target_lang,
    lips: req.lipsync ? ref.lips : null,
    source: sourceVoice(ctx.spec, ref),
    segments: segs.map((s, k) => segmentEntry(k, s, seed, s.voice ?? req.voice ?? SOURCE, s.controls, req.target_lang)),
    takes: new Map(),
  };
  const ks = dub.segments.filter((e) => e.status === 'ok').map((e) => e.index);
  const stats = await render(ctx, dub, ks, signal);
  return finish(dub, stats, began, ks.length, signal);
}

/** New takes for one segment (optionally a new text, bucket, voice or direction); the others keep theirs, but may move
 * when this one's length changes. */
export async function regenerate(ctx: Context, req: DubRegenerateRequest, signal?: AbortSignal): Promise<DubResult> {
  const began = performance.now();
  const old = dubs.get(req.dub_id);
  const seed = req.seed;
  const k = req.segment;
  if (!(seed >= 0)) throw new ApiError(400, 'seed must be >= 0');
  if (!(Number.isInteger(k) && k >= 0 && k < old.segments.length)) throw new ApiError(400, `segment must be in [0, ${old.segments.length})`);
  const seg = old.segments[k];
  const text = req.target_text.trim();
  if (text) ctx.validateText(text);
  const lips = req.lipsync ? (old.lips ?? refs.peek(old.refId)?.lips ?? null) : null;
  if (req.lipsync && !lips) throw new ApiError(400, LIPS_MISSING);
  const dub: Dub = { ...old, segments: structuredClone(old.segments), takes: new Map(old.takes), lips };
  dub.segments[k] = segmentEntry(k, { ref_start: seg.ref_start, ref_end: seg.ref_end, target_text: text, bucket: req.bucket ?? null }, seed, req.voice ?? seg.voice ?? SOURCE, req.controls ?? seg.controls, dub.targetLang);
  dub.takes.delete(k);
  if (!text && !dub.takes.size) throw new ApiError(400, 'every segment would be silent; enter target text');
  const stats = await render(ctx, dub, text ? [k] : [], signal);
  return finish(dub, stats, began, text ? 1 : 0, signal);
}

/** Offline there is no ffmpeg: the MP4 needs the dub server. */
export async function mux(dubId: string, _video: Blob): Promise<Blob> {
  dubs.get(dubId);
  throw new ApiError(501, 'MP4 export runs on the dub server (ffmpeg); offline, download the WAV');
}
