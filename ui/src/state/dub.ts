/**
 * Dub: one project is one source (a dropped video / audio file, or a Scene, Direct take or voice clip handed over
 * from the studio) dubbed into a target language. The document (undoable) holds the source, the transcript, the
 * segments (target text, speaking-rate bucket, voice, direction) and the current dub track; every generate,
 * regenerate, track choice and edit coming back from the Edit screen is one undo step.
 */
import { computed, effect, signal, untracked } from '@preact/signals';
import { stripTags } from '../lib/tags';
import * as api from '../api';
import type { DubLips, DubResult, DubSegmentIn, DubSegmentOut, DubVoice, DubWord, Dose } from '../api';
import * as player from '../audio/player';
import { resample } from '../audio/mix';
import { errorText, isAbort } from '../audio/render';
import { base64ToBytes, encodeWav, parseWav } from '../lib/files';
import { uid } from '../lib/math';
import { ipaToXsampa, pronProblem, spokenText, toSent } from '../lib/pron';
import { DUB_TO_SERVER } from '../lib/languages';
import { act, backend, notify, pronTokens, spec } from './app';
import { accentsOf, design, type Design } from './design';
import { controlDose, toDoses, type Values } from './engine';
import { History } from './history';
import { cast, saveClipVoice, savePersona, speakers, type Persona } from './library';
import { load, persist } from './persist';
import { characterOf, mixdown, renderScene, scene, scenes, timeline, type SceneLine } from './scene';
import { mergeValues } from './requests';
import { script, voiceDesign, type Take } from './script';

// ── Model ─────────────────────────────────────────────────────────────────────────────────────────

/** Who speaks: the source speaker cloned from its own audio, a library / saved voice, or a design (bench, cast). */
export type VoiceChoice = { kind: 'source' } | { kind: 'speaker'; id: string; label: string } | { kind: 'design'; label: string; design: Design };

export type SourceKind = 'file' | 'scene' | 'take' | 'voice';

/** A line of a handed-over Scene: where it is in the mixdown, its voice and direction. */
export interface SourceLine {
  lineId: string;
  start: number;
  end: number;
  text: string;
  voice: VoiceChoice;
  values: Values;
}

export interface DubSource {
  kind: SourceKind;
  name: string;
  pcm: Float32Array;
  sr: number;
  duration: number;
  /** A video file: the preview plays it muted under the dub; lip-sync and MP4 export use it. */
  video: { url: string; file: File } | null;
  lines: SourceLine[];
}

export interface DubSeg {
  id: string;
  ref_start: number;
  ref_end: number;
  ref_text: string;
  /** May hold pronunciation groups "(X-SAMPA)" on phoneme-token checkpoints. */
  target_text: string;
  /** Speaking-rate bucket; null = auto (candidates around the text's rate). */
  bucket: number | null;
  /** Overrides the project voice. */
  voice: VoiceChoice | null;
  /** Direction: UI values of spannable controls. */
  values: Values;
}

/** What a segment's take was made from: a segment differing from it is stale until regenerated. */
interface Made {
  text: string;
  bucket: number | null;
  voice: string;
  controls: string;
}

export interface DubTrack {
  id: string;
  label: string;
  dubId: string;
  pcm: Float32Array;
  sr: number;
  segments: DubSegmentOut[];
  words: DubWord[];
  metrics: DubResult['metrics'] | null;
  lipsync: DubResult['lipsync'];
  made: Made[];
  targetLang: string;
  /** Changed in the Edit screen: the server's takes no longer describe it (a regenerate starts from them). */
  edited: boolean;
}

export interface DubDoc {
  source: DubSource | null;
  refId: string | null;
  refLang: string;
  refText: string;
  /** Word timings known without ASR (handed over from the studio). */
  refWords: DubWord[] | null;
  targetLang: string;
  /** The whole translation, split across segments by /dub/analyze. */
  targetText: string;
  segments: DubSeg[];
  voice: VoiceChoice;
  track: DubTrack | null;
  /** Every dub track of the project, newest last (the versions list). */
  tracks: DubTrack[];
}

export const EMPTY_DUB: DubDoc = { source: null, refId: null, refLang: 'en', refText: '', refWords: null, targetLang: load('dub.target', 'fr'), targetText: '', segments: [], voice: { kind: 'source' }, track: null, tracks: [] };

export const docs = new History<DubDoc>(EMPTY_DUB);
export const doc = docs.state;

export const caps = computed(() => backend.value?.dub ?? null);
export const busy = signal<{ label: string; started: number; progress?: number } | null>(null);
export const error = signal<string | null>(null);
export const lips = signal<{ state: 'none' | 'uploading' | 'tracking' | 'ready' | 'failed'; progress: number; info: DubLips | null; error: string | null }>({ state: 'none', progress: 0, info: null, error: null });
export const lipsyncOn = signal<boolean>(load('dub.lipsync', true));
persist('dub.lipsync', lipsyncOn);
export const minPause = signal<number>(load('dub.minPause', 0.3));
persist('dub.minPause', minPause);
export type Listen = 'source' | 'dub' | 'mix';
export const listen = signal<Listen>('dub');
export const selected = signal<string | null>(null);
/** Seconds: where playback starts (and the playhead rests). */
export const cursor = signal(0);

let ctrl: AbortController | null = null;

// Every change keeps pins valid for the segment's language and voice mode (in the same undo step).
const commit = (fn: (d: DubDoc) => DubDoc) => docs.apply(dropInvalidPins(fn(doc.value)));

// ── Voices ────────────────────────────────────────────────────────────────────────────────────────

export const voiceKey = (v: VoiceChoice | null): string => (!v ? '' : v.kind === 'source' ? 'source' : v.kind === 'speaker' ? `speaker:${v.id}` : `design:${v.design.baseId}:${JSON.stringify(v.design.values)}`);

export function voiceLabel(v: VoiceChoice): string {
  return v.kind === 'source' ? 'Source speaker (clone)' : v.label;
}

/** The request form of a voice: a studio voice in embedding mode, with the design's steering as raw + hidden doses. */
export function dubVoice(v: VoiceChoice): DubVoice {
  if (v.kind === 'source') return { mode: 'source' };
  if (v.kind === 'speaker') return { mode: 'studio', speaker_embedding_id: v.id };
  const s = spec.value;
  const controls = toDoses(s, v.design.values, accentsOf(s, v.design), 'embedding');
  return { mode: 'studio', speaker_embedding_id: v.design.baseId, ...(Object.keys(controls).length ? { controls } : {}) };
}

/** Voices to pick from: the source clone, the Cast bench design, saved cast members, library voices. */
export const voiceOptions = computed<{ group: string; voice: VoiceChoice }[]>(() => {
  const out: { group: string; voice: VoiceChoice }[] = [{ group: '', voice: { kind: 'source' } }];
  const bench = design.value;
  const benchLabel = speakers.value.find((s) => s.id === bench.baseId)?.label;
  if (bench.baseId) out.push({ group: 'Cast bench', voice: { kind: 'design', label: `Bench: ${benchLabel ?? 'voice'}${Object.keys(bench.values).length ? ' (designed)' : ''}`, design: bench } });
  for (const p of cast.value) out.push({ group: 'Your cast', voice: { kind: 'design', label: p.name, design: p.design } });
  for (const s of speakers.value) out.push({ group: 'Library', voice: { kind: 'speaker', id: s.id, label: s.label } });
  return out;
});

/** Spannable UI values → hidden doses for a segment's direction. */
export function directionDoses(values: Values): Dose | undefined {
  const out: Dose = {};
  for (const c of spec.value.ui.controls) {
    const x = values[c.id];
    if (x && c.spannable) controlDose(c, x, out);
  }
  for (const k of Object.keys(out)) out[k] = Math.round(out[k] * 1e4) / 1e4;
  return Object.keys(out).length ? out : undefined;
}

// ── Derived ───────────────────────────────────────────────────────────────────────────────────────

const madeOf = (d: DubDoc, s: DubSeg): Made => ({ text: s.target_text.trim(), bucket: s.bucket, voice: voiceKey(s.voice ?? d.voice), controls: JSON.stringify(directionDoses(s.values) ?? {}) });

/** Segments whose text, bucket, voice or direction changed since the current track was made. */
export const stale = computed(() => {
  const d = doc.value;
  const t = d.track;
  const out = new Set<string>();
  if (!t) return out;
  d.segments.forEach((s, k) => {
    const m = t.made[k];
    const now = madeOf(d, s);
    if (!m || m.text !== now.text || m.bucket !== now.bucket || m.voice !== now.voice || m.controls !== now.controls) out.add(s.id);
  });
  return out;
});

/** Rate bucket of a text over a span (UTF-8 bytes per second, as the server measures it). */
/** A speaking-rate bucket of the dub's target language: a row bucket, or a steered one beyond the row's range. */
export interface ScaleBucket {
  index: number;
  label: string;
  /** Nominal rate (UTF-8 bytes / s); null when the server lists none (row buckets from their edges). */
  rate: number | null;
  low: number | null;
  high: number | null;
  steered: boolean;
  dose?: number;
}

/** The speaking-rate scale for the target language: steered buckets only where they were measured safe for it. */
export type VoiceMode = 'source' | 'studio';

/** A segment's voice mode: the source clone, or a studio voice (library, design). */
export const modeOf = (d: DubDoc, s: DubSeg): VoiceMode => ((s.voice ?? d.voice).kind === 'source' ? 'source' : 'studio');

/**
 * The speaking-rate scale for a target language and voice mode: steered buckets only where measured safe for that
 * language, and the slower steered ones (index < 0) for studio voices only (a cloned source voice lost words there).
 */
export function scaleOf(lang: string, mode: VoiceMode): ScaleBucket[] {
  const c = caps.value;
  if (!c) return [];
  const server = DUB_TO_SERVER[lang];
  const rows = c.rate_edges.length + 1;
  if (!c.buckets?.length) return Array.from({ length: rows }, (_, i) => ({ index: i, label: String(i), rate: null, low: i ? c.rate_edges[i - 1] : null, high: i < rows - 1 ? c.rate_edges[i] : null, steered: false }));
  // The server labels buckets by index; steered ones read as how far past the row they go.
  const label = (i: number, steered: boolean) => (!steered ? String(i) : i < 0 ? `slower−${-i}` : `faster+${i - rows + 1}`);
  return c.buckets
    .filter((b) => !b.steered || !b.languages?.length || b.languages.includes(lang) || (!!server && b.languages.includes(server)))
    .filter((b) => !b.voice_modes?.length || b.voice_modes.includes(mode))
    .map((b) => {
      const dose = b.doses?.[lang] ?? (server ? b.doses?.[server] : undefined);
      return { index: b.index, label: label(b.index, b.steered), rate: b.rate, low: b.low ?? null, high: b.high ?? null, steered: b.steered, ...(dose != null ? { dose } : {}) };
    })
    .sort((a, b) => a.index - b.index);
}

/** The target language's widest scale (studio voices): names the buckets results report. */
export const bucketScale = computed(() => scaleOf(doc.value.targetLang, 'studio'));

/** Pins the segment's language and voice mode no longer offer go back to auto. */
function dropInvalidPins(d: DubDoc): DubDoc {
  const ok = { source: new Set(scaleOf(d.targetLang, 'source').map((b) => b.index)), studio: new Set(scaleOf(d.targetLang, 'studio').map((b) => b.index)) };
  if (!d.segments.some((s) => s.bucket !== null && !ok[modeOf(d, s)].has(s.bucket))) return d;
  return { ...d, segments: d.segments.map((s) => (s.bucket !== null && !ok[modeOf(d, s)].has(s.bucket) ? { ...s, bucket: null } : s)) };
}

/** "b7", "b−2", "b11": a bucket on the extended scale. */
export const bucketTag = (i: number) => `b${i < 0 ? `−${-i}` : i}`;

/** Rate of a text over a span (UTF-8 bytes per second, as the server measures it) and its bucket on the scale. */
export function rateOf(text: string, s: { ref_start: number; ref_end: number }, mode: VoiceMode = 'studio'): { bytes: number; rate: number; bucket: number; fast: boolean; slow: boolean } | null {
  const c = caps.value;
  // Markers count no bytes (the server reads the plain text for the rate).
  const clean = spokenText(stripTags(text)).trim();
  if (!clean || !c) return null;
  const bytes = new TextEncoder().encode(clean).length;
  const rate = bytes / Math.max(0.05, s.ref_end - s.ref_start);
  const edges = c.rate_edges;
  // The server's rule: the language's lowest bucket is open below, its highest open above, otherwise low ≤ rate < high.
  const scale = scaleOf(doc.value.targetLang, mode);
  let bucket = scale[0]?.index ?? 0;
  for (const [i, b] of scale.entries()) {
    if ((i === 0 || b.low == null || rate >= b.low) && (i === scale.length - 1 || b.high == null || rate < b.high)) {
      bucket = b.index;
      break;
    }
  }
  const top = scale[scale.length - 1];
  const bottom = scale[0];
  return { bytes, rate, bucket, fast: rate > (top?.rate ?? edges[edges.length - 1]) && bucket >= (top?.index ?? edges.length), slow: rate < (bottom?.rate ?? edges[0]) && bucket <= (bottom?.index ?? 0) };
}

/** Colour of a lip-sync warp: green below 1.1×, amber below 1.3×, red beyond (WSOLA artefacts grow past ~1.1×). */
export function warpKind(o: DubSegmentOut): 'ok' | 'warn' | 'bad' | null {
  if (o.lip_stretch == null || o.lip_squeeze == null || !lipApplied(doc.value.track)) return null;
  const worst = Math.max(o.lip_stretch, o.lip_squeeze);
  return worst < 1.1 ? 'ok' : worst < 1.3 ? 'warn' : 'bad';
}

/** The track came back lip-synced (a skipped warp, `applied: false`, leaves it unwarped). */
export const lipApplied = (t: DubTrack | null | undefined) => !!t?.lipsync && t.lipsync.applied !== false;

export const RETRY_LABELS: Record<string, string> = { truncation: 'early EOS', hallucination: 'stall', skip: 'skip', pace_fast: 'too fast', pace_slow: 'too slow' };

export const retryText = (byKind: Record<string, number> | null | undefined) =>
  Object.entries(byKind ?? {})
    .sort((a, b) => b[1] - a[1])
    .map(([k, n]) => `${n} ${RETRY_LABELS[k] ?? k}`)
    .join(', ');

/** Source / dub / mix (source ducked under the dub) at the dub's rate. */
const DUCK = 0.25;
export const mixPcm = computed(() => {
  const d = doc.value;
  const t = d.track;
  if (!t || !d.source) return null;
  const src = resample(d.source.pcm, d.source.sr, t.sr);
  const out = new Float32Array(Math.max(src.length, t.pcm.length));
  for (let i = 0; i < out.length; i++) out[i] = Math.max(-1, Math.min(1, (t.pcm[i] ?? 0) + DUCK * (src[i] ?? 0)));
  return out;
});

export function listened(): { id: string; pcm: Float32Array; sr: number } | null {
  const d = doc.value;
  if (!d.source) return null;
  const mode = d.track ? listen.value : 'source';
  if (mode === 'source') return { id: 'dub:source', pcm: d.source.pcm, sr: d.source.sr };
  if (mode === 'dub') return { id: `dub:track:${d.track!.id}`, pcm: d.track!.pcm, sr: d.track!.sr };
  return { id: `dub:mix:${d.track!.id}`, pcm: mixPcm.value!, sr: d.track!.sr };
}

export function togglePlay(from?: number) {
  const l = listened();
  if (!l) return;
  if (player.nowPlaying.value && from === undefined) {
    cursor.value = player.playhead.value;
    return player.stop();
  }
  const dur = l.pcm.length / l.sr;
  const at = from ?? (cursor.value >= dur - 0.05 ? 0 : cursor.value);
  cursor.value = at;
  player.play(l.id, l.pcm, l.sr, at);
}

export function setListen(mode: Listen) {
  const playing = !!player.nowPlaying.value?.id.startsWith('dub:');
  const at = playing ? player.playhead.value : cursor.value;
  listen.value = mode;
  if (playing) togglePlay(at);
}

// ── Opening a source ──────────────────────────────────────────────────────────────────────────────

function lipsReset() {
  lips.value = { state: 'none', progress: 0, info: null, error: null };
}

/** A new project on `source`; the reference is uploaded (with progress) and, for a video, its face tracked. */
async function start(source: DubSource, init: Partial<DubDoc>): Promise<boolean> {
  const c = caps.value;
  if (!c) return false;
  if (source.duration < 0.5 || source.duration > c.max_ref_seconds) {
    notify(`Dubbing takes 0.5–${Math.round(c.max_ref_seconds)} s of audio (this is ${source.duration.toFixed(1)} s).`, 'error');
    return false;
  }
  cancel();
  player.stop();
  lipsReset();
  error.value = null;
  const prev = doc.value.source?.video?.url;
  if (prev && prev !== source.video?.url) URL.revokeObjectURL(prev);
  docs.reset({ ...EMPTY_DUB, targetLang: doc.value.targetLang, voice: init.voice ?? { kind: 'source' }, source, ...init });
  selected.value = null;
  cursor.value = 0;
  act.value = 'dub';
  return ensureRef();
}

/** Another project's dub: aborts what runs, revokes the previous video's URL, resets the screen state. */
export function loadDoc(next: DubDoc) {
  cancel();
  const prev = doc.value.source?.video?.url;
  if (prev && prev !== next.source?.video?.url) URL.revokeObjectURL(prev);
  lipsReset();
  error.value = null;
  selected.value = null;
  cursor.value = 0;
  listen.value = 'dub';
  flagged.value = new Set();
  docs.reset(next);
}

/** Server reference ids live for the server session only: a project stores none, they are kept per source here. */
const refIds = new WeakMap<DubSource, string>();
const uploads = new WeakMap<DubSource, Promise<boolean>>();

/**
 * The current source's reference on the server: uploaded (with progress) the first time it is needed, then for a
 * video its face tracked. A source restored from a project (or an undo step before the upload) has none yet.
 */
export function ensureRef(): Promise<boolean> {
  const d = doc.value;
  if (d.refId || !d.source) return Promise.resolve(!!d.refId);
  const known = refIds.get(d.source);
  if (known) {
    docs.set({ ...d, refId: known });
    return Promise.resolve(true);
  }
  let p = uploads.get(d.source);
  if (!p) {
    const source = d.source;
    p = uploadRef(source).finally(() => uploads.delete(source));
    uploads.set(source, p);
  }
  return p;
}

async function uploadRef(source: DubSource): Promise<boolean> {
  busy.value = { label: `Uploading ${source.name}…`, started: performance.now(), progress: 0 };
  let refId: string;
  try {
    refId = (await api.dubRef(encodeWav(source.pcm, source.sr), (f) => busy.value && (busy.value = { ...busy.value, progress: f }))).ref_id;
  } catch (e) {
    if (doc.value.source === source) error.value = `Couldn’t upload the reference: ${errorText(e)}`;
    return false;
  } finally {
    busy.value = null;
  }
  refIds.set(source, refId);
  if (doc.value.source !== source) return false;
  docs.set({ ...doc.value, refId });
  if (source.video && caps.value?.lipsync) void trackLips(refId, source.video.file);
  return true;
}

// A source without a reference (restored, or undone to) gets one when the Dub screen is open.
effect(() => {
  const d = doc.value;
  if (act.value === 'dub' && d.source && !d.refId && caps.value) untracked(() => void ensureRef());
});

async function trackLips(refId: string, video: File) {
  lips.value = { state: 'uploading', progress: 0, info: null, error: null };
  try {
    const info = await api.dubLips(refId, video, video.name, (f) => {
      lips.value = { ...lips.value, progress: f, state: f >= 1 ? 'tracking' : 'uploading' };
    });
    if (doc.value.refId === refId) lips.value = { state: 'ready', progress: 1, info, error: null };
  } catch (e) {
    if (doc.value.refId === refId) lips.value = { state: 'failed', progress: 0, info: null, error: errorText(e) };
  }
}

const VIDEO = /\.(mp4|mov|m4v|webm|mkv|avi)$/i;

/** A dropped / chosen file: decode its audio track in the browser, upload it, transcribe it. */
export async function openFile(f: File) {
  const c = caps.value;
  if (!c) return;
  if (f.size > c.max_video_bytes) return notify(`That file is too large (${(f.size / 1048576).toFixed(0)} MB).`, 'error');
  busy.value = { label: `Decoding ${f.name}…`, started: performance.now() };
  let pcm: Float32Array;
  let sr: number;
  try {
    ({ pcm, sr } = await player.decode(await f.arrayBuffer()));
  } catch {
    busy.value = null;
    return notify(`${f.name}: the browser couldn’t decode its audio. Convert it to MP4 (AAC), WebM or WAV.`, 'error');
  }
  busy.value = null;
  const isVideo = f.type.startsWith('video/') || VIDEO.test(f.name);
  const source: DubSource = { kind: 'file', name: f.name, pcm, sr, duration: pcm.length / sr, video: isVideo ? { url: URL.createObjectURL(f), file: f } : null, lines: [] };
  if (await start(source, {})) await transcribe(null);
}

/** Transcribe the reference (Qwen3-ASR on the server); `lang` null = detect. */
export async function transcribe(lang: string | null) {
  const d = doc.value;
  if (!d.refId) return;
  busy.value = { label: lang ? 'Transcribing…' : 'Detecting the language and transcribing…', started: performance.now() };
  error.value = null;
  try {
    const t = await api.dubTranscribe(d.refId, lang);
    commit((x) => ({ ...x, refText: t.text, refLang: t.language ?? lang ?? x.refLang, refWords: t.words?.length ? t.words : null }));
  } catch (e) {
    error.value = `Couldn’t transcribe: ${errorText(e)}`;
  } finally {
    busy.value = null;
  }
}

/** Words spread over a stretch in proportion to their length (lines whose word timings the studio doesn't have). */
function spreadWords(text: string, start: number, end: number): DubWord[] {
  const ws = text.split(/\s+/).filter(Boolean);
  const weights = ws.map((w) => w.replace(/[^\p{L}\p{N}]/gu, '').length + 1);
  const total = weights.reduce((a, b) => a + b, 0) || 1;
  const out: DubWord[] = [];
  let t = start;
  ws.forEach((w, i) => {
    const len = ((end - start) * weights[i]) / total;
    out.push({ word: w, start: +t.toFixed(3), end: +(t + len * 0.92).toFixed(3) });
    t += len;
  });
  return out;
}

/** Where speech starts and ends in a clip (RMS over 10 ms against a fraction of its peak). */
function speechSpan(pcm: Float32Array, sr: number): [number, number] {
  const hop = Math.max(1, Math.round(sr * 0.01));
  const rms: number[] = [];
  for (let i = 0; i + hop <= pcm.length; i += hop) {
    let s = 0;
    for (let j = i; j < i + hop; j++) s += pcm[j] * pcm[j];
    rms.push(Math.sqrt(s / hop));
  }
  const thr = Math.max(...rms, 1e-6) * 0.08;
  const a = rms.findIndex((x) => x > thr);
  let b = rms.length - 1;
  while (b > 0 && rms[b] <= thr) b--;
  return a < 0 ? [0, pcm.length / sr] : [(a * hop) / sr, ((b + 1) * hop) / sr];
}

/** Scene → Dub: the scene's mixdown, one segment per line with its character's voice and direction. */
export async function dubScene() {
  await renderScene();
  const clips = timeline.value.clips;
  if (!clips.length) return notify('The scene has no lines yet.', 'error');
  if (clips.some((c) => !c.audio)) return notify('Some scene lines could not be rendered; fix them first.', 'error');
  const sr = clips[0].audio!.sr;
  const mix = mixdown(sr);
  const lines: SourceLine[] = clips.map((c) => {
    const [a, b] = speechSpan(c.audio!.pcm, c.audio!.sr);
    const p = characterOf(c.line);
    return {
      lineId: c.line.id,
      start: +(c.start + a).toFixed(3),
      end: +(c.start + b).toFixed(3),
      text: spokenText(c.line.text),
      voice: p ? { kind: 'design', label: p.name, design: p.design } : { kind: 'source' },
      values: c.line.values ?? {},
    };
  });
  const source: DubSource = { kind: 'scene', name: 'Scene', pcm: mix, sr, duration: mix.length / sr, video: null, lines };
  const refWords = lines.flatMap((l) => spreadWords(l.text, l.start, l.end));
  const segments: DubSeg[] = lines.map((l) => ({ id: uid(), ref_start: l.start, ref_end: l.end, ref_text: l.text, target_text: '', bucket: null, voice: l.voice, values: { ...l.values } }));
  await start(source, { refText: lines.map((l) => l.text).join(' '), refLang: 'en', refWords, segments });
}

/** Direct → Dub: a take, its text known (no ASR), spoken by the script's voice with its baseline. */
export async function dubTake(take: Take, label: string) {
  const d = voiceDesign.value;
  const voice: VoiceChoice = { kind: 'design', label: `Direct voice${script.value.voice ? '' : ' (bench)'}`, design: { ...d, values: mergeValues(spec.value, d.values, script.value.baseline) } };
  const source: DubSource = { kind: 'take', name: label, pcm: take.audio.pcm, sr: take.audio.sr, duration: take.audio.pcm.length / take.audio.sr, video: null, lines: [] };
  await start(source, { refText: spokenText(take.text), refLang: 'en', voice });
}

/** A library voice's reference clip, with its transcript when the library has one. */
export async function dubVoiceClip(id: string) {
  const s = speakers.value.find((x) => x.id === id);
  if (!s) return;
  busy.value = { label: `Loading ${s.label}…`, started: performance.now() };
  try {
    const { pcm, sr } = await player.decode(await api.getPreview(s));
    busy.value = null;
    const source: DubSource = { kind: 'voice', name: s.label, pcm, sr, duration: pcm.length / sr, video: null, lines: [] };
    if (await start(source, { refText: s.transcript ?? '', refLang: 'en' })) if (!s.transcript) await transcribe(null);
  } catch (e) {
    busy.value = null;
    notify(`Couldn’t load ${s.label}: ${errorText(e)}`, 'error');
  }
}

// ── Script ────────────────────────────────────────────────────────────────────────────────────────

export function setRefText(text: string) {
  commit((d) => ({ ...d, refText: text, refWords: d.refWords && text.trim() === d.refWords.map((w) => w.word).join(' ').trim() ? d.refWords : null }));
}

export function setRefLang(lang: string) {
  commit((d) => ({ ...d, refLang: lang }));
}

export function setTargetLang(lang: string) {
  localStorage.setItem('zvs.dub.target', JSON.stringify(lang));
  commit((d) => ({ ...d, targetLang: lang }));
}

export function setTargetText(text: string) {
  commit((d) => ({ ...d, targetText: text }));
}

/** The voice and direction of the handed-over line that overlaps a segment most. */
function lineOf(d: DubDoc, a: number, b: number): SourceLine | null {
  let best: SourceLine | null = null;
  let most = 0;
  for (const l of d.source?.lines ?? []) {
    const o = Math.min(b, l.end) - Math.max(a, l.start);
    if (o > most) (most = o), (best = l);
  }
  return best;
}

/** Split the reference at pauses and project the translation onto the pieces (/dub/analyze). */
export async function analyze() {
  const d = doc.value;
  if (!d.refId || !d.refText.trim() || !d.targetText.trim()) return;
  const problem = pronProblem(d.targetText, pronTokens.value);
  if (problem) return void (error.value = problem);
  busy.value = { label: d.refWords ? 'Splitting with the known timings…' : 'Aligning the transcript and splitting at pauses…', started: performance.now() };
  error.value = null;
  try {
    const a = await api.dubAnalyze({
      ref_id: d.refId,
      ref_text: d.refText.trim(),
      ref_lang: d.refLang,
      target_text: toSent(d.targetText.trim(), pronTokens.value).text,
      target_lang: d.targetLang,
      min_pause_s: minPause.value,
      ...(d.refWords ? { ref_words: d.refWords } : {}),
    });
    // The server sees pronunciation groups in their sent form; show them back as written.
    const back = fromSent;
    commit((x) => ({
      ...x,
      refWords: x.refWords ?? null,
      segments: a.segments.map((s) => {
        const l = lineOf(x, s.ref_start, s.ref_end);
        return { id: uid(), ref_start: s.ref_start, ref_end: s.ref_end, ref_text: s.ref_text, target_text: back(s.target_text), bucket: null, voice: l?.voice ?? null, values: l ? { ...l.values } : {} };
      }),
    }));
    if (a.warnings.length) notify(a.warnings.join(' · '));
  } catch (e) {
    error.value = `Couldn’t split the transcript: ${errorText(e)}`;
  } finally {
    busy.value = null;
  }
}

/** U+E000 IPA U+E001 → "(X-SAMPA)" as the editor writes it. */
function fromSent(t: string): string {
  const tokens = pronTokens.value;
  if (!tokens || !t.includes(tokens.open_marker)) return t;
  const re = new RegExp(`${tokens.open_marker}([^${tokens.close_marker}]*)${tokens.close_marker}`, 'g');
  return t.replace(re, (_, ipa: string) => `(${ipaToXsampa(ipa)})`);
}

export function updateSeg(id: string, patch: Partial<DubSeg>) {
  commit((d) => ({ ...d, segments: d.segments.map((s) => (s.id === id ? { ...s, ...patch } : s)) }));
}

export function setSegValue(id: string, control: string, v: number) {
  commit((d) => ({
    ...d,
    segments: d.segments.map((s) => {
      if (s.id !== id) return s;
      const values = { ...s.values, [control]: Math.round(v * 100) / 100 };
      if (!values[control]) delete values[control];
      return { ...s, values };
    }),
  }));
}

export function setVoice(v: VoiceChoice) {
  commit((d) => ({ ...d, voice: v }));
}

/** Back to the script stage: new segments come from the next split. */
export function clearSegments() {
  commit((d) => ({ ...d, segments: [], track: null }));
}

// ── Generating ────────────────────────────────────────────────────────────────────────────────────

const lipsyncReady = () => !!(caps.value?.lipsync && doc.value.source?.video && lips.value.state === 'ready' && lipsyncOn.value);

export const canGenerate = computed(() => {
  const d = doc.value;
  return !!d.refId && d.segments.some((s) => s.target_text.trim()) && !busy.value && !segmentProblems.value.length && (!d.source?.video || !caps.value?.lipsync || !lipsyncOn.value || lips.value.state === 'ready' || lips.value.state === 'failed');
});

/** Pronunciation problems per segment (they would make the server refuse the dub). */
export const segmentProblems = computed(() =>
  doc.value.segments.flatMap((s, k) => {
    const p = pronProblem(s.target_text, pronTokens.value);
    return p ? [`Segment ${k + 1}: ${p}`] : [];
  }),
);

function segIn(d: DubDoc, s: DubSeg): DubSegmentIn {
  const controls = directionDoses(s.values);
  return {
    ref_start: s.ref_start,
    ref_end: s.ref_end,
    target_text: toSent(s.target_text.trim(), pronTokens.value).text,
    bucket: s.bucket,
    ...(s.voice && voiceKey(s.voice) !== voiceKey(d.voice) ? { voice: dubVoice(s.voice) } : {}),
    ...(controls ? { controls } : {}),
  };
}

function trackOf(d: DubDoc, res: DubResult, label: string): DubTrack {
  const { pcm, sr } = parseWav(base64ToBytes(res.audio_base64));
  return { id: uid(), label, dubId: res.dub_id, pcm, sr, segments: res.segments, words: res.words, metrics: res.metrics, lipsync: res.lipsync, made: d.segments.map((s) => madeOf(d, s)), targetLang: d.targetLang, edited: false };
}

const langLabel = (code: string) => caps.value?.target_languages.find((l) => l.code === code)?.label ?? code;

export async function generate() {
  if (busy.value) return cancel();
  const d = doc.value;
  if (!d.refId || !canGenerate.value) return;
  const n = d.segments.filter((s) => s.target_text.trim()).length;
  ctrl = new AbortController();
  busy.value = { label: `Dubbing ${n} segment${n === 1 ? '' : 's'} into ${langLabel(d.targetLang)}…`, started: performance.now() };
  error.value = null;
  player.stop();
  try {
    const res = await api.dubGenerate({ ref_id: d.refId, target_lang: d.targetLang, seed: Math.floor(Math.random() * 1_000_000), segments: d.segments.map((s) => segIn(d, s)), lipsync: lipsyncReady(), voice: dubVoice(d.voice) }, ctrl.signal);
    const track = trackOf(d, res, `${langLabel(d.targetLang)} dub`);
    commit((x) => ({ ...x, track, tracks: [...x.tracks, track] }));
    listen.value = 'dub';
  } catch (e) {
    if (!isAbort(e)) error.value = errorText(e);
  } finally {
    busy.value = null;
    ctrl = null;
  }
}

/** New takes for one segment with its current text, bucket, voice and direction; the others keep theirs. */
export async function regenerate(id: string) {
  const d = doc.value;
  const t = d.track;
  const k = d.segments.findIndex((s) => s.id === id);
  if (!t || k < 0 || busy.value) return;
  // Server dubs live for its session: a track restored from a project has no takes left to regenerate from.
  if (!t.dubId) return void (error.value = 'This dub was made in an earlier session and the server no longer has its takes: generate the dub again, then regenerate single segments.');
  const s = d.segments[k];
  const p = pronProblem(s.target_text, pronTokens.value);
  if (p) return void (error.value = `Segment ${k + 1}: ${p}`);
  ctrl = new AbortController();
  busy.value = { label: `Regenerating segment ${k + 1}…`, started: performance.now() };
  error.value = null;
  player.stop();
  try {
    const controls = directionDoses(s.values);
    const res = await api.dubRegenerate(
      // controls {} clears the segment's direction (an omitted field keeps the previous one).
      { dub_id: t.dubId, segment: k, target_text: toSent(s.target_text.trim(), pronTokens.value).text, seed: Math.floor(Math.random() * 1_000_000), bucket: s.bucket, lipsync: lipsyncReady(), voice: dubVoice(s.voice ?? d.voice), controls: controls ?? {} },
      ctrl.signal,
    );
    // The other segments' takes are kept: their "made" snapshots carry over.
    const track = { ...trackOf(d, res, `segment ${k + 1} regenerated`), made: t.made.map((m, j) => (j === k ? madeOf(d, s) : m)) };
    commit((x) => ({ ...x, track, tracks: [...x.tracks, track] }));
    if (t.edited) notify('The edits made in the Edit screen were replaced: the server regenerated from its own takes.');
  } catch (e) {
    if (!isAbort(e)) error.value = errorText(e);
  } finally {
    busy.value = null;
    ctrl = null;
  }
}

export function cancel() {
  ctrl?.abort();
}

/** Segments the user flagged as sounding wrong (this session). */
export const flagged = signal<Set<string>>(new Set());

export function toggleFlag(id: string) {
  const next = new Set(flagged.value);
  if (!next.delete(id)) next.add(id);
  flagged.value = next;
}

/**
 * The studio voice to try when the source clone says a segment badly (a fast, low-quality source segment can clone
 * unintelligibly where a studio voice is clear): the Cast bench design, else the first library voice.
 */
export const fallbackVoice = computed<VoiceChoice | null>(() => voiceOptions.value.find((o) => o.voice.kind !== 'source')?.voice ?? null);

/** One click: the segment gets the fallback studio voice and new takes. */
export async function tryStudioVoice(id: string) {
  const v = fallbackVoice.value;
  if (!v || busy.value) return;
  updateSeg(id, { voice: v });
  const next = new Set(flagged.value);
  next.delete(id);
  flagged.value = next;
  await regenerate(id);
}

/** Make an earlier track current again (one undo step). */
export function useTrack(id: string) {
  const t = doc.value.tracks.find((x) => x.id === id);
  if (!t || t === doc.value.track) return;
  player.stop();
  commit((d) => ({ ...d, track: t }));
}

// ── Hand-offs out ─────────────────────────────────────────────────────────────────────────────────

/** The dub's text as spoken (the Edit screen aligns it instead of transcribing). */
export function dubText(d: DubDoc = doc.value): string {
  return d.segments.map((s) => s.target_text.trim()).filter(Boolean).join(' ');
}

/** A track edited in the Edit screen comes back as the dub track (one undo step). */
export function receiveEdit(pcm: Float32Array, sr: number) {
  const d = doc.value;
  if (!d.track) return false;
  const track: DubTrack = { ...d.track, id: uid(), label: 'edited in Edit', pcm, sr, edited: true };
  commit((x) => ({ ...x, track, tracks: [...x.tracks, track] }));
  return true;
}

/** Dub → Scene: each dubbed segment becomes a scene line at its place, with its voice and direction. */
export async function sendToScene() {
  const d = doc.value;
  const t = d.track;
  if (!t || !d.source) return;
  const personas = new Map<string, Persona>();
  const personaFor = async (v: VoiceChoice): Promise<Persona | null> => {
    const key = voiceKey(v);
    const hit = personas.get(key);
    if (hit) return hit;
    let p: Persona | undefined;
    if (v.kind === 'design') p = cast.value.find((x) => x.name === v.label && JSON.stringify(x.design.values) === JSON.stringify(v.design.values) && x.design.baseId === v.design.baseId);
    if (!p) {
      let design: Design;
      if (v.kind === 'design') design = structuredClone(v.design);
      else if (v.kind === 'speaker') design = { ...structuredClone(doc.value.voice.kind === 'design' ? doc.value.voice.design : designBase()), baseId: v.id, mode: 'embedding', values: {} };
      else {
        // The source speaker becomes a cloned library voice first (its first 20 s).
        const clip = d.source!.pcm.subarray(0, Math.min(d.source!.pcm.length, Math.round(20 * d.source!.sr)));
        const speaker = await saveClipVoice(`${d.source!.name} (dub voice)`, encodeWav(clip, d.source!.sr), 'dub-source.wav');
        speakers.value = [...speakers.value, speaker];
        design = { ...designBase(), baseId: speaker.id, mode: 'embedding', values: {} };
      }
      p = { id: uid(), name: v.kind === 'source' ? `${d.source!.name} (dub)` : voiceLabel(v), design, signature: '', createdAt: Date.now() };
      savePersona(p);
    }
    personas.set(key, p);
    return p;
  };
  busy.value = { label: 'Sending to Scene…', started: performance.now() };
  try {
    const lines: SceneLine[] = [];
    for (const [k, s] of d.segments.entries()) {
      const o = t.segments[k];
      if (!s.target_text.trim() || !o || o.status !== 'ok') continue;
      const p = await personaFor(s.voice ?? d.voice);
      if (!p) continue;
      lines.push({ id: uid(), characterId: p.id, text: s.target_text.trim(), at: +(o.lip_start ?? o.speech_start ?? s.ref_start).toFixed(3), ...(Object.keys(s.values).length ? { values: { ...s.values } } : {}) });
    }
    scenes.apply({ ...scene.value, lines });
    notify(`Sent ${lines.length} dubbed line${lines.length === 1 ? '' : 's'} to Scene (${langLabel(t.targetLang)}); undo there to get the previous scene back.`, 'ok');
    act.value = 'scene';
  } catch (e) {
    notify(`Couldn’t send to Scene: ${errorText(e)}`, 'error');
  } finally {
    busy.value = null;
  }
}

const designBase = (): Design => ({ ...design.value, values: {} });

// ── Downloads ─────────────────────────────────────────────────────────────────────────────────────

export async function downloadMp4() {
  const d = doc.value;
  if (!d.track || !d.source?.video) return;
  if (d.track.edited) return notify('The server can only mux its own dub: this track was changed in Edit. Download the WAV instead.', 'error');
  if (!d.track.dubId) return notify('This dub was made in an earlier session and the server no longer has it: generate the dub again to export the video.', 'error');
  busy.value = { label: 'Muxing the video…', started: performance.now() };
  try {
    const blob = await api.dubMux(d.track.dubId, d.source.video.file, d.source.video.file.name);
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${d.source.name.replace(/\.[^.]+$/, '')}_dub_${d.targetLang}.mp4`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  } catch (e) {
    notify(`Couldn’t export the video: ${errorText(e)}`, 'error');
  } finally {
    busy.value = null;
  }
}
