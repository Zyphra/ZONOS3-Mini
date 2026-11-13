/**
 * Edit: transcript-first speech editing. A project is one clip with a version history (each kept take is a new
 * version, one undo step); the draft on top of the current version is the edited text, the regions planned from
 * its word diff (plus manual "regenerate as is" regions), per-region direction, and the takes of the last
 * generation.
 */
import { computed, effect, signal, untracked } from '@preact/signals';
import * as api from '../api';
import type { ControlSpec, Dose, EditRequest, EditTake } from '../api';
import * as player from '../audio/player';
import { errorText, isAbort } from '../audio/render';
import { ApiError } from '../lib/http';
import { MIN_GAP_FRAMES, contextWindows, marginFrames, normalizeTokens, planRegions, regionTargets, summarizeChanges, tokenOwners, unheard, type Change, type Word } from '../edit/regions';
import { bytesToBase64, base64ToBytes } from '../lib/files';
import { parseWav, wavBytes } from '../lib/wav';
import { levelMeter } from '../lib/level';
import { clamp, uid } from '../lib/math';
import { parsePron, pronProblem, spokenText, toSent } from '../lib/pron';
import { EDIT_LANGUAGES, langLabel } from '../lib/languages';
import { act, backend, notify, projectSettings, pronTokens, requestContext, spec } from './app';
import { controlDose, type Values } from './engine';
import { History } from './history';
import { load, persist } from './persist';
import { addTake, lineRequest, script, setLineText, takes as scriptTakes } from './script';
import { confirm } from '../components/ConfirmDialog';

export type SourceKind = 'take' | 'voice' | 'upload' | 'recording';

export interface Source {
  kind: SourceKind;
  label: string;
  lineId?: string;
  takeId?: string;
}

export interface Version {
  id: string;
  label: string;
  pcm: Float32Array;
  sr: number;
  /** WAV, plain base64, as sent to the server. */
  b64: string;
  /** Duration the server measured (what word timings and spans refer to). */
  duration: number;
  /** Empty: none yet (a server without speech recognition waits for it to be typed, then aligns it). */
  transcript: string;
  /** Word timings in use: the alignment, possibly corrected by hand. */
  words: Word[];
  /** The alignment as the server returned it ("Reset timings"). */
  alignedWords: Word[];
  /** The transcript came from ASR (worth checking). */
  automatic: boolean;
  /** The text as written, pronunciation groups included, when it differs from the (plain) transcript. */
  display?: string;
}

export interface Doc {
  name: string;
  source: Source;
  /** The clip's language (server code): every alignment (and, with speech recognition, transcription and verify) uses it. */
  language: string;
  versions: Version[];
  current: number;
}

export interface Region {
  id: string;
  /** Codec frames [start, end) in the current version. */
  start: number;
  end: number;
  changes: Change[];
  manual: boolean;
  /** Direction for this region: UI values of spannable controls. */
  values: Values;
  /** Its edges were dragged (a text edit re-plans and resets them). */
  edited?: boolean;
  /** How many regions (text changes or manual stretches) closer than the minimum gap this one absorbed. */
  mergedFrom?: number;
}

export interface Take {
  id: string;
  label: string;
  seed: number;
  pcm: Float32Array;
  sr: number;
  b64: string;
  duration: number;
  edited: [number, number][];
  fill: number[];
  stopped: (boolean | null)[];
  verify: EditTake['verify'];
  words: Word[] | null;
  /** What the model listened to, in this take's time. */
  contexts: [number, number][];
  /** How it was directed: "r1 angry ×2 · r2 whisper ×1 · all slower ×1" ('' when undirected). */
  direction: string;
  /** The server's expected fill length per region (`expected_seconds`), when it sent one. */
  expected: (number | null)[];
}

export interface Generation {
  versionId: string;
  target: string;
  /** Regions sent, in the version (server-quantized seconds). */
  spans: [number, number][];
  changes: Change[][];
  manual: boolean[];
  /** What the model listened to, in the version's time (long clips are windowed). */
  contexts: [number, number][];
  /** Normalized target tokens, and per region the ones it has to say (checked against each take's ASR). */
  targetTokens: string[];
  wants: number[][];
  /** `target` writes pronunciation groups "(X-SAMPA)", one target word each. */
  groups: boolean;
  takes: Take[];
  selected: number;
}

/** What judging a take needs from its generation. */
export type GenFacts = Pick<Generation, 'spans' | 'changes' | 'manual' | 'targetTokens' | 'wants' | 'target' | 'groups'>;

export const docs = new History<Doc | null>(null);
export const doc = docs.state;
export const version = computed(() => (doc.value ? doc.value.versions[doc.value.current] : null));

export const text = signal('');
export const regions = signal<Region[]>([]);
/** Some text region's edges were dragged (replanning from the text would undo that). */
let touched = false;
export const fixMode = signal(false);
export const wholeValues = signal<Values>({});
/** Generations by version id: stepping back to a version shows its takes again. */
export const gens = signal<Map<string, Generation>>(new Map());
export const gen = computed<Generation | null>(() => (versionId.value && gens.value.get(versionId.value)) || null);
function setGen(g: Generation | null) {
  const next = new Map(gens.value);
  const id = g?.versionId ?? versionId.value;
  if (!id) return;
  if (g) next.set(id, g);
  else next.delete(id);
  gens.value = next;
}
export const ab = signal<'version' | 'take'>('version');
export const busy = signal<{ label: string; started: number } | null>(null);
export const genError = signal<string | null>(null);
/** The error was the server's refusal (4xx), not a failure. */
export const genRefused = signal(false);
/** Where playback starts, seconds in the displayed audio. */
export const cursor = signal(0);
/** The word whose timing the keyboard edits, and which of its edges. */
export const selectedWord = signal<number | null>(null);
export const activeEdge = signal<'start' | 'end'>('start');

export const takeCount = signal<number>(load('edit.takes', 2));
persist('edit.takes', takeCount);
export const margin = signal<number>(load('edit.margin', 0.08));
persist('edit.margin', margin);
export const seedPin = signal<number | null>(null);
export const DEFAULT_CONTEXT = 4;
export const contextSeconds = signal<number>(load('edit.context', DEFAULT_CONTEXT));
persist('edit.context', contextSeconds);

export const caps = computed(() => backend.value!.edit);
export const canEdit = computed(() => !!backend.value?.edit.speech_editing);
/** The server can transcribe (and verify takes); without, a clip's transcript is typed in and aligned. */
export const canTranscribe = computed(() => !!backend.value?.edit.edit_transcribe);
/** Longest clip an edit takes: edits are planned (and long clips windowed) over an alignment, so both limits bound it. */
export const maxSeconds = computed(() => Math.min(caps.value.edit_max_seconds, caps.value.edit_align_max_seconds));
export const fps = computed(() => caps.value.edit_frame_rate || 15);
export const spannable = computed<ControlSpec[]>(() => spec.value.ui.controls.filter((c) => c.spannable));

export const selectedTake = computed(() => {
  const g = gen.value;
  return g && g.versionId === version.value?.id ? (g.takes[g.selected] ?? null) : null;
});

/** What the waveform shows and plays. */
export const displayed = computed(() => {
  const t = ab.value === 'take' ? selectedTake.value : null;
  const v = version.value;
  if (t) return { kind: 'take' as const, id: `edit:take:${t.id}`, pcm: t.pcm, sr: t.sr, duration: t.pcm.length / t.sr, words: t.words ?? [], label: t.label };
  if (!v) return null;
  return { kind: 'version' as const, id: `edit:v:${v.id}`, pcm: v.pcm, sr: v.sr, duration: v.pcm.length / v.sr, words: v.words, label: v.label };
});

/** The target text takes pronunciation groups: an edit checkpoint with phoneme tokens. */
export const pronEdit = computed(() => canEdit.value && !!pronTokens.value);

/** The current version has no transcript yet (no speech recognition): the user types what it says, then aligns. */
export const needsTranscript = computed(() => !!version.value && !version.value.transcript);

/** What the server would refuse in the target's pronunciation groups (never for the transcript being fixed). */
export const pronError = computed(() => (pronEdit.value && !fixMode.value ? pronProblem(text.value, pronTokens.value) : null));

export const regionError = computed(() => {
  const n = regions.value.length;
  const max = caps.value.edit_max_spans;
  return n > max ? `Too many separate changes (${n}). One generation handles at most ${max} regions: make fewer changes, move them closer, or remove a manual region.` : null;
});

export const seconds = (frames: number) => frames / fps.value;

// ── Opening a clip ────────────────────────────────────────────────────────────────────────────────

export async function wavOf(pcm: Float32Array, sr: number): Promise<string> {
  return bytesToBase64(wavBytes(pcm, sr).buffer);
}

/** Each clip's language, remembered with the project (by source and length). */
export const clipLanguages = signal<Record<string, string>>({});
const clipKey = (s: Source, seconds: number) => `${s.kind}:${s.label}:${seconds.toFixed(2)}`;

/** Whether speech recognition answered this session: the server's first transcription (or take check) loads it. */
let asrWarmed = false;
/** A busy label for a call that runs speech recognition: the first one loads it (about half a minute) rather than hanging. */
export const asrLabel = (label: string) => (asrWarmed || !canTranscribe.value ? label : `${label} (the server's first use loads speech recognition: about half a minute)`);

/** A starting point for typing a transcript the server cannot recognize: a known text with its groups spelled out. */
export const suggestedTranscript = signal('');

/**
 * Open a clip; its language (server code) is the one last used for this clip, else `language` (the clip's own: a
 * take's request language, a library voice's), else the project's.
 * Known text is aligned; otherwise the clip is transcribed, or, on a server without speech recognition, opened
 * without a transcript for the user to type (then aligned by saveFix).
 */
export async function openClip(source: Source, pcm: Float32Array, sr: number, knownText?: string | null, language?: string): Promise<boolean> {
  // A pronunciation group is not a word the aligner can read: such clips are transcribed (or typed) instead.
  const grouped = !!knownText && parsePron(knownText, null).groups.length > 0;
  const known = (grouped ? '' : knownText?.trim()) ?? '';
  const transcribe = !known && canTranscribe.value;
  const max = maxSeconds.value;
  const dur = pcm.length / sr;
  if (dur > max) {
    notify(`Editing opens clips up to ${Math.round(max)} s (this one is ${dur.toFixed(1)} s).`, 'error', {
      label: `Open the first ${Math.round(max)} s`,
      run: () => void openClip(source, pcm.slice(0, Math.floor((max - 0.05) * sr)), sr, knownText, language),
    });
    return false;
  }
  if (dur < 0.3) {
    notify('That clip is too short to edit.', 'error');
    return false;
  }
  if (busy.value) return false;
  const had = doc.value;
  if (had && (had.versions.length > 1 || draftDirty.value || gens.value.size > 0)) {
    const ok = await confirm(`Opening another clip closes “${had.name}” — unsaved text, regions and takes are lost. Continue?`, { ok: 'Open', danger: true });
    if (!ok) return false;
  }
  suggestedTranscript.value = grouped ? spokenText(knownText!) : '';
  const p = projectSettings.value.language;
  const lang = clipLanguages.value[clipKey(source, dur)] ?? [language, p].find((l) => l && EDIT_LANGUAGES.includes(l)) ?? 'en_us';
  ctrl = new AbortController();
  busy.value = { label: known ? 'Aligning the transcript…' : transcribe ? asrLabel('Transcribing…') : 'Opening…', started: performance.now() };
  genError.value = null;
  act.value = 'edit';
  try {
    const b64 = await wavOf(pcm, sr);
    const a = known ? await api.align(b64, known, lang, ctrl.signal) : transcribe ? await api.transcribe(b64, lang, ctrl.signal) : { text: '', duration: dur, words: [] };
    if (transcribe) asrWarmed = true;
    const v: Version = { id: uid(), label: 'Original', pcm, sr, b64, duration: a.duration || dur, transcript: a.text, words: a.words, alignedWords: a.words, automatic: transcribe };
    player.stop();
    gens.value = new Map();
    docs.reset({ name: source.label, source, language: lang, versions: [v], current: 0 });
    clipLanguages.value = { ...clipLanguages.value, [clipKey(source, dur)]: lang };
    return true;
  } catch (e) {
    if (isAbort(e)) notify('Cancelled');
    else notify(`Couldn’t open the clip: ${errorText(e)}`, 'error');
    return false;
  } finally {
    busy.value = null;
    ctrl = null;
  }
}

/** Every time the current version changes (open, keep, undo, redo): a clean draft on it. */
/** Changes when another version becomes current (open, keep, fix, undo/redo across versions), not on retiming. */
const versionId = computed(() => version.value?.id ?? null);

effect(() => {
  void versionId.value;
  const v = untracked(() => version.value);
  untracked(() => {
    selectedWord.value = null;
    genError.value = null;
    ab.value = 'version';
    // A version without a transcript opens on typing it.
    fixMode.value = !!v && !v.transcript;
    cursor.value = 0;
    touched = false;
    regions.value = [];
    text.value = v?.transcript ?? '';
    if (v) replan();
  });
});

/** The draft differs from the version (typed text or regions): leaving the version would lose it. */
export const draftDirty = computed(() => {
  const v = version.value;
  return !!v && (text.value.trim() !== (v.transcript ?? '').trim() || regions.value.length > 0);
});

/** Retiming the current version re-plans the regions against the corrected words (the draft text stays). */
let lastWords: Word[] | null = null;
effect(() => {
  const v = version.value;
  const words = v?.words ?? null;
  if (words && lastWords && words !== lastWords && untracked(() => versionId.value) === v!.id) untracked(() => replan());
  lastWords = words;
});

// ── Word timings ──────────────────────────────────────────────────────────────────────────────────

/** Shortest word the editor allows (seconds). */
export const MIN_WORD = 0.04;
const TOUCH = 0.002;

export const preRoll = signal<boolean>(load('edit.preroll', false));
persist('edit.preroll', preRoll);

const moved = (a: Word, b: Word | undefined) => !b || Math.abs(a.start - b.start) > 1e-4 || Math.abs(a.end - b.end) > 1e-4;
export const adjustedCount = computed(() => {
  const v = version.value;
  return v ? v.words.filter((w, i) => moved(w, v.alignedWords[i])).length : 0;
});
export const isAdjusted = (i: number) => {
  const v = version.value;
  return !!v && moved(v.words[i], v.alignedWords[i]);
};

function withWords(words: Word[]) {
  const d = doc.value;
  if (!d) return;
  docs.set({ ...d, versions: d.versions.map((x, i) => (i === d.current ? { ...x, words } : x)) });
}

/**
 * Move edge `edge` of word i to t. A boundary shared with a touching neighbour moves both words (`detach`
 * moves only this word, opening a gap); an edge dropped within 10 ms of a neighbour closes the gap. Clamped to
 * the neighbours, the clip and MIN_WORD.
 */
export function moveWordEdge(i: number, edge: 'start' | 'end', t: number, detach = false) {
  const v = version.value;
  if (!v || !v.words[i]) return;
  const ws = v.words.map((w) => ({ ...w }));
  const w = ws[i];
  const prev = ws[i - 1];
  const next = ws[i + 1];
  if (edge === 'start') {
    const shared = !detach && prev && Math.abs(prev.end - w.start) < TOUCH;
    const lo = shared ? prev.start + MIN_WORD : prev ? prev.end : 0;
    let x = clamp(t, lo, w.end - MIN_WORD);
    if (!shared && prev && x - prev.end < 0.01) x = prev.end;
    w.start = round4(x);
    if (shared) prev.end = w.start;
  } else {
    const shared = !detach && next && Math.abs(next.start - w.end) < TOUCH;
    const hi = shared ? next.end - MIN_WORD : next ? next.start : v.duration;
    let x = clamp(t, w.start + MIN_WORD, hi);
    if (!shared && next && next.start - x < 0.01) x = next.start;
    w.end = round4(x);
    if (shared) next.start = w.end;
  }
  withWords(ws);
}

const round4 = (x: number) => Math.round(x * 1e4) / 1e4;

/** One drag = one undo step. */
export const wordGesture = { begin: () => docs.begin(), end: () => docs.end() };

let nudgeTimer = 0;
/** Keyboard nudge of the active edge; a burst of nudges is one undo step. */
export function nudgeWord(delta: number) {
  const i = selectedWord.value;
  const v = version.value;
  if (i === null || !v) return;
  docs.begin();
  const w = v.words[i];
  moveWordEdge(i, activeEdge.value, (activeEdge.value === 'start' ? w.start : w.end) + delta);
  clearTimeout(nudgeTimer);
  nudgeTimer = window.setTimeout(() => docs.end(), 700);
}

export function resetTimings() {
  const d = doc.value;
  const v = version.value;
  if (!d || !v || !adjustedCount.value) return;
  docs.apply({ ...d, versions: d.versions.map((x, i) => (i === d.current ? { ...x, words: x.alignedWords } : x)) });
  notify('Word timings reset to the alignment');
}

/** Play word i of the displayed version, with 0.3 s of pre-roll when that option is on. */
export function playWord(i: number) {
  const d = displayed.value;
  const w = d?.words[i];
  if (!d || !w) return;
  player.play(d.id, d.pcm, d.sr, Math.max(0, w.start - (preRoll.value ? 0.3 : 0)), undefined, w.end);
}

/** The server's windowing, estimated before generating: regions ± context, cut halfway between words, merged. */
export const contextPreview = computed<[number, number][]>(() => {
  const v = version.value;
  if (!v || v.duration <= 16) return [];
  return contextWindows(v.words, regions.value.map((r) => [seconds(r.start), seconds(r.end)]), contextSeconds.value, v.duration);
});

// ── Planning ──────────────────────────────────────────────────────────────────────────────────────

const overlap = (a: { start: number; end: number }, b: { start: number; end: number }) => Math.max(0, Math.min(a.end, b.end) - Math.max(a.start, b.start));

/** Sort and merge regions closer than MIN_GAP_FRAMES; direction values survive from the larger part. */
function merged(list: Region[]): Region[] {
  const out: Region[] = [];
  for (const r of [...list].sort((a, b) => a.start - b.start || a.end - b.end)) {
    const prev = out[out.length - 1];
    if (prev && r.start - prev.end < MIN_GAP_FRAMES) {
      const keepPrev = Object.keys(prev.values).length >= Object.keys(r.values).length;
      out[out.length - 1] = {
        ...prev,
        end: Math.max(prev.end, r.end),
        changes: [...prev.changes, ...r.changes],
        // One manual part keeps the region removable (and its edges the user's).
        manual: prev.manual || r.manual,
        values: keepPrev ? prev.values : r.values,
        edited: prev.edited || r.edited,
        mergedFrom: (prev.mergedFrom ?? 1) + (r.mergedFrom ?? 1),
      };
    } else out.push({ ...r, mergedFrom: undefined });
  }
  return out;
}

/** Put the original words of a planned region back into the text (its changes are undone, the region disappears). */
export function revertRegion(id: string) {
  const v = version.value;
  const r = regions.value.find((x) => x.id === id);
  if (!v || !r || !r.changes.length) return;
  const plan = planRegions(v.words, text.value, { duration: v.duration, frameRate: fps.value, marginSeconds: margin.value, groups: pronEdit.value });
  const expanded = v.words.flatMap((w) => w.tokens.map(() => w));
  let t = text.value;
  // Later changes first, so earlier character offsets stay valid.
  for (const c of [...r.changes].sort((a, b) => b.j1 - a.j1)) {
    const from = c.j1 < plan.owners.length ? plan.owners[c.j1][0] : t.length;
    const to = c.j2 > c.j1 ? plan.owners[c.j2 - 1][1] : from;
    const words = [...new Set(expanded.slice(c.i1, c.i2))].map((w) => w.word);
    const sep = c.j1 === c.j2 && from > 0 && from < t.length && !/\s/.test(t[from - 1]) ? ' ' : '';
    t = t.slice(0, from) + sep + words.join(' ') + (c.j1 === c.j2 && words.length && from < t.length && !/\s/.test(t[from]) ? ' ' : '') + t.slice(to);
  }
  setText(t.replace(/[ \t]{2,}/g, ' '));
}

/** Regions from the text diff, plus the manual ones; direction carries over to the region it overlaps most. */
export function replan() {
  const v = version.value;
  if (!v) return;
  const plan = planRegions(v.words, text.value, { duration: v.duration, frameRate: fps.value, marginSeconds: margin.value, groups: pronEdit.value });
  const old = regions.value;
  if (touched && plan.regions.length) notify('Region edges follow the text again.');
  touched = false;
  const planned: Region[] = plan.regions.map((p) => ({ id: uid(), start: p.start, end: p.end, changes: p.changes, manual: false, values: {} }));
  const all = merged([...planned, ...old.filter((r) => r.manual).map((r) => ({ ...r }))]);
  for (const r of all) {
    if (Object.keys(r.values).length) continue;
    let best: Region | null = null;
    for (const o of old) if (overlap(o, r) > (best ? overlap(best, r) : 0)) best = o;
    if (best) r.values = best.values;
  }
  regions.value = all;
}

export function setText(t: string) {
  text.value = t;
  replan();
}

export function addManual(startSec: number, endSec: number) {
  const v = version.value;
  if (!v) return;
  // Re-say whole words: an edge inside a word moves out to that word's edge, then the margin.
  let a = Math.min(startSec, endSec);
  let b = Math.max(startSec, endSec);
  for (const w of v.words) {
    if (w.start < a && a < w.end) a = w.start;
    if (w.start < b && b < w.end) b = w.end;
  }
  const total = Math.ceil(v.duration * fps.value);
  const before = v.words.filter((w) => w.end <= a + 1e-6).pop();
  const after = v.words.find((w) => w.start >= b - 1e-6);
  const [start, end] = marginFrames(a, b, before ? before.end : null, after ? after.start : null, margin.value, fps.value, total);
  regions.value = merged([...regions.value, { id: uid(), start, end, changes: [], manual: true, values: {} }]);
}

export function moveEdges(id: string, startSec: number, endSec: number) {
  const v = version.value;
  if (!v) return;
  const total = Math.ceil(v.duration * fps.value);
  regions.value = merged(
    regions.value.map((r) => {
      if (r.id !== id) return r;
      if (!r.manual) touched = true;
      const start = clamp(Math.round(startSec * fps.value), 0, total - 1);
      return { ...r, start, end: clamp(Math.round(endSec * fps.value), start + 1, total), edited: true };
    }),
  );
}

export function removeRegion(id: string) {
  regions.value = regions.value.filter((r) => r.id !== id);
}

export function setRegionValue(id: string, control: string, v: number) {
  regions.value = regions.value.map((r) => {
    if (r.id !== id) return r;
    const values = { ...r.values, [control]: Math.round(v * 100) / 100 };
    if (!values[control]) delete values[control];
    return { ...r, values };
  });
}

export function clearRegionValues(id: string) {
  regions.value = regions.value.map((r) => (r.id === id ? { ...r, values: {} } : r));
}

export function setWholeValue(control: string, v: number) {
  const values = { ...wholeValues.value, [control]: Math.round(v * 100) / 100 };
  if (!values[control]) delete values[control];
  wholeValues.value = values;
}

export function summaryOf(r: Region): string {
  const v = version.value;
  if (r.manual && !r.changes.length) {
    // An "as is" region re-says the words under it.
    const s = seconds(r.start);
    const e = seconds(r.end);
    const ws = (v?.words ?? []).filter((w) => (w.start + w.end) / 2 >= s && (w.start + w.end) / 2 < e).map((w) => w.word.replace(/^[^\p{L}\p{N}']+|[^\p{L}\p{N}']+$/gu, ''));
    return ws.length ? `re-say “${ws.join(' ')}”` : 're-say this stretch';
  }
  return (v && summarizeChanges(r.changes, v.words, text.value, pronEdit.value)) || 'changed';
}

/** UI values → primitive doses (spannable controls only). */
export function dosesOf(values: Values): Dose | null {
  const out: Dose = {};
  for (const c of spannable.value) {
    const x = values[c.id];
    if (x) controlDose(c, x, out);
  }
  for (const k of Object.keys(out)) out[k] = Math.round(out[k] * 1e4) / 1e4;
  return Object.keys(out).length ? out : null;
}

// ── Generating ────────────────────────────────────────────────────────────────────────────────────

let ctrl: AbortController | null = null;

function takeOf(raw: EditTake, label: string): Take {
  const { pcm, sr } = parseWav(base64ToBytes(raw.audio_base64));
  return {
    id: uid(),
    label,
    seed: raw.seed,
    pcm,
    sr,
    b64: raw.audio_base64,
    duration: raw.duration,
    edited: raw.edited_regions,
    fill: raw.fill_seconds,
    stopped: raw.stopped,
    // The server reports heard words as normalized tokens ("SFEROUT"); shown like the transcript ("sferout").
    verify: raw.verify && { ...raw.verify, groups: raw.verify.groups?.map((x) => ({ ...x, heard: x.heard?.toLowerCase() ?? null })) },
    words: raw.words,
    contexts: [],
    direction: '',
    expected: [],
  };
}

/** "angry ×2, whisper ×1" for UI values of spannable controls. */
export function describeValues(values: Values): string {
  return spannable.value
    .filter((c) => values[c.id])
    .map((c) => {
      const v = values[c.id];
      const name = c.type === 'bipolar' && c.poles ? c.poles[v < 0 ? 0 : 1] : c.label.toLowerCase();
      return `${name} ×${+Math.abs(v).toFixed(2)}`;
    })
    .join(', ');
}

/** Per-region and whole-edit direction, as a take label. `indices` numbers the regions (retry: just one). */
function directionLabel(list: { values: Values }[], indices: number[]): string {
  const parts = list.map((r, i) => (Object.keys(r.values).length ? `r${indices[i] + 1} ${describeValues(r.values)}` : '')).filter(Boolean);
  if (Object.keys(wholeValues.value).length) parts.push(`all ${describeValues(wholeValues.value)}`);
  return parts.join(' · ');
}

export const directed = computed(() => regions.value.some((r) => Object.keys(r.values).length) || Object.keys(wholeValues.value).length > 0);

export interface TakeWarning {
  region: number;
  text: string;
  /** A region's words missing from the take's transcript (ranked below every other problem). */
  unheard?: boolean;
}

/** What looks broken in a take: a fill that hit the cap, ran on far past its region, or lost its new words. */
/** Per region: the words it had to say that the take's ASR transcript does not contain (raw target words). */
export function takeUnheard(t: Take, g: GenFacts): string[][] {
  if (!t.verify) return g.spans.map(() => []);
  const owners = tokenOwners(g.target, g.groups);
  return g.spans.map((_, k) => {
    const out: string[] = [];
    for (const j of unheard(g.targetTokens, t.verify!.asr_text, g.wants[k] ?? [])) {
      const o = owners[j];
      const w = o ? g.target.slice(o[0], o[1]).replace(/^[^\p{L}\p{N}']+|[^\p{L}\p{N}']+$/gu, '') : g.targetTokens[j].toLowerCase();
      if (out[out.length - 1] !== w) out.push(w);
    }
    return out;
  });
}

/** Length checks (cap, runaway, nearly empty / skipped) and the words not heard, as one list per take. */
export function takeWarnings(t: Take, g: GenFacts): TakeWarning[] {
  const { spans, changes, manual } = g;
  const out: TakeWarning[] = [];
  takeUnheard(t, g).forEach((ws, k) => ws.length && out.push({ region: k, text: `Region ${k + 1}: words not heard: ${ws.join(', ')}.`, unheard: true }));
  spans.forEach(([s, e], k) => {
    const was = e - s;
    const fill = t.fill[k] ?? 0;
    const n = k + 1;
    const exp = t.expected[k];
    if (t.stopped[k] === false) out.push({ region: k, text: `Region ${n} never stopped (it ran out of room); this take is probably broken.` });
    // The server's expectation (from the alignment and the word diff) when it sent one, else the heuristics below.
    else if (exp != null && exp > 0) {
      if (fill < 0.5 * exp) out.push({ region: k, text: `Region ${n} came out ${fill.toFixed(2)} s for about ${exp.toFixed(2)} s of words: words were probably skipped.` });
      else if (fill > 2.5 * exp + 1) out.push({ region: k, text: `Region ${n} came out ${fill.toFixed(1)} s for about ${exp.toFixed(1)} s of words: it probably ran on past them.` });
    } else if (fill > Math.max(3 * was, was + 3)) out.push({ region: k, text: `Region ${n} came out ${(fill / was).toFixed(0)}× longer than the original (${fill.toFixed(1)} s): it probably ran on past the new words.` });
    else if ((changes[k] ?? []).some((c) => c.j2 > c.j1) && fill < 0.15) out.push({ region: k, text: `Region ${n} came out nearly empty (${fill.toFixed(2)} s): the new words were probably dropped.` });
    // Re-saying the same words should take about as long; much shorter usually means words were skipped.
    else if (manual[k] && !(changes[k] ?? []).length && fill < 0.6 * was) out.push({ region: k, text: `Region ${n} came out ${Math.round((1 - fill / was) * 100)}% shorter than the words it re-says (${was.toFixed(2)} → ${fill.toFixed(2)} s): words were probably skipped.` });
  });
  return out;
}

/** The take to preselect among ts[from…]: warning-free takes first, then the lowest WER. */
/** The take to preselect among ts[from…]: every region's words heard, then no other warning, then lowest WER. */
export function bestOf(ts: Take[], g: GenFacts, from = 0): number {
  const score = (t: Take) => {
    const ws = takeWarnings(t, g);
    return [ws.some((w) => w.unheard) ? 1 : 0, ws.some((w) => !w.unheard) ? 1 : 0, t.verify?.wer ?? Infinity];
  };
  const less = (a: number[], b: number[]) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2];
  let best = from;
  for (let i = from + 1; i < ts.length; i++) if (less(score(ts[i]), score(ts[best])) < 0) best = i;
  return best;
}

export const canGenerate = computed(() => !!version.value && !fixMode.value && regions.value.length > 0 && !regionError.value && !pronError.value && !!text.value.trim() && !busy.value);

/**
 * A boundary in seconds that lands on its own codec frame however the server quantizes it (floor, ceil or
 * round): a hair inside the region from the nearest frame edge. Rounding to ms instead can push an edge out by a
 * whole frame.
 */
function onFrame(t: number, inward: 1 | -1): number {
  const f = fps.value;
  const k = Math.round(t * f);
  const snapped = Math.abs(t * f - k) < 0.02 ? k : t * f;
  return Math.round(((snapped + inward * 0.001) / f) * 1e6) / 1e6;
}

/** `target` may write pronunciation groups (sent as U+E000 IPA U+E001); `original` is plain: it describes the audio. */
export function requestFor(v: Version, target: string, list: { start: number; end: number; values: Values }[], audio = v.b64, original = v.transcript, words?: Word[]): EditRequest {
  if (pronEdit.value) target = toSent(target, pronTokens.value).text;
  const req: EditRequest = {
    audio_base64: audio,
    original_text: original,
    // Corrected timings replace the server's own alignment (windowing, expected fill lengths).
    ...(words ? { original_words: words.map((w) => ({ word: w.word, start: w.start, end: w.end, tokens: w.tokens })) } : {}),
    target_text: target,
    spans: list.map((r) => [onFrame(r.start, 1), onFrame(r.end, -1)]),
    margin_seconds: margin.value,
    num_takes: takeCount.value,
    // The take check runs speech recognition: a server without it refuses `verify`.
    verify: canTranscribe.value,
    seed: seedPin.value,
    language: doc.value?.language ?? 'en_us',
  };
  if (contextSeconds.value !== DEFAULT_CONTEXT) req.context_seconds = contextSeconds.value;
  // The project's decoder, as Direct renders with (null: the server default, left out).
  const decoder = requestContext.value.decoder;
  if (decoder) req.decoder = decoder;
  const whole = dosesOf(wholeValues.value);
  if (whole) req.controls = whole;
  const per = list.map((r) => dosesOf(r.values));
  if (per.some(Boolean)) req.region_controls = per;
  return req;
}

export async function generate() {
  if (busy.value) return cancel();
  const v = version.value;
  if (!v || !canGenerate.value) return;
  const list = regions.value;
  const target = text.value.trim();
  const req = requestFor(v, target, list.map((r) => ({ start: seconds(r.start), end: seconds(r.end), values: r.values })), v.b64, v.transcript, adjustedCount.value ? v.words : undefined);
  ctrl = new AbortController();
  busy.value = { label: asrLabel(`Generating ${req.num_takes} take${req.num_takes === 1 ? '' : 's'}…`), started: performance.now() };
  genError.value = null;
  player.stop();
  try {
    const res = await api.edit(req, ctrl.signal);
    if (req.verify) asrWarmed = true;
    const contexts = res.contexts ?? [];
    const ts = res.takes.map((t, i) => {
      const take = takeOf(t, `Take ${i + 1}`);
      take.direction = directionLabel(list, list.map((_, j) => j));
      take.expected = res.expected_seconds ?? [];
      take.contexts = contexts.map(([s, e]) => [mapBetween(s, res.spans, take.edited), mapBetween(e, res.spans, take.edited)]);
      return take;
    });
    const changes = list.map((r) => r.changes);
    const manual = list.map((r) => r.manual);
    // The text as written (groups in X-SAMPA), not the server's echo of what it was sent.
    const groups = pronEdit.value;
    const tgt = groups ? target : res.target_text || target;
    const facts: GenFacts = {
      spans: res.spans,
      changes,
      manual,
      target: tgt,
      targetTokens: normalizeTokens(tgt, groups),
      wants: regionTargets(v.words, tgt, list.map((r) => ({ start: seconds(r.start), end: seconds(r.end), changes: r.changes })), groups),
      groups,
    };
    setGen({ versionId: v.id, ...facts, contexts, takes: ts, selected: bestOf(ts, facts) });
    ab.value = 'take';
    cursor.value = 0;
  } catch (e) {
    if (!isAbort(e)) {
      genError.value = errorText(e);
      genRefused.value = e instanceof ApiError && e.status < 500;
    }
  } finally {
    busy.value = null;
    ctrl = null;
  }
}

export function cancel() {
  ctrl?.abort();
}

/** Another project's clip (or none): aborts what runs; the draft starts clean on its current version. */
export function loadDoc(next: Doc | null) {
  cancel();
  suggestedTranscript.value = '';
  gens.value = new Map();
  docs.reset(next);
}

/** Regenerate region k of the selected take (its other regions stay as they are). */
export async function retryRegion(k: number) {
  const g = gen.value;
  const t = selectedTake.value;
  const v = version.value;
  if (!g || !t || !v || busy.value) return;
  const r = regions.value[k];
  // The take says the target; its pronunciation groups, as heard, describe its audio.
  const req = requestFor(v, g.target, [{ start: t.edited[k][0], end: t.edited[k][1], values: r?.values ?? {} }], t.b64, spokenOf(g, t));
  ctrl = new AbortController();
  busy.value = { label: asrLabel(`Regenerating region ${k + 1}…`), started: performance.now() };
  genError.value = null;
  player.stop();
  try {
    const res = await api.edit(req, ctrl.signal);
    if (req.verify) asrWarmed = true;
    const fresh = res.takes.map((raw, i) => {
      const nt = takeOf(raw, `${t.label} · region ${k + 1} retry ${i + 1}`);
      // Back to the full set of regions: k is the new fill, later regions shift by its change in length.
      const [s, e] = raw.edited_regions[0];
      const delta = e - s - (t.edited[k][1] - t.edited[k][0]);
      nt.edited = t.edited.map((p, j) => (j < k ? p : j === k ? [s, e] : [p[0] + delta, p[1] + delta]));
      nt.fill = t.fill.map((f, j) => (j === k ? raw.fill_seconds[0] : f));
      nt.stopped = t.stopped.map((x, j) => (j === k ? raw.stopped[0] : x));
      nt.direction = directionLabel([{ values: r?.values ?? {} }], [k]) || `r${k + 1} undirected`;
      nt.expected = t.edited.map((_, j) => (j === k ? (res.expected_seconds?.[0] ?? null) : (t.expected[j] ?? null)));
      // Contexts come in the parent take's time.
      nt.contexts = (res.contexts ?? []).map(([cs, ce]) => [mapBetween(cs, t.edited, nt.edited), mapBetween(ce, t.edited, nt.edited)]);
      return nt;
    });
    const takes = [...g.takes, ...fresh];
    setGen({ ...g, takes, selected: bestOf(takes, g, g.takes.length) });
    ab.value = 'take';
  } catch (e) {
    if (!isAbort(e)) {
      genError.value = errorText(e);
      genRefused.value = e instanceof ApiError && e.status < 500;
    }
  } finally {
    busy.value = null;
    ctrl = null;
  }
}

export function selectTake(i: number) {
  const g = gen.value;
  if (!g || !g.takes[i]) return;
  const playing = player.nowPlaying.value;
  const at = playing ? player.playhead.value : cursor.value;
  const wasTake = ab.value === 'take';
  setGen({ ...g, selected: i });
  ab.value = 'take';
  const t = g.takes[i];
  // Same moment in the new take (from the version when we were on it).
  const from = wasTake ? (g.takes[g.selected]?.edited ?? g.spans) : g.spans;
  const mapped = mapBetween(at, from, t.edited);
  cursor.value = mapped;
  if (playing) player.play(`edit:take:${t.id}`, t.pcm, t.sr, mapped);
}

function mapBetween(t: number, from: [number, number][], to: [number, number][]) {
  let shift = 0;
  for (let k = 0; k < from.length && k < to.length; k++) {
    const [s, e] = from[k];
    const [es, ee] = to[k];
    if (t < s) return Math.max(0, t + shift);
    if (t < e) return es + ((t - s) / (e - s || 1)) * (ee - es);
    shift = ee - e;
  }
  return Math.max(0, t + shift);
}

/** A/B: the version vs the selected take, keeping the playhead on the same moment. */
export function toggleAB(to?: 'version' | 'take') {
  const t = selectedTake.value;
  const g = gen.value;
  const next = to ?? (ab.value === 'version' ? 'take' : 'version');
  if (next === ab.value) return;
  if (next === 'take' && (!t || !g)) return notify('Generate takes first, then compare.');
  const playing = player.nowPlaying.value;
  const at = playing ? player.playhead.value : cursor.value;
  const mapped = t && g ? (next === 'take' ? mapBetween(at, g.spans, t.edited) : mapBetween(at, t.edited, g.spans)) : at;
  ab.value = next;
  cursor.value = mapped;
  const d = displayed.value;
  if (playing && d) player.play(d.id, d.pcm, d.sr, mapped);
}

export function togglePlay() {
  const d = displayed.value;
  if (!d) return;
  if (player.nowPlaying.value) {
    if (player.nowPlaying.value.id === d.id) cursor.value = player.playhead.value;
    return player.stop();
  }
  player.play(d.id, d.pcm, d.sr, cursor.value >= d.duration - 0.05 ? 0 : cursor.value);
}

/** The neighbourhood of region k (pre-roll 1 s, post-roll 0.5 s) in the displayed audio. */
export function playRegion(k: number) {
  const d = displayed.value;
  const t = selectedTake.value;
  const region = regions.value[k];
  if (!d) return;
  const r = d.kind === 'take' && t ? t.edited[k] : region ? [seconds(region.start), seconds(region.end)] : null;
  if (!r) return;
  player.play(d.id, d.pcm, d.sr, Math.max(0, r[0] - 1), undefined, Math.min(d.duration, r[1] + 0.5));
}

/** The take's words in plain text: each pronunciation group as the recognizer heard it. */
function spokenOf(g: GenFacts, t: Take): string {
  return g.groups ? spokenText(g.target, (t.verify?.groups ?? []).map((x) => x.heard)) : g.target;
}

export async function keep() {
  const g = gen.value;
  const t = selectedTake.value;
  const d = doc.value;
  const v = version.value;
  if (!g || !t || !d || !v || busy.value) return;
  const later = d.versions.length - 1 - d.current;
  if (later > 0) {
    const ok = await confirm(later === 1 ? `Keeping replaces v${d.current + 1}. Continue?` : `Keeping replaces v${d.current + 1}–v${d.versions.length - 1}. Continue?`, { ok: 'Keep', danger: true });
    if (!ok) return;
  }
  ctrl = new AbortController();
  busy.value = { label: 'Keeping the take…', started: performance.now() };
  try {
    // A transcript describes the audio: pronunciation groups become the words heard there, aligned afresh.
    const spoken = spokenOf(g, t);
    const words = spoken === g.target && t.words ? t.words : (await api.align(t.b64, spoken, d.language, ctrl.signal)).words;
    const changes = g.changes.flat();
    const label = summarizeChanges(changes, v.words, g.target, g.groups) || (g.manual.some(Boolean) ? 'regenerated' : 'edit');
    const nv: Version = { id: uid(), label, pcm: t.pcm, sr: t.sr, b64: t.b64, duration: t.duration, transcript: spoken, words, alignedWords: words, automatic: false, ...(spoken !== g.target ? { display: g.target } : {}) };
    player.stop();
    // One `apply`: Undo of the keep is one `docs.undo()` (the previous version becomes current again, the new one is dropped).
    docs.apply({ ...d, versions: [...d.versions.slice(0, d.current + 1), nv], current: d.current + 1 });
    const heard = (t.verify?.groups ?? []).map((x) => x.heard ?? '?');
    notify(spoken !== g.target ? `Kept ${t.label.toLowerCase()} as v${d.current + 1}; its transcript spells the pronunciations as heard (${heard.join(', ') || 'guessed'})` : `Kept ${t.label.toLowerCase()} as v${d.current + 1}`, 'ok', {
      label: 'Undo',
      run: () => {
        player.stop();
        docs.undo();
      },
    });
  } catch (e) {
    if (isAbort(e)) notify('Cancelled');
    else notify(`Couldn’t keep the take: ${errorText(e)}`, 'error');
  } finally {
    busy.value = null;
    ctrl = null;
  }
}

/** Leaving the version loses typed text and regions: ask first. */
async function leaveVersionOk(): Promise<boolean> {
  return !draftDirty.value || confirm('Changing the version discards your unsaved text and regions. Continue?', { ok: 'Continue', danger: true });
}

export async function jumpTo(index: number) {
  const d = doc.value;
  if (!d || index === d.current || !d.versions[index]) return;
  if (!(await leaveVersionOk())) return;
  player.stop();
  docs.apply({ ...d, current: index });
}

/** ⌘Z on Edit: a step back through the document history, asking first when the draft would be lost. */
export async function undoVersion() {
  if (!docs.canUndo.value || !(await leaveVersionOk())) return;
  player.stop();
  docs.undo();
}

/** ⇧⌘Z on Edit. */
export async function redoVersion() {
  if (!docs.canRedo.value || !(await leaveVersionOk())) return;
  player.stop();
  docs.redo();
}

// ── Fixing the transcript ─────────────────────────────────────────────────────────────────────────

/** The edited words as they were before Fix the transcript opened: restored on Cancel and after a re-align. */
let draftBeforeFix: string | null = null;

export async function setFix(on: boolean) {
  if (busy.value || (!on && needsTranscript.value)) return;
  if (on) {
    draftBeforeFix = text.value;
    fixMode.value = true;
    return;
  }
  const v = version.value;
  const edited = draftBeforeFix !== null && draftBeforeFix !== (v?.transcript ?? '');
  if (edited && !(await confirm('Leave the transcript fix? Your edited words are kept.', { ok: 'Leave' }))) return;
  fixMode.value = false;
  text.value = edited ? draftBeforeFix! : (v?.transcript ?? '');
  draftBeforeFix = null;
  replan();
}

/** The reader checked an automatic transcript and found it right: it is no longer flagged. */
export function confirmTranscript() {
  const d = doc.value;
  const v = version.value;
  if (!d || !v || !v.automatic) return;
  docs.apply({ ...d, versions: d.versions.map((x, i) => (i === d.current ? { ...x, automatic: false } : x)) });
}

/**
 * The clip's language: its words came from the previous language's model, so a transcribed version is transcribed
 * again (when the server can) and a known text aligned again (undoable). A version without a transcript only
 * changes language. An unsupported language keeps the old one (the server's 400).
 */
export async function setLanguage(language: string) {
  const d = doc.value;
  const v = version.value;
  if (!d || !v || busy.value || language === d.language) return;
  const key = clipKey(d.source, d.versions[0].pcm.length / d.versions[0].sr);
  if (!v.transcript) {
    docs.apply({ ...d, language });
    clipLanguages.value = { ...clipLanguages.value, [key]: language };
    return;
  }
  if (!(await leaveVersionOk())) return;
  const retranscribe = v.automatic && canTranscribe.value;
  ctrl = new AbortController();
  busy.value = { label: retranscribe ? asrLabel('Transcribing…') : 'Aligning the transcript…', started: performance.now() };
  try {
    const a = retranscribe ? await api.transcribe(v.b64, language, ctrl.signal) : await api.align(v.b64, v.transcript, language, ctrl.signal);
    if (retranscribe) asrWarmed = true;
    const nv: Version = { ...v, id: uid(), transcript: a.text, words: a.words, alignedWords: a.words };
    docs.apply({ ...d, language, versions: d.versions.map((x, i) => (i === d.current ? nv : x)) });
    clipLanguages.value = { ...clipLanguages.value, [key]: language };
    notify(`${retranscribe ? 'Transcribed' : 'Aligned'} in ${langLabel(language)}`, 'ok');
  } catch (e) {
    if (isAbort(e)) notify('Cancelled');
    else notify(`Couldn’t switch to ${langLabel(language)}: ${errorText(e)}`, 'error');
  } finally {
    busy.value = null;
    ctrl = null;
  }
}

/** Align the corrected (or first typed) transcript: replaces the current version's words (undoable). The edited words come back on top. */
export async function saveFix(corrected: string) {
  const d = doc.value;
  const v = version.value;
  if (!d || !v || !corrected.trim()) return;
  const first = !v.transcript;
  const edited = draftBeforeFix !== null && draftBeforeFix !== (v.transcript ?? '') ? draftBeforeFix : null;
  ctrl = new AbortController();
  busy.value = { label: first ? 'Aligning the transcript…' : 'Re-aligning…', started: performance.now() };
  try {
    const a = await api.align(v.b64, corrected.trim(), d.language, ctrl.signal);
    const nv: Version = { ...v, id: uid(), transcript: a.text, words: a.words, alignedWords: a.words, automatic: false };
    // The version effect resets the draft to the new transcript; the edited words go back on top of it after.
    docs.apply({ ...d, versions: d.versions.map((x, i) => (i === d.current ? nv : x)) });
    draftBeforeFix = null;
    if (edited) {
      text.value = edited;
      replan();
    }
    notify(first ? 'Transcript aligned: edit the words to change the speech' : 'Transcript corrected and re-aligned', 'ok');
  } catch (e) {
    if (isAbort(e)) notify('Cancelled');
    else notify(`Couldn’t ${first ? 'align' : 're-align'}: ${errorText(e)}`, 'error');
  } finally {
    busy.value = null;
    ctrl = null;
  }
}

// ── Sending back ──────────────────────────────────────────────────────────────────────────────────

/** The Direct take this clip came from, when it still exists. */
export const sourceTake = computed(() => {
  const s = doc.value?.source;
  if (!s || s.kind !== 'take' || !s.lineId || !s.takeId) return null;
  const line = script.value.lines.find((l) => l.id === s.lineId);
  const take = scriptTakes.value[s.lineId]?.find((t) => t.id === s.takeId);
  return line && take ? { line, take } : null;
});

export const canSendBack = computed(() => {
  const d = doc.value;
  return !!d && d.current > 0 && d.source.kind === 'take' && !!sourceTake.value;
});

/** Put the current version where the clip came from: a new keeper take on the Direct line (the original stays). */
export async function sendBack() {
  const d = doc.value;
  const v = version.value;
  const src = sourceTake.value;
  if (!d || !v || !src || !canSendBack.value) return;
  const { line } = src;
  const audio = { pcm: v.pcm, sr: v.sr, ttfbMs: 0, totalMs: 0 };
  // The line keeps its pronunciation groups (the transcript spells them as heard).
  const said = v.display ?? v.transcript;
  const n = addTake(line.id, { id: uid(), text: said, seed: 0, request: lineRequest(line, 0), audio, createdAt: Date.now(), starred: true, edited: true });
  notify(`Added to Direct as take ${n}`, 'ok', { label: 'Go to Direct', run: () => (act.value = 'direct') });
  if (line.text !== said && (await confirm(`Also change the line's text to “${said}”?`, { ok: 'Change the text' }))) setLineText(line.id, said);
}

// ── Recording ─────────────────────────────────────────────────────────────────────────────────────

export const recording = signal<{ started: number; stop: () => void } | null>(null);
/** Microphone level while recording (0…1). */
export const recordLevel = signal(0);

export async function recordClip() {
  if (recording.value) return recording.value.stop();
  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    return notify(`Microphone unavailable: ${errorText(e)}`, 'error');
  }
  const meter = levelMeter(stream, (v) => (recordLevel.value = v));
  const rec = new MediaRecorder(stream);
  const parts: Blob[] = [];
  rec.ondataavailable = (ev) => parts.push(ev.data);
  rec.onstop = async () => {
    meter.stop();
    stream.getTracks().forEach((t) => t.stop());
    recording.value = null;
    try {
      const { pcm, sr } = await player.decode(await new Blob(parts, { type: rec.mimeType }).arrayBuffer());
      const when = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      await openClip({ kind: 'recording', label: `Recording ${when}` }, pcm, sr);
    } catch (e) {
      notify(`Couldn’t read the recording: ${errorText(e)}`, 'error');
    }
  };
  rec.start();
  const max = Math.min(120, maxSeconds.value);
  const auto = setTimeout(() => rec.state === 'recording' && rec.stop(), max * 1000);
  recording.value = {
    started: Date.now(),
    stop: () => {
      clearTimeout(auto);
      if (rec.state === 'recording') rec.stop();
    },
  };
}

export async function openFile(f: File) {
  try {
    const { pcm, sr } = await player.decode(await f.arrayBuffer());
    await openClip({ kind: 'upload', label: f.name.replace(/\.[^.]+$/, '') }, pcm, sr);
  } catch (e) {
    notify(`Couldn’t read ${f.name}: ${errorText(e)}`, 'error');
  }
}
