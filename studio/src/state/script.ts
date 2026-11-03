/** Direct: a script split into lines, delivery painted onto character spans, takes per line. */
import { computed, signal } from '@preact/signals';
import { parseTags } from '../lib/tags';
import type { GenerateRequest, Rendered } from '../api';
import { errorText, isAbort, render } from '../audio/render';
import { uid } from '../lib/math';
import { projectSettings, pronTokens, requestContext, spec } from './app';
import { pronProblem } from '../lib/pron';
import { design, type Design } from './design';
import { durationFactor, type Values } from './engine';
import { History } from './history';
import { cast } from './library';
import { buildRequest, PREVIEW_SEED, remapSpans, type Span } from './requests';

/** Codec frame rate (frames per second) used to measure a take's frames per character. */
const FPS = 15;

export interface Line {
  id: string;
  text: string;
  spans: Span[];
}

export interface Script {
  lines: Line[];
  /** Whole-script performance: UI values added to every line. */
  baseline: Values;
  /** Persona id, or null for the voice currently on the Cast bench. */
  voice: string | null;
}

export interface Take {
  id: string;
  text: string;
  seed: number;
  request: GenerateRequest;
  audio: Rendered;
  createdAt: number;
  starred: boolean;
  /** Replaced by an edit from the Edit screen (its request no longer describes the audio). */
  edited?: boolean;
}

const DEMO = [
  'Welcome back to the late show.',
  "Tonight's story starts somewhere quiet, a little after midnight.",
  "Honestly? I wasn't sure we'd ever get here.",
  "But here we are, and I couldn't be happier about it.",
];

export function splitScript(text: string): string[] {
  return text
    .split(/\n+/)
    .flatMap((para) => para.split(/(?<=[.!?…])\s+(?=\S)/))
    .map((s) => s.trim())
    .filter(Boolean);
}

const newLine = (text: string): Line => ({ id: uid(), text, spans: [] });

/** A new project's script: the demo lines. */
export const newScript = (): Script => ({ lines: DEMO.map(newLine), baseline: {}, voice: null });

export const scripts = new History<Script>(newScript());
export const script = scripts.state;

/** Takes per line id (saved with the project; the audio as media). */
export const takes = signal<Record<string, Take[]>>({});
export const lineBusy = signal<Record<string, boolean>>({});
export const lineError = signal<Record<string, string | null>>({});

export const voiceDesign = computed<Design>(() => {
  const id = script.value.voice;
  return (id && cast.value.find((p) => p.id === id)?.design) || design.value;
});

const update = (fn: (s: Script) => Script) => scripts.set(fn(script.value));
const commit = (fn: (s: Script) => Script) => scripts.apply(fn(script.value));

// ── Text editing ──────────────────────────────────────────────────────────────────────────────────

export function setLineText(id: string, text: string) {
  commit((s) => ({
    ...s,
    lines: s.lines.map((l) => (l.id === id ? { ...l, text, spans: remapSpans(l.spans, l.text, text) } : l)),
  }));
}

export function replaceScript(text: string) {
  const old = new Map(script.value.lines.map((l) => [l.text, l]));
  commit((s) => ({ ...s, lines: splitScript(text).map((t) => old.get(t) ?? newLine(t)) }));
}

export function addLine(afterId: string | null, text = '') {
  commit((s) => {
    const lines = [...s.lines];
    const i = afterId ? lines.findIndex((l) => l.id === afterId) + 1 : lines.length;
    lines.splice(i, 0, newLine(text));
    return { ...s, lines };
  });
}

export function removeLine(id: string) {
  commit((s) => ({ ...s, lines: s.lines.filter((l) => l.id !== id) }));
}

export function moveLine(id: string, delta: number) {
  commit((s) => {
    const lines = [...s.lines];
    const i = lines.findIndex((l) => l.id === id);
    const j = i + delta;
    if (i < 0 || j < 0 || j >= lines.length) return s;
    [lines[i], lines[j]] = [lines[j], lines[i]];
    return { ...s, lines };
  });
}

export function setVoice(voice: string | null) {
  commit((s) => ({ ...s, voice }));
}

export function setBaseline(id: string, v: number) {
  update((s) => ({ ...s, baseline: { ...s.baseline, [id]: Math.round(v * 100) / 100 } }));
}

// ── Painting ──────────────────────────────────────────────────────────────────────────────────────

function sameValues(a: Values, b: Values) {
  const ka = Object.keys(a);
  return ka.length === Object.keys(b).length && ka.every((k) => a[k] === b[k]);
}

function tidy(spans: Span[]): Span[] {
  const clean = spans
    .map((sp) => ({ ...sp, values: Object.fromEntries(Object.entries(sp.values).filter(([, v]) => v)) }))
    .filter((sp) => sp.end > sp.start && Object.keys(sp.values).length)
    .sort((a, b) => a.start - b.start);
  const out: Span[] = [];
  for (const sp of clean) {
    const last = out[out.length - 1];
    if (last && last.end === sp.start && sameValues(last.values, sp.values)) out[out.length - 1] = { ...last, end: sp.end };
    else out.push(sp);
  }
  return out;
}

/**
 * Set one control to `value` over [start, end): existing spans are split at the edges and keep their other
 * controls; uncovered gaps get new spans. Spans never overlap (the server refuses overlapping spans).
 * `value` null clears every control in the range.
 */
export function paintSpans(spans: Span[], start: number, end: number, control: string | null, value: number): Span[] {
  const out: Span[] = [];
  const inside: Span[] = [];
  for (const sp of spans) {
    if (sp.end <= start || sp.start >= end) {
      out.push(sp);
      continue;
    }
    if (sp.start < start) out.push({ ...sp, id: uid(), end: start });
    if (sp.end > end) out.push({ ...sp, id: uid(), start: end });
    const piece = { ...sp, start: Math.max(sp.start, start), end: Math.min(sp.end, end) };
    inside.push(piece);
  }
  inside.sort((a, b) => a.start - b.start);
  let cursor = start;
  const filled: Span[] = [];
  for (const sp of inside) {
    if (sp.start > cursor) filled.push({ id: uid(), start: cursor, end: sp.start, values: {} });
    filled.push(sp);
    cursor = sp.end;
  }
  if (cursor < end) filled.push({ id: uid(), start: cursor, end, values: {} });
  for (const sp of filled) out.push({ ...sp, values: control ? { ...sp.values, [control]: value } : {} });
  return tidy(out);
}

export function paint(lineId: string, start: number, end: number, control: string | null, value: number) {
  update((s) => ({
    ...s,
    lines: s.lines.map((l) => (l.id === lineId ? { ...l, spans: paintSpans(l.spans, start, end, control, value) } : l)),
  }));
}

/** Values in force at a character (the span covering it). */
export function valuesAt(line: Line, at: number): Values {
  return line.spans.find((sp) => at >= sp.start && at < sp.end)?.values ?? {};
}

// ── Takes ─────────────────────────────────────────────────────────────────────────────────────────

/** Frames per character measured from a take, with the server's timing model divided out. */
function measuredRate(t: Take): number | undefined {
  const s = spec.value;
  // Spoken characters of the text the server placed spans on (pronunciation groups count as sent; nonverbal
  // markers take no time, so the server's rate is per spoken character).
  const n = t.request.text.length;
  if (!n) return undefined;
  const frames = (t.audio.pcm.length / t.audio.sr) * FPS;
  const global = durationFactor(s, t.request.controls ?? {});
  const markers = parseTags(t.request.text).tokens;
  let chars = 0;
  for (let i = 0; i < n; i++) {
    if (markers.some((m) => i >= m.start && i < m.end)) continue;
    const sp = t.request.control_spans?.find((x) => i >= x.start && i < x.end);
    chars += sp ? durationFactor(s, sp.controls) : 1;
  }
  if (!chars) return undefined;
  return frames / (global * chars);
}

export function lineRequest(line: Line, seed: number): GenerateRequest {
  const prior = (takes.value[line.id] ?? []).filter((t) => t.text === line.text);
  const last = prior[prior.length - 1];
  const language = projectSettings.value.language;
  // The measured rate places spans only by estimate: not sent once the server followed the reading for this line in
  // this language (it has alignment heads for it), sent as the fallback otherwise.
  const reading = last?.audio.spanFollow === 'reading' && (last.request.language ?? 'en_us') === language;
  const spanRate = last && !reading ? measuredRate(last) : undefined;
  return buildRequest(spec.value, requestContext.value, voiceDesign.value, line.text, seed, { extra: script.value.baseline, spans: line.spans, spanRate, language });
}

export async function renderLine(lineId: string) {
  const line = script.value.lines.find((l) => l.id === lineId);
  if (!line || !line.text.trim()) return;
  const problem = pronProblem(line.text, pronTokens.value);
  if (problem) {
    lineError.value = { ...lineError.value, [lineId]: problem };
    return;
  }
  const existing = takes.value[lineId] ?? [];
  const seed = existing.length ? Math.floor(Math.random() * 1e6) : PREVIEW_SEED;
  lineBusy.value = { ...lineBusy.value, [lineId]: true };
  lineError.value = { ...lineError.value, [lineId]: null };
  try {
    const request = lineRequest(line, seed);
    const audio = await render(request);
    const take: Take = { id: uid(), text: line.text, seed, request, audio, createdAt: Date.now(), starred: false };
    const list = [...(takes.value[lineId] ?? []), take];
    if (!list.some((t) => t.starred)) take.starred = true;
    takes.value = { ...takes.value, [lineId]: list };
  } catch (e) {
    if (!isAbort(e)) lineError.value = { ...lineError.value, [lineId]: errorText(e) };
  } finally {
    lineBusy.value = { ...lineBusy.value, [lineId]: false };
  }
}

export function starTake(lineId: string, takeId: string) {
  takes.value = { ...takes.value, [lineId]: (takes.value[lineId] ?? []).map((t) => ({ ...t, starred: t.id === takeId })) };
}

export function deleteTake(lineId: string, takeId: string) {
  const list = (takes.value[lineId] ?? []).filter((t) => t.id !== takeId);
  if (list.length && !list.some((t) => t.starred)) list[list.length - 1] = { ...list[list.length - 1], starred: true };
  takes.value = { ...takes.value, [lineId]: list };
}

/** The keeper of each line in script order: the starred take (else the latest), or null. */
export const keepers = computed(() =>
  script.value.lines.map((l) => {
    const list = takes.value[l.id] ?? [];
    return { line: l, take: list.find((t) => t.starred) ?? list[list.length - 1] ?? null };
  }),
);

/** Does a take still match its line (text, painting, baseline, voice)? */
export function takeIsStale(line: Line, t: Take): boolean {
  if (t.edited) return t.text !== line.text;
  const req = lineRequest(line, t.seed);
  const strip = (r: GenerateRequest) => JSON.stringify({ ...r, control_span_rate: undefined, control_span_follow: undefined });
  return strip(req) !== strip(t.request);
}

