/** Direct: a script split into lines, each with its own voice, delivery painted onto character spans, takes per line. */
import { computed, signal } from '@preact/signals';
import { parseTags } from '../lib/tags';
import type { GenerateRequest, Rendered } from '../api';
import { sequence } from '../audio/mix';
import { errorText, isAbort, render } from '../audio/render';
import { uid } from '../lib/math';
import { projectSettings, pronTokens, requestContext, spec } from './app';
import { pronProblem } from '../lib/pron';
import { design, type Design } from './design';
import { durationFactor, type Values } from './engine';
import { History } from './history';
import { cast } from './library';
import { matchLines, newLine, staleReason, type Line } from './lines';
import { buildRequest, PREVIEW_SEED, remapSpans, type Span } from './requests';

export { matchLines, splitScript, staleReason, type Line } from './lines';

/** Codec frame rate (frames per second) used to measure a take's frames per character. */
const FPS = 15;

export interface Script {
  lines: Line[];
  /** Whole-script performance: UI values added to every line. */
  baseline: Values;
  /** Default voice: persona id, or null for the voice currently on the Cast bench. */
  voice: string | null;
  /** Silence between lines when the script plays / exports (s). */
  gap: number;
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

/** A new project's script: the demo lines. */
export const newScript = (): Script => ({ lines: DEMO.map((t) => newLine(t)), baseline: {}, voice: null, gap: 0.35 });

export const scripts = new History<Script>(newScript());
export const script = scripts.state;

/** Takes per line id (saved with the project; the audio as media). */
export const takes = signal<Record<string, Take[]>>({});
export const lineBusy = signal<Record<string, boolean>>({});
export const lineError = signal<Record<string, string | null>>({});

/** The script's default voice. */
export const voiceDesign = computed<Design>(() => {
  const id = script.value.voice;
  return (id && cast.value.find((p) => p.id === id)?.design) || design.value;
});

/** A line's persona id: its own, else the script's (null: the Cast bench). */
export function lineVoiceId(line: Line): string | null {
  return line.voice ?? script.value.voice;
}

/** The design a line renders with (the bench when its persona is gone). */
export function lineDesign(line: Line): Design {
  const id = lineVoiceId(line);
  return (id && cast.value.find((p) => p.id === id)?.design) || design.value;
}

const update = (fn: (s: Script) => Script) => scripts.set(fn(script.value));
const commit = (fn: (s: Script) => Script) => scripts.apply(fn(script.value));

// ── Text editing ──────────────────────────────────────────────────────────────────────────────────

export function setLineText(id: string, text: string) {
  commit((s) => ({
    ...s,
    lines: s.lines.map((l) => (l.id === id ? { ...l, text, spans: remapSpans(l.spans, l.text, text) } : l)),
  }));
  if (lineError.value[id]) lineError.value = { ...lineError.value, [id]: null };
}

/** Replace the lines with `rows` (see `matchLines`): unchanged and edited-in-place rows keep their takes. */
export function replaceScript(rows: string[]) {
  const { lines } = matchLines(script.value.lines, rows);
  commit((s) => ({ ...s, lines }));
}

/** Every line that speaks as `voice` goes back to the script voice (one undo step); returns their ids. */
export function clearVoice(voice: string): string[] {
  const ids = script.value.lines.filter((l) => l.voice === voice).map((l) => l.id);
  commit((s) => ({
    ...s,
    voice: s.voice === voice ? null : s.voice,
    lines: s.lines.map((l) => (l.voice === voice ? { ...l, voice: null } : l)),
  }));
  return ids;
}

/** Give lines (by id) a voice again (Undo of `clearVoice`). */
export function setLinesVoice(ids: string[], voice: string | null, scriptVoice?: string | null) {
  const set = new Set(ids);
  commit((s) => ({
    ...s,
    voice: scriptVoice === undefined ? s.voice : scriptVoice,
    lines: s.lines.map((l) => (set.has(l.id) ? { ...l, voice } : l)),
  }));
}

/** Insert a line after `afterId` (null: at the end); returns its id. */
export function addLine(afterId: string | null, text = '', voice?: string | null): string {
  const line = newLine(text, voice);
  commit((s) => {
    const lines = [...s.lines];
    const i = afterId ? lines.findIndex((l) => l.id === afterId) + 1 : lines.length;
    lines.splice(i, 0, line);
    return { ...s, lines };
  });
  return line.id;
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

/** One line's voice; null follows the script's voice. */
export function setLineVoice(id: string, voice: string | null) {
  commit((s) => ({ ...s, lines: s.lines.map((l) => (l.id === id ? { ...l, voice } : l)) }));
}

export function setGap(gap: number) {
  commit((s) => ({ ...s, gap: Math.max(0, Math.round(gap * 100) / 100) }));
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
  return buildRequest(spec.value, requestContext.value, lineDesign(line), line.text, seed, { extra: script.value.baseline, spans: line.spans, spanRate, language });
}

/** Renders in flight by line id (a line's busy button cancels its own). */
const lineCtrl = new Map<string, AbortController>();

export function cancelLine(lineId: string) {
  lineCtrl.get(lineId)?.abort();
}

export async function renderLine(lineId: string, signal?: AbortSignal): Promise<boolean> {
  const line = script.value.lines.find((l) => l.id === lineId);
  if (!line || !line.text.trim()) return false;
  const problem = pronProblem(line.text, pronTokens.value);
  if (problem) {
    lineError.value = { ...lineError.value, [lineId]: problem };
    return false;
  }
  lineCtrl.get(lineId)?.abort();
  const ctrl = new AbortController();
  lineCtrl.set(lineId, ctrl);
  signal?.addEventListener('abort', () => ctrl.abort(), { once: true });
  if (signal?.aborted) ctrl.abort();
  const existing = takes.value[lineId] ?? [];
  const seed = existing.length ? Math.floor(Math.random() * 1e6) : PREVIEW_SEED;
  lineBusy.value = { ...lineBusy.value, [lineId]: true };
  lineError.value = { ...lineError.value, [lineId]: null };
  try {
    const request = lineRequest(line, seed);
    const audio = await render(request, { signal: ctrl.signal });
    const prior = takes.value[lineId] ?? [];
    const keeper = prior.find((t) => t.starred) ?? prior[prior.length - 1];
    // The new take becomes the keeper when there is none or the keeper no longer matches the line (Regenerate).
    const starred = !keeper || takeIsStale(line, keeper);
    const take: Take = { id: uid(), text: line.text, seed, request, audio, createdAt: Date.now(), starred };
    const list = [...prior.map((t) => (starred && t.starred ? { ...t, starred: false } : t)), take];
    takes.value = { ...takes.value, [lineId]: list };
    return true;
  } catch (e) {
    if (!isAbort(e)) lineError.value = { ...lineError.value, [lineId]: errorText(e) };
    return false;
  } finally {
    if (lineCtrl.get(lineId) === ctrl) lineCtrl.delete(lineId);
    lineBusy.value = { ...lineBusy.value, [lineId]: false };
  }
}

/** A batch of renders in progress (Render missing / Regenerate stale): counts for the rail, one Cancel for all. */
export const renderQueue = signal<{ total: number; done: number; failed: number } | null>(null);
let batch: AbortController | null = null;

export function cancelRenders() {
  batch?.abort();
}

/** Render `ids` with the shared concurrency limit; the queue counts finished and failed lines until all settle. */
export async function renderLines(ids: string[]): Promise<void> {
  if (!ids.length) return;
  batch?.abort();
  const ctrl = new AbortController();
  batch = ctrl;
  renderQueue.value = { total: ids.length, done: 0, failed: 0 };
  await Promise.all(
    ids.map(async (id) => {
      const ok = await renderLine(id, ctrl.signal);
      const q = renderQueue.value;
      if (!q || batch !== ctrl) return;
      renderQueue.value = { ...q, done: q.done + 1, failed: q.failed + (ok || ctrl.signal.aborted ? 0 : 1) };
    }),
  );
  if (batch === ctrl) {
    batch = null;
    renderQueue.value = null;
  }
}

export function starTake(lineId: string, takeId: string) {
  takes.value = { ...takes.value, [lineId]: (takes.value[lineId] ?? []).map((t) => ({ ...t, starred: t.id === takeId })) };
}

/** Remove a take; returns what `restoreTake` needs to put it back. */
export function deleteTake(lineId: string, takeId: string): { take: Take; index: number; wasStarred: boolean } | null {
  const all = takes.value[lineId] ?? [];
  const index = all.findIndex((t) => t.id === takeId);
  if (index < 0) return null;
  const take = all[index];
  const list = all.filter((t) => t.id !== takeId);
  if (list.length && !list.some((t) => t.starred)) list[list.length - 1] = { ...list[list.length - 1], starred: true };
  takes.value = { ...takes.value, [lineId]: list };
  return { take, index, wasStarred: take.starred };
}

/** Put a deleted take back at `index`; `star` makes it the keeper again. */
export function restoreTake(lineId: string, take: Take, index: number, star: boolean) {
  const rest = (takes.value[lineId] ?? []).filter((t) => t.id !== take.id).map((t) => (star ? { ...t, starred: false } : t));
  const at = Math.min(index, rest.length);
  takes.value = { ...takes.value, [lineId]: [...rest.slice(0, at), { ...take, starred: star }, ...rest.slice(at)] };
}

/** Add a take to a line (an edited keeper from Edit): it becomes the keeper. Returns its number (1-based). */
export function addTake(lineId: string, take: Take): number {
  const prior = (takes.value[lineId] ?? []).map((t) => ({ ...t, starred: false }));
  takes.value = { ...takes.value, [lineId]: [...prior, { ...take, starred: true }] };
  return prior.length + 1;
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
  return takeStaleReason(line, t) !== null;
}

/** Why a take no longer matches its line ("text changed", "voice changed", …), or null when it still does. */
export function takeStaleReason(line: Line, t: Take): string | null {
  if (t.edited) return t.text !== line.text ? 'text changed' : null;
  const req = lineRequest(line, t.seed);
  const strip = (r: GenerateRequest): GenerateRequest => ({ ...r, control_span_rate: undefined, control_span_follow: undefined });
  const stored = strip(t.request);
  const fresh = strip(req);
  return staleReason(stored, fresh) ?? (JSON.stringify(stored) !== JSON.stringify(fresh) ? 'settings changed' : null);
}

/** Lines whose keeper take no longer matches them (a changed voice, text, painting or baseline). */
export const staleLines = computed(() => keepers.value.flatMap(({ line, take }) => (take && takeIsStale(line, take) ? [line] : [])));

/** A new take of every stale line (render.ts limits how many run at once). */
export function regenerateStale(): Promise<void> {
  return renderLines(staleLines.value.map((l) => l.id));
}

// ── Timeline ──────────────────────────────────────────────────────────────────────────────────────

export interface Clip {
  line: Line;
  take: Take | null;
  start: number;
  duration: number;
}

/** The keepers laid end to end with the gap; lines without a take get an estimated slot (≈15 characters per second). */
export const timeline = computed(() => {
  const gap = script.value.gap;
  const clips: Clip[] = [];
  let end = 0;
  for (const [i, { line, take }] of keepers.value.entries()) {
    const duration = take ? take.audio.pcm.length / take.audio.sr : Math.max(0.8, line.text.length / 15);
    const start = i ? end + gap : 0;
    clips.push({ line, take, start, duration });
    end = start + duration;
  }
  return { clips, total: end };
});

/** The keepers' takes in script order with the gap between them; null when no line has a take. */
export function mixdown(): { pcm: Float32Array; sr: number } | null {
  const audio = keepers.value.flatMap(({ take }) => (take ? [take.audio] : []));
  return audio.length ? sequence(audio, script.value.gap) : null;
}

