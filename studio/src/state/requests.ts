/** Design (+ script extras) → POST /v2/generate body, per the contract. */
import type { ControlSpan, ControlsSpec, Dose, GenerateRequest } from '../api';
import { laciFields, type LaciMode } from '../lib/laci';
import { clamp } from '../lib/math';
import { toSent, type PronTokens, type Sent } from '../lib/pron';
import { mixAt } from './atlas';
import type { Design } from './design';
import { controlDose, controlUsable, toDoses, type Values } from './engine';

export const PREVIEW_SEED = 4321;

/** A painted range of a line: UI values of spannable controls over [start, end). */
export interface Span {
  id: string;
  start: number;
  end: number;
  values: Values;
}

export interface RequestExtras {
  /** Added to the design's own values (clamped to range), e.g. a script's baseline performance. */
  extra?: Values;
  spans?: Span[];
  /** Frames per character for placing spans by estimate (the server's fallback without reading heads). */
  spanRate?: number;
  /** Server language code; `en_us` (the server default) is left out. */
  language?: string;
  /** The A/B reference: the same voice with no controls at all. */
  unsteered?: boolean;
  /** Speech from the first frame (`start_immediately`); sent only when the checkpoint has a start token. */
  startImmediately?: boolean;
}

/** What a request needs from the backend and the project besides the design (Studio: `requestContext`). */
export interface RequestContext {
  phonemeTokens: PronTokens | null;
  /** The `decoder` to send (`requestDecoder`): null = the server default, left out of the request. */
  decoder: string | null;
  /** The project's LACI mode (`laciFields`); undefined = off. */
  laci: LaciMode | undefined;
  /** Languages with LACI alignment heads on the server. */
  laciLanguages: readonly string[];
  /** The checkpoint has a start token (`/v2/capabilities.start_token`). */
  startToken: boolean;
}

/** Keep spans attached to their words across an edit (common prefix / suffix diff). */
export function remapSpans(spans: Span[], before: string, after: string): Span[] {
  let p = 0;
  while (p < before.length && p < after.length && before[p] === after[p]) p++;
  let s = 0;
  while (s < before.length - p && s < after.length - p && before[before.length - 1 - s] === after[after.length - 1 - s]) s++;
  const oldEnd = before.length - s;
  const newEnd = after.length - s;
  const map = (pos: number, isEnd: boolean) => {
    if (pos <= p) return pos;
    if (pos >= oldEnd) return pos + (after.length - before.length);
    return isEnd ? newEnd : p;
  };
  return spans.map((sp) => ({ ...sp, start: map(sp.start, false), end: map(sp.end, true) })).filter((sp) => sp.end > sp.start);
}

export function mergeValues(s: ControlsSpec, base: Values, extra: Values | undefined): Values {
  if (!extra) return base;
  const out = { ...base };
  for (const c of s.ui.controls) {
    const add = extra[c.id];
    if (add) out[c.id] = clamp((out[c.id] ?? 0) + add, c.range[0], c.range[1]);
  }
  return out;
}

/** `[start, end)` covers every word of a text of two or more words: no place for a span-only control. */
export function coversLine(text: string, start: number, end: number): boolean {
  return !/\S/.test(text.slice(0, start)) && !/\S/.test(text.slice(end)) && /\S\s+\S/.test(text);
}

export function spanDoses(s: ControlsSpec, span: Span, d: Design, wholeLine = false): Dose {
  const out: Dose = {};
  for (const c of s.ui.controls) {
    const v = span.values[c.id];
    if (v && c.spannable && !(wholeLine && c.span_only) && controlUsable(s, c, d.mode)) controlDose(c, v, out);
  }
  return out;
}

/**
 * Painted spans → `control_spans` on the SENT text: `map` turns display offsets into sent offsets (a boundary
 * inside a pronunciation group takes in the whole group); a span starting inside the group the previous one took
 * starts after it. Span-only controls painted over the whole line are dropped.
 */
export function controlSpans(s: ControlsSpec, spans: Span[] | undefined, d: Design, text: string, map: Sent['map'] = (at) => at): ControlSpan[] {
  const out: ControlSpan[] = [];
  for (const sp of [...(spans ?? [])].sort((a, b) => a.start - b.start)) {
    const controls = spanDoses(s, sp, d, coversLine(text, sp.start, sp.end));
    let start = map(clamp(sp.start, 0, text.length), -1);
    const end = map(clamp(sp.end, 0, text.length), 1);
    const prev = out[out.length - 1];
    if (prev && start < prev.end) start = prev.end;
    if (end > start && Object.keys(controls).length) out.push({ start, end, controls });
  }
  return out;
}

export function buildRequest(s: ControlsSpec, ctx: RequestContext, d: Design, text: string, seed: number, x: RequestExtras = {}): GenerateRequest {
  // Pronunciation groups go out as U+E000 IPA U+E001 on phoneme-token checkpoints; offsets follow the sent text.
  const sent = toSent(text, ctx.phonemeTokens);
  const req: GenerateRequest = {
    ...s.ui.request_defaults,
    text: sent.text,
    seed,
    speaker_embedding_id: d.baseId,
    speaker_mode: d.mode,
  };
  if (x.language && x.language !== 'en_us') req.language = x.language;
  // The A/B reference too: the two sides differ only in their controls.
  if (ctx.decoder) req.decoder = ctx.decoder;
  const laci = laciFields(ctx.laci, ctx.laciLanguages, x.language ?? 'en_us');
  Object.assign(req, laci);
  // Both A/B sides: the start is not a control.
  if (x.startImmediately && ctx.startToken) req.start_immediately = true;
  if (x.unsteered) return req;
  const controls = toDoses(s, mergeValues(s, d.values, x.extra), mixAt(d.pin, s.ui.accents), d.mode);
  if (Object.keys(controls).length) req.controls = controls;
  const spans = controlSpans(s, x.spans, d, text, sent.map);
  if (spans.length) {
    req.control_spans = spans;
    // Spans follow the model's reading position; the server falls back to the rate where it has no heads.
    req.control_span_follow = 'reading';
    if (x.spanRate) req.control_span_rate = Math.round(x.spanRate * 1000) / 1000;
  }
  return req;
}
