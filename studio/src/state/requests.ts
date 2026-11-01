/** Design (+ script extras) → POST /v2/generate body, per the contract. */
import type { ControlSpan, ControlsSpec, Dose, GenerateRequest } from '../api';
import { clamp } from '../lib/math';
import { toSent, type Sent } from '../lib/pron';
import { hasTags } from '../lib/tags';
import { backend, projectSettings } from './app';
import type { Design } from './design';
import { accentsOf } from './design';
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

export function spanDoses(s: ControlsSpec, span: Span, d: Design): Dose {
  const out: Dose = {};
  for (const c of s.ui.controls) {
    const v = span.values[c.id];
    if (v && c.spannable && controlUsable(s, c, d.mode)) controlDose(c, v, out);
  }
  return out;
}

/**
 * Painted spans → `control_spans` on the SENT text: `map` turns display offsets into sent offsets (a boundary
 * inside a pronunciation group takes in the whole group); a span starting inside the group the previous one took
 * starts after it.
 */
export function controlSpans(s: ControlsSpec, spans: Span[] | undefined, d: Design, textLength: number, map: Sent['map'] = (at) => at): ControlSpan[] {
  const out: ControlSpan[] = [];
  for (const sp of [...(spans ?? [])].sort((a, b) => a.start - b.start)) {
    const controls = spanDoses(s, sp, d);
    let start = map(clamp(sp.start, 0, textLength), -1);
    const end = map(clamp(sp.end, 0, textLength), 1);
    const prev = out[out.length - 1];
    if (prev && start < prev.end) start = prev.end;
    if (end > start && Object.keys(controls).length) out.push({ start, end, controls });
  }
  return out;
}

export function buildRequest(s: ControlsSpec, d: Design, text: string, seed: number, x: RequestExtras = {}): GenerateRequest {
  // Pronunciation groups go out as U+E000 IPA U+E001 on phoneme-token checkpoints; offsets follow the sent text.
  const sent = toSent(text, backend.value?.pron.phoneme_tokens ?? null);
  const req: GenerateRequest = {
    ...s.ui.request_defaults,
    text: sent.text,
    seed,
    speaker_embedding_id: d.baseId,
    speaker_mode: d.mode,
  };
  if (x.language && x.language !== 'en_us') req.language = x.language;
  if (x.unsteered) return req;
  // Tag guidance strengthens nonverbal tags; the server takes one guidance per request, so voice guidance is off.
  const tags = backend.value?.tags;
  const strength = projectSettings.value.tagStrength;
  if (tags && strength > 1 && hasTags(text)) {
    req.tag_cfg_scale = Math.round(clamp(strength, Math.max(1, tags.min_scale), tags.max_scale) * 100) / 100;
    req.speaker_cfg_scale = 1;
  }
  const controls = toDoses(s, mergeValues(s, d.values, x.extra), accentsOf(s, d), d.mode);
  if (Object.keys(controls).length) req.controls = controls;
  const spans = controlSpans(s, x.spans, d, text.length, sent.map);
  if (spans.length) {
    req.control_spans = spans;
    // Spans follow the model's reading position; the server falls back to the rate where it has no heads.
    req.control_span_follow = 'reading';
    if (x.spanRate) req.control_span_rate = Math.round(x.spanRate * 1000) / 1000;
  }
  return req;
}
