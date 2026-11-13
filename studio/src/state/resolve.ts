/**
 * Written input → studio designs and spans: control values with their safety zones, the accent pin, spans on a
 * line's text, voice lookup. Signal-free: the MCP server and the lite page's script share it.
 */
import type { ControlsSpec, Speaker, SpeakerMode } from '../api';
import { uid } from '../lib/math';
import { mixAt, pinForMix, type Pin } from './atlas';
import { zoneOf, type AccentMix, type Values } from './engine';
import type { Span } from './requests';

/** UI values from tool input: unknown ids throw, values clamp to range, zones warn; 0 leaves the key out. */
export function resolveValues(spec: ControlsSpec, input: Record<string, number>, kind: 'design' | 'delivery'): { values: Values; warnings: string[] } {
  const values: Values = {};
  const warnings: string[] = [];
  for (const [id, raw] of Object.entries(input)) {
    const c = spec.ui.controls.find((x) => x.id === id);
    if (!c) throw new Error(`Unknown control '${id}'. Controls: ${spec.ui.controls.map((x) => x.id).join(', ')}`);
    if (kind === 'delivery' && !c.spannable) throw new Error(`'${id}' is a voice control, not a delivery control: set it on the voice.`);
    let v = Math.round(raw * 100) / 100;
    const [lo, hi] = c.range;
    if (v < lo || v > hi) {
      warnings.push(`${c.label} ${v} clamped to ${lo}..${hi}`);
      v = Math.min(hi, Math.max(lo, v));
    }
    const zone = zoneOf(c, v);
    if (zone === 'strong') warnings.push(`${c.label} ${v} is outside the tested-safe range ${c.safe[0]}..${c.safe[1]}`);
    else if (zone === 'collapse') warnings.push(`${c.label} ${v} is in the collapse zone: renders may break`);
    if (v) values[id] = v;
  }
  return { values, warnings };
}

export interface AccentResolution {
  pin: Pin;
  requested: AccentMix;
  applied: AccentMix;
  warnings: string[];
}

/** An accent mix → the one pin Studio places (the strongest accent wins) and the mix it really gives. */
export function resolveAccents(spec: ControlsSpec, input: Record<string, number>): AccentResolution {
  const requested: AccentMix = {};
  const warnings: string[] = [];
  for (const [id, raw] of Object.entries(input)) {
    const a = spec.ui.accents.find((x) => x.id === id);
    if (!a) throw new Error(`Unknown accent '${id}'. Accents: ${spec.ui.accents.map((x) => x.id).join(', ')}`);
    const v = Math.min(a.range[1], Math.max(a.range[0], raw));
    if (v !== raw) warnings.push(`${a.label} ${raw} clamped to ${a.range[0]}..${a.range[1]}`);
    if (v > 0) requested[id] = v;
  }
  const pin = pinForMix(requested, spec.ui.accents, 'english');
  const applied = mixAt(pin, spec.ui.accents);
  const top = Object.entries(requested).sort((a, b) => b[1] - a[1])[0];
  if (top) {
    const got = applied[top[0]] ?? 0;
    if (Math.abs(got - top[1]) > 0.05 || Object.keys(requested).length > 1) warnings.push(`Studio places one accent pin: ${top[0]} applied at ${got} (requested ${top[1]})`);
  }
  return { pin, requested, applied, warnings };
}

export interface SpanInput {
  text: string;
  occurrence?: number;
  values: Values;
}

/**
 * Painted spans from substrings of the line. With `delivery`, the spans cover the whole line: the gaps take
 * `delivery`, each sub-span `{...delivery, ...its values}`.
 */
export function anchorSpans(text: string, delivery: Values | null, spans: SpanInput[]): Span[] {
  const placed = spans
    .map((s) => {
      const n = s.occurrence ?? 1;
      let at = -1;
      for (let k = 0; k < n; k++) {
        at = text.indexOf(s.text, at + 1);
        if (at < 0) break;
      }
      if (at < 0 || !s.text) throw new Error(`Span text "${s.text}" (occurrence ${n}) not found in the line`);
      return { start: at, end: at + s.text.length, text: s.text, values: s.values };
    })
    .sort((a, b) => a.start - b.start);
  for (let i = 1; i < placed.length; i++) {
    if (placed[i].start < placed[i - 1].end) throw new Error(`Spans "${placed[i - 1].text}" and "${placed[i].text}" overlap`);
  }
  return layerSpans(text.length, delivery, placed);
}

/**
 * Placed spans (sorted, non-overlapping) on a line of `length` characters → painted spans. With `delivery`, the
 * spans cover the whole line: the gaps take `delivery`, each sub-span `{...delivery, ...its values}`.
 */
export function layerSpans(length: number, delivery: Values | null, placed: readonly { start: number; end: number; values: Values }[]): Span[] {
  const hasDelivery = !!delivery && Object.keys(delivery).length > 0;
  if (!hasDelivery) return placed.map((p) => ({ id: uid(), start: p.start, end: p.end, values: p.values }));
  const out: Span[] = [];
  let at = 0;
  for (const p of placed) {
    if (p.start > at) out.push({ id: uid(), start: at, end: p.start, values: { ...delivery } });
    out.push({ id: uid(), start: p.start, end: p.end, values: { ...delivery, ...p.values } });
    at = p.end;
  }
  if (at < length) out.push({ id: uid(), start: at, end: length, values: { ...delivery } });
  return out;
}

/** The mode a library voice gets by default: the project's when the voice has it, else its first. */
export function modeFor(s: Speaker, preferred: SpeakerMode): SpeakerMode {
  return s.speaker_modes.includes(preferred) ? preferred : s.speaker_modes[0];
}

export function findSpeaker(speakers: Speaker[], ref: string): Speaker | undefined {
  const key = ref.toLowerCase();
  return speakers.find((s) => s.id === ref) ?? speakers.find((s) => s.label.toLowerCase() === key);
}
