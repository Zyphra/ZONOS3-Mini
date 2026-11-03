/**
 * Natural tool input → studio designs: control values with their safety zones, the accent pin, painted spans, and
 * the voice a tool names. All on the studio's own engine / atlas / request code.
 */
import type { ControlsSpec, Dose, Speaker, SpeakerMode } from '../../src/api';
import { uid } from '../../src/lib/math';
import { homePin, mixAt, pinForMix, type Pin } from '../../src/state/atlas';
import type { Design } from '../../src/state/design';
import { controlUsable, durationFactor, identityEstimate, toDoses, zoneOf, type AccentMix, type Values, type Zone } from '../../src/state/engine';
import type { Persona } from '../../src/state/library';
import type { ProjectDoc } from '../../src/state/projectSchema';
import type { Span } from '../../src/state/requests';

/** UI values from tool input: unknown ids throw, values clamp to range, zones warn; 0 leaves the key out. */
export function resolveValues(spec: ControlsSpec, input: Record<string, number>, kind: 'design' | 'delivery'): { values: Values; warnings: string[] } {
  const values: Values = {};
  const warnings: string[] = [];
  for (const [id, raw] of Object.entries(input)) {
    const c = spec.ui.controls.find((x) => x.id === id);
    if (!c) throw new Error(`Unknown control '${id}'. Controls: ${spec.ui.controls.map((x) => x.id).join(', ')}`);
    if (kind === 'delivery' && !c.spannable) throw new Error(`'${id}' is a voice control, not a delivery control: set it with zonos_design_voice.`);
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

export interface ValueView {
  id: string;
  label: string;
  group: string;
  value: number;
  zone: Zone;
  poles?: [string, string];
  dropped?: boolean;
}

export interface DesignView {
  values: ValueView[];
  accents: { id: string; label: string; applied: number }[];
  doses: Dose;
  identity: number;
  duration: number;
  warnings: string[];
}

/** What a design does: values with zones (dropped when the mode can't use them), the accent mix, doses, estimates. */
export function designView(spec: ControlsSpec, d: Design): DesignView {
  const mix = mixAt(d.pin, spec.ui.accents);
  const warnings: string[] = [];
  const values: ValueView[] = [];
  for (const c of spec.ui.controls) {
    const v = d.values[c.id];
    if (!v) continue;
    const dropped = !controlUsable(spec, c, d.mode);
    if (dropped) warnings.push(`${c.label} needs speaker_mode=embedding; dropped in reference mode`);
    values.push({ id: c.id, label: c.label, group: c.group, value: v, zone: zoneOf(c, v), ...(c.poles ? { poles: c.poles } : {}), ...(dropped ? { dropped } : {}) });
  }
  const accents = spec.ui.accents.filter((a) => mix[a.id]).map((a) => ({ id: a.id, label: a.label, applied: mix[a.id] }));
  if (d.mode === 'reference' && accents.some((a) => !controlUsable(spec, spec.ui.accents.find((x) => x.id === a.id)!, d.mode))) warnings.push('Accents need speaker_mode=embedding; dropped in reference mode');
  const doses = toDoses(spec, d.values, mix, d.mode);
  return { values, accents, doses, identity: identityEstimate(spec, d.values, mix, d.mode), duration: durationFactor(spec, doses), warnings };
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
  const hasDelivery = !!delivery && Object.keys(delivery).length > 0;
  if (!hasDelivery) return placed.map((p) => ({ id: uid(), start: p.start, end: p.end, values: p.values }));
  const out: Span[] = [];
  let at = 0;
  for (const p of placed) {
    if (p.start > at) out.push({ id: uid(), start: at, end: p.start, values: { ...delivery } });
    out.push({ id: uid(), start: p.start, end: p.end, values: { ...delivery, ...p.values } });
    at = p.end;
  }
  if (at < text.length) out.push({ id: uid(), start: at, end: text.length, values: { ...delivery } });
  return out;
}

/** The mode a library voice gets by default: the project's when the voice has it, else its first. */
export function modeFor(s: Speaker, preferred: SpeakerMode): SpeakerMode {
  return s.speaker_modes.includes(preferred) ? preferred : s.speaker_modes[0];
}

/** Add or replace a persona by name (case-insensitive); returns the stored persona. */
export function upsertPersona(doc: ProjectDoc, name: string, design: Design, signature?: string): Persona {
  const list = doc.cast.personas;
  const i = list.findIndex((p) => p.name.toLowerCase() === name.toLowerCase());
  const p: Persona = i >= 0 ? { ...list[i], design, ...(signature !== undefined ? { signature } : {}) } : { id: uid(), name, design, signature: signature ?? '', createdAt: Date.now() };
  if (i >= 0) list[i] = p;
  else list.push(p);
  return p;
}

const lower = (s: string) => s.toLowerCase();

export function findSpeaker(speakers: Speaker[], ref: string): Speaker | undefined {
  return speakers.find((s) => s.id === ref) ?? speakers.find((s) => lower(s.label) === lower(ref));
}

/**
 * The persona a tool speaks with: persona id / name, else a library voice (id / label) as a plain persona named
 * after it, else (omitted) the persona of the Cast bench. Upserts into `doc`.
 */
export function resolveVoice(doc: ProjectDoc, speakers: Speaker[], voice?: string): Persona {
  const personas = doc.cast.personas;
  if (voice) {
    const p = personas.find((x) => x.id === voice) ?? personas.find((x) => lower(x.name) === lower(voice));
    if (p) return p;
    const s = findSpeaker(speakers, voice);
    if (s) return upsertPersona(doc, s.label, { baseId: s.id, mode: modeFor(s, doc.settings.voiceMode), values: {}, pin: homePin() });
    throw new Error(`Unknown voice '${voice}'. Personas: ${personas.map((x) => x.name).join(', ') || '(none)'}. Library voices: ${speakers.map((x) => x.label).join(', ')}`);
  }
  const bench = JSON.stringify(doc.cast.design);
  const p = personas.find((x) => JSON.stringify(x.design) === bench);
  if (p) return p;
  if (!doc.cast.design.baseId) throw new Error('No voice: pass voice (see zonos_voices) or design one with zonos_design_voice');
  return upsertPersona(doc, 'Cast bench', structuredClone(doc.cast.design));
}
