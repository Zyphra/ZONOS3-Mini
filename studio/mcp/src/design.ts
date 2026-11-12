/**
 * Studio designs for tools: what a design does (values with zones, the accent mix, doses, estimates), personas,
 * and the voice a tool names. Input resolution is the studio's `state/resolve.ts`.
 */
import type { ControlsSpec, Dose, Speaker } from '../../src/api';
import { uid } from '../../src/lib/math';
import { homePin, mixAt } from '../../src/state/atlas';
import type { Design } from '../../src/state/design';
import { controlUsable, durationFactor, identityEstimate, toDoses, zoneOf, type Zone } from '../../src/state/engine';
import type { Persona } from '../../src/state/library';
import type { ProjectDoc } from '../../src/state/projectSchema';
import { findSpeaker, modeFor } from '../../src/state/resolve';

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

/** Add or replace a persona by name (case-insensitive); returns the stored persona. */
export function upsertPersona(doc: ProjectDoc, name: string, design: Design, signature?: string): Persona {
  const list = doc.cast.personas;
  const i = list.findIndex((p) => p.name.toLowerCase() === name.toLowerCase());
  const p: Persona = i >= 0 ? { ...list[i], design, ...(signature !== undefined ? { signature } : {}) } : { id: uid(), name, design, signature: signature ?? '', createdAt: Date.now() };
  if (i >= 0) list[i] = p;
  else list.push(p);
  return p;
}

/**
 * The persona a tool speaks with: persona id / name, else a library voice (id / label) as a plain persona named
 * after it, else (omitted) the persona of the Cast bench. Upserts into `doc`.
 */
export function resolveVoice(doc: ProjectDoc, speakers: Speaker[], voice?: string): Persona {
  const personas = doc.cast.personas;
  if (voice) {
    const key = voice.toLowerCase();
    const p = personas.find((x) => x.id === voice) ?? personas.find((x) => x.name.toLowerCase() === key);
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
