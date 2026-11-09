/** The voice being designed in Cast: base voice, mode, control values and the accent pin, with undo. */
import { computed } from '@preact/signals';
import { embeddingOnly, type ControlSpec, type ControlsSpec, type SpeakerMode } from '../api';
import { clamp, fmt } from '../lib/math';
import { notify, projectSettings, spec } from './app';
import { homePin, mixAt, pinForMix, placeFor, type Pin } from './atlas';
import { controlUsable, identityEstimate, presetToValues, steerEmbedding, toDoses, type AccentMix, type Values } from './engine';
import { History } from './history';
import { embeddingOf, speakers } from './library';

export interface Design {
  baseId: string;
  mode: SpeakerMode;
  values: Values;
  pin: Pin;
}

/** A neutral bench (the project sets the base voice and mode). */
export const newDesign = (mode: SpeakerMode = 'embedding'): Design => ({ baseId: '', mode, values: {}, pin: homePin() });

export const designs = new History<Design>(newDesign());
export const design = designs.state;

export const accentsOf = (s: ControlsSpec, d: Design): AccentMix => mixAt(d.pin, s.ui.accents);
export const dosesOf = (s: ControlsSpec, d: Design) => toDoses(s, d.values, accentsOf(s, d), d.mode);

export const accentMix = computed(() => accentsOf(spec.value, design.value));
export const doses = computed(() => dosesOf(spec.value, design.value));
export const identity = computed(() => identityEstimate(spec.value, design.value.values, accentMix.value, design.value.mode));
export const baseSpeaker = computed(() => speakers.value.find((s) => s.id === design.value.baseId) ?? null);
export const baseEmbedding = computed(() => embeddingOf(design.value.baseId));
/** The embedding the server will condition on (designable mode), computed exactly as the server does. */
export const steered = computed(() => {
  const e = baseEmbedding.value;
  return e && design.value.mode === 'embedding' ? steerEmbedding(spec.value, e, doses.value) : null;
});
/** Exact clone needs a reference clip: voices cached from an embedding have none. */
export const canClone = computed(() => !embeddingOnly(baseSpeaker.value));

export const usable = (c: ControlSpec) => controlUsable(spec.value, c, design.value.mode);

// ── Edits (callers wrap drags in designs.begin()/end()) ─────────────────────────────────────────────

export function setValue(id: string, v: number) {
  const c = spec.value.ui.controls.find((x) => x.id === id);
  if (!c) return;
  const d = design.value;
  designs.set({ ...d, values: { ...d.values, [id]: Math.round(clamp(v, c.range[0], c.range[1]) * 100) / 100 } });
}

export function setPin(pin: Pin) {
  designs.set({ ...design.value, pin });
}

/**
 * Start fresh from a library voice: the bench moves go back to neutral (one undo step restores them), so a saved
 * designed voice is never steered twice. Personas applied afterwards land on top of it.
 */
export function setBase(id: string) {
  const d = design.value;
  if (id === d.baseId && !Object.values(d.values).some(Boolean) && !Object.keys(accentsOf(spec.value, d)).length) return;
  // An embedding-only voice can only be designable; leaving one goes back to the project's default voice mode.
  const was = speakers.value.find((s) => s.id === d.baseId);
  designs.apply({
    baseId: id,
    mode: embeddingOnly(speakers.value.find((s) => s.id === id)) ? 'embedding' : embeddingOnly(was) ? projectSettings.value.voiceMode : d.mode,
    values: {},
    pin: homePin(d.pin.tab),
  });
}

/** The bench has moves on it that `setBase` would clear. */
export const benchMoved = (d: Design = design.value) => Object.values(d.values).some(Boolean) || Object.keys(accentsOf(spec.value, d)).length > 0;

/**
 * `setBase` from the UI (library row, map landmark, a new clip): when the bench had moves, say so and offer Undo
 * instead of clearing them silently.
 */
export function switchBase(id: string, from: 'library' | 'map' | 'clip') {
  const d = design.value;
  if (id === d.baseId && !benchMoved(d)) return;
  const moved = benchMoved(d);
  setBase(id);
  if (!moved) return;
  const label = speakers.value.find((s) => s.id === id)?.label ?? 'this voice';
  const how = from === 'clip' ? 'the new clip' : label;
  notify(`Starting from ${how}; your moves were cleared`, 'info', { label: 'Undo', run: () => designs.undo() });
}

/** Applied a design that may change the mode: say so when Exact clone turned into Designable. */
function noteModeChange(before: SpeakerMode, after: SpeakerMode) {
  if (before === 'reference' && after === 'embedding') notify('Switched to Designable');
}

export function setMode(mode: SpeakerMode) {
  if (mode !== design.value.mode) designs.apply({ ...design.value, mode });
}

/** Apply a named starting point (replaces the bench moves; the toast offers Undo). */
export function applyPreset(name: string) {
  const preset = spec.value.ui.presets[name];
  if (!preset) return;
  const { values, accents } = presetToValues(preset);
  const d = design.value;
  designs.apply({ ...d, mode: 'embedding', values, pin: pinForMix(accents, spec.value.ui.accents, d.pin.tab) });
  if (benchMoved(d)) notify(`Applied ${name}; your moves were replaced`, 'info', { label: 'Undo', run: () => designs.undo() });
  noteModeChange(d.mode, 'embedding');
}

/** Roll and apply a random voice; the toast names the new base and offers Undo. */
export function roll() {
  const before = design.value;
  const next = rollDesign();
  designs.apply(next);
  const label = speakers.value.find((s) => s.id === next.baseId)?.label ?? 'a voice';
  notify(`Rolled on ${label}`, 'info', { label: 'Undo', run: () => designs.undo() });
  noteModeChange(before.mode, next.mode);
}

/** Back to the plain base voice: in Exact clone only the usable controls (delivery, mood) reset; the rest wait for Designable. The toast offers Undo. */
export function resetDesign() {
  const d = design.value;
  const s = spec.value;
  const values: Values = {};
  for (const [id, v] of Object.entries(d.values)) {
    const c = s.ui.controls.find((x) => x.id === id);
    if (c && !controlUsable(s, c, d.mode)) values[id] = v;
  }
  const accentUsable = s.ui.accents.some((a) => controlUsable(s, a, d.mode));
  if (!benchMoved(d)) return;
  designs.apply({ ...d, values, pin: accentUsable ? homePin(d.pin.tab) : d.pin });
  notify('Controls reset', 'info', { label: 'Undo', run: () => designs.undo() });
}

// ── Randomness: roll and neighbours ───────────────────────────────────────────────────────────────

const pick = <T,>(xs: T[], r: () => number) => xs[Math.floor(r() * xs.length)];

/** A random value inside the natural range, `scale` of the way out at most, rounded to a tenth. */
function randomSafe(c: ControlSpec, r: () => number, scale: number): number {
  const [lo, hi] = c.safe;
  return Math.round((lo + (hi - lo) * r()) * scale * 10) / 10;
}

/** The control groups Roll touches: who the voice is, never how it performs (delivery and mood stay neutral). */
const ROLL_GROUPS: Record<string, true> = { who: true, texture: true, room: true };

/** `n` distinct random picks from `xs`. */
function sample<T>(xs: T[], n: number, r: () => number): T[] {
  const pool = [...xs];
  for (let i = 0; i < Math.min(n, pool.length); i++) {
    const j = i + Math.floor(r() * (pool.length - i));
    [pool[i], pool[j]] = [pool[j], pool[i]];
  }
  return pool.slice(0, Math.min(n, pool.length));
}

/**
 * A random new person: a random base voice, the who controls nudged a little, one or two texture / room moves,
 * and an accent a third of the time (mild). Delivery and mood are left alone: a rolled voice should sound like
 * someone else, not like someone upset.
 */
export function rollDesign(r: () => number = Math.random): Design {
  const s = spec.value;
  const defaults = speakers.value.filter((x) => x.is_default);
  const values: Values = {};
  for (const c of s.ui.controls) if (c.group === 'who') values[c.id] = randomSafe(c, r, 0.5);
  const others = s.ui.controls.filter((c) => ROLL_GROUPS[c.group] && c.group !== 'who');
  for (const c of sample(others, 1 + Math.floor(r() * 2), r)) values[c.id] = randomSafe(c, r, 0.5);
  let pin = homePin(design.value.pin.tab);
  if (r() < 0.33) {
    const a = pick(s.ui.accents, r);
    pin = placeFor(a, a.safe[1] * (0.3 + 0.4 * r()), s.ui.accents);
  }
  return { baseId: (pick(defaults, r) ?? speakers.value[0])?.id ?? design.value.baseId, mode: 'embedding', values, pin };
}

export interface Variation {
  design: Design;
  changes: string[];
}

export function describeChange(c: ControlSpec, before: number, after: number): string {
  const d = after - before;
  if (c.type === 'bipolar' && c.poles) return `${d > 0 ? c.poles[1] : c.poles[0]} ${Math.abs(d).toFixed(1)}`;
  return `${c.label.toLowerCase()} ${fmt(d)}`;
}

function accentLabel(mix: AccentMix): string {
  const top = Object.entries(mix).sort((a, b) => b[1] - a[1])[0];
  if (!top) return 'no accent';
  const a = spec.value.ui.accents.find((x) => x.id === top[0]);
  return `${a?.label ?? top[0]} ${top[1].toFixed(1)}`;
}

/** Five safe random moves around `d`: 1-3 controls nudged inside their safe ranges, sometimes the accent. */
export function neighbours(d: Design, n = 5, r: () => number = Math.random): Variation[] {
  const s = spec.value;
  const candidates = s.ui.controls.filter((c) => !c.span_only && controlUsable(s, c, d.mode));
  const accentsUsable = s.ui.accents.filter((a) => controlUsable(s, a, d.mode));
  const out: Variation[] = [];
  for (let i = 0; i < n; i++) {
    const values = { ...d.values };
    const changes: string[] = [];
    const k = 1 + Math.floor(r() * 3);
    const chosen = new Set<ControlSpec>();
    while (chosen.size < Math.min(k, candidates.length)) chosen.add(pick(candidates, r));
    for (const c of chosen) {
      const before = values[c.id] ?? 0;
      const span = c.safe[1] - c.safe[0];
      let after = clamp(before + (r() - 0.5) * span * 0.45, c.safe[0], c.safe[1]);
      if (Math.abs(after - before) < span * 0.06) after = clamp(before + span * 0.12 * (r() < 0.5 ? -1 : 1), c.safe[0], c.safe[1]);
      after = Math.round(after * 10) / 10;
      if (after === before) continue;
      values[c.id] = after;
      changes.push(describeChange(c, before, after));
    }
    let pin = d.pin;
    if (accentsUsable.length && (r() < 0.3 || !changes.length)) {
      const a = pick(accentsUsable, r);
      pin = placeFor(a, a.safe[1] * (0.35 + 0.6 * r()), s.ui.accents);
      changes.push(`→ ${accentLabel(mixAt(pin, s.ui.accents))}`);
    }
    out.push({ design: { ...d, values, pin }, changes });
  }
  return out;
}
