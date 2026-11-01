/**
 * Pure contract math: UI values → primitive doses, safety zones, identity estimate, steered embeddings and
 * Who-map projections. No signals here; state modules and the offline synth both build on it.
 */
import type { AccentSpec, ControlSpec, ControlsSpec, Dose, SpeakerMode } from '../api';
import { clamp, norm, piecewise } from '../lib/math';

export type Values = Record<string, number>; // UI control id → value
export type AccentMix = Record<string, number>; // accent id → strength

export type Zone = 'safe' | 'strong' | 'collapse';

export function zoneOf(c: { safe: [number, number]; collapse?: ControlSpec['collapse'] }, v: number): Zone {
  const col = c.collapse;
  if (col && ((col[0] != null && v <= col[0]) || (col[1] != null && v >= col[1]))) return 'collapse';
  return v >= c.safe[0] - 1e-9 && v <= c.safe[1] + 1e-9 ? 'safe' : 'strong';
}

export function addDose(into: Dose, map: Dose | undefined, scale: number) {
  if (!map || !scale) return;
  for (const [id, w] of Object.entries(map)) into[id] = (into[id] ?? 0) + w * scale;
}

/** v > 0 → map["+"] × v; v < 0 → map["-"] × |v| (unipolar controls only have "+"). */
export function controlDose(c: ControlSpec, v: number, into: Dose = {}): Dose {
  if (v > 0) addDose(into, c.map['+'], v);
  else if (v < 0) addDose(into, c.map['-'], -v);
  return into;
}

export function modeAllows(spec: ControlsSpec, dirId: string, mode: SpeakerMode): boolean {
  const d = spec.directions[dirId];
  return !!d && (spec.sets[d.set]?.modes ?? []).includes(mode);
}

/** A control is usable in a mode when every direction it maps to is. */
export function controlUsable(spec: ControlsSpec, c: ControlSpec | AccentSpec, mode: SpeakerMode): boolean {
  const maps = 'type' in c ? [c.map['+'], c.map['-']] : [c.map];
  return maps.every((m) => !m || Object.keys(m).every((id) => modeAllows(spec, id, mode)));
}

export function isRaw(spec: ControlsSpec, dirId: string) {
  return spec.directions[dirId]?.space === 'raw';
}

/** All UI values + accent mix → primitive `controls` for a request in `mode` (unusable ones dropped). */
export function toDoses(spec: ControlsSpec, values: Values, accents: AccentMix, mode: SpeakerMode): Dose {
  const out: Dose = {};
  for (const c of spec.ui.controls) {
    const v = values[c.id] ?? 0;
    if (v && controlUsable(spec, c, mode)) controlDose(c, v, out);
  }
  for (const a of spec.ui.accents) {
    const s = accents[a.id] ?? 0;
    if (s && controlUsable(spec, a, mode)) addDose(out, a.map, s);
  }
  for (const id of Object.keys(out)) {
    out[id] = Math.round(out[id] * 1e4) / 1e4;
    if (!out[id]) delete out[id];
  }
  return out;
}

/** Presets reference UI control ids (`accent:<id>` for accents). */
export function presetToValues(preset: Record<string, number>): { values: Values; accents: AccentMix } {
  const values: Values = {};
  const accents: AccentMix = {};
  for (const [k, v] of Object.entries(preset)) {
    if (k.startsWith('accent:')) accents[k.slice(7)] = v;
    else values[k] = v;
  }
  return { values, accents };
}

// L1 accents carry no curve: the README's SIM drop (≈8 points at ×1, 20 at ×1.5) for speaker-embedding accents.
const ACCENT_IDENTITY: [number, number][] = [
  [0, 0.95],
  [1, 0.87],
  [1.5, 0.76],
  [2, 0.68],
];

/** Identity factor of one control at |v|, relative to the curve's unsteered value. */
function factor(curve: [number, number][], x: number): number {
  return piecewise(curve, x) / (piecewise(curve, 0) || 1);
}

/**
 * Estimated identity kept: 0.95 × Π f_c(|v_c|) / f_c(0) over the controls usable in `mode` (controls whose
 * identity was not measured count as 1). An estimate, labelled as such in the UI.
 */
export function identityEstimate(spec: ControlsSpec, values: Values, accents: AccentMix, mode: SpeakerMode): number {
  let k = 0.95;
  for (const c of spec.ui.controls) {
    const v = values[c.id] ?? 0;
    if (v && c.identity && controlUsable(spec, c, mode)) k *= factor(c.identity, Math.abs(v));
  }
  for (const a of spec.ui.accents) {
    const s = accents[a.id] ?? 0;
    if (s && controlUsable(spec, a, mode)) k *= factor(a.identity ?? ACCENT_IDENTITY, s);
  }
  return clamp(k, 0, 1);
}

/** Expected duration multiplier of a set of doses: Π timing[id](dose), as the server computes it. */
export function durationFactor(spec: ControlsSpec, doses: Dose): number {
  let f = 1;
  for (const [id, table] of Object.entries(spec.timing ?? {})) {
    const d = doses[id];
    if (d) f *= piecewise(table, d);
  }
  return f;
}

/** e' = (e + Σ dose·vector) × |e| / |e + Σ dose·vector| over the raw-space doses. */
export function steerEmbedding(spec: ControlsSpec, e: Float32Array, doses: Dose): Float32Array {
  const out = Float32Array.from(e);
  let moved = false;
  for (const [id, dose] of Object.entries(doses)) {
    const v = spec.directions[id]?.vector;
    if (!v || spec.directions[id].space !== 'raw' || !dose) continue;
    for (let i = 0; i < out.length; i++) out[i] += dose * v[i];
    moved = true;
  }
  if (!moved) return out;
  const scale = norm(e) / (norm(out) || 1);
  for (let i = 0; i < out.length; i++) out[i] *= scale;
  return out;
}

/** Collapse a dose map to stable JSON for cache keys. */
export function stableDose(d: Dose | undefined): string {
  if (!d) return '';
  return Object.keys(d)
    .sort()
    .map((k) => `${k}=${d[k]}`)
    .join(',');
}
