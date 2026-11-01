/**
 * Who map in perceptual coordinates (ui.who): a fixed world in dose units where every library voice sits at its
 * measured spot (x = clamp(logit P(female), ±clip) / logit_per_dose, y = (age − centre) / years_per_dose). The
 * designed voice is its base voice's spot plus the gender / age doses.
 */
import { computed, signal } from '@preact/signals';
import type { Speaker, WhoSpec } from '../api';
import { spec } from './app';

/** Fallback when a bundle carries no ui.who (calibration of the release sweep). */
const DEFAULT_WHO: WhoSpec = {
  gender: { control: 'gender', logit_per_dose: 1.31, clip_logit: 3 },
  age: { control: 'age', center_years: 42, years_per_dose: 8.7 },
  landmarks: {},
};

export const who = computed<WhoSpec>(() => spec.value.ui.who ?? DEFAULT_WHO);

export interface Spot {
  x: number;
  y: number;
  /** false: no measurement, the voice is placed at the centre. */
  measured: boolean;
  /** Saved designed voice: where its position came from. */
  from?: string;
}

/** Positions of designed voices saved from a library voice: source spot + the doses baked in. */
export const voiceOrigins = signal<Record<string, Spot>>({});

const logit = (p: number) => Math.log(Math.max(1e-6, p) / Math.max(1e-6, 1 - p));

export function measuredSpot(female: number, age: number): Spot {
  const w = who.value;
  const l = Math.max(-w.gender.clip_logit, Math.min(w.gender.clip_logit, logit(female)));
  return { x: l / w.gender.logit_per_dose, y: (age - w.age.center_years) / w.age.years_per_dose, measured: true };
}

export function spotOf(s: Speaker | null | undefined): Spot {
  if (!s) return { x: 0, y: 0, measured: false };
  const m = who.value.landmarks[s.label];
  if (m) return measuredSpot(m.female, m.age);
  return voiceOrigins.value[s.id] ?? { x: 0, y: 0, measured: false };
}

export const yearsAt = (y: number) => who.value.age.center_years + y * who.value.age.years_per_dose;
export const yAtYears = (years: number) => (years - who.value.age.center_years) / who.value.age.years_per_dose;
/** P(female) at an x position (saturating at the clip). */
export const femaleAt = (x: number) => 1 / (1 + Math.exp(-x * who.value.gender.logit_per_dose));

/** The map's fixed extent in dose units: a margin beyond the clipped landmarks, ages ~14…84. */
export const extent = computed(() => {
  const xMax = who.value.gender.clip_logit / who.value.gender.logit_per_dose + 1.1;
  return { x: [-xMax, xMax] as [number, number], y: [yAtYears(14), yAtYears(84)] as [number, number] };
});

export function rememberOrigin(saved: Speaker, base: Speaker | null, dx: number, dy: number) {
  const s = spotOf(base);
  voiceOrigins.value = { ...voiceOrigins.value, [saved.id]: { x: s.x + dx, y: s.y + dy, measured: s.measured, from: base?.label } };
}
