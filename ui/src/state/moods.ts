/**
 * Named moods: the unipolar mood controls, their measured landmarks on the valence / arousal pad (`ui.moods`) and
 * default doses.
 */
import { computed } from '@preact/signals';
import type { ControlSpec } from '../api';
import { spec } from './app';

export interface Mood {
  control: ControlSpec;
  label: string;
  /** Measured place on the pad (−1…1) at `dose`; null when the bundle has no landmark for it. */
  at: { valence: number; arousal: number } | null;
  dose: number;
  effect: string;
}

const half = (x: number) => Math.max(0.5, Math.round(x * 2) / 2);

/** The valence (x) and arousal (y) pad controls: the first two bipolar mood controls. */
export const padAxes = computed(() => {
  const [x, y] = spec.value.ui.controls.filter((c) => c.group === 'mood' && c.type === 'bipolar');
  return x && y ? { x, y } : null;
});

/** Every named mood (unipolar mood control) with its landmark when measured: most aroused first, like the pad. */
export const moods = computed<Mood[]>(() => {
  const marks = new Map((spec.value.ui.moods ?? []).map((m) => [m.control, m]));
  const list = spec.value.ui.controls
    .filter((c) => c.group === 'mood' && c.type === 'unipolar')
    .map((c): Mood => {
      const m = marks.get(c.id);
      const at = m && Number.isFinite(m.valence) && Number.isFinite(m.arousal) ? { valence: m.valence, arousal: m.arousal } : null;
      return { control: c, label: m?.label ?? c.label, at, dose: m?.dose && m.dose > 0 ? m.dose : half(c.safe[1] * 0.5), effect: m?.effect ?? c.effect ?? '' };
    });
  return list.every((m) => m.at) ? list.sort((a, b) => b.at!.arousal - a.at!.arousal) : list;
});

export const moodIds = computed(() => new Set(moods.value.map((m) => m.control.id)));
