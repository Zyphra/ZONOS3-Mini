import { computed, signal } from '@preact/signals';
import type { Backend, ControlSpec, ControlsSpec, SpeakerMode } from '../api';
import type { History } from './history';

export const backend = signal<Backend | null>(null);
export const bootError = signal<string | null>(null);
/** Boot-screen status while a backend loads (the in-browser server's model download). */
export const bootProgress = signal<string | null>(null);

/**
 * Settings of the open project: the language (Direct / Scene renders, Edit clips, pronunciation lookups), the
 * default voice mode, and the tag strength (tag guidance for lines with nonverbal tags; 1 = off).
 */
export interface ProjectSettings {
  language: string;
  voiceMode: SpeakerMode;
  tagStrength: number;
}
export const DEFAULT_SETTINGS: ProjectSettings = { language: 'en_us', voiceMode: 'embedding', tagStrength: 1.5 };
export const projectSettings = signal<ProjectSettings>(DEFAULT_SETTINGS);

/** The controls spec; only read after boot. */
export const spec = computed<ControlsSpec>(() => backend.value!.spec);

/** The checkpoint's phoneme tokens (pronunciation input), or null. */
export const pronTokens = computed(() => backend.value?.pron.phoneme_tokens ?? null);

export const controlById = computed(() => new Map<string, ControlSpec>(spec.value.ui.controls.map((c) => [c.id, c])));

export const groupHue = computed(() => {
  const m = new Map<string, string>();
  spec.value.ui.groups.forEach((g, i) => m.set(g.id, g.hue ?? `c${i + 1}`));
  return m;
});

/** CSS colour of a control family (groups carry hue tokens c1..c6). */
export function hue(group: string): string {
  return `var(--${groupHue.value.get(group) ?? 'c1'})`;
}

export type Act = 'cast' | 'direct' | 'scene' | 'edit' | 'dub';
export const ACT_IDS: Act[] = ['cast', 'direct', 'scene', 'edit', 'dub'];
const initialAct = location.hash.slice(1) as Act;
export const act = signal<Act>(ACT_IDS.includes(initialAct) ? initialAct : 'cast');

/** The history ⌘Z acts on (each act registers its own). */
export const activeHistory = signal<History<unknown> | null>(null);
/** A dialog is open: act switching by number keys is off. */
export const modalOpen = signal(false);
/** The current act's own keys (called first; return true when handled). */
export const screenKeys = signal<((e: KeyboardEvent) => boolean) | null>(null);
/** What space does on the current act. */
export const primaryAction = signal<(() => void) | null>(null);

export interface Toast {
  id: number;
  text: string;
  kind: 'info' | 'error' | 'ok';
}
export const toasts = signal<Toast[]>([]);
let toastId = 0;

export function notify(text: string, kind: Toast['kind'] = 'info') {
  const id = ++toastId;
  toasts.value = [...toasts.value.slice(-3), { id, text, kind }];
  setTimeout(() => (toasts.value = toasts.value.filter((t) => t.id !== id)), kind === 'error' ? 7000 : 3500);
}
