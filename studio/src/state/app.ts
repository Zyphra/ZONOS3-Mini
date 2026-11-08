import { computed, signal } from '@preact/signals';
import type { Backend, ControlSpec, ControlsSpec } from '../api';
import type { Download } from '../lib/web-engine';
import type { History } from './history';
import { DEFAULT_SETTINGS, type ProjectSettings } from './projectSchema';
import type { RequestContext } from './requests';

export const backend = signal<Backend | null>(null);
export const bootError = signal<string | null>(null);
/** Boot-screen download while a backend loads (the in-browser server's models). */
export const bootProgress = signal<Download | null>(null);
/** The encoder models a clip voice's first use loads on the in-browser server; null once they are in. */
export const encoderDownload = signal<Download | null>(null);

export const projectSettings = signal<ProjectSettings>(DEFAULT_SETTINGS);

/** The controls spec; only read after boot. */
export const spec = computed<ControlsSpec>(() => backend.value!.spec);

/** The checkpoint's phoneme tokens (pronunciation input), or null. */
export const pronTokens = computed(() => backend.value?.pron.phoneme_tokens ?? null);

/** Backend and project inputs of every generate request (`buildRequest`). */
export const requestContext = computed<RequestContext>(() => ({
  phonemeTokens: pronTokens.value,
  tags: backend.value?.tags ?? null,
  tagStrength: projectSettings.value.tagStrength,
}));

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

export type Act = 'cast' | 'direct' | 'edit';
export const ACT_IDS: Act[] = ['cast', 'direct', 'edit'];
/** The retired Scene act's hash opens Direct, where its lines now live. */
const initialAct = (location.hash.slice(1) === 'scene' ? 'direct' : location.hash.slice(1)) as Act;
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
  /** One thing the reader can do about it (Undo, Retry, Go to…). */
  action?: { label: string; run: () => void };
}
export const toasts = signal<Toast[]>([]);
let toastId = 0;
const timers = new Map<number, number>();

export function dismissToast(id: number) {
  clearTimeout(timers.get(id));
  timers.delete(id);
  toasts.value = toasts.value.filter((t) => t.id !== id);
}

/** Show a toast; returns its id. Errors stay 15 s, toasts with an action 8 s, the rest 3.5 s. Repeating the last text restarts its timer instead of stacking. */
export function notify(text: string, kind: Toast['kind'] = 'info', action?: Toast['action']): number {
  const ms = kind === 'error' ? 15000 : action ? 8000 : 3500;
  const same = toasts.value.find((t) => t.text === text && t.kind === kind);
  if (same) {
    clearTimeout(timers.get(same.id));
    toasts.value = toasts.value.map((t) => (t === same ? { ...t, action } : t));
    timers.set(same.id, window.setTimeout(() => dismissToast(same.id), ms));
    return same.id;
  }
  const id = ++toastId;
  toasts.value = [...toasts.value.slice(-3), { id, text, kind, action }];
  timers.set(id, window.setTimeout(() => dismissToast(id), ms));
  return id;
}
