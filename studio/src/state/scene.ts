/** Scene: dialogue lines assigned to saved cast members, rendered with each character's voice and delivery. */
import { computed, signal } from '@preact/signals';
import type { GenerateRequest, Rendered } from '../api';
import { cached, errorText, isAbort, render, requestKey } from '../audio/render';
import { hash32, uid } from '../lib/math';
import { projectSettings, pronTokens, requestContext, spec } from './app';
import { pronProblem } from '../lib/pron';
import { History } from './history';
import { cast, type Persona } from './library';
import { buildRequest } from './requests';
import { resample } from '../audio/mix';
import type { Values } from './engine';

export interface SceneLine {
  id: string;
  characterId: string;
  text: string;
  /** Start in the scene (seconds), e.g. a dubbed line kept at its place; else it follows the previous line + gap. */
  at?: number;
  /** Direction on top of the character's design (UI values of spannable controls), e.g. from a dub segment. */
  values?: Values;
}

export interface Scene {
  lines: SceneLine[];
  /** Silence between lines, seconds. */
  gap: number;
}

export const newScene = (): Scene => ({ lines: [], gap: 0.35 });

export const scenes = new History<Scene>(newScene());
export const scene = scenes.state;

export interface LineRender {
  key: string;
  audio: Rendered | null;
  busy: boolean;
  error: string | null;
}
export const renders = signal<Record<string, LineRender>>({});

const commit = (fn: (s: Scene) => Scene) => scenes.apply(fn(scene.value));

export function addSceneLine(characterId: string, text = '', afterId?: string) {
  commit((s) => {
    const lines = [...s.lines];
    const i = afterId ? lines.findIndex((l) => l.id === afterId) + 1 : lines.length;
    lines.splice(i, 0, { id: uid(), characterId, text });
    return { ...s, lines };
  });
}

export function updateSceneLine(id: string, patch: Partial<SceneLine>) {
  commit((s) => ({ ...s, lines: s.lines.map((l) => (l.id === id ? { ...l, ...patch } : l)) }));
}

export function removeSceneLine(id: string) {
  commit((s) => ({ ...s, lines: s.lines.filter((l) => l.id !== id) }));
}

export function moveSceneLine(id: string, delta: number) {
  commit((s) => {
    const lines = [...s.lines];
    const i = lines.findIndex((l) => l.id === id);
    const j = i + delta;
    if (i < 0 || j < 0 || j >= lines.length) return s;
    [lines[i], lines[j]] = [lines[j], lines[i]];
    return { ...s, lines };
  });
}

export function setGap(gap: number) {
  commit((s) => ({ ...s, gap }));
}

export const characterOf = (l: SceneLine): Persona | undefined => cast.value.find((p) => p.id === l.characterId);

/** Each line keeps a fixed seed, so unchanged lines re-use their render. */
export function sceneRequest(l: SceneLine): GenerateRequest | null {
  const p = characterOf(l);
  if (!p || !l.text.trim()) return null;
  return buildRequest(spec.value, requestContext.value, p.design, l.text, hash32(l.id) % 1_000_000, { extra: l.values, language: projectSettings.value.language });
}

export function lineState(l: SceneLine): LineRender | null {
  const req = sceneRequest(l);
  if (!req) return null;
  const key = requestKey(req);
  const r = renders.value[l.id];
  if (r && r.key === key) return r;
  const hit = cached(req);
  return { key, audio: hit ?? null, busy: false, error: null };
}

export async function renderSceneLine(l: SceneLine) {
  const req = sceneRequest(l);
  if (!req) return;
  const key = requestKey(req);
  const set = (r: Partial<LineRender>) => {
    const prev = renders.value[l.id];
    const base: LineRender = prev?.key === key ? prev : { key, audio: null, busy: false, error: null };
    renders.value = { ...renders.value, [l.id]: { ...base, ...r } };
  };
  const problem = pronProblem(l.text, pronTokens.value);
  if (problem) return set({ busy: false, error: problem });
  set({ busy: true, error: null, audio: null });
  try {
    const audio = await render(req);
    if (renders.value[l.id]?.key === key) set({ audio, busy: false });
  } catch (e) {
    if (renders.value[l.id]?.key === key) set({ busy: false, error: isAbort(e) ? null : errorText(e) });
  }
}

export function renderScene() {
  return Promise.all(scene.value.lines.filter((l) => !lineState(l)?.audio).map(renderSceneLine));
}

export interface Clip {
  line: SceneLine;
  start: number;
  duration: number;
  audio: Rendered | null;
  persona: Persona | undefined;
}

/**
 * Lines laid end to end with the gap, or at their own start (`at`) when they have one, never overlapping the line
 * before; unrendered lines get an estimated slot (≈15 characters per second).
 */
export const timeline = computed(() => {
  void renders.value;
  const clips: Clip[] = [];
  let end = 0;
  for (const [i, line] of scene.value.lines.entries()) {
    const st = lineState(line);
    const duration = st?.audio ? st.audio.pcm.length / st.audio.sr : Math.max(0.8, line.text.length / 15);
    const follow = i ? end + scene.value.gap : 0;
    const start = line.at != null ? Math.max(line.at, i ? end + 0.05 : 0) : follow;
    clips.push({ line, start, duration, audio: st?.audio ?? null, persona: characterOf(line) });
    end = start + duration;
  }
  return { clips, total: end };
});

/** The rendered lines placed on the timeline, mixed at `sr`. */
export function mixdown(sr: number): Float32Array {
  const { clips, total } = timeline.value;
  const out = new Float32Array(Math.ceil((total + 0.05) * sr));
  for (const c of clips) {
    if (!c.audio) continue;
    const pcm = resample(c.audio.pcm, c.audio.sr, sr);
    const at = Math.round(c.start * sr);
    for (let i = 0; i < pcm.length && at + i < out.length; i++) out[at + i] += pcm[i];
  }
  for (let i = 0; i < out.length; i++) out[i] = Math.max(-1, Math.min(1, out[i]));
  return out;
}
