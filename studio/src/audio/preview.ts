/**
 * The Cast preview: the design saying the preview phrase (fixed seed), streamed into the speakers, cached by
 * request, stale renders cancelled, with the unsteered render of the same voice ready for A/B. With auto-generate
 * on, every design move re-renders after a 250 ms pause; otherwise (and always for what a load sets: boot, opening
 * a project) the take is marked out of date until Speak, A or Space asks for it.
 */
import { computed, effect, signal, untracked } from '@preact/signals';
import type { GenerateRequest, Rendered } from '../api';
import { notify, projectSettings, pronTokens, requestContext, spec } from '../state/app';
import { pronProblem } from '../lib/pron';
import { design } from '../state/design';
import { load, persist } from '../state/persist';
import { buildRequest, PREVIEW_SEED } from '../state/requests';
import * as player from './player';
import { cached, errorText, isAbort, render, requestKey } from './render';

const DEBOUNCE_MS = 250;

export type Side = 'A' | 'B';
export const side = signal<Side>('A');
export const autoplay = signal(load('autoplay', true));
persist('autoplay', autoplay);
/** Render on every design / text change (else only on Speak). */
export const autoRender = signal(load('autoRender', true));
persist('autoRender', autoRender);

/** The project's preview line ('' = the bundle's until the reader typed; once touched, '' means empty). */
export const previewText = signal('');
/** The reader edited the preview line in this session: an empty field stays empty (no fallback to the bundle's text). */
export const previewTouched = signal(false);
export const phrase = computed(() => {
  const typed = previewText.value.trim();
  return previewTouched.value ? typed : typed || spec.value.ui.preview_text;
});

export interface PreviewState {
  status: 'idle' | 'waiting' | 'rendering' | 'ready' | 'error';
  key: string;
  a: Rendered | null;
  /** Audio received so far while streaming. */
  partial: Float32Array | null;
  error: string | null;
}

export const preview = signal<PreviewState>({ status: 'idle', key: '', a: null, partial: null, error: null });
export const reference = signal<{ key: string; take: Rendered | null; error: string | null }>({ key: '', take: null, error: null });

export const ID_A = 'preview:A';
export const ID_B = 'preview:B';

let timer = 0;
let ctrl: AbortController | null = null;
/** The current design's requests (null while the text can't be rendered). */
let reqA: GenerateRequest | null = null;
let reqB: GenerateRequest | null = null;
/** Speak asked for the render still coming: play it even with auto-play off. */
let playNext = false;
/** What changes now came from a load, not the user: they don't render. */
let holding = false;
/** autoRender as the follower last saw it (turning it on renders the out-of-date take). */
let wasAuto = false;

/** The changes made in this task (a project being restored) don't render. */
export function holdPreview() {
  holding = true;
  queueMicrotask(() => (holding = false));
}

/** Whether a render arriving now plays (and clears a pending Speak). */
function claimPlay(): boolean {
  if (side.value !== 'A' || !(autoplay.value || playNext)) return false;
  playNext = false;
  return true;
}

function playIfWanted(r: Rendered) {
  if (claimPlay()) player.play(ID_A, r.pcm, r.sr);
}

async function run(req: GenerateRequest, key: string) {
  ctrl?.abort();
  const mine = (ctrl = new AbortController());
  preview.value = { ...preview.value, status: 'rendering', key, partial: null, error: null };
  let live: player.LiveStream | null = null;
  const parts: Float32Array[] = [];
  let n = 0;
  try {
    const r = await render(req, {
      signal: mine.signal,
      onChunk: (chunk, sr) => {
        if (mine.signal.aborted) return;
        if (!live && claimPlay()) live = player.stream(ID_A, sr);
        live?.push(chunk);
        parts.push(chunk);
        n += chunk.length;
        const partial = new Float32Array(n);
        let o = 0;
        for (const p of parts) {
          partial.set(p, o);
          o += p.length;
        }
        preview.value = { ...preview.value, partial };
      },
    });
    if (mine.signal.aborted) return;
    (live as player.LiveStream | null)?.end();
    preview.value = { status: 'ready', key, a: r, partial: null, error: null };
    void loadReference();
  } catch (e) {
    if (isAbort(e) || mine.signal.aborted) return;
    playNext = false;
    if (player.isPlaying(ID_A)) player.stop();
    preview.value = { ...preview.value, status: 'error', partial: null, error: errorText(e) };
  }
}

async function loadReference() {
  const req = reqB;
  if (!req) return;
  const key = requestKey(req);
  if (reference.value.key === key && (reference.value.take || reference.value.error)) return;
  reference.value = { key, take: cached(req) ?? null, error: null };
  if (reference.value.take) return;
  try {
    const take = await render(req);
    if (reference.value.key === key) reference.value = { key, take, error: null };
  } catch (e) {
    if (reference.value.key === key && !isAbort(e)) reference.value = { key, take: null, error: errorText(e) };
  }
}

/** Forget a failed B render and try it again. */
export function retryReference() {
  reference.value = { key: '', take: null, error: null };
  void loadReference();
}

function schedule(key: string) {
  clearTimeout(timer);
  const hit = cached(reqA!);
  if (hit) {
    ctrl?.abort();
    preview.value = { status: 'ready', key, a: hit, partial: null, error: null };
    playIfWanted(hit);
    void loadReference();
    return;
  }
  preview.value = { ...preview.value, status: 'waiting', key, error: null };
  const req = reqA!;
  timer = window.setTimeout(() => void run(req, key), DEBOUNCE_MS);
}

/** The design moved but nothing renders: the last take stays, out of date (a cached take is shown as is). */
function markStale(key: string) {
  clearTimeout(timer);
  ctrl?.abort();
  playNext = false;
  const hit = cached(reqA!);
  preview.value = hit ? { status: 'ready', key, a: hit, partial: null, error: null } : { ...preview.value, status: 'idle', key, partial: null, error: null };
}

/** Start following the design (call once after boot; what boot restored doesn't render). */
export function startPreview() {
  holding = true;
  wasAuto = autoRender.peek();
  effect(() => {
    const d = design.value;
    const text = phrase.value;
    const s = spec.value;
    const auto = autoRender.value;
    if (!d.baseId || !text) return;
    const problem = pronProblem(text, pronTokens.value);
    if (problem) {
      untracked(() => {
        reqA = reqB = null;
        clearTimeout(timer);
        ctrl?.abort();
        preview.value = { ...preview.value, status: 'error', key: '', partial: null, error: problem };
      });
      return;
    }
    const language = projectSettings.value.language;
    reqA = buildRequest(s, requestContext.value, d, text, PREVIEW_SEED, { language });
    reqB = buildRequest(s, requestContext.value, d, text, PREVIEW_SEED, { unsteered: true, language });
    untracked(() => {
      const key = requestKey(reqA!);
      const p = preview.value;
      const live = auto && !holding;
      const switchedOn = auto && !wasAuto;
      wasAuto = auto;
      // Same request: nothing to do, unless auto-generate was just turned on over a take that is out of date or failed.
      if (key === p.key && !(live && switchedOn && (p.status === 'idle' || p.status === 'error'))) return;
      // A genuinely new request shows (and, with auto-play, plays) the design again: B is a comparison, not a mode.
      if (key !== p.key && !holding) side.value = 'A';
      if (live) schedule(key);
      else markStale(key);
    });
  });
  holding = false;
}

/** Force a fresh render of the current settings (after an error). */
export function retry() {
  const d = design.value;
  const req = buildRequest(spec.value, requestContext.value, d, phrase.value, PREVIEW_SEED, { language: projectSettings.value.language });
  preview.value = { ...preview.value, key: '' };
  void run(req, requestKey(req));
}

export function playSide(which: Side) {
  if (which === 'A') return speak();
  side.value = 'B';
  const b = reference.value.take;
  if (b) player.play(ID_B, b.pcm, b.sr);
  else void loadReference().then(() => reference.value.take && side.value === 'B' && player.play(ID_B, reference.value.take.pcm, reference.value.take.sr));
}

export function togglePreview() {
  if (player.isPlaying(ID_A) || player.isPlaying(ID_B)) player.stop();
  else playSide(side.value);
}

/**
 * Hear the design say the current text: now if its take is ready, else rendered at once (no debounce) and played
 * when it lands, auto-play or not.
 */
export function speak() {
  side.value = 'A';
  const p = preview.value;
  if (p.status === 'ready' && p.a) return player.play(ID_A, p.a.pcm, p.a.sr);
  if (!reqA) {
    if (!design.value.baseId) notify('Pick a voice first');
    return;
  }
  playNext = true;
  if (p.status === 'rendering') return;
  clearTimeout(timer);
  void run(reqA, requestKey(reqA));
}
