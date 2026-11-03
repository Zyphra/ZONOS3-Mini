/**
 * Always-on preview: every design move re-renders the preview phrase (fixed seed) after a 250 ms pause,
 * streams it into the speakers, caches it by request, cancels stale renders, and keeps the unsteered render
 * of the same voice ready for A/B.
 */
import { computed, effect, signal, untracked } from '@preact/signals';
import type { GenerateRequest, Rendered } from '../api';
import { pronTokens, requestContext, spec } from '../state/app';
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

/** The project's preview line ('' = the bundle's). */
export const previewText = signal('');
export const phrase = computed(() => previewText.value.trim() || spec.value.ui.preview_text);

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
let reqB: GenerateRequest | null = null;

function playIfWanted(r: Rendered) {
  if (autoplay.value && side.value === 'A') player.play(ID_A, r.pcm, r.sr);
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
        if (!live && autoplay.value && side.value === 'A') live = player.stream(ID_A, sr);
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

function schedule(reqA: GenerateRequest, b: GenerateRequest) {
  reqB = b;
  const key = requestKey(reqA);
  if (key === preview.value.key && preview.value.status !== 'error') return;
  clearTimeout(timer);
  const hit = cached(reqA);
  if (hit) {
    ctrl?.abort();
    preview.value = { status: 'ready', key, a: hit, partial: null, error: null };
    playIfWanted(hit);
    void loadReference();
    return;
  }
  preview.value = { ...preview.value, status: 'waiting', key, error: null };
  timer = window.setTimeout(() => void run(reqA, key), DEBOUNCE_MS);
}

/** Start following the design (call once after boot). */
export function startPreview() {
  effect(() => {
    const d = design.value;
    const text = phrase.value;
    const s = spec.value;
    if (!d.baseId || !text) return;
    const problem = pronProblem(text, pronTokens.value);
    if (problem) {
      untracked(() => {
        clearTimeout(timer);
        ctrl?.abort();
        preview.value = { ...preview.value, status: 'error', key: '', partial: null, error: problem };
      });
      return;
    }
    const a = buildRequest(s, requestContext.value, d, text, PREVIEW_SEED);
    const b = buildRequest(s, requestContext.value, d, text, PREVIEW_SEED, { unsteered: true });
    untracked(() => schedule(a, b));
  });
}

/** Force a fresh render of the current settings (after an error). */
export function retry() {
  const d = design.value;
  const req = buildRequest(spec.value, requestContext.value, d, phrase.value, PREVIEW_SEED);
  preview.value = { ...preview.value, key: '' };
  void run(req, requestKey(req));
}

export function playSide(which: Side) {
  side.value = which;
  if (which === 'A') {
    const a = preview.value.a;
    if (a && preview.value.status === 'ready') player.play(ID_A, a.pcm, a.sr);
  } else {
    const b = reference.value.take;
    if (b) player.play(ID_B, b.pcm, b.sr);
    else void loadReference().then(() => reference.value.take && side.value === 'B' && player.play(ID_B, reference.value.take.pcm, reference.value.take.sr));
  }
}

export function togglePreview() {
  if (player.isPlaying(ID_A) || player.isPlaying(ID_B)) player.stop();
  else playSide(side.value);
}
