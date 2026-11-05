/**
 * The in-browser server (servers/web): ZONOS3 on WebGPU in a Web Worker, serving the v2 API through a
 * fetch-compatible transport. Its module (`dist/engine.js`, exporting `{engine, localFetch}`) loads at run time from
 * `?engine=URL` (else VITE_ZONOS_WEB_ENGINE, else /web-engine/engine.js) and is never bundled; it reads its own
 * `?models=URL`. Shared by the studio (`../api.ts`) and the lite page (`../lite/api.ts`).
 */

import { withZeroGpu } from './zerogpu';

/** The engine's model download readout (`servers/web/src/gpu/weights.ts` LoadProgress). */
export interface EngineProgress {
  models: string[];
  loaded: number;
  total: number;
  /** Bytes that came over the network (the rest from Cache Storage). */
  fetched: number;
}

interface WebEngine {
  init(progress: (p: EngineProgress) => void, opts?: { base?: string; cache?: boolean }): Promise<void>;
  /** Progress of the models loaded after init, on a clip voice's first use (speaker / reference encoders). */
  onEncoderProgress(cb: (p: EngineProgress) => void): void;
}

interface WebEngineModule {
  engine: WebEngine;
  localFetch: typeof fetch;
}

export const engineUrl =
  new URLSearchParams(location.search).get('engine') || import.meta.env.VITE_ZONOS_WEB_ENGINE || '/web-engine/engine.js';

let engineModule: Promise<WebEngineModule> | null = null;
const loadWebEngine = (): Promise<WebEngineModule> => (engineModule ??= import(/* @vite-ignore */ engineUrl));

let useWeb = false;

/** Whether the in-browser server serves this tab's requests. */
export const usingWebEngine = () => useWeb;

/** Every later request goes to the in-browser server (its module loads on first use). */
export function selectWebEngine(): void {
  useWeb = true;
}

/**
 * Every request goes through here: the in-browser server once it is picked, else the network (GPU routes with the
 * visitor's ZeroGPU headers when a huggingface.co page frames a Space: `./zerogpu.ts`).
 */
export const transport: typeof fetch = (input, init) =>
  useWeb ? loadWebEngine().then((m) => m.localFetch(input, init)) : withZeroGpu(input, init).then((i) => fetch(input, i));

/** A model download as a bar and a line: what loads, how far, and (once the network rate is known) how long is left. */
export interface Download {
  /** e.g. "Downloading backbone + zac" or "Loading backbone + zac" (from this browser's cache). */
  label: string;
  /** 0…1 */
  fraction: number;
  /** "420 MB of 1.9 GB · 18 MB/s · about 1 min 20 s left" */
  detail: string;
}

const MB = 1e6;
const size = (n: number) => (n >= 1e9 ? `${(n / 1e9).toFixed(1)} GB` : `${Math.max(1, Math.round(n / MB))} MB`);

/** The time left, phrased to follow "about" (or standing alone under 5 s). */
function timeLeft(s: number): string {
  if (s < 5) return 'a few seconds left';
  if (s < 60) return `about ${Math.ceil(s / 5) * 5} s left`;
  const m = Math.floor(s / 60);
  const r = Math.round((s % 60) / 10) * 10;
  return `about ${r && m < 10 ? `${m} min ${r} s` : `${Math.round(s / 60)} min`} left`;
}

/**
 * Readouts → Downloads. The rate is the network bytes over a sliding 8 s window (cached bytes arrive at disk speed
 * and would make the ETA lie), shown after 2 s of data; the ETA assumes the rest comes over the network.
 */
function tracker(): (p: EngineProgress) => Download {
  let key = '';
  let samples: { t: number; fetched: number }[] = [];
  return (p) => {
    const k = p.models.join('+');
    if (k !== key) {
      key = k;
      samples = [];
    }
    const now = performance.now();
    samples.push({ t: now, fetched: p.fetched });
    while (samples.length > 2 && now - samples[1].t > 8000) samples.shift();
    const span = (now - samples[0].t) / 1000;
    const rate = span >= 2 ? (p.fetched - samples[0].fetched) / span : 0;
    const left = Math.max(0, p.total - p.loaded);
    const parts = [`${size(p.loaded)} of ${size(p.total)}`];
    if (rate > 0.05 * MB && left > 0) parts.push(`${(rate / MB).toFixed(rate >= 10 * MB ? 0 : 1)} MB/s`, timeLeft(left / rate));
    return {
      label: `${p.fetched > 0 ? 'Downloading' : 'Loading'} ${p.models.join(' + ')}`,
      fraction: p.total ? Math.min(1, p.loaded / p.total) : 0,
      detail: parts.join(' · '),
    };
  };
}

/** Picks the in-browser server and loads its models; `progress` gets the download as it goes. */
export async function startWebEngine(progress: (d: Download) => void): Promise<void> {
  useWeb = true;
  const { engine } = await loadWebEngine();
  const track = tracker();
  await engine.init((p) => progress(track(p)));
}

/** The download of the encoder models a clip voice's first use loads (only once the in-browser server is started). */
export async function onEncoderProgress(progress: (d: Download) => void): Promise<void> {
  const { engine } = await loadWebEngine();
  const track = tracker();
  engine.onEncoderProgress((p) => progress(track(p)));
}

/** `auto` when no ZONOS server answers: this browser has WebGPU and the site deploys the engine. */
export const webEngineAvailable = async (): Promise<boolean> =>
  'gpu' in navigator && (await fetch(engineUrl, { method: 'HEAD' }).then((r) => r.ok, () => false));
