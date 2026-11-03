/**
 * The in-browser server (servers/web): ZONOS3 on WebGPU in a Web Worker, serving the v2 API through a
 * fetch-compatible transport. Its module (`dist/engine.js`, exporting `{engine, localFetch}`) loads at run time from
 * `?engine=URL` (else VITE_ZONOS_WEB_ENGINE, else /web-engine/engine.js) and is never bundled; it reads its own
 * `?models=URL`. Shared by the studio (`../api.ts`) and the lite page (`../lite/api.ts`).
 */

export interface EngineProgress {
  model: string;
  loaded: number;
  total: number;
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

/** Every request goes through here: the in-browser server once it is picked, else the network. */
export const transport: typeof fetch = (input, init) =>
  useWeb ? loadWebEngine().then((m) => m.localFetch(input, init)) : fetch(input, init);

const progressText = (p: EngineProgress) =>
  `Loading ${p.model} … ${Math.round((100 * p.loaded) / Math.max(1, p.total))}%`;

/** Picks the in-browser server and loads its models; `progress` gets a one-line readout as they download. */
export async function startWebEngine(progress: (text: string) => void): Promise<void> {
  useWeb = true;
  const { engine } = await loadWebEngine();
  await engine.init((p) => progress(progressText(p)));
}

/** A readout of the encoder models a clip voice's first use loads (only once the in-browser server is started). */
export async function onEncoderProgress(progress: (text: string) => void): Promise<void> {
  const { engine } = await loadWebEngine();
  engine.onEncoderProgress((p) => progress(progressText(p)));
}

/** `auto` when no ZONOS server answers: this browser has WebGPU and the site deploys the engine. */
export const webEngineAvailable = async (): Promise<boolean> =>
  'gpu' in navigator && (await fetch(engineUrl, { method: 'HEAD' }).then((r) => r.ok, () => false));
