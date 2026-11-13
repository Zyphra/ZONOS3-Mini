/** What the server offers, cached for 30 s: capabilities, the steering spec, and the request context. */
import type { ControlsSpec, GenerateRequest } from '../../src/api';
import fixtureSpec from '../../src/fixtures/controls.json';
import { requestDecoder } from '../../src/lib/decoders';
import type { ProjectDoc } from '../../src/state/projectSchema';
import { DEFAULT_SETTINGS } from '../../src/state/projectSchema';
import type { RequestContext } from '../../src/state/requests';
import * as zonos from './zonos';

const TTL_MS = 30_000;

interface Cached<T> {
  at: number;
  value: Promise<T>;
}

function cached<T>(load: () => Promise<T>): () => Promise<T> {
  let c: Cached<T> | null = null;
  return () => {
    if (!c || Date.now() - c.at > TTL_MS) {
      const value = load();
      c = { at: Date.now(), value };
      // A failed load is not kept.
      value.catch(() => (c = null));
    }
    return c.value;
  };
}

export const caps = cached(zonos.capabilities);

export interface SpecInfo {
  spec: ControlsSpec;
  /** The server applies `controls` / `control_spans`; false: the fixture spec only names the controls. */
  steering: boolean;
}

/** The server's steering bundle, else the studio's fixture spec (what the studio's fixture backend does). */
export const spec = cached(async (): Promise<SpecInfo> => {
  const live = await zonos.controls();
  return live ? { spec: live, steering: true } : { spec: fixtureSpec as unknown as ControlsSpec, steering: false };
});

export const NO_STEERING =
  'This server has no steering bundle (start it with STEER_BUNDLE=assets/steering/studio_bundle_mect60k_x10.json, or the MLX server with steer.gguf next to the backbone).';

export function requestContext(c: zonos.Capabilities, doc: ProjectDoc | null): RequestContext {
  return {
    phonemeTokens: c.phoneme_tokens ?? null,
    // The project's decoder (set in the Studio's project settings) where this server serves it, else the default.
    decoder: requestDecoder(c.decoders ?? [], doc?.settings.decoder),
    // The project's LACI mode (set in the Studio's project settings); languages without heads here run without it.
    laci: doc?.settings.laci ?? DEFAULT_SETTINGS.laci,
    laciLanguages: c.laci_languages,
    startToken: c.start_token != null,
  };
}

/** The body as sent: without steering fields the server can't apply (the stored request keeps them, as Studio's). */
export function sendable(req: GenerateRequest, steering: boolean): GenerateRequest {
  if (steering) return req;
  const { controls: _c, control_spans: _s, control_span_follow: _f, ...rest } = req;
  return rest;
}

