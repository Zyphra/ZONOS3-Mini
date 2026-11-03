// What the API routes need from the loaded models (the worker builds it from the GPU classes; tests use stubs).
import type { Backbone, ProbeInput } from '../backbone';
import type { Conditioning, ModelHp } from '../prompt';
import type { LaciHeads } from '../request/laci';
import type { SteerBundle } from '../request/steer';
import type { Tokenizer } from '../tokenizer';

/** The backbone surface the generation loop drives (src/backbone.ts Backbone). */
export type BackboneApi = Pick<Backbone, 'steerSources'> & {
  prefill(slot: 0 | 1, ids: Int32Array, cond: Conditioning): Promise<{ logits: Float32Array }>;
  step(slot: 0 | 1, next: Int32Array, steer: Float32Array | null, probe: ProbeInput | null):
    Promise<{ logits: Float32Array; probe: Float32Array | null }>;
};

/** Streaming codec decoder (src/dac.ts DacStream). */
export interface CodecApi {
  readonly gain: number;
  reset(): void;
  chunk(idx: Int32Array, p0: number, n: number): Promise<Float32Array>;
  flush(): Float32Array;
}

/** A shipped default voice: its clip decoded to 44.1 kHz mono by the client and its shipped reference codes. */
export interface DefaultVoice {
  id: string;
  label: string;
  file: string;
  pcm: Float32Array | null;     // null when the client could not decode the clip
  codes: Int32Array | null;     // ordinary [T][10]
}

export interface Engine {
  hp: ModelHp;
  modelFile: string;                       // GET /v1/models root
  backbone: BackboneApi;
  codec: CodecApi;
  tokenizer: Tokenizer;
  steer: SteerBundle | null;               // companions/steer.gguf
  controlsDoc: unknown;                    // GET /v2/controls body (null without steer)
  laci: Map<string, LaciHeads>;            // language -> alignment heads (laci/index.json)
  voices: DefaultVoice[];
  headroomDb: number;                      // dac meta headroom_db
  /** Speaker encoder (lazy-loaded): 44.1 kHz mono -> [speakerDim]. Throws HttpError-compatible messages. */
  embedSpeaker(pcm44k: Float32Array): Promise<Float32Array>;
  /** Reference encoder (lazy-loaded): 44.1 kHz mono -> ordinary codes [T][10]. */
  encodeReference(pcm44k: Float32Array): Promise<Int32Array>;
}
