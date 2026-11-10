// What the API routes need from the loaded models (the worker builds it from the GPU classes; tests use stubs).
import type { AlignedWord } from '../aligner';
import type { Backbone, ProbeInput } from '../backbone';
import type { SamplingSetup } from '../gpu/sampler';
import type { Conditioning, ModelHp } from '../prompt';
import type { LaciHeads } from '../request/laci';
import type { SteerBundle } from '../request/steer';
import type { Tokenizer } from '../tokenizer';

/** The backbone surface the generation loop drives (src/backbone.ts Backbone). */
export type BackboneApi = Pick<Backbone, 'steerSources'> & {
  /** Concurrent calls on the two slots are allowed (each submits all of its GPU work before awaiting). */
  prefill(slot: 0 | 1, ids: Int32Array, cond: Conditioning): Promise<{ logits: Float32Array }>;
  /** Starts a request's sampling (random stream, repetition history, CFG scale: non-null = stepSample runs both slots). */
  startSampling(s: SamplingSetup): void;
  /** The frame at stream row `row` sampled from the last prefill(s)' logits: ids [NCB + 1], the next decode input. */
  sampleFirst(row: number): Promise<Int32Array>;
  /** Decodes the last sampled frame and samples the frame at stream row `row` (ids [NCB + 1]); submits before awaiting,
   *  so the next call may start before this one resolves. `twinSteer` is `steer` or null. */
  stepSample(row: number, steer: Float32Array | null, twinSteer: Float32Array | null, probe: ProbeInput | null):
    Promise<{ frame: Int32Array; probe: Float32Array | null }>;
};

/** Streaming codec decoder (src/zac.ts ZacStream). */
export interface CodecApi {
  readonly gain: number;
  /** Takes a residual code -1 as missing (early partial-frame decode; false: complete frames only). */
  readonly partialFrames: boolean;
  reset(): void;
  /** Submits its GPU work before returning; results resolve in call order. */
  chunk(idx: Int32Array, p0: number, n: number): Promise<Float32Array>;
  flush(): Promise<Float32Array>;
}

/** An audio decoder a request may pick with `decoder` (Capabilities.decoders). */
export interface Decoder {
  /** The decoder head variant (manifest meta `head.variant`): `keep2_plain` or `full_mr`. */
  id: string;
  /** The decoder GGUF the model directory was exported from. */
  checkpoint: string;
  /** Meta headroom_db: the output gain of a request without headroom_db. */
  headroomDb: number;
  /** The streaming decoder: loaded at startup for the default, on the first request that picks it otherwise (its
   *  download reported like the encoders'). */
  codec(): Promise<CodecApi>;
}

/** A shipped default voice: its clip decoded to 44.1 kHz mono by the client, its shipped reference codes and its
 *  shipped speaker embedding (so neither encoder loads for a default voice). */
export interface DefaultVoice {
  id: string;
  label: string;
  file: string;
  pcm: Float32Array | null;     // null when the client could not decode the clip
  codes: Int32Array | null;     // ordinary [T][10]
  embedding: Float32Array | null; // [speakerDim]
}

/** Speech editing (/v2/edit/align, /v2/edit): the speech-edit LM (the speech-edit adapter of model directory `edit` on
 *  the release backbone's weights) and FuseAlign (model directory `aligner`), each loaded on first use. */
export interface EditModels {
  /** The speech-edit LM's hyper-parameters (the `edit` manifest meta; nothing is loaded for them). */
  hp: ModelHp;
  /** The speech-edit LM (its adapter lazy-loaded), steered by the same bundle as the release backbone. */
  backbone(): Promise<BackboneApi>;
  /** FuseAlign (lazy-loaded): word intervals of `tokens` in 16 kHz mono audio (src/aligner Aligner.align; HttpError 400
   *  past its limits). */
  align(pcm16k: Float32Array, tokens: string[]): Promise<AlignedWord[]>;
}

export interface Engine {
  hp: ModelHp;
  modelFile: string;                       // GET /v1/models root
  backbone: BackboneApi;
  /** Every decoder a request may pick, the default (the one a request without `decoder` gets) first. */
  decoders: Decoder[];
  tokenizer: Tokenizer;
  steer: SteerBundle | null;               // companions/steer.gguf
  controlsDoc: unknown;                    // GET /v2/controls body (null without steer)
  laci: Map<string, LaciHeads>;            // language -> alignment heads (laci/index.json)
  voices: DefaultVoice[];
  /** Speaker encoder (lazy-loaded): 44.1 kHz mono -> [speakerDim]. Throws HttpError-compatible messages. */
  embedSpeaker(pcm44k: Float32Array): Promise<Float32Array>;
  /** Reference encoder (lazy-loaded): 44.1 kHz mono -> ordinary codes [T][10]. */
  encodeReference(pcm44k: Float32Array): Promise<Int32Array>;
  /** Speech editing; null when the model export has no `edit` backbone or no `aligner`. */
  edit: EditModels | null;
}
