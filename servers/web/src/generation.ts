// The generation registry (model/src/zonos3/registries/generation.json, Python zonos3.generation): the request defaults
// and limits every server shares. The ggml server embeds the same file at build time; servers/api/openapi.yaml repeats the
// sampling defaults.
import registry from '../../../model/src/zonos3/registries/generation.json';

/** The API's audio rate (Hz): generated audio, uploads and default-voice clips. */
export const SAMPLE_RATE = registry.sample_rate;
/** The v2 sampling defaults (temperature, topk, top_p, min_p, repetition_penalty / _window / _codebooks). */
export const SAMPLING: Readonly<typeof registry.sampling> = Object.freeze({ ...registry.sampling });
/** The silence prefill a request without silence_prefill_frames gets, and the longest a request may ask for. */
export const SILENCE_PREFILL_DEFAULT_FRAMES = registry.silence_prefill_frames.default;
export const SILENCE_PREFILL_MAX_FRAMES = registry.silence_prefill_frames.max;
/** The reference tail decoded ahead of the stream when decode_context_frames is unset or 0. */
export const DECODE_CONTEXT_FRAMES = registry.decode_context_frames;
/** The reference prefix limit: longer reference codes keep their first REFERENCE_MAX_FRAMES frames. */
export const REFERENCE_MAX_FRAMES = registry.reference_max_frames;
/** The frame budget of a request (its max_tokens is capped here). */
export const MAX_FRAMES = registry.max_frames;
/** Early partial-frame prefix decode: a request without stream_codebooks / stream_codebooks_scope decodes a frame once
 *  this many of its codebooks arrived, on the first prefix chunk only ('first') or every chunk ('all'). */
export const STREAM_CODEBOOKS = registry.stream_codebooks.default;
export const STREAM_CODEBOOKS_SCOPE = registry.stream_codebooks.scope as 'first' | 'all';
/** Samples per codec frame (44.1 kHz). */
export const HOP = 2940;
/** Rows per slot: native context_capacity = min(context_length, max_frames + 4096). */
export const CONTEXT_CAPACITY = MAX_FRAMES + 4096;
/** Text rows the alignment probe reads: probe_max_text = min(2048, context_capacity) (sgl --align-probe-max-text). */
export const ALIGN_PROBE_MAX_TEXT = 2048;
/** The longest reference clip (seconds) the reference and speaker encoders take. */
export const REFERENCE_AUDIO_MAX_SECONDS = registry.audio_max_seconds.reference;
/** The longest speech-edit source (seconds) the capabilities advertise. */
export const EDIT_MAX_SECONDS = registry.audio_max_seconds.edit;
/** The knee (dBFS) of the soft limiter after the decoder's output gain (output.ts outputPcm). */
export const OUTPUT_LIMIT_DBFS = registry.output_limit_dbfs;
