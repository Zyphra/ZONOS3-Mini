// GET /v2/capabilities and GET /v2/dub/capabilities of the in-browser server (servers/ggml/src/server/capabilities.cpp,
// edit.cpp edit_capabilities, phonemize.cpp phoneme_capabilities, dub.cpp DubRuntime::capabilities) with this server's
// feature set: no text normalization, speech editing, dubbing, phonemizer or server projects.
import languages from '../../../../model/src/zonos3/registries/languages.json';
import { NONVERBAL_MARKUP_TAGS } from '../request/markup';
import type { Engine } from './engine';

const AUDIO_CODEC = 'opus_dual_3n_600k_sched600k_r4096_step600000';
const SPEAKER_EMBEDDING_CODEC = 'ownmect_s2';
const HOP = 2940;
/** The server language codes (zonos3.languages SERVER), in order. */
export const TEXT_NORM_LANGUAGES = languages.server.map((l) => l.code);
export const MAX_FRAMES = 2000;
export const EDIT_ALIGN_MAX_SECONDS = 150;
const DECODER_CHECKPOINT = 'dac.gguf';

/** float_decimal: a float32 metadata value as the shortest decimal that round-trips it. */
export function floatDecimal(v: number): number {
  const f = Math.fround(v);
  for (let p = 1; p < 10; p++) {
    const d = Number(f.toPrecision(p));
    if (Math.fround(d) === f) return d;
  }
  return f;
}

/** The silence-prefill default (configure_silence_prefill with the defaults: 3 frames, on). */
export const SILENCE_PREFILL = { frames: 3, on: true };

export function laciLanguages(e: Engine): string[] {
  return [...e.laci.keys()].sort();
}

/** Accent adapters a request can pick with `accent` (`accents.gguf`): the web export ships none. */
export const ACCENT_ADAPTERS: string[] = [];

function conditionTokens(e: Engine): unknown {
  const hp = e.hp;
  if (hp.nCond === 0) return null;
  const edges: Record<string, number[]> = {}, units: Record<string, string> = {};
  let layout = '[reference prefix | speaker row]';
  hp.condNames.forEach((name, i) => {
    edges[name] = (hp.condEdges[i] ?? []).map(floatDecimal);
    if (i < hp.condUnits.length) units[name] = hp.condUnits[i];
    layout += `[${name}]`;
  });
  return {
    enabled: true, names: hp.condNames, flags: {}, count: hp.nCond, n_buckets: hp.nCondBuckets,
    unknown_index: hp.nCondBuckets, edges, units, layout: layout + ' BOS text EOS',
  };
}

export function capabilities(e: Engine, speakerModes: string[]): Record<string, unknown> {
  const hp = e.hp;
  const embedding = hp.speakerDim > 0;
  const cfg = speakerModes.length > 0;   // one request at a time: the CFG twin always has the backbone's second slot
  const frames = SILENCE_PREFILL.frames;
  return {
    text_tokenizer_spec: JSON.parse(hp.textTokenizerSpec),
    text_normalization_enabled: false,
    text_norm_languages: TEXT_NORM_LANGUAGES,
    accents: ACCENT_ADAPTERS,
    speaker_enabled: speakerModes.length > 0,
    speaker_conditioning_type: 'raw_reference_prefix',
    speaker_embedding_dim: embedding ? hp.speakerDim : null,
    speaker_embedding_codec: embedding ? SPEAKER_EMBEDDING_CODEC : null,
    n_codebooks: hp.nCodebooks,
    codebook_sizes: new Array(hp.nCodebooks).fill(4096),
    audio_codec: AUDIO_CODEC,
    max_tokens: MAX_FRAMES,
    stream_decode_modes: ['prefix', 'whole'],
    stream_decode_default: 'prefix',
    silence_prefill: {
      available: true, default_frames: frames, default_on: SILENCE_PREFILL.on,
      default_seconds: Math.round(frames * HOP / 44100 * 1000) / 1000, frame_seconds: HOP / 44100, codec: AUDIO_CODEC,
      note: 'codec-matched silence frames open the audio stream before the model starts (their audio is not ' +
        'emitted); a request without silence_prefill_frames gets the server default, silence_prefill_frames=0 ' +
        'turns it off and silence_prefill_frames=N sets the length',
    },
    tag_cfg: {
      available: true, trained: hp.tagCfgTrained, tags: NONVERBAL_MARKUP_TAGS,
      min_scale: 1.0, max_scale: 4.0, default_scale: 1.0,
    },
    decode_context: {
      available: true, default_frames: 30, max_frames: 450,
      note: 'decode the generated frames behind the last decode_context_frames of the raw-reference prompt ' +
        "(their audio dropped), so the first generated frame is not the decoder's position 0; " +
        'no-op without a reference prompt',
    },
    speaker_modes: speakerModes,
    speaker_mode_default: speakerModes.length ? speakerModes[0] : null,
    decoder: {
      kind: 'istft_mr', label: `Multi-resolution iSTFT head (${DECODER_CHECKPOINT}); causal`, checkpoint: DECODER_CHECKPOINT,
      expect_step: null, headroom_db: e.headroomDb, causal: true, codec: AUDIO_CODEC,
    },
    condition_tokens: conditionTokens(e),
    speaker_audio_upload: speakerModes.length > 0,
    speaker_cfg_supported: cfg,
    speaker_cfg_modes: cfg ? speakerModes : [],
    speaker_cfg_unconditional: 'remove_reference_prefix_or_speaker_row_condition_rows_unknown_keep_text_and_generated_history',
    speaker_cfg_formula: 'uncond + scale * (cond - uncond)',
    laci_languages: laciLanguages(e),
    dub: false,
    speaker_embedding_upload: embedding,
    speaker_embedding_cache: speakerModes.length > 0,
    speaker_embedding_blend: embedding,
    default_voices_enabled: true,
    projects: { enabled: false },
    speech_editing: false,
    edit_max_spans: 0,
    edit_frame_rate: null,
    edit_max_seconds: EDIT_ALIGN_MAX_SECONDS,
    edit_align_max_seconds: EDIT_ALIGN_MAX_SECONDS,
    phoneme_tokens: null,
    phonemize_enabled: false,
    phonemize_languages: [],
    phonemize_guess_languages: [],
  };
}

/** Dub reference languages (zonos3.languages SPOKEN): code → name. */
const REF_LANGUAGES = languages.spoken.map(({ code, name }) => ({ code, name }));

/** /v2/dub/capabilities with the dub routes off (DubRuntime::capabilities; no alignment worker in a browser). */
export function dubCapabilities(e: Engine): Record<string, unknown> {
  const hp = e.hp;
  const rate = hp.condNames.indexOf('speaking_rate');
  return {
    available: false,
    ref_languages: REF_LANGUAGES,
    target_languages: [],
    max_ref_seconds: 180.0,
    max_video_bytes: 2 ** 30,
    max_segments: 120,
    default_min_pause_s: 0.30,
    min_pause_range: [0.10, 2.00],
    sample_rate: 44100,
    frame_rate: 15.0,
    rate_edges: rate >= 0 ? (hp.condEdges[rate] ?? []).map(floatDecimal) : [],
    n_buckets: hp.nCondBuckets,
    buckets: [],
    candidate_bucket_offsets: [0, -1, 1, -2, 2],
    candidate_seeds: 2,
    model: e.modelFile,
    asr_model: 'Qwen/Qwen3-ASR-1.7B',
    aligner_model: '',   // FuseAlign's checkpoint; empty when its worker cannot start (no worker in a browser)
    word_aligner: 'google-bert/bert-base-multilingual-cased',
    align_worker: { installed: false, running: false },
    lipsync: false,
    voice_modes: [],
    segment_controls: true,
    phoneme_groups: false,
    ref_words: true,
    enabled: false,
  };
}
