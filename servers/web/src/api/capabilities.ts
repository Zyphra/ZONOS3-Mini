// GET /v2/capabilities of the in-browser server (servers/ggml/src/server/capabilities.cpp, edit.cpp edit_capabilities,
// phonemize.cpp phoneme_capabilities, zonos3.server.capabilities text_marker_capabilities) with this server's feature
// set: no text normalization, speech recognition, phonemizer, server projects or plugins; speech editing with the
// clip's transcript when the export has the speech-edit backbone and FuseAlign.
import { ALIGN_MAX_SECONDS } from '../aligner/host';
import languages from '../../../../model/src/zonos3/registries/languages.json';
import {
  DECODE_CONTEXT_FRAMES, EDIT_MAX_SECONDS, HOP, MAX_FRAMES, SAMPLE_RATE, SILENCE_PREFILL_DEFAULT_FRAMES,
} from '../generation';
import { releaseDocument } from '../request/steer';
import { markerRows } from '../tokenizer';
import type { Decoder, Engine } from './engine';

const AUDIO_CODEC = 'opus_dual_3n_600k_sched600k_r4096_step600000';
const SPEAKER_EMBEDDING_CODEC = 'ownmect_s2';
/** The server language codes (zonos3.languages SERVER), in order. */
export const TEXT_NORM_LANGUAGES = languages.server.map((l) => l.code);

/** DecoderInfo of a decoder (servers/ggml/src/server/capabilities.cpp). */
function decoderInfo(d: Decoder): Record<string, unknown> {
  return {
    id: d.id, kind: 'istft_mr', label: `Multi-resolution iSTFT head (${d.checkpoint}); causal`, checkpoint: d.checkpoint,
    expect_step: null, headroom_db: d.headroomDb, causal: true, codec: AUDIO_CODEC,
  };
}

/** float_decimal: a float32 metadata value as the shortest decimal that round-trips it. */
export function floatDecimal(v: number): number {
  const f = Math.fround(v);
  for (let p = 1; p < 10; p++) {
    const d = Number(f.toPrecision(p));
    if (Math.fround(d) === f) return d;
  }
  return f;
}

/** The silence-prefill default (configure_silence_prefill with the registry default, on). */
export const SILENCE_PREFILL = { frames: SILENCE_PREFILL_DEFAULT_FRAMES, on: true };

export function laciLanguages(e: Engine): string[] {
  return [...e.laci.keys()].sort();
}

/** Accent adapters a request can pick with `accent` (`accents.gguf`): the web export ships none (its one adapter, the
 *  speech-edit LoRA of `edit/`, serves edit fills only). */
export const ACCENT_ADAPTERS: string[] = [];

/** Edit regions per request: the speech-edit backbone's marker rows (zonos3.tts.edit edit_marker_rows, ggml
 *  zonos3_edit_marker_rows): the extra text rows before the tokenizer's phoneme / start / pause rows (modelHp checked
 *  that those rows end the extra rows), all of them without any. */
export function editMaxSpans(e: Engine): number {
  if (!e.edit) return 0;
  const hp = e.edit.hp;
  return hp.textExtraTokens - markerRows(hp.textMarkers);
}

function conditionTokens(e: Engine): unknown {
  const hp = e.hp;
  if (hp.nCond === 0) return null;
  const edges: Record<string, number[]> = {}, units: Record<string, string> = {}, counts: Record<string, number> = {};
  let layout = '[reference prefix | speaker row]';
  hp.condNames.forEach((name, i) => {
    edges[name] = (hp.condEdges[i] ?? []).map(floatDecimal);
    counts[name] = hp.condBucketCounts[i];
    if (i < hp.condUnits.length) units[name] = hp.condUnits[i];
    layout += `[${name}]`;
  });
  return {
    enabled: true, names: hp.condNames, flags: {}, count: hp.nCond, bucket_counts: counts,
    unknown_ids: { ...counts }, edges, units, layout: layout + ' BOS text EOS',
  };
}

export function capabilities(e: Engine, speakerModes: string[]): Record<string, unknown> {
  const hp = e.hp, m = hp.textMarkers;
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
      default_seconds: Math.round(frames * HOP / SAMPLE_RATE * 1000) / 1000, frame_seconds: HOP / SAMPLE_RATE,
      codec: AUDIO_CODEC,
      note: 'codec-matched silence frames open the audio stream before the model starts (their audio is not ' +
        'emitted); a request without silence_prefill_frames gets the server default, silence_prefill_frames=0 ' +
        'turns it off and silence_prefill_frames=N sets the length',
    },
    decode_context: {
      available: true, default_frames: DECODE_CONTEXT_FRAMES, max_frames: 450,
      note: 'decode the generated frames behind the last decode_context_frames of the raw-reference prompt ' +
        "(their audio dropped), so the first generated frame is not the decoder's position 0; " +
        'no-op without a reference prompt',
    },
    speaker_modes: speakerModes,
    speaker_mode_default: speakerModes.length ? speakerModes[0] : null,
    decoder: decoderInfo(e.decoders[0]),
    decoders: e.decoders.map(decoderInfo),
    condition_tokens: conditionTokens(e),
    speaker_audio_upload: speakerModes.length > 0,
    speaker_cfg_supported: cfg,
    speaker_cfg_modes: cfg ? speakerModes : [],
    speaker_cfg_unconditional: 'remove_reference_prefix_or_speaker_row_condition_rows_unknown_keep_text_and_generated_history',
    speaker_cfg_formula: 'uncond + scale * (cond - uncond)',
    laci_languages: laciLanguages(e),
    plugins: [],
    speaker_embedding_upload: embedding,
    speaker_embedding_cache: speakerModes.length > 0,
    speaker_embedding_blend: embedding,
    default_voices_enabled: true,
    projects: { enabled: false },
    // Steer then hold: the defaults of steer_hold / steer_onset_seconds / steer_ramp_seconds and the directions that stay
    // constant (the bundle's steer_release); null = request-level controls are constant.
    steer_release: releaseDocument(e.steer),
    speech_editing: e.edit !== null && editMaxSpans(e) > 0,
    edit_transcribe: false,
    edit_max_spans: editMaxSpans(e),
    edit_frame_rate: e.edit ? SAMPLE_RATE / HOP : null,
    edit_max_seconds: EDIT_MAX_SECONDS,
    edit_align_max_seconds: ALIGN_MAX_SECONDS,
    phoneme_tokens: m.phonemes ? { open_marker: m.phonemes.open, close_marker: m.phonemes.close, symbols: m.phonemes.symbols } : null,
    start_token: m.start ? { marker: m.start.marker } : null,
    pause_tokens: m.pauses ? { markers: m.pauses.markers, seconds: m.pauses.seconds } : null,
    phonemize_enabled: false,
    phonemize_languages: [],
    phonemize_guess_languages: [],
  };
}
