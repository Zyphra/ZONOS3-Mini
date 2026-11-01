/**
 * The only module that talks to the ZONOS3 server. Everything else sees the contract types below.
 *
 * Backends (picked once by `connect()`):
 *  - live:    the server serves GET /tts/controls; requests carry `controls` / `control_spans`.
 *  - fixture: the server answers but has no /tts/controls (404): the bundled fixture describes the controls,
 *             speakers / previews / renders are real, and steering fields are stripped (the server can't apply them).
 *  - offline: no server answers: fixture controls, fixture voices, renders from a small client-side synthesizer
 *             that applies the contract's steering math so every control is audible.
 * Force one with `?api=live|fixture|offline` (or VITE_ZONOS_API); point at another server with `?server=URL`.
 */
import fixtureSpec from './fixtures/controls.json';
import { fixtureSpeakers, syntheticEmbedding } from './fixtures/embeddings';
import { synthesizeWords } from './fixtures/synth';
import * as fixtureEdit from './fixtures/edit';
import * as fixturePron from './fixtures/phonemize';
import * as fixtureDub from './fixtures/dub';
import * as fixtureProjects from './fixtures/projects';
import * as fixtureNv from './fixtures/nv';
import fixturePhonemes from './fixtures/phonemes.json';
import type { PronTokens } from './lib/pron';
import { encodeNpy, encodeWav, bufferToDataUrl, blobToDataUrl, slug } from './lib/files';

// ── Contract types ────────────────────────────────────────────────────────────────────────────────

export type SpeakerMode = 'embedding' | 'reference';
/** Primitive direction id (`set.direction`) → raw dose. */
export type Dose = Record<string, number>;

export interface SetSpec {
  space: string;
  source_index?: number;
  modes: SpeakerMode[];
}

export interface DirectionSpec {
  set: string;
  space: string;
  kind: string;
  norm: number;
  vector?: number[];
}

export interface ControlSpec {
  id: string;
  group: string;
  label: string;
  type: 'bipolar' | 'unipolar';
  poles?: [string, string];
  map: { '-'?: Dose; '+': Dose };
  range: [number, number];
  safe: [number, number];
  /** [lo, hi]: v ≤ lo or v ≥ hi collapses renders; either side may be null. */
  collapse: [number | null, number | null] | null;
  spannable: boolean;
  /** Measured identity kept vs |v| (piecewise linear); null = not measured. */
  identity: [number, number][] | null;
  effect?: string;
}

export interface AccentSpec {
  id: string;
  label: string;
  tab: string;
  region: string;
  strength: string;
  map: Dose;
  range: [number, number];
  safe: [number, number];
  identity: [number, number][] | null;
  effect?: string;
}

/** Duration multiplier vs dose, per primitive direction (piecewise linear, clamped at the ends). */
export type Timing = Record<string, [number, number][]>;

export interface ControlsSpec {
  bundle: string;
  embedding_dim: number;
  sets: Record<string, SetSpec>;
  directions: Record<string, DirectionSpec>;
  timing: Timing;
  ui: {
    groups: { id: string; label: string; hue?: string }[];
    controls: ControlSpec[];
    accents: AccentSpec[];
    presets: Record<string, Record<string, number>>;
    preview_text: string;
    request_defaults: Partial<GenerateRequest>;
    /** Identity the Who-map ring encloses. */
    identity_threshold: number;
    /** Perceptual Who-map calibration and the measured positions of the library voices (by label). */
    who?: WhoSpec;
    /** Named moods with their measured place on the valence / arousal pad at their default dose. */
    moods?: MoodSpec[];
  };
}

/**
 * A named mood (`ui.moods`): `control` is its unipolar mood control; `valence` / `arousal` its MEASURED shift vs
 * unsteered at `dose`, on the pad's −1…1 scale.
 */
export interface MoodSpec {
  id: string;
  label: string;
  control: string;
  valence: number;
  arousal: number;
  dose: number;
  effect?: string;
}

export interface WhoSpec {
  gender: { control: string; logit_per_dose: number; clip_logit: number };
  age: { control: string; center_years: number; years_per_dose: number };
  landmarks: Record<string, { female: number; age: number }>;
  note?: string;
}

export interface Speaker {
  id: string;
  label: string;
  is_default: boolean;
  source_type?: string;
  scope?: string;
  has_preview?: boolean;
  /** Known text of the reference clip (a `<voice>.txt` next to it), for speech editing. */
  transcript?: string | null;
}

/** Voices cached from an embedding file have no reference clip: no Exact clone, no reference preview. */
export const embeddingOnly = (s: Speaker | null | undefined) => !!s?.source_type?.startsWith('embedding');

export interface ControlSpan {
  start: number;
  end: number;
  controls: Dose;
}

export interface GenerateRequest {
  text: string;
  speaker_embedding_id?: string | null;
  seed?: number | null;
  sr_out?: number;
  stream?: boolean;
  speaker_mode?: SpeakerMode;
  speaker_cfg_scale?: number;
  topk?: number;
  stream_decode?: string;
  controls?: Dose;
  control_spans?: ControlSpan[];
  /** Frames per character of `text` (measured from an earlier take) for placing spans by estimate. */
  control_span_rate?: number;
  /** "reading": spans follow the model's reading position (alignment heads of `language`), else the rate estimate. */
  control_span_follow?: SpanFollow;
  /** Server language code (`en_us` when absent): text normalization and, for spans, the alignment heads. */
  language?: string;
  /** Tag guidance (1–4): a markup-free twin mixed in to strengthen NV tags; > 1 needs a tag and no speaker CFG. */
  tag_cfg_scale?: number;
}

export type SpanFollow = 'rate' | 'reading';

export interface Rendered {
  pcm: Float32Array;
  sr: number;
  ttfbMs: number;
  totalMs: number;
  /** How the server placed `control_spans` (X-Control-Span-Follow); absent without spans. */
  spanFollow?: SpanFollow;
}

export type BackendKind = 'live' | 'fixture' | 'offline';

export interface Backend {
  kind: BackendKind;
  server: string;
  spec: ControlsSpec;
  /** The server applies `controls` / `control_spans` (live), or the offline synth emulates them. */
  steering: boolean;
  edit: EditCaps;
  pron: PronCaps;
  /** Dubbing (`/dub/capabilities`); null when the server has no dub setup. */
  dub: DubCaps | null;
  /** Shared projects on the server (`/tts/capabilities.projects`); null: projects stay in this browser. */
  projects: ProjectCaps | null;
  /** Nonverbal markup + tag guidance (`/tts/capabilities.tag_cfg`); null: tags are plain text to this server. */
  tags: TagCaps | null;
}

/** `/tts/capabilities.tag_cfg`; `fixture`: renders run on the client-side synth, which voices a few tags. */
export interface TagCaps {
  tags: string[];
  min_scale: number;
  max_scale: number;
  default_scale: number;
  /** Whether the served checkpoint was trained on the tags (null: the server doesn't say). */
  trained: boolean | null;
  fixture: boolean;
}

/** `/tts/capabilities.projects`; `fixture`: /projects and /media run on the client-side fixture store. */
export interface ProjectCaps {
  max_media_bytes: number;
  max_project_bytes: number;
  fixture: boolean;
}

/** A project revision: opaque, echoed back as `base_revision`. */
export type Revision = number | string;

export interface ServerProjectEntry {
  id: string;
  name: string;
  /** Epoch seconds, epoch ms or ISO text. */
  updated_at: number | string;
  revision: Revision;
  bytes: number;
  media_bytes: number;
  summary?: unknown;
}

export interface ServerProject {
  id: string;
  name: string;
  revision: Revision;
  updated_at: number | string;
  data: unknown;
}


/** Pronunciation input (`/tts/capabilities`); `fixture`: /tts/phonemize runs on the client-side fixture. */
export interface PronCaps {
  phoneme_tokens: PronTokens | null;
  phonemize_enabled: boolean;
  phonemize_languages: string[];
  phonemize_guess_languages: string[];
  fixture: boolean;
}

export type PhonemeStatus = 'ok' | 'guessed' | 'ambiguous' | 'unknown' | 'skipped' | 'unsupported';

export interface PhonemizeResponse {
  language: string;
  /** Concatenated texts give back the input; words carry a status (and IPA when ok / guessed). */
  segments: { text: string; status: PhonemeStatus | null; ipa?: string | null }[];
}

/** `/tts/capabilities` speech-editing fields; `fixture`: edits run on the client-side fixture. */
export interface EditCaps {
  speech_editing: boolean;
  edit_max_spans: number;
  edit_frame_rate: number;
  edit_max_seconds: number;
  edit_align_max_seconds: number;
  fixture: boolean;
}

export interface EditWord {
  word: string;
  start: number;
  end: number;
  confidence: number;
  tokens: string[];
}

/** `/tts/edit/transcribe` and `/tts/edit/align`. */
export interface Alignment {
  text: string;
  duration: number;
  words: EditWord[];
}

export interface EditRequest {
  /** The clip (plain base64 of a WAV). */
  audio_base64: string;
  /** The transcript the clip's word timings belong to (after any correction). */
  original_text?: string;
  /** Corrected word timings of original_text (in order); the server uses them instead of its own alignment. */
  original_words?: { word: string; start: number; end: number; tokens?: string[] }[];
  target_text: string;
  /** Regions of the original, seconds, ascending, non-overlapping, at most edit_max_spans. */
  spans: [number, number][];
  margin_seconds?: number;
  max_span_seconds?: number;
  language?: string;
  text_normalization?: boolean;
  seed?: number | null;
  num_takes?: number;
  verify?: boolean;
  speaker_embedding?: boolean | null;
  /** Applied to every regenerated region. */
  controls?: Dose;
  /** One entry per `spans` entry, summed with `controls`. */
  region_controls?: (Dose | null)[];
  /** Clips longer than this are edited on word-snapped windows around the regions (server default 16). */
  max_context_seconds?: number;
  /** Audio kept on each side of a region inside such a window (server default 4). */
  context_seconds?: number;
  /** Span-end is masked until a fill reaches this share of its expected length (server default 0.7). */
  min_fill_ratio?: number;
}

export interface EditTake {
  seed: number;
  audio_base64: string;
  duration: number;
  /** The regions in the edited audio. */
  edited_regions: [number, number][];
  fill_seconds: number[];
  /** false: that region's fill hit max_span_seconds (probably broken). */
  stopped: boolean[];
  /** `groups`: per pronunciation group of the target, the word recognized at its place (null: none). */
  verify: { asr_text: string; wer: number; cer: number; groups?: { ipa: string; heard: string | null }[] | null } | null;
  words: EditWord[] | null;
}

export interface EditResponse {
  sample_rate: number;
  original_text: string;
  target_text: string;
  words: EditWord[] | null;
  spans: [number, number][];
  /** The audio the model saw, full-clip seconds (present when the clip was windowed). */
  contexts?: [number, number][] | null;
  /** Per region: the fill length the server expected (alignment of original_text + word diff). */
  expected_seconds?: number[] | null;
  takes: EditTake[];
}

// ── Dubbing (/dub/*) ──────────────────────────────────────────────────────────────────────────────

/** `GET /dub/capabilities`; `fixture`: every /dub call runs on the client-side fixture engine. */
export interface DubCaps {
  ref_languages: { code: string; name: string }[];
  target_languages: { code: string; label: string }[];
  max_ref_seconds: number;
  max_video_bytes: number;
  default_min_pause_s: number;
  min_pause_range: [number, number];
  sample_rate: number;
  frame_rate: number;
  /** Speaking-rate bucket i covers [edge(i−1), edge(i)) UTF-8 bytes per second of the reference segment. */
  rate_edges: number[];
  candidate_bucket_offsets: number[];
  candidate_seeds: number;
  /**
   * The full speaking-rate scale: the row buckets 0…n−1 plus steered ones below and above (rate row unknown + a
   * tempo dose), each listing the target languages it was measured safe for. Absent: row buckets only.
   */
  buckets?: DubBucket[];
  model?: string;
  /** AV-sync is set up: video sources can be warped onto the speaker's lips. */
  lipsync: boolean;
  voice_modes?: string[];
  fixture: boolean;
}

export interface DubBucket {
  index: number;
  /** str(index). */
  label: string;
  /** Nominal speaking rate (bucket midpoint), UTF-8 bytes per second. */
  rate: number;
  /** Nominal edges (bytes / s): low ≤ rate < high; null at the open ends of the widest scale. */
  low?: number | null;
  high?: number | null;
  steered: boolean;
  /** Dub target languages that have this bucket (row buckets: every target language). */
  languages: string[];
  /** Steered buckets: the `delivery.tempo` dose per target language. */
  doses?: Record<string, number>;
  /** Voice modes the bucket serves: the slower steered buckets are studio-voice only. */
  voice_modes?: string[];
}

export interface DubWord {
  word: string;
  start: number;
  end: number;
  segment?: number;
}

export interface DubRef {
  ref_id: string;
  duration: number;
  sample_rate: number;
}

/** The video's first face track (`POST /dub/lips`), seconds. */
export interface DubLips {
  start: number;
  end: number;
  track_frames: number;
  video_frames: number;
  tracks: number;
}

export interface DubTranscript {
  text: string;
  language: string | null;
  language_name: string;
  /** ForcedAligner word timings of `text`: usable as `ref_words` (no second alignment). */
  words?: DubWord[];
}

export interface DubAnalyzeRequest {
  ref_id: string;
  ref_text: string;
  ref_lang: string;
  target_text: string;
  target_lang: string;
  min_pause_s?: number;
  /** Word timings the studio already knows (a Scene mixdown, a Direct take): no ASR / forced alignment runs. */
  ref_words?: DubWord[];
}

export interface DubPlanSegment {
  index: number;
  ref_start: number;
  ref_end: number;
  ref_text: string;
  target_text: string;
  ref_word_range?: [number, number];
  target_unit_range?: [number, number];
  bucket_auto?: number | null;
}

export interface DubAnalysis {
  duration: number;
  ref_words: DubWord[];
  segments: DubPlanSegment[];
  warnings: string[];
}

/** Who speaks the dub: the reference speaker cloned from its own audio, or a studio voice with Cast steering. */
export type DubVoice = { mode: 'source' } | { mode: 'studio'; speaker_embedding_id: string; controls?: Dose };

export interface DubSegmentIn {
  ref_start: number;
  ref_end: number;
  target_text: string;
  /** Speaking-rate bucket; null = from the text's rate (candidates around it). */
  bucket: number | null;
  voice?: DubVoice;
  /** Direction: spannable hidden controls, e.g. {"emotions.angry": 4}. */
  controls?: Dose;
}

export interface DubGenerateRequest {
  ref_id: string;
  target_lang: string;
  seed: number;
  segments: DubSegmentIn[];
  lipsync: boolean;
  voice?: DubVoice;
}

export interface DubRegenerateRequest {
  dub_id: string;
  segment: number;
  target_text: string;
  seed: number;
  bucket: number | null;
  lipsync: boolean;
  voice?: DubVoice;
  controls?: Dose;
}

export interface DubCandidate {
  bucket: number;
  offset: number;
  seed: number;
  speech_s: number | null;
  attempts: number | null;
  /** LACI retries by kind: truncation (early EOS), hallucination (stall), skip, pace_fast, pace_slow. */
  retries: Record<string, number> | null;
  final_bucket: number | null;
  end: string | null;
  capped: boolean | null;
  score: number | null;
}

export interface DubSegmentOut {
  index: number;
  ref_start: number;
  ref_end: number;
  target_text: string;
  status: 'ok' | 'silent';
  seed: number;
  bytes: number;
  rate: number | null;
  bucket: number | null;
  pinned: boolean;
  candidates: DubCandidate[];
  chosen: number | null;
  speech_start: number | null;
  speech_end: number | null;
  shift: number | null;
  start_error: number | null;
  end_error: number | null;
  duration_error: number | null;
  lip_start: number | null;
  lip_end: number | null;
  lip_scale: number | null;
  lip_stretch: number | null;
  lip_squeeze: number | null;
  lip_warp: number | null;
  voice?: DubVoice;
  controls?: Dose;
}

export interface DubMetrics {
  mean_abs_start_error_ms: number;
  mean_abs_end_error_ms: number;
  mean_abs_duration_error_ms: number;
  max_shift_ms: number;
  laci_retries: number;
  retries_by_kind: Record<string, number>;
  all_retries: number;
  all_retries_by_kind: Record<string, number>;
  exhausted_takes: number;
  candidates: number;
  generation_s: number;
}

export interface DubLipsync {
  start: number;
  end: number;
  max_shift_ms: number;
  mean_abs_shift_ms: number;
  max_stretch: number;
  max_squeeze: number;
  mean_warp: number;
  sync_distance_before: number;
  sync_distance_after: number;
  /** The bounds the server held the warp to: local rate within 1/max_rate…max_rate, every shift ≤ max_shift_s. */
  guard?: { max_rate: number; max_shift_s: number } | null;
  /** False: the warp did not lower the SyncNet distance, so the dub came back unwarped (`reason` says why). */
  applied?: boolean;
  reason?: string;
}

export interface DubResult {
  dub_id: string;
  audio_base64: string;
  sample_rate: number;
  duration: number;
  segments: DubSegmentOut[];
  words: DubWord[];
  metrics: DubMetrics;
  lipsync: DubLipsync | null;
}

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string,
  ) {
    super(detail);
  }
}

/** 409 from `PUT /projects/{id}`: someone saved since `base_revision`; `revision` is the stored one (null: deleted). */
export class RevisionConflict extends ApiError {
  constructor(readonly revision: Revision | null) {
    super(409, 'The project changed elsewhere since it was opened');
  }
}

/** 400 from `PUT /projects/{id}`: media the document names that the server lacks (upload them, retry). */
export class MissingMedia extends ApiError {
  constructor(readonly missing: string[]) {
    super(400, `${missing.length} media file${missing.length === 1 ? ' is' : 's are'} missing on the server`);
  }
}

// ── Configuration ─────────────────────────────────────────────────────────────────────────────────

const params = new URLSearchParams(location.search);
const forced = (params.get('api') || import.meta.env.VITE_ZONOS_API || 'auto') as BackendKind | 'auto';
const server = (params.get('server') || import.meta.env.VITE_ZONOS_SERVER_URL || '').replace(/\/+$/, '');
export const SAMPLE_RATE = 44100;

function sessionId(): string {
  let id = localStorage.getItem('zvs.session');
  if (!id) {
    // crypto.randomUUID exists only in secure contexts (https, localhost); a cluster host over http has none.
    id = typeof crypto.randomUUID === 'function'
      ? crypto.randomUUID()
      : Array.from(crypto.getRandomValues(new Uint8Array(16)), (b) => b.toString(16).padStart(2, '0')).join('');
    localStorage.setItem('zvs.session', id);
  }
  return id;
}

/**
 * Long POSTs the server answers in keep-alive form (`X-Keepalive: 1`): 200 at once, whitespace heartbeats, then the
 * JSON — an error as `{detail, status_code}`. A proxy in front (a Cloudflare quick tunnel drops requests silent for
 * 100 s) then never times them out.
 */
const KEEPALIVE = /^\/(tts\/edit(\/transcribe|\/align)?|dub\/(ref|lips\/[^?]*|transcribe|analyze|generate|regenerate))(\?|$)/;

const headers = (json = false, path = ''): HeadersInit => ({
  'X-TTS-Session-ID': sessionId(),
  ...(json ? { 'Content-Type': 'application/json' } : {}),
  ...(KEEPALIVE.test(path) ? { 'X-Keepalive': '1' } : {}),
});

/** An error response as a short message: the server's `detail`, never a proxy's HTML page. */
function errorOf(status: number, statusText: string, text: string): ApiError {
  try {
    const body = JSON.parse(text);
    const detail = body?.detail ?? body;
    return new ApiError(typeof body?.status_code === 'number' ? body.status_code : status, typeof detail === 'string' ? detail : JSON.stringify(detail));
  } catch {
    if (status === 524 || status === 504 || status === 408) return new ApiError(status, `The server didn’t answer in time (proxy ${status}) — try again`);
    if (/^\s*</.test(text) || !text.trim()) return new ApiError(status, status >= 500 ? `Server error (${status})` : `${status} ${statusText}`.trim());
    return new ApiError(status, text.slice(0, 300));
  }
}

/** A successful response's JSON; a keep-alive body carrying `status_code` ≥ 400 is the error it reports. */
function bodyOf<T>(status: number, text: string): T {
  let body: unknown;
  try {
    body = JSON.parse(text);
  } catch {
    throw /^\s*</.test(text) ? new ApiError(status, 'The server sent a web page instead of an answer (a proxy error?) — try again') : new ApiError(status, 'unreadable response');
  }
  const code = (body as { status_code?: unknown } | null)?.status_code;
  if (typeof code === 'number' && code >= 400) throw errorOf(code, '', text);
  return body as T;
}

async function fail(res: Response): Promise<never> {
  let text = '';
  try {
    text = await res.text();
  } catch {
    /* body unreadable: the status line */
  }
  throw errorOf(res.status, res.statusText, text);
}

/** JSON of a fetch response, errors (HTTP or keep-alive `status_code`) as ApiError. */
async function readJson<T>(res: Response): Promise<T> {
  if (!res.ok) await fail(res);
  return bodyOf<T>(res.status, await res.text());
}

async function getJson<T>(path: string): Promise<T> {
  return readJson<T>(await fetch(server + path, { headers: headers() }));
}

async function postJson<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  return readJson<T>(await fetch(server + path, { method: 'POST', headers: headers(true, path), body: JSON.stringify(body), signal }));
}

// ── Backend detection ─────────────────────────────────────────────────────────────────────────────

let backend: Backend | null = null;
const fixture = fixtureSpec as unknown as ControlsSpec;

export function current(): Backend {
  if (!backend) throw new Error('api.connect() has not run');
  return backend;
}

const FIXTURE_EDIT: EditCaps = {
  speech_editing: true,
  edit_max_spans: 3,
  edit_frame_rate: 15,
  edit_max_seconds: 150,
  edit_align_max_seconds: 150,
  fixture: true,
};
const NO_EDIT: EditCaps = { ...FIXTURE_EDIT, speech_editing: false, fixture: false };

/**
 * Speech editing: the server's `/tts/capabilities` when one answers, the fixture engine offline.
 * `?edit=fixture` forces the fixture engine, `?edit=0` a checkpoint that cannot edit.
 */
async function editCaps(kind: BackendKind, c: Partial<EditCaps> | null): Promise<EditCaps> {
  const want = params.get('edit');
  if (want === '0' || want === 'off') return { ...NO_EDIT, fixture: kind !== 'live' };
  if (want === 'fixture' || kind === 'offline') return FIXTURE_EDIT;
  try {
    if (!c) throw new Error('no capabilities');
    return {
      speech_editing: !!c.speech_editing,
      edit_max_spans: c.edit_max_spans ?? 3,
      edit_frame_rate: c.edit_frame_rate ?? 15,
      edit_max_seconds: c.edit_max_seconds ?? 150,
      edit_align_max_seconds: c.edit_align_max_seconds ?? 150,
      fixture: false,
    };
  } catch {
    return NO_EDIT;
  }
}

const FIXTURE_PRON: PronCaps = { ...(fixturePhonemes as Omit<PronCaps, 'fixture'>), fixture: true };
const NO_PRON: PronCaps = { phoneme_tokens: null, phonemize_enabled: false, phonemize_languages: [], phonemize_guess_languages: [], fixture: false };

/**
 * Pronunciation input: the server's capabilities when one answers, the fixture offline (the real 136-symbol
 * inventory of the phoneme-token checkpoint). `?phon=fixture` forces the fixture, `?phon=0` a checkpoint without.
 */
function pronCaps(kind: BackendKind, c: Partial<PronCaps> | null): PronCaps {
  const want = params.get('phon');
  if (want === '0' || want === 'off') return { ...NO_PRON, fixture: kind !== 'live' };
  if (want === 'fixture' || kind === 'offline') return FIXTURE_PRON;
  if (!c) return NO_PRON;
  const t = c.phoneme_tokens;
  return {
    phoneme_tokens: t && Array.isArray(t.symbols) ? t : null,
    phonemize_enabled: !!c.phonemize_enabled,
    phonemize_languages: Array.isArray(c.phonemize_languages) ? c.phonemize_languages : [],
    phonemize_guess_languages: Array.isArray(c.phonemize_guess_languages) ? c.phonemize_guess_languages : [],
    fixture: false,
  };
}

/**
 * Dubbing: the server's `/dub/capabilities` when it has the dub setup, the fixture engine offline.
 * `?dub=fixture` forces the fixture engine, `?dub=0` a server without dubbing.
 */
async function dubCaps(kind: BackendKind): Promise<DubCaps | null> {
  const want = params.get('dub');
  if (want === '0' || want === 'off') return null;
  if (want === 'fixture' || kind === 'offline') return { ...fixtureDub.CAPS, fixture: true };
  try {
    return { ...(await getJson<Omit<DubCaps, 'fixture'>>('/dub/capabilities')), fixture: false };
  } catch {
    return null;
  }
}

type RawTagCaps = Partial<Omit<TagCaps, 'fixture' | 'trained'>> & { available?: boolean; trained?: boolean };

const FIXTURE_TAGS: TagCaps = { tags: ['laugh', 'breath', 'sigh', 'cough', 'panting', 'shouting', 'crying'], min_scale: 1, max_scale: 4, default_scale: 1, trained: true, fixture: true };

/**
 * Nonverbal tags: the server's `tag_cfg` when available, the fixture synth offline.
 * `?nv=fixture` forces the fixture (renders then run on the client-side synth), `?nv=0` a server without tags.
 */
function tagCaps(kind: BackendKind, c: RawTagCaps | undefined): TagCaps | null {
  const want = params.get('nv');
  if (want === '0' || want === 'off') return null;
  if (want === 'fixture' || kind === 'offline') return FIXTURE_TAGS;
  // Tag guidance runs on any byte / SuperBPE checkpoint, but only one trained on tags reads them: else none at all.
  if (!c?.available || c.trained === false) return null;
  return {
    tags: Array.isArray(c.tags) && c.tags.length ? c.tags : FIXTURE_TAGS.tags,
    min_scale: c.min_scale ?? 1,
    max_scale: c.max_scale ?? 4,
    default_scale: c.default_scale ?? 1,
    trained: typeof c.trained === 'boolean' ? c.trained : null,
    fixture: false,
  };
}

const FIXTURE_PROJECTS: ProjectCaps = { max_media_bytes: 2e9, max_project_bytes: 20e6, fixture: true };

/**
 * Projects on the server when its capabilities say so, else in this browser (IndexedDB).
 * `?projects=fixture` forces the fixture server store, `?projects=browser` the browser store.
 */
function projectCaps(c: (Partial<ProjectCaps> & { enabled?: boolean }) | undefined): ProjectCaps | null {
  const want = params.get('projects');
  if (want === 'fixture') return FIXTURE_PROJECTS;
  if (want === 'browser' || want === '0' || !c?.enabled) return null;
  return { max_media_bytes: c.max_media_bytes ?? 2e9, max_project_bytes: c.max_project_bytes ?? 20e6, fixture: false };
}

export async function connect(): Promise<Backend> {
  const make = async (kind: BackendKind, spec: ControlsSpec): Promise<Backend> => {
    const caps = kind === 'offline' ? null : await getJson<Partial<EditCaps> & Partial<PronCaps> & { projects?: Partial<ProjectCaps> & { enabled?: boolean }; tag_cfg?: RawTagCaps }>('/tts/capabilities').catch(() => null);
    return (backend = { kind, server, spec, steering: kind !== 'fixture', edit: await editCaps(kind, caps), pron: pronCaps(kind, caps), dub: await dubCaps(kind), projects: projectCaps(caps?.projects), tags: tagCaps(kind, caps?.tag_cfg) });
  };
  if (forced === 'offline') return make('offline', fixture);
  try {
    return await make('live', await getControls());
  } catch (err) {
    if (forced === 'live') throw err;
    if (err instanceof ApiError && err.status === 404) return make('fixture', fixture);
    if (forced === 'fixture') return make('fixture', fixture);
    // Nothing (or not a ZONOS server) answered: run on the client-side fixture.
    return make('offline', fixture);
  }
}

export function getControls(): Promise<ControlsSpec> {
  if (forced === 'fixture') return Promise.reject(new ApiError(404, 'fixture forced'));
  return getJson<ControlsSpec>('/tts/controls');
}

// ── Voices ────────────────────────────────────────────────────────────────────────────────────────

const offlineVoices = new Map<string, { speaker: Speaker; embedding?: Float32Array }>();

export async function getSpeakers(): Promise<Speaker[]> {
  if (current().kind === 'offline') {
    // Default voices carry their clip's transcript (as `<voice>.txt` does on the server); two are left without
    // one so the transcribe path is exercised too.
    const defaults = fixtureSpeakers().map((s) => ({ ...s, transcript: /Skyler|Yelling/.test(s.label) ? null : offlinePreviewText(s) }));
    return [...defaults, ...[...offlineVoices.values()].map((v) => v.speaker)];
  }
  const body = await getJson<{ speakers: Speaker[] }>('/tts/speakers');
  return Array.isArray(body.speakers) ? body.speakers : [];
}

/** The vector the server conditions on. Synthetic (and flagged) when the server can't provide it. */
export async function getEmbedding(speaker: Speaker): Promise<{ embedding: Float32Array; synthetic: boolean }> {
  const b = current();
  if (b.kind !== 'offline') {
    try {
      const body = await getJson<{ embedding: number[] }>(`/tts/speakers/${encodeURIComponent(speaker.id)}/embedding`);
      return { embedding: Float32Array.from(body.embedding), synthetic: false };
    } catch (err) {
      if (!(err instanceof ApiError) || err.status !== 404 || b.kind === 'live') throw err;
    }
  }
  const own = offlineVoices.get(speaker.id)?.embedding;
  return { embedding: own ?? syntheticEmbedding(b.spec, speaker.id, speaker.label), synthetic: b.kind !== 'offline' };
}

const offlinePreviewText = (s: Speaker) => `Hi, I'm ${s.label}. This is how I sound when I read a short line.`;

/** The voice's reference clip, as an encoded audio file. */
export async function getPreview(speaker: Speaker): Promise<ArrayBuffer> {
  if (current().kind === 'offline') {
    const { embedding } = await getEmbedding(speaker);
    const text = offlinePreviewText(speaker);
    const { pcm, words } = synthesizeWords(current().spec, { text, embedding, seed: 7 });
    fixtureEdit.remember(pcm, SAMPLE_RATE, { text, words, embedding });
    return encodeWav(pcm, SAMPLE_RATE);
  }
  const res = await fetch(`${server}/tts/speakers/${encodeURIComponent(speaker.id)}/preview`, { headers: headers() });
  if (!res.ok) await fail(res);
  return res.arrayBuffer();
}

/** Cache an uploaded / recorded clip as a voice (server computes its embedding). */
export async function uploadVoice(label: string, audio: Blob, fileName: string): Promise<Speaker> {
  if (current().kind === 'offline') {
    const speaker: Speaker = { id: `local_${slug(label)}_${Date.now().toString(36)}`, label, is_default: false, source_type: 'audio', has_preview: true };
    offlineVoices.set(speaker.id, { speaker });
    return speaker;
  }
  return postJson<Speaker>('/tts/speakers', {
    label,
    speaker_audio_base64: await blobToDataUrl(audio),
    speaker_audio_name: fileName,
  });
}

/** Cache a designed voice: its steered embedding, as a .npy. */
export async function saveEmbeddingVoice(label: string, embedding: Float32Array): Promise<Speaker> {
  if (current().kind === 'offline') {
    const speaker: Speaker = { id: `local_${slug(label)}_${Date.now().toString(36)}`, label, is_default: false, source_type: 'embedding_file', has_preview: false };
    offlineVoices.set(speaker.id, { speaker, embedding: embedding.slice() });
    return speaker;
  }
  return postJson<Speaker>('/tts/speakers', {
    label,
    speaker_embedding_base64: await bufferToDataUrl(encodeNpy(embedding), 'application/octet-stream'),
    speaker_embedding_name: `${slug(label)}.npy`,
  });
}

// ── Generation ────────────────────────────────────────────────────────────────────────────────────

export interface GenerateOptions {
  signal?: AbortSignal;
  /** Called with each float32 chunk as it streams in. */
  onChunk?: (chunk: Float32Array, sr: number) => void;
}

/**
 * Spans following the reading position: unknown until a render with spans answers. A server without it either
 * rejects the field (422) or ignores it (no X-Control-Span-Follow header): then requests go out as before, spans
 * placed by the rate estimate.
 */
let spanFollowOk: boolean | null = null;

/** Languages the offline fixture "has alignment heads" for: the others fall back to the estimate, as on the server. */
const FIXTURE_READING_LANGUAGES = new Set(['en_us', 'fr_fr']);

export async function generate(req: GenerateRequest, opts: GenerateOptions = {}): Promise<Rendered> {
  const b = current();
  const t0 = performance.now();
  const body: GenerateRequest = { sr_out: SAMPLE_RATE, stream: true, ...req };
  if (!b.steering) {
    delete body.controls;
    delete body.control_spans;
  }
  if (!body.control_spans?.length || spanFollowOk === false) delete body.control_span_follow;
  if (!b.tags || (body.tag_cfg_scale ?? 1) <= 1) delete body.tag_cfg_scale;
  if (b.kind === 'offline' || b.tags?.fixture) return offlineGenerate(body, opts, t0);

  const post = () => fetch(`${server}/tts/generate`, { method: 'POST', headers: headers(true), body: JSON.stringify(body), signal: opts.signal });
  let res = await post();
  if (res.status === 422 && body.control_span_follow) {
    // A server from before reading-position spans validates the field away: today's request instead.
    spanFollowOk = false;
    delete body.control_span_follow;
    res = await post();
  }
  if (!res.ok) await fail(res);
  let spanFollow: SpanFollow | undefined;
  if (body.control_spans?.length) {
    const h = res.headers.get('X-Control-Span-Follow');
    if (body.control_span_follow) spanFollowOk = h === 'reading' || h === 'rate';
    spanFollow = h === 'reading' ? 'reading' : 'rate';
  }
  const sr = Number(res.headers.get('X-Audio-Sample-Rate')) || body.sr_out || SAMPLE_RATE;
  if (!res.body) throw new ApiError(502, 'Empty audio stream');

  const reader = res.body.getReader();
  const chunks: Float32Array[] = [];
  let carry = new Uint8Array(0);
  let total = 0;
  let ttfbMs = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (!ttfbMs) ttfbMs = performance.now() - t0;
    const merged = new Uint8Array(carry.length + value.length);
    merged.set(carry);
    merged.set(value, carry.length);
    const aligned = merged.length - (merged.length % 4);
    if (aligned) {
      const chunk = new Float32Array(merged.buffer.slice(0, aligned));
      chunks.push(chunk);
      total += chunk.length;
      opts.onChunk?.(chunk, sr);
    }
    carry = merged.slice(aligned);
  }
  const pcm = new Float32Array(total);
  let o = 0;
  for (const c of chunks) {
    pcm.set(c, o);
    o += c.length;
  }
  return { pcm, sr, ttfbMs, totalMs: performance.now() - t0, ...(spanFollow ? { spanFollow } : {}) };
}

async function offlineGenerate(req: GenerateRequest, opts: GenerateOptions, t0: number): Promise<Rendered> {
  const b = current();
  fixturePron.validateGroups(req.text, b.pron.phoneme_tokens);
  const speakers = await getSpeakers();
  const speaker = speakers.find((s) => s.id === req.speaker_embedding_id);
  if (!speaker) throw new ApiError(400, `Unknown speaker_embedding_id '${req.speaker_embedding_id}'`);
  const controls = req.controls ?? {};
  for (const id of [...Object.keys(controls), ...(req.control_spans ?? []).flatMap((s) => Object.keys(s.controls))]) {
    const dir = b.spec.directions[id];
    if (!dir) throw new ApiError(400, `Unknown control direction '${id}'`);
    const mode = req.speaker_mode ?? 'embedding';
    if (!b.spec.sets[dir.set]?.modes.includes(mode)) throw new ApiError(400, `Control '${id}' moves the speaker embedding and needs speaker_mode=embedding.`);
  }
  for (const span of req.control_spans ?? []) {
    if (Object.keys(span.controls).some((id) => b.spec.directions[id].space === 'raw')) throw new ApiError(400, 'control_spans take hidden-space directions only.');
    if (!(span.start >= 0 && span.end > span.start && span.end <= req.text.length)) throw new ApiError(400, `control_spans [${span.start}, ${span.end}) is outside the text.`);
  }
  const sorted = [...(req.control_spans ?? [])].sort((a, b) => a.start - b.start);
  for (let i = 1; i < sorted.length; i++) {
    if (sorted[i].start < sorted[i - 1].end) throw new ApiError(400, `control spans [${sorted[i - 1].start}, ${sorted[i - 1].end}) and [${sorted[i].start}, ...) overlap.`);
  }
  // The server's tag-guidance rules: a recognized tag, and one twin per request (no speaker / emotion CFG).
  const tagScale = req.tag_cfg_scale ?? 1;
  const nv = b.tags ? fixtureNv.prepare(req.text, req.control_spans) : null;
  if (tagScale > 1) {
    if (!nv?.tagged) throw new ApiError(400, 'tag_cfg_scale > 1 needs a recognized nonverbal tag in the text.');
    if ((req.speaker_cfg_scale ?? 1) !== 1) throw new ApiError(400, 'tag_cfg_scale cannot be combined with speaker_cfg_scale ≠ 1.');
  }
  const { embedding } = await getEmbedding(speaker);
  const spoken = nv ?? { text: req.text, spans: req.control_spans };
  const said = synthesizeWords(b.spec, { text: spoken.text, embedding, controls, spans: spoken.spans, seed: req.seed ?? 0 });
  const { pcm, words } = nv ? fixtureNv.voice(said.pcm, SAMPLE_RATE, said.words, nv, tagScale, req.seed ?? 0) : said;
  fixtureEdit.remember(pcm, SAMPLE_RATE, { text: spoken.text, words, embedding, controls });
  // Emulate the measured latency and streaming so the UI's scheduling is exercised.
  const wait = (ms: number) => {
    const { promise, resolve, reject } = Promise.withResolvers<void>();
    const t = setTimeout(resolve, ms);
    opts.signal?.addEventListener('abort', () => {
      clearTimeout(t);
      reject(new DOMException('Aborted', 'AbortError'));
    });
    return promise;
  };
  await wait(220 + Math.random() * 180);
  const ttfbMs = performance.now() - t0;
  const step = Math.round(SAMPLE_RATE * 0.4);
  for (let i = 0; i < pcm.length; i += step) {
    opts.onChunk?.(pcm.slice(i, i + step), SAMPLE_RATE);
    if (i + step < pcm.length) await wait(40);
  }
  // The server's header, emulated: reading where the fixture "has heads" for the language, else the estimate.
  const spanFollow: SpanFollow | undefined = req.control_spans?.length ? (req.control_span_follow === 'reading' && FIXTURE_READING_LANGUAGES.has(req.language ?? 'en_us') ? 'reading' : 'rate') : undefined;
  return { pcm, sr: SAMPLE_RATE, ttfbMs, totalMs: performance.now() - t0, ...(spanFollow ? { spanFollow } : {}) };
}

// ── Speech editing ────────────────────────────────────────────────────────────────────────────────

/** MMS greedy transcript + word timings of a clip (plain base64 WAV). */
/** `language`: the clip's language (server code); it picks the MMS adapter (a new one loads on first use). */
export function transcribe(audio_base64: string, language = 'en_us'): Promise<Alignment> {
  if (current().edit.fixture) return fixtureEdit.transcribe(audio_base64, language);
  return postJson<Alignment>('/tts/edit/transcribe', { audio_base64, language });
}

/** Forced alignment of known text. */
export function align(audio_base64: string, text: string, language = 'en_us'): Promise<Alignment> {
  if (current().edit.fixture) return fixtureEdit.align(audio_base64, text, language);
  return postJson<Alignment>('/tts/edit/align', { audio_base64, text, language });
}

export async function edit(req: EditRequest, signal?: AbortSignal): Promise<EditResponse> {
  if (current().edit.fixture) {
    const tokens = current().pron.phoneme_tokens;
    fixturePron.validateGroups(req.target_text, tokens);
    if (/[\ue000\ue001]/.test(req.original_text ?? '')) throw new ApiError(400, 'original_text describes the recorded audio: it cannot hold pronunciation groups.');
    return fixtureEdit.edit(current().spec, current().edit, req, signal);
  }
  return postJson<EditResponse>('/tts/edit', req, signal);
}

// ── Pronunciation ─────────────────────────────────────────────────────────────────────────────────

/** Lexicon IPA for the words of `text` (`guess`: espeak-ng fills in the languages that allow it). */
export function phonemize(text: string, language: string, guess = true): Promise<PhonemizeResponse> {
  const b = current();
  if (b.pron.fixture) return fixturePron.phonemize(b.pron, text, language, guess);
  return postJson<PhonemizeResponse>('/tts/phonemize', { text, language, guess });
}

// ── Dubbing ───────────────────────────────────────────────────────────────────────────────────────

const dubFixture = () => !!current().dub?.fixture;

/** POST raw bytes with upload progress (fetch has none); JSON back, errors as ApiError. */
function postBytes<T>(path: string, body: Blob | ArrayBuffer, onProgress?: (f: number) => void, signal?: AbortSignal): Promise<T> {
  const { promise, resolve, reject } = Promise.withResolvers<T>();
  const xhr = new XMLHttpRequest();
  xhr.open('POST', server + path);
  for (const [k, v] of Object.entries(headers(false, path))) xhr.setRequestHeader(k, v as string);
  xhr.setRequestHeader('Content-Type', 'application/octet-stream');
  xhr.upload.onprogress = (e) => e.lengthComputable && onProgress?.(e.loaded / e.total);
  xhr.onload = () => {
    try {
      if (xhr.status >= 200 && xhr.status < 300) resolve(bodyOf<T>(xhr.status, xhr.responseText));
      else reject(errorOf(xhr.status, xhr.statusText, xhr.responseText));
    } catch (e) {
      reject(e);
    }
  };
  xhr.onerror = () => reject(new ApiError(0, 'network error'));
  xhr.onabort = () => reject(new DOMException('aborted', 'AbortError'));
  signal?.addEventListener('abort', () => xhr.abort());
  xhr.send(body);
  return promise;
}

/** The reference audio (a WAV) → ref_id. */
export function dubRef(wav: ArrayBuffer, onProgress?: (f: number) => void): Promise<DubRef> {
  if (dubFixture()) return fixtureDub.addRef(wav, onProgress);
  return postBytes<DubRef>('/dub/ref', wav, onProgress);
}

/** Face track + lip embeddings of the reference's video, for lip-sync. */
export function dubLips(refId: string, video: Blob, name: string, onProgress?: (f: number) => void): Promise<DubLips> {
  if (dubFixture()) return fixtureDub.lips(refId, video, onProgress);
  return postBytes<DubLips>(`/dub/lips/${encodeURIComponent(refId)}?name=${encodeURIComponent(name)}`, video, onProgress);
}

export function dubTranscribe(refId: string, refLang?: string | null): Promise<DubTranscript> {
  if (dubFixture()) return fixtureDub.transcribe(refId, refLang ?? null);
  return postJson<DubTranscript>('/dub/transcribe', { ref_id: refId, ...(refLang ? { ref_lang: refLang } : {}) });
}

export function dubAnalyze(req: DubAnalyzeRequest): Promise<DubAnalysis> {
  if (dubFixture()) return fixtureDub.analyze(req);
  return postJson<DubAnalysis>('/dub/analyze', req);
}


/** The fixture engine's voice for a dub: a studio speaker's embedding, or the reference speaker's own. */
const dubContext = (): fixtureDub.Context => ({
  spec: current().spec,
  embeddingOf: async (id: string) => {
    const s = (await getSpeakers()).find((x) => x.id === id);
    return s ? (await getEmbedding(s)).embedding : syntheticEmbedding(current().spec, id, id);
  },
  validateText: (text: string) => fixturePron.validateGroups(text, current().pron.phoneme_tokens),
});

export function dubGenerate(req: DubGenerateRequest, signal?: AbortSignal): Promise<DubResult> {
  if (dubFixture()) return fixtureDub.generate(dubContext(), req, signal);
  return postJson<DubResult>('/dub/generate', req, signal);
}

export function dubRegenerate(req: DubRegenerateRequest, signal?: AbortSignal): Promise<DubResult> {
  if (dubFixture()) return fixtureDub.regenerate(dubContext(), req, signal);
  return postJson<DubResult>('/dub/regenerate', req, signal);
}

/** The video with the dub as its audio track (MP4). */
export async function dubMux(dubId: string, video: Blob, name: string): Promise<Blob> {
  if (dubFixture()) return fixtureDub.mux(dubId, video);
  const res = await fetch(`${server}/dub/mux/${encodeURIComponent(dubId)}?name=${encodeURIComponent(name)}`, { method: 'POST', headers: { ...headers(), 'Content-Type': 'application/octet-stream' }, body: video });
  if (!res.ok) await fail(res);
  return res.blob();
}

// ── Projects (shared on the server) ───────────────────────────────────────────────────────────────

const projectsFixture = () => current().projects?.fixture === true;

/** A fixture error as the server's would be. */
async function viaFixture<T>(fn: () => Promise<T>): Promise<T> {
  try {
    return await fn();
  } catch (e) {
    if (!(e instanceof fixtureProjects.FixtureError)) throw e;
    if (e.status === 409) throw new RevisionConflict((e.body.revision as Revision | null | undefined) ?? null);
    if (Array.isArray(e.body.missing)) throw new MissingMedia(e.body.missing as string[]);
    throw new ApiError(e.status, e.detail);
  }
}

const projectPath = (id: string) => `/projects/${encodeURIComponent(id)}`;
const mediaPath = (hash: string) => `/media/${encodeURIComponent(hash)}`;

/** Every project on the server, newest first. */
export function listProjects(): Promise<ServerProjectEntry[]> {
  if (projectsFixture()) return viaFixture(fixtureProjects.list);
  return getJson<ServerProjectEntry[]>('/projects');
}

export function getProject(id: string): Promise<ServerProject> {
  if (projectsFixture()) return viaFixture(() => fixtureProjects.get(id));
  return getJson<ServerProject>(projectPath(id));
}

/** Save a project; `base_revision` is the revision it was loaded at (null: a new project). 409 → RevisionConflict. */
export async function putProject(id: string, body: { name: string; data: unknown; base_revision: Revision | null }): Promise<{ revision: Revision; updated_at: number | string }> {
  if (projectsFixture()) return viaFixture(() => fixtureProjects.put(id, body as { name: string; data: unknown; base_revision: number | null }));
  const res = await fetch(server + projectPath(id), { method: 'PUT', headers: headers(true), body: JSON.stringify(body) });
  if (res.status === 409 || res.status === 400) {
    const text = await res.text();
    let body: { revision?: Revision | null; missing?: unknown } = {};
    try {
      body = JSON.parse(text);
    } catch {
      /* not JSON: a plain error below */
    }
    if (res.status === 409) throw new RevisionConflict(body.revision ?? null);
    if (Array.isArray(body.missing)) throw new MissingMedia(body.missing as string[]);
    throw errorOf(res.status, res.statusText, text);
  }
  return readJson(res);
}

export async function deleteProject(id: string): Promise<void> {
  if (projectsFixture()) return viaFixture(() => fixtureProjects.remove(id));
  const res = await fetch(server + projectPath(id), { method: 'DELETE', headers: headers() });
  if (!res.ok && res.status !== 404) await fail(res);
}

/** Whether the server holds media `hash` (HEAD: 200 / 404). */
export async function hasMedia(hash: string): Promise<boolean> {
  if (projectsFixture()) return viaFixture(() => fixtureProjects.hasMedia(hash));
  const res = await fetch(server + mediaPath(hash), { method: 'HEAD', headers: headers() });
  if (res.status === 404) return false;
  if (!res.ok) await fail(res);
  return true;
}

export async function getMedia(hash: string, type?: string): Promise<Blob> {
  if (projectsFixture()) return viaFixture(() => fixtureProjects.getMedia(hash));
  const res = await fetch(server + mediaPath(hash), { headers: headers() });
  if (!res.ok) await fail(res);
  const blob = await res.blob();
  // The stored type when the server kept one, else the document's.
  return type && blob.type !== type ? new Blob([blob], { type }) : blob;
}

/** Upload media under its SHA-256 (the server checks it; idempotent). */
export async function putMedia(hash: string, blob: Blob): Promise<void> {
  if (projectsFixture()) return viaFixture(() => fixtureProjects.putMedia(hash, blob));
  const res = await fetch(server + mediaPath(hash), { method: 'PUT', headers: { ...headers(), 'Content-Type': blob.type || 'application/octet-stream' }, body: blob });
  if (!res.ok) await fail(res);
}
