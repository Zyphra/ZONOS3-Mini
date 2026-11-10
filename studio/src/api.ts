/**
 * The only module that talks to the ZONOS3 server. Everything else sees the contract types below, which alias the
 * schemas of the v2 API spec (`servers/api/openapi.json`, generated into `./api/schema.ts` by `pnpm gen:api`).
 *
 * Backends (picked once by `connect()`):
 *  - live:    the server serves GET /v2/controls; requests carry `controls` / `control_spans`.
 *  - fixture: the server answers but has no /v2/controls (404): the bundled fixture describes the controls,
 *             speakers / previews / renders are real, and steering fields are stripped (the server can't apply them).
 *  - offline: no server answers: fixture controls, fixture voices, renders from a small client-side synthesizer
 *             that applies the contract's steering math so every control is audible.
 *  - web:     the in-browser server (servers/web): ZONOS3 runs in this tab on WebGPU behind a fetch-compatible
 *             transport loaded from `?engine=URL` (default /web-engine/engine.js); models stream from /models/.
 *             Picked in `auto` when no server answers and the browser has WebGPU and the engine is deployed.
 * Force one with `?api=live|fixture|offline|web` (or VITE_ZONOS_API); point at another server with `?server=URL`.
 */
import type { components, paths } from './api/schema';
import fixtureSpec from './fixtures/controls.json';
import { FIXTURE_DECODERS } from './fixtures/decoders';
import { fixtureSpeakers, syntheticEmbedding } from './fixtures/embeddings';
import { synthesizeWords } from './fixtures/synth';
import * as fixtureEdit from './fixtures/edit';
import * as fixturePron from './fixtures/phonemize';
import * as fixtureProjects from './fixtures/projects';
import * as fixtureNv from './fixtures/nv';
import fixturePhonemes from './fixtures/phonemes.json';
import { unknownDecoder } from './lib/decoders';
import { ApiError, bodyOf, createHttp, errorOf, MissingMedia, newSessionId, readError, readFloat32Stream, RevisionConflict } from './lib/http';
import { wavBytes } from './lib/wav';
import { encodeNpy, bufferToDataUrl, blobToDataUrl, slug } from './lib/files';
import { onEncoderProgress, selectWebEngine, startWebEngine, transport, webEngineAvailable } from './lib/web-engine';
import { bootProgress, bootStage, encoderDownload } from './state/app';

// ── Contract types ────────────────────────────────────────────────────────────────────────────────

type Schemas = components['schemas'];

export type SpeakerMode = Schemas['Speaker']['speaker_modes'][number];
/** Primitive direction id (`set.direction`) → raw dose. */
export type Dose = Record<string, number>;

export type SetSpec = Schemas['ControlSet'];
export type DirectionSpec = Schemas['ControlDirection'];

// The spec passes the bundle's `ui` block through untyped: its shape is the studio's own (the bundle writer's) contract.

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
  /** Painted on a word or a few, never on a whole voice or line (emphasis: the line's prominent word). */
  span_only?: boolean;
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
export type Timing = Schemas['ControlsDocument']['timing'];

/** `GET /v2/controls`, with the studio's typing of the bundle's `ui` block. */
export interface ControlsSpec extends Omit<Schemas['ControlsDocument'], 'ui'> {
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

/** A voice of `GET /v2/speakers`; `transcript` is the known text of its reference clip (for speech editing). */
export type Speaker = Schemas['Speaker'];

/** Voices cached from an embedding file have no reference clip: no Exact clone, no reference preview. */
export const embeddingOnly = (s: Speaker | null | undefined) => !!s?.source_type?.startsWith('embedding');

export type ControlSpan = Schemas['ControlSpan'];
export type GenerateRequest = Schemas['GenerateRequest'];
export type SpanFollow = NonNullable<GenerateRequest['control_span_follow']>;

export interface Rendered {
  pcm: Float32Array;
  sr: number;
  ttfbMs: number;
  totalMs: number;
  /** How the server placed `control_spans` (X-Control-Span-Follow); absent without spans. */
  spanFollow?: SpanFollow;
  /** The decoder that made the audio (X-Decoder); absent from servers that don't say. */
  decoder?: string;
}

export type BackendKind = 'live' | 'fixture' | 'offline' | 'web';

export interface Backend {
  kind: BackendKind;
  server: string;
  spec: ControlsSpec;
  /** The server applies `controls` / `control_spans` (live), or the offline synth emulates them. */
  steering: boolean;
  edit: EditCaps;
  pron: PronCaps;
  /** Shared projects on the server (`/v2/capabilities.projects`); null: projects stay in this browser. */
  projects: ProjectCaps | null;
  /** Nonverbal markup + tag guidance (`/v2/capabilities.tag_cfg`); null: tags are plain text to this server. */
  tags: TagCaps | null;
  /** Audio decoders a request may pick with `decoder` (`/v2/capabilities.decoders`, the default first); [] when none. */
  decoders: DecoderInfo[];
}

type Capabilities = Schemas['Capabilities'];

/** An audio decoder of the server's codec (`/v2/capabilities.decoders`). */
export type DecoderInfo = Schemas['DecoderInfo'];

/** `/v2/capabilities.tag_cfg` of a checkpoint trained on tags; `fixture`: renders run on the client-side synth. */
export type TagCaps = Omit<Schemas['TagCfg'], 'available' | 'trained'> & { fixture: boolean };

/** `/v2/capabilities.projects` when enabled; `fixture`: /v2/projects and /v2/media run on the client-side fixture store. */
export type ProjectCaps = Omit<Extract<Schemas['ProjectsCapability'], { enabled: true }>, 'enabled'> & { fixture: boolean };

/** A project revision, echoed back as `base_revision`. */
export type Revision = Schemas['Project']['revision'];

export type ServerProjectEntry = Schemas['ProjectSummary'];
export type ServerProject = Schemas['Project'];
export type ProjectPut = Schemas['ProjectPutRequest'];
export type ProjectSaved = Schemas['ProjectSaved'];

/** Pronunciation input (`/v2/capabilities`); `fixture`: /v2/phonemize runs on the client-side fixture. */
export type PronCaps = Pick<Capabilities, 'phoneme_tokens' | 'phonemize_enabled' | 'phonemize_languages' | 'phonemize_guess_languages'> & { fixture: boolean };

export type PhonemeStatus = NonNullable<Schemas['PhonemeSegment']['status']>;
/** Concatenated segment texts give back the input; words carry a status (and IPA when ok / guessed). */
export type PhonemizeResponse = Schemas['PhonemizeResponse'];

/** `/v2/capabilities` speech-editing fields; `fixture`: edits run on the client-side fixture. `edit_transcribe`: the
 * server has speech recognition (without it, a clip's transcript is typed in and only aligned). */
export type EditCaps = Pick<Capabilities, 'speech_editing' | 'edit_transcribe' | 'edit_max_spans' | 'edit_max_seconds' | 'edit_align_max_seconds'> & {
  edit_frame_rate: number;
  fixture: boolean;
};

export type EditWord = Schemas['EditWord'];
/** `/v2/edit/transcribe` and `/v2/edit/align`. */
export type Alignment = Schemas['EditTranscript'];
export type EditRequest = Schemas['EditRequest'];
export type EditTake = Schemas['EditTake'];
export type EditResponse = Schemas['EditResponse'];

// ── Configuration ─────────────────────────────────────────────────────────────────────────────────

const params = new URLSearchParams(location.search);
const forced = (params.get('api') || import.meta.env.VITE_ZONOS_API || 'auto') as BackendKind | 'auto';
const server = (params.get('server') || import.meta.env.VITE_ZONOS_SERVER_URL || '').replace(/\/+$/, '');

if (forced === 'web') selectWebEngine();

export const SAMPLE_RATE = 44100;

function sessionId(): string {
  let id = localStorage.getItem('zvs.session');
  if (!id) {
    id = newSessionId();
    localStorage.setItem('zvs.session', id);
  }
  return id;
}

const { client, headers, ok, send } = createHttp<paths>({ baseUrl: server, session: sessionId, fetch: transport });

// ── Backend detection ─────────────────────────────────────────────────────────────────────────────

let backend: Backend | null = null;
const fixture = fixtureSpec as unknown as ControlsSpec;

export function current(): Backend {
  if (!backend) throw new Error('api.connect() has not run');
  return backend;
}

const FIXTURE_EDIT: EditCaps = {
  speech_editing: true,
  edit_transcribe: true,
  edit_max_spans: 3,
  edit_frame_rate: 15,
  edit_max_seconds: 150,
  edit_align_max_seconds: 150,
  fixture: true,
};
const NO_EDIT: EditCaps = { ...FIXTURE_EDIT, speech_editing: false, edit_transcribe: false, fixture: false };

/**
 * Speech editing: the server's `/v2/capabilities` when one answers, the fixture engine offline.
 * `?edit=fixture` forces the fixture engine, `?edit=0` a checkpoint that cannot edit, `?edit_transcribe=0` a server
 * without speech recognition (transcripts are typed in).
 */
function editCaps(kind: BackendKind, c: Capabilities | null): EditCaps {
  const want = params.get('edit');
  if (want === '0' || want === 'off') return { ...NO_EDIT, fixture: kind !== 'live' };
  const asr = !/^(0|off)$/.test(params.get('edit_transcribe') ?? '');
  if (want === 'fixture' || kind === 'offline') return { ...FIXTURE_EDIT, edit_transcribe: asr };
  if (!c) return NO_EDIT;
  return {
    speech_editing: c.speech_editing,
    edit_transcribe: c.edit_transcribe && asr,
    edit_max_spans: c.edit_max_spans,
    edit_frame_rate: c.edit_frame_rate ?? FIXTURE_EDIT.edit_frame_rate,
    edit_max_seconds: c.edit_max_seconds,
    edit_align_max_seconds: c.edit_align_max_seconds,
    fixture: false,
  };
}

const FIXTURE_PRON: PronCaps = { ...(fixturePhonemes as Omit<PronCaps, 'fixture'>), fixture: true };
const NO_PRON: PronCaps = { phoneme_tokens: null, phonemize_enabled: false, phonemize_languages: [], phonemize_guess_languages: [], fixture: false };

/**
 * Pronunciation input: the server's capabilities when one answers, the fixture offline (the real 136-symbol
 * inventory of the phoneme-token checkpoint). `?phon=fixture` forces the fixture, `?phon=0` a checkpoint without.
 */
function pronCaps(kind: BackendKind, c: Capabilities | null): PronCaps {
  const want = params.get('phon');
  if (want === '0' || want === 'off') return { ...NO_PRON, fixture: kind !== 'live' };
  if (want === 'fixture' || kind === 'offline') return FIXTURE_PRON;
  if (!c) return NO_PRON;
  return {
    phoneme_tokens: c.phoneme_tokens,
    phonemize_enabled: c.phonemize_enabled,
    phonemize_languages: c.phonemize_languages,
    phonemize_guess_languages: c.phonemize_guess_languages,
    fixture: false,
  };
}

const FIXTURE_TAGS: TagCaps = { tags: ['laugh', 'breath', 'sigh', 'cough', 'panting', 'shouting', 'crying'], min_scale: 1, max_scale: 4, default_scale: 1, fixture: true };

/**
 * Nonverbal tags: the server's `tag_cfg` when available, the fixture synth offline.
 * `?nv=fixture` forces the fixture (renders then run on the client-side synth), `?nv=0` a server without tags.
 */
function tagCaps(kind: BackendKind, c: Schemas['TagCfg'] | undefined): TagCaps | null {
  const want = params.get('nv');
  if (want === '0' || want === 'off') return null;
  if (want === 'fixture' || kind === 'offline') return FIXTURE_TAGS;
  // Tag guidance runs on any byte / SuperBPE checkpoint, but only one trained on tags reads them: else none at all.
  if (!c?.available || !c.trained) return null;
  return {
    tags: c.tags.length ? c.tags : FIXTURE_TAGS.tags,
    min_scale: c.min_scale,
    max_scale: c.max_scale,
    default_scale: c.default_scale,
    fixture: false,
  };
}

const FIXTURE_PROJECTS: ProjectCaps = { max_media_bytes: 2e9, max_project_bytes: 20e6, fixture: true };

/**
 * Projects on the server when its capabilities say so, else in this browser (IndexedDB).
 * `?projects=fixture` forces the fixture server store, `?projects=browser` the browser store.
 */
function projectCaps(c: Schemas['ProjectsCapability'] | undefined): ProjectCaps | null {
  const want = params.get('projects');
  if (want === 'fixture') return FIXTURE_PROJECTS;
  if (want === 'browser' || want === '0' || !c?.enabled) return null;
  return { max_media_bytes: c.max_media_bytes, max_project_bytes: c.max_project_bytes, fixture: false };
}

/** Where the studio looks for a server: `?server=`, else this site. */
export const serverLabel = () => server || location.origin;

/** The in-browser server: load the engine and its models (progress on the boot screen), then serve from this tab. */
async function startWeb(make: (kind: BackendKind, spec: ControlsSpec) => Promise<Backend>): Promise<Backend> {
  bootStage.value = 'Loading the in-browser engine…';
  await startWebEngine((d) => (bootProgress.value = d));
  bootProgress.value = null;
  bootStage.value = 'Starting the voice engine…';
  // A clip voice's first use downloads the encoders, the first full-size decoder request that decoder: a bar in the corner until they are in.
  void onEncoderProgress((d) => (encoderDownload.value = d.fraction >= 1 ? null : d));
  return make('web', await getControls());
}

export interface ConnectOptions {
  /** No server answered and this browser could run the engine: asked before the (large) download. `false` → offline. */
  confirmWeb?: () => Promise<boolean>;
}

export async function connect(opts: ConnectOptions = {}): Promise<Backend> {
  const make = async (kind: BackendKind, spec: ControlsSpec): Promise<Backend> => {
    const caps = kind === 'offline' ? null : await ok<Capabilities>(client.GET('/v2/capabilities')).catch(() => null);
    // The in-browser server answers like a live one; its capabilities say which features are off.
    const as = kind === 'web' ? 'live' : kind;
    return (backend = {
      kind,
      server,
      spec,
      steering: kind !== 'fixture',
      edit: editCaps(as, caps),
      pron: pronCaps(as, caps),
      projects: projectCaps(caps?.projects),
      tags: tagCaps(as, caps?.tag_cfg),
      // Audio decoders: the server's when one answers, the fixture's two offline.
      decoders: kind === 'offline' ? FIXTURE_DECODERS : (caps?.decoders ?? []),
    });
  };
  if (forced === 'offline') return make('offline', fixture);
  if (forced === 'web') return startWeb(make);
  bootStage.value = `Connecting to ${serverLabel()}…`;
  try {
    return await make('live', await getControls());
  } catch (err) {
    if (forced === 'live') throw err;
    if (err instanceof ApiError && err.status === 404) return make('fixture', fixture);
    if (forced === 'fixture') return make('fixture', fixture);
    // Nothing (or not a ZONOS server) answered: the in-browser server when this browser and site can run it (asked first).
    if (forced === 'auto' && (await webEngineAvailable()) && (!opts.confirmWeb || (await opts.confirmWeb()))) {
      selectWebEngine();
      return startWeb(make);
    }
    // Else run on the client-side fixture.
    return make('offline', fixture);
  }
}

/** The forced mode (`?api=`), for the shell's messages. */
export const forcedMode = () => forced;
/** This browser and site could run the in-browser engine. */
export const canRunWeb = webEngineAvailable;

export async function getControls(): Promise<ControlsSpec> {
  if (forced === 'fixture') throw new ApiError(404, 'fixture forced');
  // The bundle's `ui` block is the studio's own contract (the spec passes it through untyped).
  return (await ok<Schemas['ControlsDocument']>(client.GET('/v2/controls'))) as ControlsSpec;
}

// ── Voices ────────────────────────────────────────────────────────────────────────────────────────

const offlineVoices = new Map<string, { speaker: Speaker; embedding?: Float32Array }>();

/** A voice cached in this browser (offline backend), shaped as the server's session voices are. */
const offlineSpeaker = (label: string, source_type: Speaker['source_type'], name: string): Speaker => ({
  id: `local_${slug(label)}_${Date.now().toString(36)}`,
  label,
  source_type,
  scope: 'session',
  is_default: false,
  original_name: name,
  created_at: Date.now() / 1000,
  has_preview: source_type === 'audio',
  speaker_modes: source_type === 'audio' ? ['embedding', 'reference'] : ['embedding'],
  embedding_dim: current().spec.embedding_dim,
  transcript: null,
  disabled_controls: [],
});

export async function getSpeakers(): Promise<Speaker[]> {
  if (current().kind === 'offline') {
    // Default voices carry their clip's transcript (as `<voice>.txt` does on the server); two are left without
    // one so the transcribe path is exercised too.
    const defaults = fixtureSpeakers(current().spec).map((s) => ({ ...s, transcript: /Skyler|Yelling/.test(s.label) ? null : offlinePreviewText(s) }));
    return [...defaults, ...[...offlineVoices.values()].map((v) => v.speaker)];
  }
  return (await ok<Schemas['SpeakerList']>(client.GET('/v2/speakers'))).speakers;
}

/** The vector the server conditions on. Synthetic (and flagged) when the server can't provide it. */
export async function getEmbedding(speaker: Speaker): Promise<{ embedding: Float32Array; synthetic: boolean }> {
  const b = current();
  if (b.kind !== 'offline') {
    try {
      const body = await ok<Schemas['SpeakerEmbedding']>(client.GET('/v2/speakers/{speaker_id}/embedding', { params: { path: { speaker_id: speaker.id } } }));
      return { embedding: Float32Array.from(body.embedding), synthetic: false };
    } catch (err) {
      if (!(err instanceof ApiError) || err.status !== 404 || b.kind === 'live' || b.kind === 'web') throw err;
    }
  }
  const own = offlineVoices.get(speaker.id)?.embedding;
  return { embedding: own ?? syntheticEmbedding(b.spec, speaker.id, speaker.label), synthetic: b.kind !== 'offline' };
}

const offlinePreviewText = (s: Speaker) => `Hi, I'm ${s.label}. This is how I sound when I read a short line.`;

/** The voice's reference clip, as an encoded audio file (`GET /v2/speakers/{id}/preview`, binary). */
export async function getPreview(speaker: Speaker): Promise<ArrayBuffer> {
  if (current().kind === 'offline') {
    const { embedding } = await getEmbedding(speaker);
    const text = offlinePreviewText(speaker);
    const { pcm, words } = synthesizeWords(current().spec, { text, embedding, seed: 7 });
    fixtureEdit.remember(pcm, SAMPLE_RATE, { text, words, embedding });
    return wavBytes(pcm, SAMPLE_RATE).buffer;
  }
  const res = await transport(`${server}/v2/speakers/${encodeURIComponent(speaker.id)}/preview`, { headers: headers() });
  if (!res.ok) throw await readError(res);
  return res.arrayBuffer();
}

/** Cache an uploaded / recorded clip as a voice (server computes its embedding). */
export async function uploadVoice(label: string, audio: Blob, fileName: string): Promise<Speaker> {
  if (current().kind === 'offline') {
    const speaker = offlineSpeaker(label, 'audio', fileName);
    offlineVoices.set(speaker.id, { speaker });
    return speaker;
  }
  const body: Schemas['SpeakerCreateRequest'] = { label, speaker_audio_base64: await blobToDataUrl(audio), speaker_audio_name: fileName };
  return ok<Speaker>(client.POST('/v2/speakers', { body }));
}

/** Cache a designed voice: its steered embedding, as a .npy. */
export async function saveEmbeddingVoice(label: string, embedding: Float32Array): Promise<Speaker> {
  const name = `${slug(label)}.npy`;
  if (current().kind === 'offline') {
    const speaker = offlineSpeaker(label, 'embedding', name);
    offlineVoices.set(speaker.id, { speaker, embedding: embedding.slice() });
    return speaker;
  }
  const body: Schemas['SpeakerCreateRequest'] = {
    label,
    speaker_embedding_base64: await bufferToDataUrl(encodeNpy(embedding), 'application/octet-stream'),
    speaker_embedding_name: name,
  };
  return ok<Speaker>(client.POST('/v2/speakers', { body }));
}

// ── Generation ────────────────────────────────────────────────────────────────────────────────────

export interface GenerateOptions {
  signal?: AbortSignal;
  /** Called with each float32 chunk as it streams in. */
  onChunk?: (chunk: Float32Array, sr: number) => void;
}

/** Languages the offline fixture "has alignment heads" for: the others fall back to the estimate, as on the server. */
const FIXTURE_READING_LANGUAGES = new Set(['en_us', 'fr_fr']);

/** `POST /v2/generate`, streamed float32 PCM (raw fetch: openapi-fetch has no streaming body reader). */
export async function generate(req: GenerateRequest, opts: GenerateOptions = {}): Promise<Rendered> {
  const b = current();
  const t0 = performance.now();
  const body: GenerateRequest = { stream: true, ...req };
  if (!b.steering) {
    delete body.controls;
    delete body.control_spans;
  }
  if (!body.control_spans?.length) delete body.control_span_follow;
  if (!b.tags || (body.tag_cfg_scale ?? 1) <= 1) delete body.tag_cfg_scale;
  if (b.kind === 'offline' || b.tags?.fixture) return offlineGenerate(body, opts, t0);

  const res = await transport(`${server}/v2/generate`, { method: 'POST', headers: headers({ 'Content-Type': 'application/json' }), body: JSON.stringify(body), signal: opts.signal });
  if (!res.ok) throw await readError(res);
  const spanFollow: SpanFollow | undefined = body.control_spans?.length ? (res.headers.get('X-Control-Span-Follow') === 'reading' ? 'reading' : 'rate') : undefined;
  const decoder = res.headers.get('X-Decoder');
  const sr = Number(res.headers.get('X-Audio-Sample-Rate')) || SAMPLE_RATE;
  if (!res.body) throw new ApiError(502, 'Empty audio stream');

  const chunks: Float32Array[] = [];
  let total = 0;
  let ttfbMs = 0;
  await readFloat32Stream(
    res.body,
    (chunk) => {
      chunks.push(chunk);
      total += chunk.length;
      opts.onChunk?.(chunk, sr);
    },
    () => {
      if (!ttfbMs) ttfbMs = performance.now() - t0;
    },
  );
  const pcm = new Float32Array(total);
  let o = 0;
  for (const c of chunks) {
    pcm.set(c, o);
    o += c.length;
  }
  return { pcm, sr, ttfbMs, totalMs: performance.now() - t0, ...(spanFollow ? { spanFollow } : {}), ...(decoder ? { decoder } : {}) };
}

async function offlineGenerate(req: GenerateRequest, opts: GenerateOptions, t0: number): Promise<Rendered> {
  const b = current();
  fixturePron.validateGroups(req.text, b.pron.phoneme_tokens);
  const decoder = fixtureDecoder(req.decoder);
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
  // The server's tag-guidance rules: a recognized tag, and one twin per request (no speaker / controls CFG).
  const tagScale = req.tag_cfg_scale ?? 1;
  const nv = b.tags ? fixtureNv.prepare(req.text, req.control_spans ?? undefined) : null;
  if (tagScale > 1) {
    if (!nv?.tagged) throw new ApiError(400, 'tag_cfg_scale > 1 needs a recognized nonverbal tag in the text.');
    if ((req.speaker_cfg_scale ?? 1) !== 1) throw new ApiError(400, 'tag_cfg_scale cannot be combined with speaker_cfg_scale ≠ 1.');
  }
  const { embedding } = await getEmbedding(speaker);
  const spoken = nv ?? { text: req.text, spans: req.control_spans ?? undefined };
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
  return { pcm, sr: SAMPLE_RATE, ttfbMs, totalMs: performance.now() - t0, ...(spanFollow ? { spanFollow } : {}), ...(decoder ? { decoder } : {}) };
}

/** The fixture's answer to `decoder`: the one served by that id (null: the default), else the server's 400. The synth sounds the same either way. */
function fixtureDecoder(id: string | null | undefined): string | undefined {
  const served = current().decoders;
  if (id && !served.some((d) => d.id === id)) throw new ApiError(400, unknownDecoder(id, served));
  return id ?? served[0]?.id;
}

// ── Speech editing ────────────────────────────────────────────────────────────────────────────────

/** Qwen3-ASR transcript + FuseAlign word timings of a clip (plain base64 WAV). */
/** `language`: the clip's language (server code); it picks the recognition language and the alignment tokenization. */
export function transcribe(audio_base64: string, language = 'en_us', signal?: AbortSignal): Promise<Alignment> {
  if (current().edit.fixture) return fixtureEdit.transcribe(current().edit, audio_base64, language, signal);
  return ok<Alignment>(client.POST('/v2/edit/transcribe', { body: { audio_base64, language }, signal }));
}

/** Forced alignment of known text. */
export function align(audio_base64: string, text: string, language = 'en_us', signal?: AbortSignal): Promise<Alignment> {
  if (current().edit.fixture) return fixtureEdit.align(audio_base64, text, language, signal);
  return ok<Alignment>(client.POST('/v2/edit/align', { body: { audio_base64, text, language }, signal }));
}

export async function edit(req: EditRequest, signal?: AbortSignal): Promise<EditResponse> {
  if (current().edit.fixture) {
    const tokens = current().pron.phoneme_tokens;
    fixturePron.validateGroups(req.target_text, tokens);
    if (/[\ue000\ue001]/.test(req.original_text ?? '')) throw new ApiError(400, 'original_text describes the recorded audio: it cannot hold pronunciation groups.');
    fixtureDecoder(req.decoder);
    return fixtureEdit.edit(current().spec, current().edit, req, signal);
  }
  return ok<EditResponse>(client.POST('/v2/edit', { body: req, signal }));
}

// ── Pronunciation ─────────────────────────────────────────────────────────────────────────────────

/** Lexicon IPA for the words of `text` (`guess`: espeak-ng fills in the languages that allow it). */
export function phonemize(text: string, language: string, guess = true): Promise<PhonemizeResponse> {
  const b = current();
  if (b.pron.fixture) return fixturePron.phonemize(b.pron, text, language, guess);
  return ok<PhonemizeResponse>(client.POST('/v2/phonemize', { body: { text, language, guess } }));
}

// ── Projects (shared on the server) ───────────────────────────────────────────────────────────────

const projectsFixture = () => current().projects?.fixture === true;

/** A fixture error as the server's would be. */
async function viaFixture<T>(fn: () => Promise<T>): Promise<T> {
  try {
    return await fn();
  } catch (e) {
    if (!(e instanceof fixtureProjects.FixtureError)) throw e;
    if (e.status === 409) throw new RevisionConflict(e.body.revision ?? null);
    if (e.body.missing) throw new MissingMedia(e.body.missing);
    throw new ApiError(e.status, e.detail);
  }
}

const projectParams = (id: string) => ({ params: { path: { project_id: id } } });
/** `/v2/media/{sha256}` (raw bodies: kept on fetch). */
const mediaUrl = (hash: string) => `${server}/v2/media/${encodeURIComponent(hash)}`;

/** Every project on the server, newest first. */
export function listProjects(): Promise<ServerProjectEntry[]> {
  if (projectsFixture()) return viaFixture(fixtureProjects.list);
  return ok<ServerProjectEntry[]>(client.GET('/v2/projects'));
}

export function getProject(id: string): Promise<ServerProject> {
  if (projectsFixture()) return viaFixture(() => fixtureProjects.get(id));
  return ok<ServerProject>(client.GET('/v2/projects/{project_id}', projectParams(id)));
}

/** Save a project; `base_revision` is the revision it was loaded at (null: a new project). 409 → RevisionConflict. */
export async function putProject(id: string, body: ProjectPut): Promise<ProjectSaved> {
  if (projectsFixture()) return viaFixture(() => fixtureProjects.put(id, body));
  const { response, text } = await send(client.PUT('/v2/projects/{project_id}', { ...projectParams(id), body }));
  if (response.ok) return bodyOf<ProjectSaved>(response.status, text);
  // 409 / 400 carry the spec's Error body with `revision` / `missing`.
  let err: Partial<Schemas['Error']> = {};
  try {
    err = JSON.parse(text) ?? {};
  } catch {
    /* not JSON: errorOf below */
  }
  if (response.status === 409) throw new RevisionConflict(err.revision ?? null);
  if (response.status === 400 && err.missing) throw new MissingMedia(err.missing);
  throw errorOf(response.status, response.statusText, text);
}

export async function deleteProject(id: string): Promise<void> {
  if (projectsFixture()) return viaFixture(() => fixtureProjects.remove(id));
  const { response, text } = await send(client.DELETE('/v2/projects/{project_id}', projectParams(id)));
  if (!response.ok && response.status !== 404) throw errorOf(response.status, response.statusText, text);
}

/** Whether the server holds media `hash` (HEAD: 200 / 404). */
export async function hasMedia(hash: string): Promise<boolean> {
  if (projectsFixture()) return viaFixture(() => fixtureProjects.hasMedia(hash));
  const res = await transport(mediaUrl(hash), { method: 'HEAD', headers: headers() });
  if (res.status === 404) return false;
  if (!res.ok) throw await readError(res);
  return true;
}

export async function getMedia(hash: string, type?: string): Promise<Blob> {
  if (projectsFixture()) return viaFixture(() => fixtureProjects.getMedia(hash));
  const res = await transport(mediaUrl(hash), { headers: headers() });
  if (!res.ok) throw await readError(res);
  const blob = await res.blob();
  // The stored type when the server kept one, else the document's.
  return type && blob.type !== type ? new Blob([blob], { type }) : blob;
}

/** Upload media under its SHA-256 (the server checks it; idempotent: 201 stored, 200 already held). */
export async function putMedia(hash: string, blob: Blob): Promise<void> {
  if (projectsFixture()) return viaFixture(() => fixtureProjects.putMedia(hash, blob));
  const res = await transport(mediaUrl(hash), { method: 'PUT', headers: headers({ 'Content-Type': blob.type || 'application/octet-stream' }), body: blob });
  if (!res.ok) throw await readError(res);
}
