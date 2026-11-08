// POST /v2/generate and POST /v1/audio/speech: a GenerateRequest resolved against the loaded models, run by a
// one-request scheduler (CFG twin on the backbone's second slot, LACI, span follow, hidden steering) and answered as
// streamed / buffered float32 PCM or a buffered 16-bit WAV. Ports servers/ggml/src/server/generate.cpp (resolve,
// plan_generate, make_job, the routes) and scheduler.cpp (attach_probe, laci_resume, step_steer, the worker tick,
// the ZAC lane's prefix / whole decode).
import type { ProbeInput } from '../backbone';
import { locateTextSpan, probeReduce } from '../probe';
import {
  ALIGN_PROBE_MAX_TEXT, CONTEXT_CAPACITY, DECODE_CONTEXT_FRAMES, HOP, MAX_FRAMES, SAMPLE_RATE, SAMPLING, SILENCE_PREFILL_MAX_FRAMES,
  STREAM_CODEBOOKS, STREAM_CODEBOOKS_SCOPE, TAG_CFG_MAX_SCALE, TAG_CFG_MIN_SCALE,
} from '../generation';
import { buildCfgTwin, buildPrompt, conditionBucket, NCB, prefillCodes, type Conditioning, type Prompt } from '../prompt';
import { LaciSession, defaultLaciConfig, RateBuckets, type LaciHeads, type ProbeControl, type Windows } from '../request/laci';
import { hasNonverbalMarkup, NONVERBAL_MARKUP_TAGS, nonverbalMarkerSpans, stripNonverbalMarkup } from '../request/markup';
import { defaultFollowConfig, SpanFollower, spanTokenRanges, tokenCharRanges, type Range } from '../request/spanFollow';
import { NO_STEERING, resolveControls, Steering, textChars, type ControlSpan } from '../request/steer';
import { defaultSampling, vocabIds, type Sampling } from '../sampler';
import { buildIdx, BLOCK } from '../schedule';
import { acceptFrame, newSlot, type SlotState } from '../slot';
import { wavBytes } from '../../../../studio/src/lib/wav';
import { applyFade } from '../wav';
import { ACCENT_ADAPTERS, laciLanguages, SILENCE_PREFILL, TEXT_NORM_LANGUAGES } from './capabilities';
import type { Engine } from './engine';
import {
  bool, fail, HttpError, isSet, num, parseBody, pyFloat, pyList, pyRepr, sessionId, str, truthy, type EngineRequest,
  type Sink,
} from './http';
import type { Speakers } from './speakers';

/** Prefix streaming (PrefixStreamSettings defaults): chunks ramp from the first chunk, doubling (1, 2, 4, 8 frames),
 *  up to the steady chunk (15), with no frame hold-back (ZONOS3_PREFIX_STREAM_* on the native servers). */
const PREFIX_CHUNK_FRAMES = 15, PREFIX_HOLD_BACK_FRAMES = 0, PREFIX_FIRST_CHUNK_FRAMES = 1;

/** A validated GenerateRequest resolved into what the scheduler runs (generate.cpp GenReq + GeneratePlan). */
interface Plan {
  text: string;                       // tokenized text (no normalization here)
  rawText: string;
  language: string;
  sp: Sampling;
  maxFrames: number;
  speakerMode: string;
  referenceCodes: Int32Array | null;
  speakerEmbedding: Float32Array | null;
  conditionBuckets: number[];
  silencePrefill: number;
  decodeContext: Int32Array | null;   // the reference's last frames, ordinary [K][10]
  cfgScale: number;
  cfgKind: '' | 'speaker' | 'tag' | 'controls';
  twinSteered: boolean;
  headroomDb: number;
  fadeOutMs: number;
  stream: boolean;
  wav: boolean;
  prefix: boolean;
  streamCodebooks: number;            // early partial-frame decode: codebooks a frame needs (NCB = complete frames)
  streamFirstOnly: boolean;           // stream_codebooks_scope 'first': only the first prefix chunk decodes early
  steering: Steering;
  spanFollow: '' | 'rate' | 'reading';
  laci: LaciHeads | null;
  follow: LaciHeads | null;
  followTokens: Range[];
  followLead: number;
  followLag: number;
  prompt: Prompt;
  twin: Prompt | null;
}

type Body = Record<string, unknown>;

/** Library errors carry the C++ 400 details: re-raise them as 400 (HttpError passes through). */
export function as400<T>(f: () => T): T {
  try {
    return f();
  } catch (e) {
    if (e instanceof HttpError) throw e;
    fail(400, e instanceof Error ? e.message : String(e));
  }
}

// ------------------------------------------------------------------------------------------------ resolve

/** zonos3.tts.plan validate_tag_cfg (R1). */
function validateTagCfg(text: string, scale: number, speakerCfg: number, controlsCfg: number): void {
  if (!Number.isFinite(scale) || scale < TAG_CFG_MIN_SCALE || scale > TAG_CFG_MAX_SCALE)
    fail(400, `tag_cfg_scale must be finite and between ${pyFloat(TAG_CFG_MIN_SCALE)} and ${pyFloat(TAG_CFG_MAX_SCALE)}.`);
  if (scale === 1.0) return;
  if (!hasNonverbalMarkup(text) || stripNonverbalMarkup(text) === text) {
    const tags = NONVERBAL_MARKUP_TAGS.map(t => `<${t}/>`).join(', ');
    fail(400, `tag_cfg_scale above 1.0 requires recognized nonverbal markup in the text: point tags ${tags}` +
      ' or the same names as span tags (<shouting>...</shouting>).');
  }
  if (speakerCfg !== 1.0) fail(400, 'tag_cfg_scale above 1.0 cannot be combined with speaker_cfg_scale; set speaker_cfg_scale to 1.0.');
  if (controlsCfg !== 1.0) fail(400, 'tag_cfg_scale above 1.0 cannot be combined with controls_cfg_scale; set controls_cfg_scale to 1.0.');
}

/** zonos3.tts.plan resolve_condition_buckets (R3): explicit ids win, raw values map through the edges, anything else is
 *  unknown (the row's bucket count n; ids 0..n-1 are buckets). */
export function resolveConditionBuckets(e: Engine, body: Body): number[] {
  const hp = e.hp;
  const buckets = isSet(body, 'condition_buckets') ? body.condition_buckets : null;
  const values = isSet(body, 'condition_values') ? body.condition_values : null;
  const nonEmpty = (v: unknown) => v !== null && (Array.isArray(v) ? v.length > 0 : Object.keys(v as object).length > 0);
  if (hp.nCond === 0) {
    if (nonEmpty(buckets) || nonEmpty(values)) fail(400, 'This checkpoint has no condition rows (loudness / fmax / speaking rate)');
    return [];
  }
  const names = hp.condNames;
  const asMap = (raw: unknown, what: string): Map<string, unknown> => {
    const out = new Map<string, unknown>();
    if (raw === null) return out;
    if (!Array.isArray(raw)) {
      const unknown: string[] = [];
      for (const [k, v] of Object.entries(raw as Record<string, unknown>)) {
        if (!names.includes(k)) unknown.push(k);
        else out.set(k, v);
      }
      if (unknown.length) fail(400, `${what}: unknown condition name(s) ${pyList(unknown)}; expected ${pyList(names)}`);
      return out;
    }
    if (raw.length !== names.length) fail(400, `${what}: expected ${names.length} entries in the order ${pyList(names)}`);
    names.forEach((n, i) => out.set(n, raw[i]));
    return out;
  };
  const ids = asMap(buckets, 'condition_buckets'), vals = asMap(values, 'condition_values');
  return names.map((name, i) => {
    const id = ids.get(name), val = vals.get(name), nBuckets = hp.condBucketCounts[i];
    if (id !== undefined && id !== null) {
      const b = Number(id);
      if (b < 0 || b > nBuckets) fail(400, `condition_buckets[${name}] must be in [0, ${nBuckets - 1}] or ${nBuckets} (= unknown)`);
      return b;
    }
    if (val !== undefined && val !== null) {
      const v = Number(val);
      if (!Number.isFinite(v)) fail(400, `condition_values[${name}] must be finite`);
      const edges = hp.condEdges[i] ?? [];
      if (!edges.length) fail(400, `the checkpoint carries no edges for ${name}; send condition_buckets[${name}] instead`);
      return conditionBucket(edges, v);
    }
    return nBuckets;
  });
}

/** zonos3.languages normalize (R4). */
export function resolveLanguage(body: Body): string {
  const given = str(body, 'language') ?? 'en_us';
  const lang = given.trim().toLowerCase().replaceAll('-', '_');
  if (!TEXT_NORM_LANGUAGES.includes(lang))
    fail(400, `Unsupported language code: ${pyRepr(given)}. Supported: ${TEXT_NORM_LANGUAGES.join(', ')}.`);
  return lang;
}

function requestEmbedding(p: Plan): Float32Array | null {
  return p.speakerMode === 'embedding' && p.speakerEmbedding ? p.speakerEmbedding : null;
}

/** zonos3.tts.plan resolve_accent (R9): `accent` picks an accent adapter (ACCENT_ADAPTERS). */
function checkAccent(body: Body): void {
  const accent = str(body, 'accent');
  if (accent === null || accent === '' || accent === 'none') return;
  if (!ACCENT_ADAPTERS.includes(accent)) {
    const loaded = ACCENT_ADAPTERS.length ? ACCENT_ADAPTERS.join(', ') : 'none (load accent adapters)';
    fail(400, `Unknown accent ${pyRepr(accent)}; loaded accents: ${loaded}.`);
  }
}

/** zonos3.tts.plan request_rules' steering controls (R10). */
function applyControls(e: Engine, body: Body, p: Plan): void {
  if (!truthy(body, 'controls') && !truthy(body, 'control_spans')) return;
  if (!e.steer)
    fail(400, `${NO_STEERING}; \`controls\` are unavailable.`);
  const controls = isSet(body, 'controls') ? body.controls as Record<string, number> : {};
  const spans = isSet(body, 'control_spans') ? body.control_spans as ControlSpan[] : [];
  const rate = num(body, 'control_span_rate', 0);
  const embedding = p.speakerMode === 'embedding' ? p.speakerEmbedding : null;
  p.steering = as400(() => resolveControls(e.steer!, controls, spans, textChars(p.rawText), rate, embedding));
}

/** zonos3.tts.plan check_laci (R14). */
function checkLaci(e: Engine, body: Body, language: string): void {
  if (!bool(body, 'laci')) return;
  if (!e.laci.size) fail(400, 'LACI is not enabled on this server (start it with --laci-heads LANG=PATH)');
  if (!e.laci.has(language))
    fail(400, `No LACI alignment heads for language ${pyRepr(language)} (configured: ${laciLanguages(e).join(', ')})`);
  if (num(body, 'speaker_cfg_scale', 1) !== 1 || num(body, 'controls_cfg_scale', 1) !== 1)
    fail(400, 'LACI needs speaker_cfg_scale = controls_cfg_scale = 1');
  if (num(body, 'tag_cfg_scale', 1) !== 1) fail(400, 'LACI needs tag_cfg_scale = 1');
}

/** The tokenizer's follow tokens (span_follow.span_token_ranges). */
function resolveFollowTokens(e: Engine, p: Plan): void {
  if (!p.follow) return;
  const tok = e.tokenizer;
  const enc = new TextEncoder();
  // SuperBPE offsets: a token's characters run from the character of its first byte to the one of its last.
  const bpe = (piece: string): Range[] => {
    const out: Range[] = [];
    const utf8 = enc.encode(piece);
    const nChars = Array.from(piece).length;
    const charOf = new Array<number>(utf8.length + 1).fill(nChars);
    for (let b = 0, c = 0; b < utf8.length; b++) {
      if (b > 0 && (utf8[b] & 0xc0) !== 0x80) c++;
      charOf[b] = c;
    }
    const ids = tok.encode(piece);
    let at = 0;
    for (let i = 1; i + 1 < ids.length; i++) {
      const n = tok.tokenBytes(ids[i]);
      if (n === 0) continue;
      out.push([charOf[at], charOf[at + n - 1] + 1]);
      at += n;
    }
    return out;
  };
  const spans: Range[] = p.steering.spans.map(s => [s.charStart, s.charEnd]);
  const ranges = tokenCharRanges(p.text, bpe, null);
  p.followTokens = spanTokenRanges(p.rawText, p.text, spans, ranges, nonverbalMarkerSpans(p.text));
}

/** GenerateRequest -> Plan (generate.cpp resolve + plan_generate). */
export async function planGenerate(e: Engine, spk: Speakers, body: Body, session: string | null,
                                   pcm: Float32Array | undefined): Promise<Plan> {
  const text = str(body, 'text')!;
  const speakerCfg = num(body, 'speaker_cfg_scale', 1), controlsCfg = num(body, 'controls_cfg_scale', 1);
  validateTagCfg(text, num(body, 'tag_cfg_scale', 1), speakerCfg, controlsCfg);
  // zonos3.tts.plan resolve_speaker_mode (R2)
  const modes = spk.checkpointModes();
  let mode = modes.length ? modes[0] : '';
  if (isSet(body, 'speaker_mode')) {
    mode = str(body, 'speaker_mode')!;
    if (!modes.includes(mode))
      fail(400, `speaker_mode=${pyRepr(mode)} is not available for this checkpoint (supported: ${pyList(modes)})`);
  }
  const conditionBuckets = resolveConditionBuckets(e, body);
  const rs = await spk.resolveRequest(body, session, mode, pcm);
  const p: Plan = {
    text, rawText: text, language: 'en_us', sp: defaultSampling(), maxFrames: MAX_FRAMES, speakerMode: rs.mode,
    referenceCodes: rs.codes, speakerEmbedding: rs.embedding ? Float32Array.from(rs.embedding) : null,
    conditionBuckets, silencePrefill: 0, decodeContext: null,
    cfgScale: 1, cfgKind: '', twinSteered: true, headroomDb: -1, fadeOutMs: 0, stream: true, wav: false, prefix: false,
    streamCodebooks: STREAM_CODEBOOKS, streamFirstOnly: STREAM_CODEBOOKS_SCOPE === 'first',
    steering: new Steering(), spanFollow: '', laci: null, follow: null, followTokens: [],
    followLead: 1, followLag: 0, prompt: null!, twin: null,
  };
  p.language = resolveLanguage(body);
  // zonos3.tts.plan max_tokens_budget (R5; fit to the slot after the prompt is built) and silence_prefill_frames (R6)
  if (isSet(body, 'max_tokens')) {
    const requested = Number(body.max_tokens);
    if (requested <= 0) fail(400, 'max_tokens must be positive.');
    p.maxFrames = Math.min(requested, MAX_FRAMES);
  }
  p.silencePrefill = SILENCE_PREFILL.on ? SILENCE_PREFILL.frames : 0;
  if (isSet(body, 'silence_prefill_frames')) {
    const frames = Number(body.silence_prefill_frames);
    if (!Number.isInteger(frames) || frames < 0 || frames > SILENCE_PREFILL_MAX_FRAMES)
      fail(400, `silence_prefill_frames must be 0..${SILENCE_PREFILL_MAX_FRAMES}`);
    p.silencePrefill = frames;
  }
  checkAccent(body);
  applyControls(e, body, p);
  // zonos3.tts.plan request_rules' span follow (R11)
  if (str(body, 'control_span_follow') === 'reading' && p.steering.spans.length) {
    const heads = e.laci.get(p.language);
    if (heads) {
      const d = defaultFollowConfig();
      p.follow = heads;
      p.followLead = num(body, 'control_span_lead', d.lead);
      p.followLag = num(body, 'control_span_lag', d.lag);
    }
  }
  if (truthy(body, 'control_spans')) p.spanFollow = p.follow ? 'reading' : 'rate';
  p.stream = bool(body, 'stream') ?? true;
  p.prefix = p.stream && (isSet(body, 'stream_decode') ? str(body, 'stream_decode') === 'prefix' : true);
  // the spec validates the type, minimum 1 and the scope enum (422); above n_codebooks is a 400
  if (isSet(body, 'stream_codebooks')) {
    p.streamCodebooks = Number(body.stream_codebooks);
    if (p.streamCodebooks > NCB) fail(400, `stream_codebooks must be 1..${NCB}`);
  }
  if (isSet(body, 'stream_codebooks_scope')) p.streamFirstOnly = str(body, 'stream_codebooks_scope') === 'first';
  if (speakerCfg !== 1.0) {
    const hasPrompt = (p.speakerMode === 'reference' && !!p.referenceCodes?.length) ||
      (p.speakerMode === 'embedding' && !!p.speakerEmbedding?.length);
    if (!hasPrompt)
      fail(400, 'Speaker CFG needs a speaker prompt: speaker_mode=reference with reference audio or codes, or ' +
        'speaker_mode=embedding with a clip or an embedding');
    if (controlsCfg !== 1.0) fail(400, 'Speaker and controls CFG cannot be combined');
    p.cfgScale = speakerCfg;
    p.cfgKind = 'speaker';
  }
  const tagCfg = num(body, 'tag_cfg_scale', 1);
  if (tagCfg > 1.0) {
    p.cfgScale = tagCfg;
    p.cfgKind = 'tag';
  }
  if (controlsCfg !== 1.0 && p.steering.hidden && requestEmbedding(p)) {
    p.cfgScale = controlsCfg;
    p.twinSteered = false;
    p.cfgKind = 'controls';
  }
  // zonos3.tts.plan validate_phoneme_groups (R13): this checkpoint has no phoneme tokens
  if (text.includes('\uE000') || text.includes('\uE001'))
    fail(400, 'This checkpoint has no phoneme tokens: remove the U+E000 / U+E001 pronunciation groups');
  checkLaci(e, body, p.language);
  if (bool(body, 'laci')) p.laci = e.laci.get(p.language)!;
  // sampling (TTSSamplingParams; registry defaults)
  const sp = p.sp;
  sp.temperature = num(body, 'temperature', SAMPLING.temperature);
  sp.top_k = num(body, 'topk', SAMPLING.topk);
  sp.top_p = num(body, 'top_p', SAMPLING.top_p);
  sp.min_p = num(body, 'min_p', SAMPLING.min_p);
  sp.rep_penalty = num(body, 'repetition_penalty', SAMPLING.repetition_penalty);
  sp.rep_window = num(body, 'repetition_window', SAMPLING.repetition_window);
  sp.rep_codebooks = num(body, 'repetition_codebooks', SAMPLING.repetition_codebooks);
  if (sp.rep_codebooks < 0) sp.rep_codebooks = e.hp.nCodebooks;
  if (isSet(body, 'seed')) sp.seed = Number(body.seed) >>> 0;
  else sp.seed = crypto.getRandomValues(new Uint32Array(1))[0] & 0x7fffffff;  // secrets.randbelow(2**31)
  sp.greedy = sp.temperature <= 0;
  p.fadeOutMs = num(body, 'fade_out_ms', 0);
  p.headroomDb = num(body, 'headroom_db', -1);
  if (bool(body, 'decode_context') && p.referenceCodes?.length) {
    // unset or 0 = the registry's decode-context length
    const frames = num(body, 'decode_context_frames', 0) || DECODE_CONTEXT_FRAMES;
    const refFrames = p.referenceCodes.length / NCB;
    const k = Math.max(0, Math.min(frames, refFrames));
    p.decodeContext = p.referenceCodes.slice(p.referenceCodes.length - k * NCB);
  }
  // text normalization is not available in the browser: the text is tokenized as sent
  resolveFollowTokens(e, p);

  // plan_generate: prompt and CFG twin
  const opt = {
    referenceCodes: p.speakerMode === 'reference' && p.referenceCodes?.length ? p.referenceCodes : undefined,
    speakerEmbedding: p.speakerMode === 'embedding' && p.speakerEmbedding?.length ? p.speakerEmbedding : undefined,
    buckets: p.conditionBuckets,
    silencePrefill: p.silencePrefill,
  };
  const build = (t: string, what: string) => {
    try {
      return buildPrompt(e.tokenizer, t, opt, e.hp);
    } catch (err) {
      fail(400, `${what}: ${err instanceof Error ? err.message : String(err)}`);
    }
  };
  p.prompt = build(p.text, 'could not build the prompt');
  if (p.cfgKind === 'speaker') {
    try {
      p.twin = buildCfgTwin(p.prompt, e.hp.condBucketCounts);
    } catch {
      fail(400, 'could not build the speaker-CFG twin prompt');
    }
  } else if (p.cfgKind === 'tag') {
    p.twin = build(stripNonverbalMarkup(p.text), 'could not build the tag-CFG twin prompt');
  } else if (p.cfgKind === 'controls') {
    // The same prompt and speaker, run without the hidden steering (twinSteered = false).
    p.twin = { ...p.prompt, cond: { ...p.prompt.cond } };
  }
  // make_job: the prompt takes its rows of the slot; the frame budget shrinks to what is left
  const rows = Math.max(p.prompt.nRows, p.twin?.nRows ?? 0);
  if (rows >= CONTEXT_CAPACITY) fail(400, `TTS input len ${rows} exceeds ${CONTEXT_CAPACITY}`);
  p.maxFrames = Math.min(p.maxFrames, CONTEXT_CAPACITY - rows);
  return p;
}

// ------------------------------------------------------------------------------------------------ job

interface Probe { textStart: number; textLen: number; heads: { layer: number; head: number }[]; windows: Windows }

/** attach_probe: the probed text span, the span follower and the LACI session (400 when the request cannot run it). */
function attachProbe(e: Engine, p: Plan): { probe: Probe | null; follower: SpanFollower | null; laci: LaciSession | null } {
  const heads = p.laci ?? p.follow;
  if (!heads) return { probe: null, follower: null, laci: null };
  const span = locateTextSpan(p.prompt.ids, NCB);
  if (!span) fail(400, 'the prompt has no complete text span for the alignment probe');
  if (span.len > ALIGN_PROBE_MAX_TEXT)
    fail(400, `the text is too long for the alignment probe (${span.len} text rows; at most ${ALIGN_PROBE_MAX_TEXT})`);
  const probe: Probe = { textStart: span.start, textLen: span.len, heads: heads.heads.map(h => ({ ...h })), windows: [] };
  let follower: SpanFollower | null = null;
  if (p.follow && p.steering.spans.length) {
    const cfg = { ...defaultFollowConfig(), lead: p.followLead, lag: p.followLag };
    follower = new SpanFollower(p.followTokens.map(r => [r[0], r[1]] as Range), p.steering.spans.map(s => s.delta), cfg);
  }
  if (!p.laci) return { probe, follower, laci: null };
  const W = NCB + 1, P = p.prompt.prefillFrames;
  const prefillRows: Int32Array[] = [];
  for (let r = p.prompt.nRows - P; r < p.prompt.nRows; r++) prefillRows.push(p.prompt.ids.slice(r * W, (r + 1) * W));
  let rates: RateBuckets | null = null;
  const hp = e.hp;
  hp.condNames.forEach((name, i) => {
    if (name === 'speaking_rate' && hp.condEdges[i]?.length) rates = new RateBuckets(i, hp.condEdges[i].slice(), hp.condBucketCounts[i]);
  });
  const laci = as400(() => new LaciSession({
    heads: p.laci!, cfg: defaultLaciConfig(), prefillRows, seed: p.sp.seed, maxTokens: p.maxFrames, nCodebooks: NCB,
    rates, conditionValues: hp.nCond > 0 ? p.prompt.cond.buckets.slice() : null, bucket: null, rateDeltas: null, follower,
  }));
  return { probe, follower, laci };
}

/** The first H delayed rows of the stream behind the decode-context frames (scheduler.cpp decode_rows). */
function decodeRows(codes: number[], H: number, ctx: Int32Array | null, K: number): number[] {
  if (K === 0) return codes.slice(0, H * NCB);
  const rows = new Array<number>((K + H) * NCB).fill(4097);
  const c0 = ctx!.length / NCB - K;
  for (let r = 0; r < K + H; r++) {
    if (r >= K) for (let c = 0; c < NCB; c++) rows[r * NCB + c] = codes[(r - K) * NCB + c];
    for (let c = 0; c < NCB; c++) {
      const frame = r - c;
      if (frame >= 0 && frame < K) rows[r * NCB + c] = ctx![(c0 + frame) * NCB + c];
    }
  }
  return rows;
}

/** Runs one planned request to completion; PCM goes to `emit` (gain applied, prefill/context dropped). */
export async function runJob(e: Engine, p: Plan, emit: (pcm: Float32Array) => void, aborted: () => boolean): Promise<void> {
  const { probe, follower, laci } = attachProbe(e, p);
  const bb = e.backbone, codec = e.codec;
  const ncb = NCB, W = ncb + 1;
  const ctxFrames = p.decodeContext ? p.decodeContext.length / ncb : 0;
  let skip = (p.prompt.prefillFrames + ctxFrames) * HOP;
  const gain = p.headroomDb < 0 ? codec.gain : Math.fround(10 ** (-p.headroomDb / 20));
  const finish = (pcm: Float32Array): Float32Array => {
    let out = pcm;
    if (skip > 0) {
      const n = Math.min(out.length, skip);
      out = out.subarray(n);
      skip -= n;
    }
    if (gain !== 1) {
      const g = new Float32Array(out.length);
      for (let i = 0; i < out.length; i++) g[i] = Math.fround(out[i] * gain);
      out = g;
    }
    return out;
  };

  const slot: SlotState = newSlot(p.prompt.prefillFrames);
  const stopId = vocabIds(e.hp.audioVocab).stopId;
  const codes: number[] = Array.from(prefillCodes(p.prompt));   // stream rows: silence prefill first
  let eosFrame = -1;
  let attemptMax = p.maxFrames, historyRows = 0;
  const twin = p.twin;
  const twinSteered = p.twinSteered;
  const cfgScale = twin ? p.cfgScale : null;
  let windows: Windows = laci?.probeMask() ?? [];
  const probeCtl: ProbeControl = { textLen: probe?.textLen ?? 0, mask: null, stop: false };
  let hidden = p.steering.hidden;

  // codec stream state (prefix streaming: incremental; whole: decoded at the end through the same causal stream).
  // Prefix chunks are queued, not awaited: their GPU work is submitted before the next backbone step and their host
  // half (codec worker) runs while the GPU runs the following steps; pcmChain emits them in order.
  codec.reset();
  let decoded = 0;         // frames handed to the codec
  let pcmChain: Promise<void> = Promise.resolve();
  let codecErr: unknown = null;
  const decodeUpTo = (rows: number[], H: number, fHi: number, missing = false) => {
    while (decoded < fHi) {
      const n = Math.min(BLOCK, fHi - decoded);
      const out = codec.chunk(buildIdx(rows, H, decoded, n, undefined, missing), decoded, n);
      out.catch(() => {});   // handled through pcmChain; avoids an unhandled-rejection report
      decoded += n;
      pcmChain = pcmChain.then(() => out).then(pcm => { const f = finish(pcm); if (f.length) emit(f); })
        .catch(e => { codecErr ??= e; });
    }
  };
  let emitted = p.prompt.prefillFrames;   // prefix streaming: frames handed to the codec (scheduler job->emitted)
  // early partial-frame decode (stream_codebooks < NCB; a codec without per-codebook biases decodes complete frames)
  const k = codec.partialFrames ? p.streamCodebooks : ncb;
  const pad = e.hp.audioPad;
  const prefixReady = () => {
    const H = codes.length / ncb;
    let complete = H - (ncb - 1);
    if (eosFrame >= 0) complete = Math.min(complete, eosFrame);
    // zonos3.tts.stream PrefixDecoder.ready: frames whose first k codebooks arrived (codebook j of frame f rides row
    // f + j), capped at EOS and walked back before a frame whose semantic code is EOA / padding
    let ready = complete;
    if (k < ncb && !(p.streamFirstOnly && emitted > p.prompt.prefillFrames)) {
      ready = H - (k - 1);
      if (eosFrame >= 0) ready = Math.min(ready, eosFrame);
      while (ready > Math.max(complete, emitted) && codes[(ready - 1) * ncb] >= pad - 1) ready--;
    }
    const decodable = ready - PREFIX_HOLD_BACK_FRAMES;
    const need = Math.min(PREFIX_CHUNK_FRAMES, PREFIX_FIRST_CHUNK_FRAMES + emitted - p.prompt.prefillFrames);
    if (decodable - emitted < need) return;
    decodeUpTo(decodeRows(codes, H, p.decodeContext, ctxFrames), H + ctxFrames, decodable + ctxFrames, decodable > complete);
    emitted = decodable;
  };

  const probeInput = (): ProbeInput | null => probe ? { ...probe, windows } : null;

  // step_steer: the hidden steering delta of the decode that yields the attempt's frame k (k >= 1)
  const steerAt = (k: number): Float32Array | null => {
    const frame = historyRows + k;
    if (follower) {
      const span = follower.delta(frame);
      if (!span) return hidden;
      if (hidden) for (let i = 0; i < hidden.length && i < span.length; i++) span[i] += hidden[i];
      return span;
    }
    if (p.steering.spans.length) {
      const out = { v: null as Float32Array | null };
      p.steering.deltaAt(frame, out);
      return out.v;
    }
    return hidden;
  };

  // Sampling runs on the GPU after each decode (bb.stepSample). Without a probe readout nothing the host learns from
  // frame k changes step k + 1, so step k + 1 is submitted before frame k is read back (one step in flight); with a
  // probe (LACI, span follow) the readout steers the next step and the loop stays serial.
  type StepOut = { frame: Int32Array; probe: Float32Array | null };
  const lookahead = probe === null;
  const inflight: Promise<StepOut>[] = [];
  let issued = 0;   // frames of this attempt whose sample is submitted
  const submit = () => {
    const k = issued++;
    const out: Promise<StepOut> = k === 0
      ? bb.sampleFirst(slot.prefill).then(frame => ({ frame, probe: null }))
      : (() => { const steer = steerAt(k); return bb.stepSample(slot.prefill + k, steer, twinSteered ? steer : null, probeInput()); })();
    out.catch(() => {});   // awaited through `inflight`; a step discarded after an error is not reported
    inflight.push(out);
  };
  // a finished attempt drops the step it ran ahead (its KV row lies past the attempt; the next prefill resets it)
  const discard = async () => {
    await Promise.allSettled(inflight);
    inflight.length = 0;
  };

  // admit: prefill the conditional slot and the twin concurrently (each submits all of its work before awaiting)
  await Promise.all([bb.prefill(0, p.prompt.ids, p.prompt.cond), twin ? bb.prefill(1, twin.ids, twin.cond) : null]);
  bb.startSampling({ sp: p.sp, minFrames: 0, cfgScale });

  for (;;) {
    if (aborted()) { await discard(); await pcmChain; return; }
    if (codecErr) { await discard(); throw codecErr; }
    // 3. the next frame (submitted now, or a step ago), then the step after it when running ahead
    if (!inflight.length) submit();
    if (lookahead && inflight.length < 2 && issued < attemptMax) submit();
    const r = await inflight.shift()!;
    if (probe && r.probe) {
      const ro = probeReduce(r.probe, probe.textLen);
      if (laci) {
        try {
          laci.onStep(probeCtl, slot.step, ro.argmax, ro.centroid);
        } catch (err) {
          throw new Error(`LACI: ${err instanceof Error ? err.message : String(err)}`);
        }
        windows = probeCtl.mask ?? [];
      } else if (follower) {
        follower.onStep(slot.step, ro.argmax);
      }
    }
    if (laci) {
      acceptFrame(slot, r.frame, stopId, attemptMax);
      const eos = slot.eosFrame >= 0 ? Math.max(0, slot.eosFrame - laci.attemptStart) : null;
      laci.append(Int32Array.from(slot.nextIds), eos);
      if (probeCtl.stop) slot.done = true;
      if (!slot.done) {
        const r = laci.drain();
        for (let i = r.begin; i < r.end; i++) for (let c = 0; c < ncb; c++) codes.push(laci.stream[i][c]);
      }
    } else {
      const frame = acceptFrame(slot, r.frame, stopId, attemptMax);
      for (const c of frame) codes.push(c);
      eosFrame = slot.eosFrame;
    }
    if (p.prefix) prefixReady();

    // 4. a finished slot: LACI rollback, or the final decode
    if (slot.done) {
      await discard();
      if (laci) {
        const c = laci.finishAttempt();
        if (c) {
          // laci_resume: re-prefill the prompt (without its silence rows) plus the kept stream rows; the twin keeps its
          // state and last logits
          const base = p.prompt.nRows - p.prompt.prefillFrames;
          const ids = new Int32Array((base + c.nRows) * W);
          ids.set(p.prompt.ids.subarray(0, base * W));
          for (let r = 0; r < c.nRows; r++) ids.set(laci.stream[r].subarray(0, W), (base + r) * W);
          const cond: Conditioning = { ...p.prompt.cond };
          if (c.conditionValues) cond.buckets = c.conditionValues.slice();
          if (c.swapDelta) hidden = c.steeringDelta;
          Object.assign(slot, newSlot(c.nRows));
          issued = 0;
          windows = laci.probeMask() ?? [];
          probeCtl.mask = null;
          probeCtl.stop = false;
          historyRows = c.historyRows;
          attemptMax = c.maxTokens;
          await bb.prefill(0, ids, cond);
          bb.startSampling({
            sp: { ...p.sp, seed: (c.seed ?? p.sp.seed) >>> 0 }, minFrames: 0, cfgScale,
            history: laci.stream.slice(p.prompt.prefillFrames, c.nRows),
          });
          const r = laci.drain();
          for (let i = r.begin; i < r.end; i++) for (let k = 0; k < ncb; k++) codes.push(laci.stream[i][k]);
          if (p.prefix) prefixReady();
          continue;
        }
        const r = laci.drain(true);
        for (let i = r.begin; i < r.end; i++) for (let k = 0; k < ncb; k++) codes.push(laci.stream[i][k]);
        eosFrame = laci.end && laci.end[1] !== null ? laci.end[1] : -1;
      }
      // submit_finalize: EOS inside the silence prefill leaves no audio
      if (eosFrame >= 0 && eosFrame <= p.prompt.prefillFrames) throw new Error('generation failed');
      await pcmChain;
      if (codecErr) throw codecErr;
      const nf = codes.length / ncb;
      const H = nf + ctxFrames;
      const E = Math.min(eosFrame >= 0 ? eosFrame + ctxFrames : H, H - (ncb - 1));
      if (E > 0) {
        const rows = decodeRows(codes, nf, p.decodeContext, ctxFrames);
        const tail: Float32Array[] = [];
        const keep = (pcm: Float32Array) => { if (pcm.length) tail.push(pcm); };
        // the frames prefix streaming has not decoded yet, then the codec's tail
        while (decoded < E) {
          const n = Math.min(BLOCK, E - decoded);
          keep(finish(await codec.chunk(buildIdx(rows, H, decoded, n), decoded, n)));
          decoded += n;
        }
        keep(finish(await codec.flush()));
        const total = tail.reduce((a, b) => a + b.length, 0);
        const out = new Float32Array(total);
        let o = 0;
        for (const t of tail) { out.set(t, o); o += t.length; }
        applyFade(out, p.fadeOutMs, SAMPLE_RATE);
        if (out.length) emit(out);
      }
      return;
    }
  }
}

// ------------------------------------------------------------------------------------------------ routes

async function generate(e: Engine, spk: Speakers, body: Body, req: EngineRequest, sink: Sink, aborted: () => boolean): Promise<void> {
  const wav = str(body, 'format') === 'wav';
  const stream = bool(body, 'stream') ?? true;
  if (stream && wav) fail(400, 'format=wav needs stream=false');
  const p = await planGenerate(e, spk, body, sessionId(req), req.pcm);
  p.wav = wav;
  const headers: Record<string, string> = {
    'x-audio-sample-rate': String(SAMPLE_RATE),
    'x-audio-channels': '1',
    'x-audio-format': wav ? 'pcm16' : 'float32',
    'x-seed': String(p.sp.seed),
  };
  if (p.spanFollow) headers['x-control-span-follow'] = p.spanFollow;
  if (p.stream) {
    let started = false;
    const start = () => { if (!started) { started = true; sink.start(200, { ...headers, 'content-type': 'audio/pcm' }); } };
    // an error before the first byte answers as JSON; afterwards the stream ends (the client sees a short body)
    try {
      await runJob(e, p, pcm => { start(); sink.chunk(new Uint8Array(pcm.buffer, pcm.byteOffset, pcm.byteLength)); }, aborted);
    } catch (err) {
      if (!started) throw err;
      console.error('generate:', err);
    }
    start();
    sink.end();
    return;
  }
  const parts: Float32Array[] = [];
  try {
    await runJob(e, p, pcm => parts.push(pcm.slice()), aborted);
  } catch (err) {
    if (err instanceof HttpError) throw err;
    console.error('generate:', err);
    fail(500, 'generation failed');
  }
  const audio = new Float32Array(parts.reduce((a, b) => a + b.length, 0));
  let o = 0;
  for (const part of parts) { audio.set(part, o); o += part.length; }
  if (wav) {
    sink.start(200, { ...headers, 'content-type': 'audio/wav' });
    sink.chunk(wavBytes(audio, SAMPLE_RATE));
  } else {
    sink.start(200, { ...headers, 'content-type': 'audio/pcm' });
    sink.chunk(new Uint8Array(audio.buffer));
  }
  sink.end();
}

const SPEAKER_INPUT_FIELDS = [
  'speaker_audio_base64', 'speaker_audio_name', 'speaker_reference_codes', 'speaker_embedding_base64',
  'speaker_embedding_name', 'speaker_embedding_id', 'speaker_blend_embedding_id_a', 'speaker_blend_embedding_id_b',
  'speaker_blend_t',
];

export async function generateRoute(e: Engine, spk: Speakers, req: EngineRequest, sink: Sink, aborted: () => boolean): Promise<void> {
  await generate(e, spk, parseBody(req, 'GenerateRequest'), req, sink, aborted);
}

/** OpenAI-compatible: the /v2/generate request of `input`, with `voice` as its speaker_embedding_id. */
export async function speechRoute(e: Engine, spk: Speakers, req: EngineRequest, sink: Sink, aborted: () => boolean): Promise<void> {
  const body = parseBody(req, 'SpeechRequest');
  const session = sessionId(req);
  const format = str(body, 'response_format') ?? 'pcm';
  if (isSet(body, 'voice')) {
    for (const field of SPEAKER_INPUT_FIELDS) if (isSet(body, field)) fail(400, 'voice cannot be combined with speaker_* fields');
    const voice = str(body, 'voice')!;
    if (!spk.known(session, voice)) fail(400, `unknown voice '${voice}'`);
    body.speaker_embedding_id = voice;
  }
  body.text = body.input;
  body.stream = format === 'pcm';
  body.format = format;
  for (const k of ['model', 'input', 'voice', 'response_format']) delete body[k];
  await generate(e, spk, body, req, sink, aborted);
}
