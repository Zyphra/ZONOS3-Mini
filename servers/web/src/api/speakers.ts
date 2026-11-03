// Voices of the engine API: default voices, session uploads, embedding files, slerp blends and the /v2/speakers routes
// (port of servers/ggml/src/server/speakers.cpp; same 400 details).
import { b64DecodeStrict } from '../b64';
import { MAX_REFERENCE_FRAMES } from '../prompt';
import { disabledControls } from '../request/steer';
import { wavBytes } from '../wav';
import type { Engine } from './engine';
import { fail, isSet, parseBody, pyRepr, sendBytes, sendJson, sessionId, str, type EngineRequest, type Sink } from './http';

const NO_SESSION = 'Cached speaker operations require the X-TTS-Session-ID header.';
const MAX_SECONDS = 60;

export interface Voice {
  id: string;
  label: string;
  sourceType: 'audio' | 'embedding';
  isDefault: boolean;
  originalName: string | null;
  created: number;
  pcm: Float32Array | null;       // 44.1 kHz mono (audio voices)
  codes: Int32Array | null;       // ordinary [T][10]
  embedding: Float32Array | null; // [speakerDim]
}

export interface ResolvedSpeaker {
  mode: string;                    // '' = no speaker prompt
  codes: Int32Array | null;
  embedding: Float32Array | null;
}

/** Session-scoped uploads plus the engine's default voices. */
export class Speakers {
  private sessions = new Map<string, Map<string, Voice>>();
  private defaults: Voice[];

  constructor(private e: Engine, now = 0) {
    this.defaults = e.voices.map(v => ({
      id: v.id, label: v.label, sourceType: 'audio', isDefault: true, originalName: v.file.split('/').pop() ?? v.file,
      created: now, pcm: v.pcm, codes: v.codes, embedding: null,
    }));
  }

  /** The checkpoint's speaker modes (speakers.cpp checkpoint_speaker_modes). */
  checkpointModes(): string[] {
    return this.e.hp.speakerDim > 0 ? ['reference', 'embedding'] : ['reference'];
  }

  private voiceModes(v: Voice): string[] {
    const out: string[] = [];
    for (const m of this.checkpointModes()) {
      if (v.sourceType === 'embedding') { if (m === 'embedding') out.push(m); continue; }
      // The speaker encoder and reference encoder both ship with the web export: every audio voice takes both modes.
      out.push(m);
    }
    return out;
  }

  json(v: Voice): Record<string, unknown> {
    return {
      id: v.id,
      label: v.label,
      source_type: v.sourceType,
      scope: v.isDefault ? 'default' : 'session',
      is_default: v.isDefault,
      original_name: v.originalName,
      created_at: v.created,
      has_preview: v.isDefault ? v.sourceType === 'audio' && v.pcm !== null : v.pcm !== null,
      speaker_modes: this.voiceModes(v),
      embedding_dim: this.e.hp.speakerDim > 0 ? this.e.hp.speakerDim : null,
      transcript: null,
      disabled_controls: disabledControls(this.e.steer, v.id),
    };
  }

  /** A voice by id (default voices first, then the session's), or 400 like sgl _get_cached_speaker. */
  lookup(session: string | null, id: string): Voice {
    const d = this.defaults.find(v => v.id === id);
    if (d) return d;
    if (!session) fail(400, NO_SESSION);
    const v = this.sessions.get(session)?.get(id);
    if (!v) fail(400, `Unknown cached speaker id '${id}'.`);
    return v;
  }

  known(session: string | null, id: string): boolean {
    if (this.defaults.some(v => v.id === id)) return true;
    return !!session && !!this.sessions.get(session)?.has(id);
  }

  private pcmOf(v: Voice): Float32Array {
    if (v.pcm) return v.pcm;
    if (v.isDefault) fail(400, `Failed to read default voice '${v.label}'.`);
    fail(400, `Cached speaker '${v.label}' has no reference audio.`);
  }

  /** speakers.cpp voice_embedding: an embedding file's vector, or the speaker encoder's embedding of the clip, once. */
  async voiceEmbedding(session: string | null, id: string): Promise<Float32Array> {
    const v = this.lookup(session, id);
    const dim = this.e.hp.speakerDim;
    if (!v.embedding) v.embedding = await this.embed(this.pcmOf(v));
    if (v.embedding.length !== dim)
      fail(400, `Cached speaker '${v.label}' has ${v.embedding.length} dimensions, but the current model expects ${dim}.`);
    return v.embedding;
  }

  /** speakers.cpp voice_codes: shipped codes, else the reference encoder's codes of the clip, once. */
  async voiceCodes(session: string | null, id: string): Promise<Int32Array> {
    const v = this.lookup(session, id);
    if (v.sourceType !== 'audio') {
      if (v.isDefault) fail(400, `Default voice '${v.label}' is an embedding file; this checkpoint needs reference audio.`);
      fail(400, `Cached speaker '${v.label}' has no reference audio.`);
    }
    if (!v.codes) v.codes = await this.encode(this.pcmOf(v));
    return v.codes;
  }

  /** embed_speaker_locked: the speaker encoder's embedding (400 past 60 s, or with the encoder's own message when the
   *  device clamps clips shorter or the computation fails). */
  async embed(pcm: Float32Array): Promise<Float32Array> {
    const limitMsg = `speaker embedding computation failed (maximum ${MAX_SECONDS} seconds)`;
    if (!pcm.length || pcm.length > MAX_SECONDS * 44100) fail(400, limitMsg);
    let emb: Float32Array;
    try {
      emb = await this.e.embedSpeaker(pcm);
    } catch (e) {
      const msg = e instanceof Error ? e.message : '';
      fail(400, msg.startsWith('speaker embedding computation failed') ? msg : limitMsg);
    }
    if (emb.length !== this.e.hp.speakerDim) fail(400, limitMsg);
    if (!emb.every(Number.isFinite)) fail(400, 'speaker embedding computation failed');
    return emb;
  }

  /** encode_reference_locked: ordinary reference codes of the clip (400 on failure). */
  async encode(pcm: Float32Array, maxSeconds = MAX_SECONDS, maxFrames = MAX_REFERENCE_FRAMES): Promise<Int32Array> {
    const msg = `reference audio encoding failed (maximum ${maxSeconds} seconds)`;
    if (!pcm.length || pcm.length > maxSeconds * 44100) fail(400, msg);
    let codes: Int32Array;
    try {
      codes = await this.e.encodeReference(pcm);
    } catch {
      fail(400, msg);
    }
    if (!codes.length || codes.length % 10 || codes.length > maxFrames * 10)
      fail(400, msg);
    return codes;
  }

  /** resolve_request_speaker (speakers.cpp:583-651). `pcm` is the client-decoded speaker_audio_base64. */
  async resolveRequest(body: Record<string, unknown>, session: string | null, mode: string,
                       pcm: Float32Array | undefined): Promise<ResolvedSpeaker> {
    const out: ResolvedSpeaker = { mode: '', codes: null, embedding: null };
    const nonempty = (k: string) => (str(body, k) ?? '') !== '';
    const audio = nonempty('speaker_audio_base64'), embFile = nonempty('speaker_embedding_base64');
    const id = nonempty('speaker_embedding_id');
    const blendA = nonempty('speaker_blend_embedding_id_a'), blendB = nonempty('speaker_blend_embedding_id_b');
    const hasCodes = isSet(body, 'speaker_reference_codes');
    const uploaded = () => decodedAudio(body, pcm);
    if (mode === 'reference') {
      if (embFile || blendA || blendB)
        fail(400, 'This checkpoint conditions on raw reference audio and does not accept speaker embedding files or embedding blends.');
      if (hasCodes) {
        if (audio || id) fail(400, 'Provide either speaker_reference_codes or a reference clip (audio or a speaker id), not both.');
        const rows = body.speaker_reference_codes as unknown[][];
        const ncb = this.e.hp.nCodebooks;
        if (rows.length > MAX_REFERENCE_FRAMES)
          fail(400, `speaker_reference_codes must be a non-empty [frames, codebooks] list of at most ${MAX_REFERENCE_FRAMES} frames`);
        const codes: number[] = [];
        for (const row of rows) {
          if (row.length !== ncb) fail(400, 'speaker_reference_codes must be a non-empty [frames, codebooks] list');
          for (const c of row) {
            if (Number(c) >= 4096) fail(400, 'speaker_reference_codes must be ordinary codec ids 0..4095');
            codes.push(Number(c));
          }
        }
        out.codes = Int32Array.from(codes);
        out.mode = mode;
        return out;
      }
      if (id) {
        if (audio) fail(400, 'Provide either a reference audio file or a cached speaker id, not both.');
        out.codes = await this.voiceCodes(session, str(body, 'speaker_embedding_id')!);
      } else if (audio) {
        out.codes = await this.encode(uploaded());
      } else {
        return out;
      }
      out.mode = mode;
      return out;
    }
    if (mode !== 'embedding') return out;
    if (hasCodes) fail(400, 'speaker_reference_codes need speaker_mode=reference.');
    if (audio && embFile) fail(400, 'Provide either a reference audio file or a saved embedding file, not both.');
    const selected = Number(audio || embFile) + Number(id) + Number(blendA || blendB);
    if (selected === 0) return out;
    if (selected > 1) fail(400, 'Provide speaker conditioning via upload, one cached speaker id, or two cached ids for blending.');
    const dim = this.e.hp.speakerDim;
    if (blendA || blendB) {
      if (!blendA || !blendB) fail(400, 'Provide both cached speaker ids to blend between two speakers.');
      const a = await this.voiceEmbedding(session, str(body, 'speaker_blend_embedding_id_a')!);
      const b = await this.voiceEmbedding(session, str(body, 'speaker_blend_embedding_id_b')!);
      out.embedding = slerp(a, b, typeof body.speaker_blend_t === 'number' ? body.speaker_blend_t : 0.5);
    } else if (id) {
      out.embedding = await this.voiceEmbedding(session, str(body, 'speaker_embedding_id')!);
    } else if (embFile) {
      out.embedding = loadEmbeddingVector(decodeB64Field(body, 'speaker_embedding_base64'), dim, str(body, 'speaker_embedding_name'));
    } else {
      out.embedding = await this.embed(uploaded());
    }
    out.mode = mode;
    return out;
  }

  // ------------------------------------------------------------------------------------------------ routes

  create(req: EngineRequest, sink: Sink, now: number): void {
    const body = parseBody(req, 'SpeakerCreateRequest');
    const session = sessionId(req);
    if (!session) fail(400, NO_SESSION);
    const v: Voice = {
      id: 'spk_' + randomHex(12), label: '', sourceType: 'audio', isDefault: false, originalName: null, created: now,
      pcm: null, codes: null, embedding: null,
    };
    const audio = (str(body, 'speaker_audio_base64') ?? '') !== '';
    if ((str(body, 'speaker_embedding_base64') ?? '') !== '') {
      if (audio) fail(400, 'Provide either a reference audio file or a saved embedding file, not both.');
      v.originalName = str(body, 'speaker_embedding_name');
      v.embedding = loadEmbeddingVector(decodeB64Field(body, 'speaker_embedding_base64'), this.e.hp.speakerDim, v.originalName);
      v.sourceType = 'embedding';
      v.label = uploadLabel(str(body, 'label'), v.originalName, 'Speaker embedding');
    } else {
      if (!audio) fail(400, 'Provide an audio file to cache as the speaker reference.');
      v.originalName = str(body, 'speaker_audio_name');
      v.pcm = decodedAudio(body, req.pcm);
      v.label = uploadLabel(str(body, 'label'), v.originalName, 'Speaker audio');
    }
    let voices = this.sessions.get(session);
    if (!voices) { voices = new Map(); this.sessions.set(session, voices); }
    voices.set(v.id, v);
    sendJson(sink, this.json(v));
  }

  list(req: EngineRequest, sink: Sink): void {
    const session = sessionId(req);
    const defaults = [...this.defaults].sort((a, b) => cmp(a.label.toLowerCase(), b.label.toLowerCase()));
    const out = defaults.map(v => this.json(v));
    const cached = session ? [...(this.sessions.get(session)?.values() ?? [])] : [];
    cached.sort((a, b) => b.created - a.created);
    for (const v of cached) out.push(this.json(v));
    sendJson(sink, { speakers: out });
  }

  preview(req: EngineRequest, sink: Sink, id: string): void {
    const v = this.lookup(sessionId(req), id);
    if (v.isDefault && v.sourceType !== 'audio') fail(400, 'Default embedding speakers do not include preview audio.');
    if (v.isDefault && !v.pcm) fail(400, `Failed to read default voice '${v.label}'.`);
    if (!v.pcm) fail(404, 'Cached speaker preview audio is not available.');
    sendBytes(sink, wavBytes(v.pcm, 44100, 'rint'), 'audio/wav');
  }

  async embedding(req: EngineRequest, sink: Sink, id: string): Promise<void> {
    if (this.e.hp.speakerDim === 0) fail(404, `No speaker embedding for ${pyRepr(id)}.`);
    const emb = await this.voiceEmbedding(sessionId(req), id);
    sendJson(sink, { id, embedding: Array.from(emb, x => Math.round(x * 1e6) / 1e6) });
  }
}

function cmp(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}

/** uuid4().hex[:n] */
function randomHex(n: number): string {
  const b = new Uint8Array(Math.ceil(n / 2));
  crypto.getRandomValues(b);
  return Array.from(b, x => x.toString(16).padStart(2, '0')).join('').slice(0, n);
}

/** sgl _default_speaker_label: the explicit label, else the file name's stem, else the fallback. */
function uploadLabel(label: string | null, fileName: string | null, fallback: string): string {
  if (label && label.trim()) return label.trim();
  if (fileName) {
    const base = fileName.split(/[\\/]/).pop() ?? '';
    const dot = base.lastIndexOf('.');
    const stem = (dot > 0 ? base.slice(0, dot) : base).trim();
    if (stem) return stem;
  }
  return fallback;
}

/** decode_b64_field: strict base64 (or a data URL) of a request field. */
function decodeB64Field(body: Record<string, unknown>, field: string): Uint8Array {
  const raw = str(body, field) ?? '';
  const payload = raw.trim();
  if (payload.startsWith('data:') && !payload.includes(',')) fail(400, `Invalid data URL for ${field}.`);
  const out = b64DecodeStrict(raw);
  if (!out) fail(400, `${field} must be valid base64-encoded data.`);
  return out;
}

/** speaker_audio_base64 as the client decoded it (decode_audio_bytes). */
function decodedAudio(body: Record<string, unknown>, pcm: Float32Array | undefined): Float32Array {
  decodeB64Field(body, 'speaker_audio_base64');
  if (!pcm || !pcm.length)
    fail(400, `could not decode the reference audio (any container ffmpeg reads, at most ${MAX_SECONDS} seconds)`);
  return pcm;
}

/** sgl _slerp_embeddings (float32). */
export function slerp(v0: Float32Array, v1: Float32Array, t: number): Float32Array {
  const f = Math.fround;
  if (v0.length !== v1.length)
    fail(400, `Cannot blend speaker embeddings with different shapes (${v0.length},) and (${v1.length},).`);
  if (t < 0 || t > 1) fail(400, 'speaker_blend_t must be between 0.0 and 1.0.');
  const blend = f(t);
  let n0 = 0, n1 = 0;
  for (const x of v0) n0 = f(n0 + f(x * x));
  for (const x of v1) n1 = f(n1 + f(x * x));
  n0 = f(f(Math.sqrt(n0)) + f(1e-8));
  n1 = f(f(Math.sqrt(n1)) + f(1e-8));
  let dot = 0;
  for (let i = 0; i < v0.length; i++) dot = f(dot + f(f(v0[i] / n0) * f(v1[i] / n1)));
  const omega = f(Math.acos(Math.min(1, Math.max(-1, dot))));
  const out = new Float32Array(v0.length);
  if (omega < f(1e-6)) {
    for (let i = 0; i < out.length; i++) out[i] = f(f(f(1 - blend) * v0[i]) + f(blend * v1[i]));
    return out;
  }
  const so = f(Math.sin(omega));
  const a = f(f(Math.sin(f(f(1 - blend) * omega))) / so), b = f(f(Math.sin(f(blend * omega))) / so);
  for (let i = 0; i < out.length; i++) out[i] = f(f(a * v0[i]) + f(b * v1[i]));
  return out;
}

// ------------------------------------------------------------------------------------------------ embedding files

interface NpyData { shape: number[]; data: Float64Array }

function halfToFloat(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >> 10) & 0x1f, mant = h & 0x3ff;
  if (exp === 0) return sign * mant * 2 ** -24;
  if (exp === 31) return mant ? NaN : sign * Infinity;
  return sign * (1 + mant / 1024) * 2 ** (exp - 15);
}

/** speakers.cpp parse_npy: one .npy payload (versions 1-3) as doubles; a reason string when unsupported. */
function parseNpy(p: Uint8Array): NpyData | string {
  const dv = new DataView(p.buffer, p.byteOffset, p.byteLength);
  if (p.length < 10 || p[0] !== 0x93 || String.fromCharCode(...p.subarray(1, 6)) !== 'NUMPY') return 'not an NPY payload';
  const major = p[6];
  let hlen: number, off: number;
  if (major === 1) { hlen = dv.getUint16(8, true); off = 10; }
  else if ((major === 2 || major === 3) && p.length >= 12) { hlen = dv.getUint32(8, true); off = 12; }
  else return 'unsupported NPY version';
  if (off + hlen > p.length) return 'truncated NPY header';
  const header = String.fromCharCode(...p.subarray(off, off + hlen));
  off += hlen;
  const field = (key: string) => {
    const k = header.indexOf(`'${key}'`);
    return k < 0 ? -1 : header.indexOf(':', k);
  };
  const d = field('descr'), f = field('fortran_order'), sh = field('shape');
  if (d < 0 || f < 0 || sh < 0) return 'invalid NPY header';
  const q = header.slice(d + 1).search(/[^ ]/) + d + 1;
  if (q <= d || (header[q] !== "'" && header[q] !== '"')) return 'structured NPY arrays are not supported';
  const qe = header.indexOf(header[q], q + 1);
  const descr = header.slice(q + 1, qe);
  const fs = header.slice(f + 1).search(/[^ ]/) + f + 1;
  const fortran = header.startsWith('True', fs);
  const ps = header.indexOf('(', sh), pe = header.indexOf(')', sh);
  if (ps < 0 || pe < 0) return 'invalid NPY shape';
  const shape = (header.slice(ps + 1, pe).match(/\d+/g) ?? []).map(Number);
  const count = shape.reduce((a, b) => a * b, 1);
  if (descr.length < 3 || !'<|='.includes(descr[0])) return 'big-endian NPY data';
  const kind = descr[1], bytes = parseInt(descr.slice(2), 10);
  if (!((kind === 'f' && [2, 4, 8].includes(bytes)) || ((kind === 'i' || kind === 'u') && [1, 2, 4, 8].includes(bytes)) ||
        (kind === 'b' && bytes === 1))) return 'unsupported NPY dtype ' + descr;
  if (off + count * bytes > p.length) return 'truncated NPY data';
  let flat = new Float64Array(count);
  for (let i = 0; i < count; i++) {
    const o = off + i * bytes;
    let v: number;
    if (kind === 'f') v = bytes === 2 ? halfToFloat(dv.getUint16(o, true)) : bytes === 4 ? dv.getFloat32(o, true) : dv.getFloat64(o, true);
    else if (kind === 'i') v = bytes === 1 ? dv.getInt8(o) : bytes === 2 ? dv.getInt16(o, true) : bytes === 4 ? dv.getInt32(o, true) : Number(dv.getBigInt64(o, true));
    else v = bytes === 1 ? dv.getUint8(o) : bytes === 2 ? dv.getUint16(o, true) : bytes === 4 ? dv.getUint32(o, true) : Number(dv.getBigUint64(o, true));
    flat[i] = v;
  }
  if (fortran && shape.length > 1) {
    const c = new Float64Array(count);
    const idx = new Array<number>(shape.length).fill(0);
    for (let lin = 0; lin < count; lin++) {
      let fpos = 0, stride = 1;
      for (let k = 0; k < shape.length; k++) { fpos += idx[k] * stride; stride *= shape[k]; }
      c[lin] = flat[fpos];
      for (let k = shape.length - 1; k >= 0; k--) { if (++idx[k] < shape[k]) break; idx[k] = 0; }
    }
    flat = c;
  }
  return { shape, data: flat };
}

/** speakers.cpp parse_npz: the stored (uncompressed) members of a zip archive; a reason string otherwise. */
function parseNpz(z: Uint8Array): [string, Uint8Array][] | string {
  const dv = new DataView(z.buffer, z.byteOffset, z.byteLength);
  if (z.length < 22) return 'not a zip archive';
  let eocd = -1;
  for (let i = z.length - 22; i >= 0 && z.length - i <= 22 + 65535; i--) if (dv.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
  if (eocd < 0) return 'not a zip archive';
  const entries = dv.getUint16(eocd + 10, true);
  let cd = dv.getUint32(eocd + 16, true);
  const out: [string, Uint8Array][] = [];
  for (let e = 0; e < entries; e++) {
    if (cd + 46 > z.length || dv.getUint32(cd, true) !== 0x02014b50) return 'corrupt zip directory';
    const method = dv.getUint16(cd + 10, true), csize = dv.getUint32(cd + 20, true);
    const nameLen = dv.getUint16(cd + 28, true), extraLen = dv.getUint16(cd + 30, true), commentLen = dv.getUint16(cd + 32, true);
    const local = dv.getUint32(cd + 42, true);
    const name = new TextDecoder().decode(z.subarray(cd + 46, cd + 46 + nameLen));
    cd += 46 + nameLen + extraLen + commentLen;
    if (local + 30 > z.length || dv.getUint32(local, true) !== 0x04034b50) return 'corrupt zip entry';
    const data = local + 30 + dv.getUint16(local + 26, true) + dv.getUint16(local + 28, true);
    if (method !== 0) return 'compressed .npz archives (numpy.savez_compressed) are not supported; use numpy.savez or .npy';
    if (data + csize > z.length) return 'truncated zip entry';
    out.push([name, z.subarray(data, data + csize)]);
  }
  return out;
}

/** sgl _load_embedding_vector: .npy, or .npz with `emb` (or exactly one array); squeezed; (D,) or (N, D) averaged. */
export function loadEmbeddingVector(bytes: Uint8Array, expectedDim: number, fileName: string | null): Float32Array {
  const desc = fileName ? fileName : 'Speaker embedding file';
  const invalid = `${desc} must be a valid .npy or .npz file saved without pickle.`;
  let arr: NpyData | string;
  if (bytes.length >= 4 && bytes[0] === 0x50 && bytes[1] === 0x4b) {
    const members = parseNpz(bytes);
    if (typeof members === 'string') fail(400, `${invalid} (${members})`);
    let pick = members.find(([n]) => n === 'emb.npy')?.[1];
    if (!pick && members.length === 1) pick = members[0][1];
    if (!pick) fail(400, "Embedding archive must contain an 'emb' array or exactly one array.");
    arr = parseNpy(pick);
  } else {
    arr = parseNpy(bytes);
  }
  if (typeof arr === 'string') fail(400, invalid);
  const shape = arr.shape.filter(d => d !== 1);
  let vec: Float32Array;
  if (shape.length === 1) {
    vec = Float32Array.from(arr.data);
  } else if (shape.length === 2) {
    if (shape[0] === 0) fail(400, 'Speaker embedding file is empty.');
    const [rows, dim] = shape;
    vec = new Float32Array(dim);
    for (let j = 0; j < dim; j++) {
      let acc = 0;   // numpy float32 mean: accumulate in float32
      for (let i = 0; i < rows; i++) acc = Math.fround(acc + Math.fround(arr.data[i * dim + j]));
      vec[j] = Math.fround(acc / rows);
    }
  } else {
    fail(400, `Unsupported embedding array shape (${shape.join(', ')}${shape.length === 1 ? ',' : ''}). Expected (D,) or (N, D).`);
  }
  if (vec.length !== expectedDim)
    fail(400, `Reference embedding dimension mismatch. Model expects ${expectedDim}, but loaded file has ${vec.length}.`);
  return vec;
}
