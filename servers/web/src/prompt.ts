// Port of zonos3-prompt.cpp: [speaker row | delayed reference prefix] [condition rows] BOS text EOS [silence rows].
import type { Tokenizer } from './tokenizer';

export const NCB = 10;
export const W = NCB + 1;
export const AUDIO_PAD = 4097;
export const TEXT_PAD = 8196;
export const EOA = 4096;
export const CODEBOOK_SIZE = 4096;
export const MAX_REFERENCE_FRAMES = 900;
export const MAX_SILENCE_PREFILL = 30;

export interface ModelHp {
  nCodebooks: number; audioPad: number; textVocab: number; nCond: number; nCondBuckets: number;
  condNames: string[]; condEdges: number[][]; silenceFrame: number[]; speakerDim: number; contextLength: number;
  condUnits: string[];          // condition.units (capabilities)
  textTokenizerSpec: string;    // text_tokenizer_spec JSON (capabilities)
  tagCfgTrained: boolean;       // tag_cfg_trained (absent = false)
  nEmbd: number; nLayer: number;
}

/** Conditioning rows: speakerRow/conditionRow are absolute prompt rows (-1 = none). */
export interface Conditioning {
  speakerRow: number; speaker: Float32Array | null; conditionRow: number; buckets: number[];
}

/** ids is row-major [nRows][11]. */
export interface Prompt { ids: Int32Array; nRows: number; speakerRows: number; prefillFrames: number; cond: Conditioning }

function num(meta: Record<string, unknown>, key: string): number {
  const v = meta[key];
  if (typeof v !== 'number') throw new Error(`backbone: missing meta ${key}`);
  return v;
}

function nums(meta: Record<string, unknown>, key: string): number[] {
  const v = meta[key];
  if (!Array.isArray(v) || v.some(x => typeof x !== 'number')) throw new Error(`backbone: missing meta ${key}`);
  return v.map(Number);
}

export function modelHp(meta: Record<string, unknown>): ModelHp {
  const names = meta['condition.names'];
  if (!Array.isArray(names) || names.some(x => typeof x !== 'string')) throw new Error('backbone: missing meta condition.names');
  const condNames = names.map(String);
  return {
    nCodebooks: num(meta, 'n_codebooks'),
    audioPad: num(meta, 'audio_pad_id'),
    textVocab: num(meta, 'text_vocab'),
    nCond: num(meta, 'condition.count'),
    nCondBuckets: num(meta, 'condition.bucket_count'),
    condNames,
    condEdges: condNames.map(n => nums(meta, `condition.edges.${n}`)),
    silenceFrame: nums(meta, 'silence_frame'),
    speakerDim: num(meta, 'speaker.embedding_length'),
    contextLength: num(meta, 'context_length'),
    condUnits: Array.isArray(meta['condition.units']) ? meta['condition.units'].map(String) : [],
    textTokenizerSpec: typeof meta.text_tokenizer_spec === 'string' ? meta.text_tokenizer_spec : '{}',
    tagCfgTrained: meta.tag_cfg_trained === true,
    nEmbd: num(meta, 'embedding_length'),
    nLayer: num(meta, 'block_count'),
  };
}

export function buildPrompt(
  tok: Tokenizer, text: string,
  opt: { referenceCodes?: Int32Array; speakerEmbedding?: Float32Array; buckets: number[]; silencePrefill: number },
  model: ModelHp,
): Prompt {
  const ncb = model.nCodebooks, w = ncb + 1;
  const ref = opt.referenceCodes;
  const refFrames = ref ? Math.floor(ref.length / ncb) : 0;
  const spk = opt.speakerEmbedding;
  if (spk && ref) throw new Error('a speaker embedding and a reference prefix are mutually exclusive');
  if (refFrames > MAX_REFERENCE_FRAMES || (ref !== undefined && refFrames === 0)) throw new Error('invalid reference-code prefix');
  const prefill = opt.silencePrefill;
  if (!Number.isInteger(prefill) || prefill < 0 || prefill > MAX_SILENCE_PREFILL)
    throw new Error(`silence prefill must be 0..${MAX_SILENCE_PREFILL} frames`);
  if (spk) {
    for (let i = 0; i < model.speakerDim; i++)
      if (!Number.isFinite(spk[i])) throw new Error('speaker embedding contains a non-finite value');
  }
  const buckets = opt.buckets.length ? opt.buckets.slice() : new Array<number>(model.nCond).fill(model.nCondBuckets);
  if (buckets.length !== model.nCond) throw new Error(`expected ${model.nCond} condition buckets`);
  for (let i = 0; i < model.nCond; i++) {
    if (buckets[i] < 0 || buckets[i] > model.nCondBuckets)
      throw new Error(`condition bucket ${buckets[i]} for ${model.condNames[i]} outside 0..${model.nCondBuckets - 1} (${model.nCondBuckets} = unknown)`);
  }
  const ids = tok.encode(text);
  const referenceRows = refFrames ? refFrames + ncb - 1 : 0;
  const speakerRows = spk ? 1 : referenceRows;
  const rows = speakerRows + model.nCond + ids.length + prefill;
  if (rows === 0 || rows > model.contextLength) throw new Error(`prompt exceeds the ${model.contextLength}-position context`);
  for (let i = 0; i < refFrames * ncb; i++) {
    if (ref![i] < 0 || ref![i] >= CODEBOOK_SIZE) throw new Error('reference contains a nonordinary audio code');
  }
  const out = new Int32Array(rows * w).fill(model.audioPad);
  for (let t = 0; t < rows; t++) out[t * w + ncb] = model.textVocab;
  for (let t = 0; t < referenceRows; t++) {
    for (let cb = 0; cb < ncb; cb++) {
      const f = t - cb;
      if (f >= 0 && f < refFrames) out[t * w + cb] = ref![f * ncb + cb];
    }
  }
  const textStart = speakerRows + model.nCond;
  for (let t = 0; t < ids.length; t++) out[(textStart + t) * w + ncb] = ids[t];
  const silenceStart = textStart + ids.length;
  for (let k = 0; k < prefill; k++)
    for (let cb = 0; cb < ncb; cb++)
      if (k - cb >= 0) out[(silenceStart + k) * w + cb] = model.silenceFrame[cb];
  return {
    ids: out, nRows: rows, speakerRows, prefillFrames: prefill,
    cond: {
      speakerRow: spk ? 0 : -1,
      speaker: spk ? Float32Array.from(spk.subarray(0, model.speakerDim)) : null,
      conditionRow: speakerRows, buckets,
    },
  };
}

export function buildCfgTwin(p: Prompt, nBuckets: number): Prompt {
  if (p.speakerRows <= 0) throw new Error('speaker CFG needs a speaker prompt (speaker embedding or reference)');
  if (p.speakerRows >= p.nRows || p.ids.length !== p.nRows * W) throw new Error('inconsistent prompt for speaker CFG');
  return {
    ids: p.ids.slice(p.speakerRows * W),
    nRows: p.nRows - p.speakerRows,
    speakerRows: 0,
    prefillFrames: p.prefillFrames,
    cond: {
      speakerRow: -1, speaker: null, conditionRow: p.cond.conditionRow - p.speakerRows,
      buckets: new Array<number>(p.cond.buckets.length).fill(nBuckets),
    },
  };
}

/** bisect_right(edges, v): number of edges <= v; -1 for a non-finite value. */
export function conditionBucket(edges: number[], v: number): number {
  if (!Number.isFinite(v)) return -1;
  let b = 0;
  while (b < edges.length && edges[b] <= v) b++;
  return b;
}

export function prefillCodes(p: Prompt): Int32Array {
  const out = new Int32Array(p.prefillFrames * NCB);
  for (let k = 0; k < p.prefillFrames; k++) {
    const row = (p.nRows - p.prefillFrames + k) * W;
    out.set(p.ids.subarray(row, row + NCB), k * NCB);
  }
  return out;
}
