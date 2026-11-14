import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { parseNpyF32 } from '../src/npy';
import {
  AUDIO_PAD, buildCfgTwin, buildPrompt, conditionBucket, conditionBucketCounts, modelHp, NCB, prefillCodes, TEXT_PAD, W,
  type ModelHp,
} from '../src/prompt';
import { createTokenizer, NO_TEXT_MARKERS } from '../src/tokenizer';

const tok = createTokenizer();
const silence = [1918, 746, 3544, 2300, 11, 12, 13, 14, 15, 16];
const RATE_EDGES = [
  6.546, 8.182, 9.273, 10.31, 11.054, 11.667, 12.193, 12.656, 13.083, 13.476, 13.846, 14.196, 14.53, 14.887, 15.193,
  15.517, 15.865, 16.208, 16.579, 16.972, 17.411, 17.898, 18.434, 19.069, 19.818, 20.816, 22.286, 24.722, 29.25,
];
const hp: ModelHp = {
  nCodebooks: 10, audioPad: 4097, textVocab: 8196, audioVocab: 4098, spanEndId: null, textExtraTokens: 0, nCond: 3,
  condBucketCounts: [10, 10, 30],
  condNames: ['loudness', 'fmax', 'speaking_rate'],
  condEdges: [[1, 2, 3], [1, 2, 3], RATE_EDGES], silenceFrame: silence, speakerDim: 192, contextLength: 32768,
  condUnits: ['LUFS', 'Hz', 'utf-8 bytes per second'], textTokenizerSpec: '{}', textMarkers: NO_TEXT_MARKERS,
  nEmbd: 1024, nLayer: 11,
};
const row = (ids: Int32Array, r: number) => Array.from(ids.subarray(r * W, r * W + W));
const padRow = [...new Array<number>(NCB).fill(AUDIO_PAD), TEXT_PAD];

describe('buildPrompt', () => {
  it('lays out condition rows, text and delayed silence', () => {
    const text = tok.encode('Hello world');
    const p = buildPrompt(tok, 'Hello world', { buckets: [3, 10, 10], silencePrefill: 3 }, hp);
    expect(p.nRows).toBe(3 + text.length + 3);
    expect(p.speakerRows).toBe(0);
    expect(p.cond.speakerRow).toBe(-1);
    expect(p.cond.conditionRow).toBe(0);
    for (let r = 0; r < 3; r++) expect(row(p.ids, r)).toEqual(padRow);
    for (let i = 0; i < text.length; i++) expect(p.ids[(3 + i) * W + NCB]).toBe(text[i]);
    const s0 = 3 + text.length;
    for (let k = 0; k < 3; k++) {
      const r = row(p.ids, s0 + k);
      expect(r[NCB]).toBe(TEXT_PAD);
      for (let cb = 0; cb < NCB; cb++) expect(r[cb]).toBe(k >= cb ? silence[cb] : AUDIO_PAD);
    }
    expect(Array.from(prefillCodes(p))).toEqual([0, 1, 2].flatMap(k => row(p.ids, s0 + k).slice(0, NCB)));
  });

  it('puts an embedding speaker row first', () => {
    const p = buildPrompt(tok, 'Hi', { speakerEmbedding: new Float32Array(192).fill(0.1), buckets: [1, 2, 3], silencePrefill: 0 }, hp);
    expect(p.cond.speakerRow).toBe(0);
    expect(p.cond.conditionRow).toBe(1);
    expect(p.speakerRows).toBe(1);
    expect(row(p.ids, 0)).toEqual(padRow);
  });

  it('prepends a delayed reference prefix', () => {
    const ref = Int32Array.from({ length: 20 }, (_, i) => i);
    const p = buildPrompt(tok, 'Hi', { referenceCodes: ref, buckets: [], silencePrefill: 0 }, hp);
    expect(p.speakerRows).toBe(11);
    expect(p.cond.conditionRow).toBe(11);
    expect(p.cond.buckets).toEqual([10, 10, 30]);
    expect(row(p.ids, 0)).toEqual([0, ...new Array<number>(9).fill(AUDIO_PAD), TEXT_PAD]);
    expect(row(p.ids, 10)[9]).toBe(19);
    expect(() => buildPrompt(tok, 'Hi', { referenceCodes: ref, speakerEmbedding: new Float32Array(192), buckets: [], silencePrefill: 0 }, hp))
      .toThrow();
    expect(() => buildPrompt(tok, 'Hi', { referenceCodes: Int32Array.from([4096, ...ref.slice(1)]), buckets: [], silencePrefill: 0 }, hp))
      .toThrow();
    expect(() => buildPrompt(tok, 'Hi', { buckets: [1], silencePrefill: 0 }, hp)).toThrow();
    expect(() => buildPrompt(tok, 'Hi', { buckets: [], silencePrefill: 31 }, hp)).toThrow();
  });

  it('checks each condition row against its own bucket count', () => {
    expect(buildPrompt(tok, 'Hi', { buckets: [1, 2, 29], silencePrefill: 0 }, hp).cond.buckets).toEqual([1, 2, 29]);
    expect(buildPrompt(tok, 'Hi', { buckets: [1, 2, 30], silencePrefill: 0 }, hp).cond.buckets).toEqual([1, 2, 30]);
    expect(() => buildPrompt(tok, 'Hi', { buckets: [1, 2, 31], silencePrefill: 0 }, hp))
      .toThrow();
    expect(() => buildPrompt(tok, 'Hi', { buckets: [11, 2, 3], silencePrefill: 0 }, hp))
      .toThrow();
  });

  it('reads per-table bucket counts, falling back to the legacy scalar', () => {
    expect(conditionBucketCounts({ 'condition.count': 3, 'condition.bucket_counts': [10, 10, 30] })).toEqual([10, 10, 30]);
    expect(conditionBucketCounts({ 'condition.count': 3, 'condition.bucket_count': 10 })).toEqual([10, 10, 10]);
  });

  it('builds the CFG twin without speaker rows', () => {
    const p = buildPrompt(tok, 'Hi', { speakerEmbedding: new Float32Array(192), buckets: [1, 2, 3], silencePrefill: 2 }, hp);
    const t = buildCfgTwin(p, hp.condBucketCounts);
    expect(t.nRows).toBe(p.nRows - 1);
    expect(t.speakerRows).toBe(0);
    expect(t.cond.speakerRow).toBe(-1);
    expect(t.cond.conditionRow).toBe(0);
    expect(t.cond.buckets).toEqual([10, 10, 30]);
    expect(Array.from(t.ids)).toEqual(Array.from(p.ids.subarray(W)));
  });

  it('bisects condition edges to the right', () => {
    expect(conditionBucket([1, 2, 3], 2)).toBe(2);
    expect(conditionBucket([1, 2, 3], 0)).toBe(0);
    expect(conditionBucket([1, 2, 3], 9)).toBe(3);
  });
});

const G = new URL('../public/goldens/', import.meta.url);
const M = new URL('../public/models/backbone/manifest.json', import.meta.url);
const have = existsSync(new URL('ids.npy', G)) && existsSync(M);
const npy = (name: string) => {
  const b = readFileSync(new URL(name, G));
  return parseNpyF32(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength))!;
};

describe('buildPrompt golden', () => {
  it.skipIf(!have)('matches zonos3-cli --build-prompt', () => {
    const meta = JSON.parse(readFileSync(M, 'utf8')).meta;
    const p = buildPrompt(tok, 'Hello from the browser engine.',
      { speakerEmbedding: npy('speaker-embedding.npy').data, buckets: [3, 5, 30], silencePrefill: 3 }, modelHp(meta));
    const ids = npy('ids.npy');
    expect(ids.shape).toEqual([p.nRows, W]);
    expect(Array.from(p.ids)).toEqual(Array.from(ids.data));
    const c = JSON.parse(readFileSync(new URL('ids.cond.json', G), 'utf8'));
    expect(p.cond.speakerRow).toBe(c.speaker_row);
    expect(p.cond.conditionRow).toBe(c.condition_row);
    expect(p.cond.buckets).toEqual(c.condition_buckets);
    expect(p.speakerRows).toBe(c.speaker_rows);
    expect(p.prefillFrames).toBe(c.prefill_frames);
    expect(Array.from(p.cond.speaker!)).toEqual(Array.from(Float32Array.from(c.speaker_embedding)));
  });
});
