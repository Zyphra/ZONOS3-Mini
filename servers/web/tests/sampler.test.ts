import { describe, expect, it } from 'vitest';
import { defaultSampling, Sampler, type Sampling } from '../src/sampler';

const V = 8;
const cfg = (o: Partial<Sampling>): Sampling => ({ ...defaultSampling(), rep_penalty: 1, top_k: 0, top_p: 0, ...o });

/** Distinct indices drawn across many seeds. */
function drawn(logits: number[], o: Partial<Sampling>): Set<number> {
  const out = new Set<number>();
  for (let seed = 0; seed < 400; seed++) out.add(new Sampler(cfg({ ...o, seed }), 1, V).sample(Float32Array.from(logits), 0));
  return out;
}

describe('sampler', () => {
  it('never returns the audio pad id', () => {
    const lg = [0, 1, 5, 2, 0, 0, 0, 9];
    expect(new Sampler(cfg({ greedy: true }), 1, V).sample(Float32Array.from(lg), 0)).toBe(2);
    expect(drawn(lg, { temperature: 1 }).has(7)).toBe(false);
  });

  it('top-k keeps every logit tied with the k-th largest', () => {
    const got = drawn([3, 1, 3, 0, 3, 2, 0, 0], { top_k: 2, temperature: 1 });
    expect([...got].sort()).toEqual([0, 2, 4]);
  });

  it('top-p cuts on the mass before each token', () => {
    const lg = [Math.log(0.6), Math.log(0.3), Math.log(0.1), -1e9, -1e9, -1e9, -1e9, 0];
    expect([...drawn(lg, { top_p: 0.69, temperature: 1 })].sort()).toEqual([0, 1]);
    // Two equal halves at top_p 0.5: mass before token 1 is exactly 0.5, not > 0.5, so it survives
    // (an inclusive cumulative cut stops once the running mass reaches 0.5 and drops it).
    const half = [0, 0, -1e9, -1e9, -1e9, -1e9, -1e9, 0];
    expect([...drawn(half, { top_p: 0.5, temperature: 1 })].sort()).toEqual([0, 1]);
  });
});

describe('sampler stop id and min frames', () => {
  it('never samples the stop id before generated >= minFrames + cb', () => {
    const lg = Float32Array.from([0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 9, 0]); // 2 codebooks, stop = 6
    const s = new Sampler(cfg({ greedy: true }), 2, V);
    s.stopId = 6; s.minFrames = 2;
    const frame = () => [s.sample(lg, 0), s.sample(lg, 1)];
    expect(frame()).toEqual([0, 0]); s.accept([0, 0]);
    expect(frame()).toEqual([0, 0]); s.accept([0, 0]);
    expect(frame()).toEqual([6, 0]); s.accept([0, 0]);
    expect(s.generated).toBe(3);
    expect(frame()).toEqual([6, 6]);
  });

  it('masks special ids other than the stop id', () => {
    const lg = Float32Array.from([0, 1, 0, 0, 0, 0, 9, 9]);
    const s = new Sampler(cfg({ greedy: true }), 1, V);
    expect([s.eoaId, s.padId, s.spanEndId, s.stopId]).toEqual([6, 7, -1, 6]);
    expect(s.sample(lg, 0)).toBe(6); // EOA is the stop id; the pad (7) is masked
    s.stopId = 4096;                 // a stop id outside this vocab: EOA masked too
    expect(s.sample(lg, 0)).toBe(1);
  });
});

describe('sampler span-end vocabulary', () => {
  // 9 ids: codes 0..5, EOA 6, pad 7, span-end 8
  const lg = Float32Array.from([0, 1, 0, 0, 0, 0, 9, 9, 8]);

  it('lays out EOA, pad and span-end; plain stop on EOA masks span-end', () => {
    const s = new Sampler(cfg({ greedy: true }), 1, 9, { spanEnd: true });
    expect([s.eoaId, s.padId, s.spanEndId, s.stopId]).toEqual([6, 7, 8, 6]);
    expect(s.sample(Float32Array.from([0, 1, 0, 0, 0, 0, 0, 9, 8]), 0)).toBe(1);
    expect(s.sample(lg, 0)).toBe(6);
  });

  it('spanEndStop stops on span-end, masks EOA and waits for minFrames', () => {
    const s = new Sampler(cfg({ greedy: true }), 1, 9, { spanEndStop: true });
    expect([s.eoaId, s.padId, s.spanEndId, s.stopId]).toEqual([6, 7, 8, 8]);
    s.minFrames = 1;
    expect(s.sample(lg, 0)).toBe(1);
    s.accept([1]);
    expect(s.sample(lg, 0)).toBe(8);
  });
});
