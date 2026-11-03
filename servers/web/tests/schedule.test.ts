import { describe, expect, it } from 'vitest';
import { buildIdx, finalBlocks, readyBlocks, type Range } from '../src/schedule';

describe('un-shear', () => {
  it('reads codebook c from frame t+c and clamps right padding to 4095', () => {
    const H = 12;
    const codes = Array.from({ length: H * 10 }, (_, i) => 100 + i);
    const idx = buildIdx(codes, H, 0, 4);
    expect(idx[9 * 16 + 2]).toBe(codes[11 * 10 + 9]);
    expect(idx[9 * 16 + 3]).toBe(4095);
  });
  it('clamps eoa/pad ids to 4095', () => {
    const idx = buildIdx([4096, 4097, 0, 0, 0, 0, 0, 0, 0, 0], 1, 0, 1);
    expect(idx[0]).toBe(4095);
    expect(idx[16]).toBe(4095);
  });
});

function stream(Hmax: number, eos: number, eosAt = 0): Range[] {
  const chunks: Range[] = [];
  let emitted = 0;
  for (let H = 1; H <= Hmax; H++) {
    const r = readyBlocks(H, emitted, H >= eosAt ? eos : -1);
    chunks.push(...r.blocks);
    emitted = r.emitted;
  }
  chunks.push(...finalBlocks(Hmax, emitted, eos));
  return chunks;
}

describe('streaming scheduler', () => {
  it('emits 4 then 16-frame blocks and flushes to eos', () => {
    expect(stream(60, 50, 59)).toEqual([[0, 4], [4, 20], [20, 36], [36, 50]]);
  });
  it('clamps a ready block at eos and emits nothing past it', () => {
    expect(stream(40, 10)).toEqual([[0, 4], [4, 10]]);
  });
  it('flushes in ≤16-frame chunks to H - 9 when no eos', () => {
    expect(finalBlocks(40, 0, -1)).toEqual([[0, 16], [16, 31]]);
  });
});
