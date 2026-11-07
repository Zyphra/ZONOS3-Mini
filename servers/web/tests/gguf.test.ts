// parseGguf against the ggml test fixtures.
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { GGUF_TYPE, parseGguf } from '../src/gguf';

const FIX = new URL('../../ggml/tests/fixtures/', import.meta.url);
const load = (name: string) => {
  const b = readFileSync(new URL(name, FIX));
  return parseGguf(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
};

describe('parseGguf', () => {
  it('reads kv, types and F32 tensors', () => {
    const g = load('dummy-steer.gguf');
    expect(g.kv.get('general.architecture')).toBe('zonos3_steer');
    expect(g.kv.get('zonos3_steer.speaker_dim')).toBe(192);
    expect(g.types.get('zonos3_steer.speaker_dim')).toEqual({ type: GGUF_TYPE.UINT32, arrType: null });
    expect(g.kv.get('zonos3_steer.set.sources')).toEqual([-1, 3, 5]);
    expect(g.types.get('zonos3_steer.set.sources')).toEqual({ type: GGUF_TYPE.ARRAY, arrType: GGUF_TYPE.INT32 });
    expect(g.tensors.size).toBe(5);
    expect(g.shapes.get('steer.dir.0')).toEqual([192]);
    expect(g.tensors.get('steer.dir.2')!.length).toBe(32);
    expect(g.tensors.get('steer.dir.0')!.every(Number.isFinite)).toBe(true);
  });
  it('reads 2-D shapes', () => {
    const g = load('dummy-accents.gguf');
    expect(g.shapes.get('acc.0.blk.0.attn_output.lora_down')).toEqual([128, 4]);
  });
  it('rejects non-F32 tensors', () => {
    expect(() => load('dummy-zonos3.gguf')).toThrow();
  });
});
