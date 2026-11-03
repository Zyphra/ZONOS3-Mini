import { describe, expect, it } from 'vitest';
import { REFERENCE_MAX_FRAMES } from '../src/generation';
import { loadReferenceCodes, saveReferenceCodes } from '../src/npy';

/** v1.0 .npy bytes with the same header layout saveReferenceCodes uses (64-byte aligned, trailing '\n'). */
function npyBytes(dict: string, payload: Uint8Array): ArrayBuffer {
  const header = dict + ' '.repeat((64 - (10 + dict.length + 1) % 64) % 64) + '\n';
  const out = new Uint8Array(10 + header.length + payload.length);
  out.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0]);
  new DataView(out.buffer).setUint16(8, header.length, true);
  for (let i = 0; i < header.length; i++) out[10 + i] = header.charCodeAt(i);
  out.set(payload, 10 + header.length);
  return out.buffer;
}

function i4(values: number[]): Uint8Array {
  const out = new Uint8Array(values.length * 4);
  const dv = new DataView(out.buffer);
  values.forEach((v, i) => dv.setInt32(i * 4, v, true));
  return out;
}

describe('reference codes npy', () => {
  it('round-trips [3,10] codes with an aligned header', () => {
    const codes = Int32Array.from({ length: 30 }, (_, i) => (i * 137) % 4096);
    const bytes = saveReferenceCodes(codes);
    expect((10 + new DataView(bytes.buffer).getUint16(8, true)) % 64).toBe(0);
    expect(Array.from(loadReferenceCodes(bytes.buffer as ArrayBuffer))).toEqual(Array.from(codes));
  });

  it('keeps the first REFERENCE_MAX_FRAMES frames of a longer file, validating all of it', () => {
    const frames = REFERENCE_MAX_FRAMES + 50;
    const dict = `{'descr': '<i4', 'fortran_order': False, 'shape': (${frames}, 10), }`;
    const vals = Array.from({ length: frames * 10 }, (_, i) => Math.floor(i / 10));
    const codes = loadReferenceCodes(npyBytes(dict, i4(vals)));
    expect(codes.length).toBe(REFERENCE_MAX_FRAMES * 10);
    expect(Array.from(codes)).toEqual(vals.slice(0, REFERENCE_MAX_FRAMES * 10));
    vals[vals.length - 1] = 4096;
    expect(() => loadReferenceCodes(npyBytes(dict, i4(vals)))).toThrow('reference codes must be ordinary integer codec IDs 0..4095');
  });

  it('rejects codes outside 0..4095', () => {
    const vals = new Array(10).fill(0);
    vals[3] = 4096;
    const buf = npyBytes("{'descr': '<i4', 'fortran_order': False, 'shape': (1, 10), }", i4(vals));
    expect(() => loadReferenceCodes(buf)).toThrow('reference codes must be ordinary integer codec IDs 0..4095');
  });

  it('loads integral <f8 values', () => {
    const payload = new Uint8Array(80);
    const dv = new DataView(payload.buffer);
    for (let i = 0; i < 10; i++) dv.setFloat64(i * 8, 7.0, true);
    const buf = npyBytes("{'descr': '<f8', 'fortran_order': False, 'shape': (1, 10), }", payload);
    expect(Array.from(loadReferenceCodes(buf))).toEqual(new Array(10).fill(7));
  });

  it('rejects Fortran order', () => {
    const buf = npyBytes("{'descr': '<i4', 'fortran_order': True, 'shape': (1, 10), }", i4(new Array(10).fill(0)));
    expect(() => loadReferenceCodes(buf)).toThrow('reference NPY must be C-order');
  });
});
