// Minimal .npy codecs: a little-endian f32 reader (goldens) and ports of mp_load_reference_codes /
// mp_save_reference_codes (model-paths.h) for ordinary [T,10] reference codes, with native error messages.
export interface NpyArray { data: Float32Array; shape: number[] }

export function parseNpyF32(buf: ArrayBuffer): NpyArray | null {
  const u8 = new Uint8Array(buf);
  if (u8.length < 10 || u8[0] !== 0x93 || String.fromCharCode(...u8.subarray(1, 6)) !== 'NUMPY') return null;
  const major = u8[6];
  const dv = new DataView(buf);
  let hlen: number, hoff: number;
  if (major === 1) { hlen = dv.getUint16(8, true); hoff = 10; }
  else if (major === 2 || major === 3) { if (u8.length < 12) return null; hlen = dv.getUint32(8, true); hoff = 12; }
  else return null;
  if (hoff + hlen > u8.length) return null;
  const header = new TextDecoder().decode(u8.subarray(hoff, hoff + hlen));
  const descr = /'descr'\s*:\s*'([^']*)'/.exec(header)?.[1];
  const fortran = /'fortran_order'\s*:\s*(True|False)/.exec(header)?.[1];
  const shapeStr = /'shape'\s*:\s*\(([^)]*)\)/.exec(header)?.[1];
  if (descr !== '<f4' || fortran !== 'False' || shapeStr === undefined) return null;
  const shape = shapeStr.split(',').map(s => s.trim()).filter(s => s.length).map(Number);
  if (shape.some(n => !Number.isInteger(n) || n < 0)) return null;
  const n = shape.reduce((a, b) => a * b, 1);
  const off = hoff + hlen;
  if (off + n * 4 > u8.length) return null;
  const data = new Float32Array(n);
  for (let i = 0; i < n; i++) data[i] = dv.getFloat32(off + i * 4, true);
  return { data, shape };
}

export const MAX_REFERENCE_FRAMES = 900;
const DTYPES = ['<i4', '<i8', '<u4', '<u8', '<f4', '<f8'];

/** Ordinary C-order reference codes [T][10] (1 <= T <= 900, integer IDs 0..4095) from .npy bytes. */
export function loadReferenceCodes(buf: ArrayBuffer): Int32Array {
  const u8 = new Uint8Array(buf);
  const fail = (why: string): never => { throw new Error(why); };
  if (u8.length < 8 || u8[0] !== 0x93 || String.fromCharCode(...u8.subarray(1, 6)) !== 'NUMPY' ||
      u8[7] !== 0 || (u8[6] !== 1 && u8[6] !== 2 && u8[6] !== 3))
    fail('invalid NPY reference header');
  const hbytes = u8[6] === 1 ? 2 : 4;
  if (u8.length < 8 + hbytes) fail('truncated NPY header');
  let nheader = 0;
  for (let i = hbytes - 1; i >= 0; i--) nheader = nheader * 256 + u8[8 + i];
  if (nheader > 65536) fail('oversized NPY reference header');
  const hoff = 8 + hbytes;
  if (u8.length < hoff + nheader) fail('truncated NPY header');
  const header = String.fromCharCode(...u8.subarray(hoff, hoff + nheader));
  const order = header.indexOf('fortran_order'), shape = header.indexOf('shape'), descr = header.indexOf('descr');
  if (order < 0 || shape < 0 || descr < 0) fail('invalid NPY reference metadata');
  let p = header.indexOf(':', order);
  if (p < 0) fail('invalid NPY array order');
  p = p + 1;
  while (p < header.length && (header[p] === ' ' || header[p] === '\t')) p++;
  if (p >= header.length || !header.startsWith('False', p)) fail('reference NPY must be C-order');
  p = header.indexOf(':', descr);
  if (p < 0) fail('invalid NPY dtype');
  const q0 = header.slice(p + 1).search(/['"]/);
  if (q0 < 0) fail('invalid NPY dtype');
  const qs = p + 1 + q0;
  const qe = header.indexOf(header[qs], qs + 1);
  if (qe < 0) fail('invalid NPY dtype');
  const dtype = header.slice(qs + 1, qe);
  if (!DTYPES.includes(dtype)) fail('reference NPY needs little-endian i4/i8/u4/u8/f4/f8 dtype');
  const shapeMsg = 'reference codes must have shape [T,10], 1 <= T <= 900 (60 seconds)';
  p = header.indexOf('(', shape);
  // sscanf("( %lld , %lld %n"): whitespace anywhere around the two integers
  const m = p < 0 ? null : /^\(\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*/.exec(header.slice(p));
  if (!m) return fail(shapeMsg);
  const frames = Number(m[1]), width = Number(m[2]);
  if (frames < 1 || frames > MAX_REFERENCE_FRAMES || width !== 10) fail(shapeMsg);
  p += m[0].length;
  while (p < header.length && (header[p] === ' ' || header[p] === ',')) p++;
  if (p === header.length || header[p] !== ')') fail('reference codes must be two-dimensional');
  const bytes = dtype.charCodeAt(2) - 48;
  const off = hoff + nheader;
  const n = frames * 10;
  if (u8.length < off + n * bytes) fail('truncated reference codes');
  const dv = new DataView(buf, off);
  const codes = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const o = i * bytes;
    let v: number;
    if (dtype[1] === 'f') v = bytes === 4 ? dv.getFloat32(o, true) : dv.getFloat64(o, true);
    else if (dtype[1] === 'i') v = bytes === 4 ? dv.getInt32(o, true) : Number(dv.getBigInt64(o, true));
    else v = bytes === 4 ? dv.getUint32(o, true) : Number(dv.getBigUint64(o, true));
    if (!Number.isFinite(v) || v < 0 || v > 4095 || v !== Math.floor(v))
      fail('reference codes must be ordinary integer codec IDs 0..4095');
    codes[i] = v;
  }
  if (u8.length > off + n * bytes) fail('unexpected trailing data in reference NPY');
  return codes;
}

/** .npy (v1.0, <i4, header padded to a 64-byte multiple) bytes for flat reference codes [T][10]. */
export function saveReferenceCodes(codes: Int32Array): Uint8Array {
  if (!codes.length || codes.length % 10) throw new Error('reference codes must be [T,10]');
  let header = `{'descr': '<i4', 'fortran_order': False, 'shape': (${codes.length / 10}, 10), }`;
  header += ' '.repeat((64 - (10 + header.length + 1) % 64) % 64) + '\n';
  const out = new Uint8Array(10 + header.length + codes.length * 4);
  const dv = new DataView(out.buffer);
  out.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0]);
  dv.setUint16(8, header.length, true);
  for (let i = 0; i < header.length; i++) out[10 + i] = header.charCodeAt(i);
  for (let i = 0; i < codes.length; i++) dv.setInt32(10 + header.length + i * 4, codes[i], true);
  return out;
}
