// Minimal GGUF v3 reader for the companion file (steer.gguf): metadata of every kv type, tensor infos, F32 tensor data
// only.
//
// kv values: integers and floats are numbers (64-bit integers through Number), bool is boolean, string is string,
// arrays are plain JS arrays (nested arrays allowed). `types` keeps each key's GGUF type (and the element type of an
// array) so loaders can check types the way gguf_get_kv_type / gguf_get_arr_type do. `shapes` are ggml ne order.

export const GGUF_TYPE = {
  UINT8: 0, INT8: 1, UINT16: 2, INT16: 3, UINT32: 4, INT32: 5, FLOAT32: 6, BOOL: 7, STRING: 8, ARRAY: 9,
  UINT64: 10, INT64: 11, FLOAT64: 12,
} as const;

export interface GgufKvType { type: number; arrType: number | null }

export interface Gguf {
  kv: Map<string, unknown>;
  types: Map<string, GgufKvType>;
  tensors: Map<string, Float32Array>;
  shapes: Map<string, number[]>;
}

const GGML_TYPE_F32 = 0;

class Reader {
  off = 0;
  private dv: DataView;
  private bytes: Uint8Array;
  private utf8 = new TextDecoder('utf-8');
  constructor(buf: ArrayBuffer) {
    this.dv = new DataView(buf);
    this.bytes = new Uint8Array(buf);
  }
  need(n: number): void {
    if (this.off + n > this.bytes.length) throw new Error('gguf: truncated file');
  }
  u8(): number { this.need(1); return this.dv.getUint8(this.off++); }
  i8(): number { this.need(1); return this.dv.getInt8(this.off++); }
  u16(): number { this.need(2); const v = this.dv.getUint16(this.off, true); this.off += 2; return v; }
  i16(): number { this.need(2); const v = this.dv.getInt16(this.off, true); this.off += 2; return v; }
  u32(): number { this.need(4); const v = this.dv.getUint32(this.off, true); this.off += 4; return v; }
  i32(): number { this.need(4); const v = this.dv.getInt32(this.off, true); this.off += 4; return v; }
  f32(): number { this.need(4); const v = this.dv.getFloat32(this.off, true); this.off += 4; return v; }
  f64(): number { this.need(8); const v = this.dv.getFloat64(this.off, true); this.off += 8; return v; }
  u64(): number { this.need(8); const v = this.dv.getBigUint64(this.off, true); this.off += 8; return Number(v); }
  i64(): number { this.need(8); const v = this.dv.getBigInt64(this.off, true); this.off += 8; return Number(v); }
  str(): string {
    const n = this.u64();
    this.need(n);
    const s = this.utf8.decode(this.bytes.subarray(this.off, this.off + n));
    this.off += n;
    return s;
  }
}

function readValue(r: Reader, type: number): unknown {
  switch (type) {
    case GGUF_TYPE.UINT8: return r.u8();
    case GGUF_TYPE.INT8: return r.i8();
    case GGUF_TYPE.UINT16: return r.u16();
    case GGUF_TYPE.INT16: return r.i16();
    case GGUF_TYPE.UINT32: return r.u32();
    case GGUF_TYPE.INT32: return r.i32();
    case GGUF_TYPE.FLOAT32: return r.f32();
    case GGUF_TYPE.BOOL: return r.u8() !== 0;
    case GGUF_TYPE.STRING: return r.str();
    case GGUF_TYPE.UINT64: return r.u64();
    case GGUF_TYPE.INT64: return r.i64();
    case GGUF_TYPE.FLOAT64: return r.f64();
    case GGUF_TYPE.ARRAY: {
      const t = r.u32();
      const n = r.u64();
      const out: unknown[] = new Array(n);
      for (let i = 0; i < n; i++) out[i] = readValue(r, t);
      return out;
    }
    default: throw new Error(`gguf: unknown kv type ${type}`);
  }
}

export function parseGguf(buf: ArrayBuffer): Gguf {
  const r = new Reader(buf);
  if (r.u32() !== 0x46554747) throw new Error('gguf: bad magic');
  const version = r.u32();
  if (version !== 3) throw new Error(`gguf: unsupported version ${version}`);
  const nTensors = r.u64();
  const nKv = r.u64();
  const kv = new Map<string, unknown>();
  const types = new Map<string, GgufKvType>();
  for (let i = 0; i < nKv; i++) {
    const key = r.str();
    const type = r.u32();
    let arrType: number | null = null;
    if (type === GGUF_TYPE.ARRAY) {
      const save = r.off;
      arrType = r.u32();
      r.off = save;
    }
    kv.set(key, readValue(r, type));
    types.set(key, { type, arrType });
  }
  const infos: { name: string; ne: number[]; type: number; offset: number }[] = [];
  for (let i = 0; i < nTensors; i++) {
    const name = r.str();
    const nd = r.u32();
    const ne: number[] = [];
    for (let d = 0; d < nd; d++) ne.push(r.u64());
    const type = r.u32();
    const offset = r.u64();
    infos.push({ name, ne, type, offset });
  }
  const al = kv.get('general.alignment');
  const alignment = typeof al === 'number' && al > 0 ? al : 32;
  const dataStart = Math.ceil(r.off / alignment) * alignment;
  const tensors = new Map<string, Float32Array>();
  const shapes = new Map<string, number[]>();
  for (const t of infos) {
    if (t.type !== GGML_TYPE_F32) throw new Error('gguf: only F32 companion tensors are supported');
    const n = t.ne.reduce((a, b) => a * b, 1);
    const start = dataStart + t.offset;
    if (start + 4 * n > buf.byteLength) throw new Error('gguf: truncated file');
    // copy: the offset need not be 4-byte aligned relative to the buffer and callers may keep the vectors
    tensors.set(t.name, new Float32Array(buf.slice(start, start + 4 * n)));
    shapes.set(t.name, t.ne);
  }
  return { kv, types, tensors, shapes };
}
