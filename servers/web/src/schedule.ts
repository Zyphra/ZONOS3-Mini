// Streaming codec scheduling + un-shear (server.cpp stream blocks, zac_decode frame count, decode_chunk ids).
export const NCB = 10;
export const SHEAR = NCB - 1;
export const FIRST_BLOCK = 4;
export const BLOCK = 16;
const MAX_CODE = 4095;

export type Range = [number, number];

/** Blocks ready to render after H frames exist, capped at eos (-1 = none); advances `emitted` (returned). */
export function readyBlocks(H: number, emitted: number, eosFrame: number): { blocks: Range[]; emitted: number } {
  const blocks: Range[] = [];
  for (;;) {
    const B = emitted === 0 ? FIRST_BLOCK : BLOCK;
    if (H - emitted < B + SHEAR) break;
    const limit = eosFrame >= 0 ? Math.min(H - SHEAR, eosFrame) : H - SHEAR;
    if (limit <= emitted) break;
    const end = Math.min(emitted + B, limit);
    blocks.push([emitted, end]);
    emitted = end;
  }
  return { blocks, emitted };
}

/** Final blocks (≤16 frames each) after generation ends: decode up to min(eos, H - 9). */
export function finalBlocks(H: number, emitted: number, eosFrame: number): Range[] {
  const end = eosFrame >= 0 && eosFrame < H - SHEAR ? eosFrame : Math.max(0, H - SHEAR);
  const out: Range[] = [];
  for (let f = emitted; f < end; f += BLOCK) out.push([f, Math.min(f + BLOCK, end)]);
  return out;
}

/** Un-sheared indices [10][16] clamped to [0, 4095] for frames [p0, p0+n) of flat codes [H][10]. `missing`: a residual
 *  cell whose row is past H (not arrived; early partial-frame decode) is -1, which the codec leaves out. */
export function buildIdx(codes: ArrayLike<number>, H: number, p0: number, n: number, out = new Int32Array(NCB * 16),
                         missing = false): Int32Array {
  out.fill(0);
  for (let c = 0; c < NCB; c++)
    for (let i = 0; i < n; i++) {
      const t = p0 + i + c;
      if (t >= H && missing && c > 0) { out[c * 16 + i] = -1; continue; }
      const raw = t < H ? codes[t * NCB + c] : 4097;
      out[c * 16 + i] = Math.min(Math.max(raw, 0), MAX_CODE);
    }
  return out;
}
