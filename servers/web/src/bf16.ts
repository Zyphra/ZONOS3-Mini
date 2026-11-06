// Rounding to the nearest bf16 (ties to even) and back to f32, as torch's .to(torch.bfloat16) and ggml_fp32_to_bf16;
// NaN and Inf pass through unchanged.

const scratch = new Float32Array(1);
const scratchBits = new Uint32Array(scratch.buffer);

/** The f32 bits `u` rounded to bf16 (NaN / Inf unchanged). */
function roundBits(u: number): number {
  if ((u & 0x7fffffff) > 0x7f800000) return u;
  return ((u + 0x7fff + ((u >>> 16) & 1)) & 0xffff0000) >>> 0;
}

export function bf16Round(x: number): number {
  scratch[0] = x;
  scratchBits[0] = roundBits(scratchBits[0]);
  return scratch[0];
}

export function bf16RoundInPlace(a: Float32Array): void {
  const u = new Uint32Array(a.buffer, a.byteOffset, a.length);
  for (let i = 0; i < u.length; i++) u[i] = roundBits(u[i]);
}
