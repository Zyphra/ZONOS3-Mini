/** Interleaved (cos, sin) table [cap][hd] for GGML_ROPE_TYPE_NORMAL; theta advanced in f32 like ggml. */
export function ropeTable(cap: number, base: number, hd: number): Float32Array {
  const f = Math.fround;
  const out = new Float32Array(cap * hd);
  const thetaScale = f(Math.pow(base, -2 / hd));
  for (let p = 0; p < cap; p++) {
    let theta = f(p);
    for (let i = 0; i < hd / 2; i++) {
      out[p * hd + 2 * i] = Math.cos(theta);
      out[p * hd + 2 * i + 1] = Math.sin(theta);
      theta = f(theta * thetaScale);
    }
  }
  return out;
}
