// k-major f16 ("h8") codec matrices for the register-tiled gemm_h8 kernels (ported from the OpenCL engine's
// gemm_h8.cl), and the load-time timing that picks, per row class, the fastest of the q8 kernels and the h8 variants.
import type { Gpu } from './device';
import { gemm } from './ops';
import { Builder, linearWg } from './program';
import type { GpuTensor, Model } from './weights';

/** Row-tiled variants for many rows and split-K variants for few (ks16's 16 KB reduction buffer sits at the default
 *  workgroup-storage limit, so it is left out). */
const H8_SKINNY = ['gemm_h8_ks4', 'gemm_h8_ks8', 'gemm_h8_r2_8x8'];
const H8_LARGE = ['gemm_h8_r4_8x8', 'gemm_h8_r8_8x8', 'gemm_h8_r4_16x4'];

/** The gemm kernel per codec row class: 'q8' (the existing q8 kernels) or a gemm_h8 variant. */
export interface GemmPlan { skinny: string; large: string }

const ceil32 = (n: number) => Math.ceil(n / 32) * 32;

/** Repacks a q8 [O][IC] matrix (K = 1) into a new h8 buffer; submitted, not waited on. */
export function repackH8(gpu: Gpu, W: GpuTensor, O: number, IC: number): GPUBuffer {
  if (W.dtype !== 'q8' || !W.scales) throw new Error(`repackH8: expected q8 weights, got ${W.dtype}`);
  const words = IC * ceil32(O) / 2;
  const out = gpu.buffer(words * 4, 'h8');
  const b = new Builder(gpu);
  b.dispatch('repack_h8', { Wq: W.buf, Wsc: W.scales, outp: out }, { O, IC, K: 1, words }, linearWg(Math.ceil(words / 256)));
  const prog = b.finish();
  const enc = gpu.device.createCommandEncoder();
  prog.encode(enc);
  gpu.device.queue.submit([enc.finish()]);
  void gpu.device.queue.onSubmittedWorkDone().then(() => prog.destroy());
  return out;
}

/** Milliseconds per gemm of `W` with `kernel` ('q8': the q8 path) over T rows: best of 3 timed submissions of 4
 *  identical dispatches, after a warm-up. */
export async function timeGemm(gpu: Gpu, W: GpuTensor, kernel: string, T: number, O: number, IC: number): Promise<number> {
  const X = gpu.buffer(T * IC * 4, 'tune.x');
  const Y = gpu.buffer(T * O * 4, 'tune.y');
  const b = new Builder(gpu);
  for (let i = 0; i < 4; i++) gemm(b, X, W, Y, { T, O, IC, kernel: kernel === 'q8' ? undefined : kernel });
  const prog = b.finish();
  const run = async () => {
    const enc = gpu.device.createCommandEncoder();
    prog.encode(enc);
    gpu.device.queue.submit([enc.finish()]);
    await gpu.device.queue.onSubmittedWorkDone();
  };
  try {
    await run();
    let best = Infinity;
    for (let i = 0; i < 3; i++) {
      const t = performance.now();
      await run();
      best = Math.min(best, performance.now() - t);
    }
    return best / 4;
  } finally {
    prog.destroy();
    X.destroy();
    Y.destroy();
  }
}

/** The fastest kernel of one class, timed on a representative codec weight [O][IC] at T rows. */
async function pick(gpu: Gpu, w: Model, name: string, T: number, candidates: string[]): Promise<{ kernel: string; ms: number }> {
  const W = w.get(name);
  const [IC, O] = W.shape;
  let best = { kernel: 'q8', ms: await timeGemm(gpu, W, 'q8', T, O, IC) };
  const h8: GpuTensor = { buf: repackH8(gpu, W, O, IC), dtype: 'h8', shape: W.shape };
  try {
    for (const k of candidates) {
      const ms = await timeGemm(gpu, h8, k, T, O, IC);
      if (ms < best.ms) best = { kernel: k, ms };
    }
  } finally {
    h8.buf.destroy();
  }
  return best;
}

/** Times the q8 kernels against the h8 variants on one codec matrix per row class and returns the plan. */
export async function tuneZacGemms(gpu: Gpu, w: Model): Promise<GemmPlan> {
  const skinny = await pick(gpu, w, 'stage.1.l0.ffn1.weight', 32, H8_SKINNY);
  const large = await pick(gpu, w, 'head.0.pw1.weight', 256, H8_LARGE);
  console.info(`zac: gemm skinny=${skinny.kernel} ${skinny.ms.toFixed(3)} ms, large=${large.kernel} ${large.ms.toFixed(3)} ms`);
  return { skinny: skinny.kernel, large: large.kernel };
}
