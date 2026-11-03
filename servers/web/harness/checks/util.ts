// Shared helpers for the parity checks (harness/parity.ts).
import type { Gpu } from '../../src/gpu/device';
import { parseNpyF32, type NpyArray } from '../../src/npy';

export interface CheckResult { ok: boolean; metric: Record<string, unknown> }
export interface Ctx {
  gpu: Gpu;
  base: string;                                   // models base URL (…/models/)
  log: (s: string) => void;
  golden: (path: string) => Promise<ArrayBuffer>; // public/goldens/<path>
  npy: (path: string) => Promise<NpyArray>;       // float32 .npy under public/goldens
}
export type Check = (ctx: Ctx) => Promise<CheckResult>;

export async function goldenBytes(path: string): Promise<ArrayBuffer> {
  const res = await fetch('/goldens/' + path);
  if (!res.ok) throw new Error(`missing golden ${path}: ${res.status} (run scripts/make_goldens.sh)`);
  return res.arrayBuffer();
}

export async function goldenNpy(path: string): Promise<NpyArray> {
  const a = parseNpyF32(await goldenBytes(path));
  if (!a) throw new Error(`golden ${path} is not a float32 npy`);
  return a;
}

export function cosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return d / Math.sqrt(na * nb + 1e-30);
}

export function argmax(a: ArrayLike<number>, off = 0, n = a.length - off): number {
  let best = 0;
  for (let i = 1; i < n; i++) if (a[off + i] > a[off + best]) best = i;
  return best;
}

export function maxAbs(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}
