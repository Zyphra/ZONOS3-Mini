// The output fade of scheduler.cpp apply_fade.

/** Cosine fade-out over the final fadeMs milliseconds, in place. */
export function applyFade(a: Float32Array, fadeMs: number, sr: number): void {
  if (fadeMs <= 0 || a.length === 0) return;
  const n = Math.min(a.length, Math.floor(fadeMs * 0.001 * sr));
  for (let i = 0; i < n; i++) {
    const t = Math.fround(i / n);
    const g = Math.fround(0.5 * (1 + Math.fround(Math.cos(Math.fround(Math.PI * t)))));
    a[a.length - n + i] = Math.fround(a[a.length - n + i] * g);
  }
}
