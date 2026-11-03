/**
 * The studio's bar waveform on a canvas (`components/Waveform.tsx`, the lite page): min/max bars every 3 px, a
 * hairline centre, auto-gained to the clip's peak. Drawn against `length` samples when given (a growing stream),
 * else stretched over the clip.
 */
export function drawWave(el: HTMLCanvasElement, pcm: Float32Array | null | undefined, width: number, height: number, ink: string, length?: number) {
  const dpr = window.devicePixelRatio || 1;
  el.width = width * dpr;
  el.height = height * dpr;
  const g = el.getContext('2d')!;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, width, height);
  const mid = height / 2;
  g.fillStyle = ink;
  g.globalAlpha = 0.18;
  g.fillRect(0, mid - 0.5, width, 1);
  if (!pcm || !pcm.length) return;
  g.globalAlpha = 0.95;
  const total = Math.max(length ?? pcm.length, pcm.length);
  const bar = 2;
  const stepPx = 3;
  let peak = 0.05;
  for (let i = 0; i < pcm.length; i += 64) peak = Math.max(peak, Math.abs(pcm[i]));
  const gain = Math.min(4, 0.95 / peak);
  for (let x = 0; x < width; x += stepPx) {
    const a = Math.floor((x / width) * total);
    const b = Math.min(pcm.length, Math.floor(((x + stepPx) / width) * total));
    if (a >= pcm.length) break;
    let lo = 0;
    let hi = 0;
    for (let i = a; i < b; i += 4) {
      const v = pcm[i];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    const top = mid - Math.max(0.5, hi * gain * mid);
    const bottom = mid - Math.min(-0.5, lo * gain * mid);
    g.fillRect(x, top, bar, bottom - top);
  }
}
