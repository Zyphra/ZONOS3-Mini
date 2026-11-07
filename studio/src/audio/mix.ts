import type { Rendered } from '../api';
import { download } from '../lib/files';
import { wavBytes } from '../lib/wav';

export function resample(pcm: Float32Array, from: number, to: number): Float32Array {
  if (from === to) return pcm;
  const out = new Float32Array(Math.round((pcm.length * to) / from));
  const k = from / to;
  for (let i = 0; i < out.length; i++) {
    const x = i * k;
    const j = Math.floor(x);
    const f = x - j;
    out[i] = (pcm[j] ?? 0) * (1 - f) + (pcm[j + 1] ?? 0) * f;
  }
  return out;
}

/** Clips end to end with `gap` seconds of silence between them, at the first clip's rate. */
export function sequence(clips: Pick<Rendered, 'pcm' | 'sr'>[], gap: number): { pcm: Float32Array; sr: number } {
  const sr = clips[0]?.sr ?? 44100;
  const parts = clips.map((c) => resample(c.pcm, c.sr, sr));
  const silence = Math.round(gap * sr);
  const total = parts.reduce((n, p) => n + p.length, 0) + silence * Math.max(0, parts.length - 1);
  const pcm = new Float32Array(total);
  let o = 0;
  parts.forEach((p, i) => {
    pcm.set(p, o);
    o += p.length + (i < parts.length - 1 ? silence : 0);
  });
  return { pcm, sr };
}

export function exportWav(pcm: Float32Array, sr: number, name: string) {
  download(wavBytes(pcm, sr).buffer, name.endsWith('.wav') ? name : `${name}.wav`);
}
