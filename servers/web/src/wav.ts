// 16-bit mono WAV writers and the output fade: zac_wav_bytes (lrintf rounding; speaker previews) and generate.cpp
// pcm16_wav (truncating (int16_t) cast; /v2/generate format=wav), and scheduler.cpp apply_fade.

/** Rounding of a float sample to int16: 'rint' (lrintf, half to even) or 'trunc' (C cast). */
export type WavRounding = 'rint' | 'trunc';

export function wavBytes(pcm: Float32Array, sr: number, rounding: WavRounding = 'rint'): Uint8Array {
  const ns = pcm.length;
  const out = new Uint8Array(44 + ns * 2);
  const dv = new DataView(out.buffer);
  const tag = (o: number, s: string) => { for (let i = 0; i < s.length; i++) out[o + i] = s.charCodeAt(i); };
  tag(0, 'RIFF'); dv.setUint32(4, 36 + ns * 2, true); tag(8, 'WAVE');
  tag(12, 'fmt '); dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 1, true);
  dv.setUint32(24, sr, true); dv.setUint32(28, sr * 2, true); dv.setUint16(32, 2, true); dv.setUint16(34, 16, true);
  tag(36, 'data'); dv.setUint32(40, ns * 2, true);
  for (let i = 0; i < ns; i++) {
    const v = Math.fround(Math.min(1, Math.max(-1, pcm[i])) * 32767);
    let q: number;
    if (rounding === 'trunc') {
      q = Math.trunc(v);
    } else {
      q = Math.round(v);  // ties toward +inf; an odd tie result moves down to the even neighbour
      if (Math.abs(v - Math.trunc(v)) === 0.5 && q % 2 !== 0) q -= 1;
    }
    dv.setInt16(44 + i * 2, q, true);
  }
  return out;
}

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
