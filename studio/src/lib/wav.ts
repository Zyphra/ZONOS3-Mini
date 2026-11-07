/**
 * 16-bit PCM mono WAV, as every ZONOS3 server writes it (Python, C++, the in-browser server): per sample NaN → 0,
 * clip to [-1, 1], × 32767 in float32, round half to even (libsndfile's `lrintf`). No DOM: the in-browser server and
 * the MCP server use it too.
 */
export function wavBytes(pcm: Float32Array, sampleRate: number): Uint8Array<ArrayBuffer> {
  const n = pcm.length;
  const out = new Uint8Array(44 + n * 2);
  const dv = new DataView(out.buffer);
  const tag = (o: number, s: string) => {
    for (let i = 0; i < s.length; i++) out[o + i] = s.charCodeAt(i);
  };
  tag(0, 'RIFF');
  dv.setUint32(4, 36 + n * 2, true);
  tag(8, 'WAVE');
  tag(12, 'fmt ');
  dv.setUint32(16, 16, true);
  dv.setUint16(20, 1, true);
  dv.setUint16(22, 1, true);
  dv.setUint32(24, sampleRate, true);
  dv.setUint32(28, sampleRate * 2, true);
  dv.setUint16(32, 2, true);
  dv.setUint16(34, 16, true);
  tag(36, 'data');
  dv.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) {
    const x = pcm[i];
    if (Number.isNaN(x)) continue; // already 0
    const v = Math.fround(Math.min(1, Math.max(-1, x)) * 32767);
    let q = Math.round(v); // ties toward +inf; an odd tie result moves down to the even neighbour
    if (Math.abs(v % 1) === 0.5 && q % 2 !== 0) q -= 1;
    dv.setInt16(44 + i * 2, q, true);
  }
  return out;
}

/** Whether `bytes` start like a WAV file (`RIFF` … `WAVE`). */
export function isWav(bytes: Uint8Array): boolean {
  const at = (o: number, s: string) => [...s].every((c, i) => bytes[o + i] === c.charCodeAt(0));
  return bytes.length >= 12 && at(0, 'RIFF') && at(8, 'WAVE');
}

/** Mono float32 from a PCM WAV (8/16/24/32-bit integer or 32-bit float; channels averaged). */
export function parseWav(buf: ArrayBuffer): { pcm: Float32Array; sr: number } {
  const v = new DataView(buf);
  const tag = (o: number) => String.fromCharCode(v.getUint8(o), v.getUint8(o + 1), v.getUint8(o + 2), v.getUint8(o + 3));
  if (buf.byteLength < 12 || tag(0) !== 'RIFF' || tag(8) !== 'WAVE') throw new Error('Not a WAV file');
  let fmt = 0;
  let channels = 1;
  let sr = 44100;
  let bits = 16;
  let o = 12;
  while (o + 8 <= buf.byteLength) {
    const id = tag(o);
    const size = v.getUint32(o + 4, true);
    if (id === 'fmt ') {
      fmt = v.getUint16(o + 8, true);
      channels = v.getUint16(o + 10, true);
      sr = v.getUint32(o + 12, true);
      bits = v.getUint16(o + 22, true);
    } else if (id === 'data') {
      const bytes = bits / 8;
      const n = Math.floor(Math.min(size, buf.byteLength - o - 8) / (bytes * channels));
      const pcm = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        let acc = 0;
        for (let c = 0; c < channels; c++) {
          const p = o + 8 + (i * channels + c) * bytes;
          acc +=
            fmt === 3 ? v.getFloat32(p, true) : bits === 16 ? v.getInt16(p, true) / 32768 : bits === 8 ? (v.getUint8(p) - 128) / 128 : bits === 24 ? ((v.getUint8(p) | (v.getUint8(p + 1) << 8) | (v.getInt8(p + 2) << 16)) / 8388608) : v.getInt32(p, true) / 2147483648;
        }
        pcm[i] = acc / channels;
      }
      return { pcm, sr };
    }
    o += 8 + size + (size & 1);
  }
  throw new Error('WAV has no data');
}
