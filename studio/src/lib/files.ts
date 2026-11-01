/** 16-bit PCM mono WAV. */
export function encodeWav(pcm: Float32Array, sr: number): ArrayBuffer {
  const buf = new ArrayBuffer(44 + pcm.length * 2);
  const v = new DataView(buf);
  const str = (o: number, s: string) => {
    for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i));
  };
  str(0, 'RIFF');
  v.setUint32(4, 36 + pcm.length * 2, true);
  str(8, 'WAVE');
  str(12, 'fmt ');
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, sr, true);
  v.setUint32(28, sr * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  str(36, 'data');
  v.setUint32(40, pcm.length * 2, true);
  for (let i = 0; i < pcm.length; i++) {
    const s = Math.max(-1, Math.min(1, pcm[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buf;
}

/** A float32 1-D .npy (format 1.0), as numpy.save writes it. */
export function encodeNpy(vec: Float32Array): ArrayBuffer {
  let header = `{'descr': '<f4', 'fortran_order': False, 'shape': (${vec.length},), }`;
  const pre = 10; // magic(6) + version(2) + header len(2)
  const total = Math.ceil((pre + header.length + 1) / 64) * 64;
  header = header.padEnd(total - pre - 1, ' ') + '\n';
  const buf = new ArrayBuffer(total + vec.length * 4);
  const u8 = new Uint8Array(buf);
  u8.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0]);
  new DataView(buf).setUint16(8, header.length, true);
  for (let i = 0; i < header.length; i++) u8[pre + i] = header.charCodeAt(i);
  new Float32Array(buf, total).set(vec);
  return buf;
}

export function bufferToDataUrl(buf: ArrayBuffer, mime: string): Promise<string> {
  return blobToDataUrl(new Blob([buf], { type: mime }));
}

export function blobToDataUrl(blob: Blob): Promise<string> {
  const { promise, resolve, reject } = Promise.withResolvers<string>();
  const r = new FileReader();
  r.onload = () => resolve(String(r.result));
  r.onerror = () => reject(r.error ?? new Error('Could not read file'));
  r.readAsDataURL(blob);
  return promise;
}

export function download(buf: ArrayBuffer, name: string, mime = 'audio/wav') {
  const url = URL.createObjectURL(new Blob([buf], { type: mime }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

export const slug = (s: string) => s.trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'voice';

/** Plain base64 of bytes (no data-URL prefix), chunked to stay under argument-length limits. */
export function bytesToBase64(buf: ArrayBuffer): string {
  const u8 = new Uint8Array(buf);
  let s = '';
  for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode(...u8.subarray(i, i + 0x8000));
  return btoa(s);
}

export function base64ToBytes(b64: string): ArrayBuffer {
  const bin = atob(b64.includes(',') ? b64.slice(b64.indexOf(',') + 1) : b64);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return u8.buffer;
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
