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

/** A file-name-safe form of a label: letters and digits of any script kept, runs of anything else become one dash. */
export const slug = (s: string) => s.trim().toLowerCase().replace(/[^\p{L}\p{N}]+/gu, '-').replace(/^-|-$/g, '') || 'voice';

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
