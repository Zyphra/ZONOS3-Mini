const K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

/**
 * SHA-256 of bytes as 64 hex characters: the content address of project media (the server verifies it on upload).
 * Plain JS because crypto.subtle exists only in secure contexts, and the studio on a cluster host over http is not one.
 */
export function sha256(bytes: Uint8Array): string {
  const h = new Uint32Array([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]);
  const w = new Uint32Array(64);
  const len = bytes.length;
  // The last one or two blocks: the tail, 0x80, zeros, the bit length (big-endian).
  const tailStart = len - (len % 64);
  const tail = new Uint8Array(len % 64 < 56 ? 64 : 128);
  tail.set(bytes.subarray(tailStart));
  tail[len - tailStart] = 0x80;
  const tv = new DataView(tail.buffer);
  tv.setUint32(tail.length - 8, Math.floor(len / 0x20000000));
  tv.setUint32(tail.length - 4, (len << 3) >>> 0);
  const block = (src: Uint8Array, off: number) => {
    for (let i = 0; i < 16; i++) {
      const j = off + i * 4;
      w[i] = (src[j] << 24) | (src[j + 1] << 16) | (src[j + 2] << 8) | src[j + 3];
    }
    for (let i = 16; i < 64; i++) {
      const a = w[i - 15];
      const b = w[i - 2];
      const s0 = ((a >>> 7) | (a << 25)) ^ ((a >>> 18) | (a << 14)) ^ (a >>> 3);
      const s1 = ((b >>> 17) | (b << 15)) ^ ((b >>> 19) | (b << 13)) ^ (b >>> 10);
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) | 0;
    }
    let a = h[0];
    let b = h[1];
    let c = h[2];
    let d = h[3];
    let e = h[4];
    let f = h[5];
    let g = h[6];
    let k = h[7];
    for (let i = 0; i < 64; i++) {
      const S1 = ((e >>> 6) | (e << 26)) ^ ((e >>> 11) | (e << 21)) ^ ((e >>> 25) | (e << 7));
      const t1 = (k + S1 + ((e & f) ^ (~e & g)) + K[i] + w[i]) | 0;
      const S0 = ((a >>> 2) | (a << 30)) ^ ((a >>> 13) | (a << 19)) ^ ((a >>> 22) | (a << 10));
      const t2 = (S0 + ((a & b) ^ (a & c) ^ (b & c))) | 0;
      k = g;
      g = f;
      f = e;
      e = (d + t1) | 0;
      d = c;
      c = b;
      b = a;
      a = (t1 + t2) | 0;
    }
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
    h[5] += f;
    h[6] += g;
    h[7] += k;
  };
  for (let off = 0; off < tailStart; off += 64) block(bytes, off);
  for (let off = 0; off < tail.length; off += 64) block(tail, off);
  return Array.from(h, (x) => x.toString(16).padStart(8, '0')).join('');
}

export const isSha256 = (h: string) => /^[0-9a-f]{64}$/.test(h);

/** SHA-256 off the main thread's hot path where the browser offers it (secure contexts); big files (videos) use this. */
export async function sha256Async(bytes: Uint8Array<ArrayBuffer>): Promise<string> {
  if (!globalThis.crypto?.subtle) return sha256(bytes);
  const d = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
  return Array.from(d, (x) => x.toString(16).padStart(2, '0')).join('');
}

/**
 * MurmurHash3 x86 128-bit as 32 hex characters: how media were addressed before SHA-256 (browser projects and zips
 * from then). Only checks those zips; moving such media to the server re-addresses them.
 */
export function hash128(bytes: Uint8Array): string {
  const c1 = 0x239b961b;
  const c2 = 0xab0e9789;
  const c3 = 0x38b34ae5;
  const c4 = 0xa1e38b93;
  const rotl = (x: number, r: number) => (x << r) | (x >>> (32 - r));
  const len = bytes.length;
  const blocks = len >>> 4;
  // Whole 16-byte blocks as little-endian words (every browser platform is little-endian); copy when unaligned.
  const words = bytes.byteOffset % 4 === 0 ? new Uint32Array(bytes.buffer, bytes.byteOffset, blocks * 4) : new Uint32Array(bytes.slice(0, blocks * 16).buffer);
  let h1 = 0;
  let h2 = 0;
  let h3 = 0;
  let h4 = 0;
  for (let i = 0; i < blocks * 4; i += 4) {
    let k1 = words[i];
    let k2 = words[i + 1];
    let k3 = words[i + 2];
    let k4 = words[i + 3];
    k1 = Math.imul(rotl(Math.imul(k1, c1), 15), c2);
    h1 ^= k1;
    h1 = (Math.imul(rotl(h1, 19) + h2, 5) + 0x561ccd1b) | 0;
    k2 = Math.imul(rotl(Math.imul(k2, c2), 16), c3);
    h2 ^= k2;
    h2 = (Math.imul(rotl(h2, 17) + h3, 5) + 0x0bcaa747) | 0;
    k3 = Math.imul(rotl(Math.imul(k3, c3), 17), c4);
    h3 ^= k3;
    h3 = (Math.imul(rotl(h3, 15) + h4, 5) + 0x96cd1c35) | 0;
    k4 = Math.imul(rotl(Math.imul(k4, c4), 18), c1);
    h4 ^= k4;
    h4 = (Math.imul(rotl(h4, 13) + h1, 5) + 0x32ac3b17) | 0;
  }
  const tail = blocks * 16;
  const k = [0, 0, 0, 0];
  for (let i = len - 1; i >= tail; i--) k[(i - tail) >>> 2] ^= bytes[i] << (((i - tail) & 3) * 8);
  const rest = len & 15;
  if (rest > 12) h4 ^= Math.imul(rotl(Math.imul(k[3], c4), 18), c1);
  if (rest > 8) h3 ^= Math.imul(rotl(Math.imul(k[2], c3), 17), c4);
  if (rest > 4) h2 ^= Math.imul(rotl(Math.imul(k[1], c2), 16), c3);
  if (rest > 0) h1 ^= Math.imul(rotl(Math.imul(k[0], c1), 15), c2);
  h1 ^= len;
  h2 ^= len;
  h3 ^= len;
  h4 ^= len;
  h1 = (h1 + h2 + h3 + h4) | 0;
  h2 = (h2 + h1) | 0;
  h3 = (h3 + h1) | 0;
  h4 = (h4 + h1) | 0;
  const fmix = (h: number) => {
    h = Math.imul(h ^ (h >>> 16), 0x85ebca6b);
    h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35);
    return h ^ (h >>> 16);
  };
  h1 = fmix(h1);
  h2 = fmix(h2);
  h3 = fmix(h3);
  h4 = fmix(h4);
  h1 = (h1 + h2 + h3 + h4) | 0;
  h2 = (h2 + h1) | 0;
  h3 = (h3 + h1) | 0;
  h4 = (h4 + h1) | 0;
  return [h1, h2, h3, h4].map((h) => (h >>> 0).toString(16).padStart(8, '0')).join('');
}
