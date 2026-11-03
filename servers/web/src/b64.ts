// Base64 as servers/ggml/src/server/util.cpp b64_decode_strict / b64_encode (Python's base64.b64decode(validate=True)
// after stripping and an optional "data:...," prefix).
const ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
const TABLE = new Int8Array(256).fill(-1);
for (let i = 0; i < 64; i++) TABLE[ALPHABET.charCodeAt(i)] = i;

/** Decoded bytes, or null when the payload is not strict base64 (or a data URL has no comma). */
export function b64DecodeStrict(raw: string): Uint8Array | null {
  let payload = raw.trim();
  if (payload.startsWith('data:')) {
    const comma = payload.indexOf(',');
    if (comma < 0) return null;
    payload = payload.slice(comma + 1);
  }
  let dataLen = payload.length;
  while (dataLen > 0 && payload.charCodeAt(dataLen - 1) === 61) dataLen--;
  if (payload.length - dataLen > 2 || payload.length % 4 !== 0) return null;
  const out = new Uint8Array(Math.floor(dataLen * 3 / 4));
  let n = 0, val = 0, bits = 0;
  for (let i = 0; i < dataLen; i++) {
    const c = payload.charCodeAt(i);
    const d = c < 256 ? TABLE[c] : -1;
    if (d < 0) return null;
    val = ((val << 6) | d) & 0xffffff;
    bits += 6;
    if (bits >= 8) { bits -= 8; out[n++] = (val >> bits) & 0xff; }
  }
  return out.subarray(0, n);
}

export function b64Encode(data: Uint8Array): string {
  let out = '';
  for (let i = 0; i < data.length; i += 3) {
    const a = data[i], b = i + 1 < data.length ? data[i + 1] : 0, c = i + 2 < data.length ? data[i + 2] : 0;
    out += ALPHABET[a >> 2] + ALPHABET[((a & 3) << 4) | (b >> 4)];
    out += i + 1 < data.length ? ALPHABET[((b & 15) << 2) | (c >> 6)] : '=';
    out += i + 2 < data.length ? ALPHABET[c & 63] : '=';
  }
  return out;
}
