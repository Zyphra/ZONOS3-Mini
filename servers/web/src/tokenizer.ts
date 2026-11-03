// SuperBPE-8192 text tokenizer, a port of zonos3.cpp-superbpe8k/src/zonos2-tokenizer.cpp:
// added-token split, the Split pre-tokenizer
//   \p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S)   (behavior Isolated)
// as a hand-written matcher over Unicode 15.0 tables, GPT-2 byte-level symbols, rank-ordered BPE merges.
import raw from './superbpe8192.json?raw';
import { UNI_L, UNI_N } from './unicode-data';

interface TokenizerJson {
  added_tokens: { id: number; content: string }[];
  model: { vocab: Record<string, number>; merges: [string, string][] };
}

/** Binary search over flat sorted inclusive [lo, hi] pairs. */
function inRanges(r: Uint32Array, cp: number): boolean {
  let lo = 0, hi = r.length >> 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (cp < r[2 * mid]) hi = mid;
    else if (cp > r[2 * mid + 1]) lo = mid + 1;
    else return true;
  }
  return false;
}

// Character classes of the native matcher: \p{N}, \s, and [^\s\p{L}\p{N}].
const isN = (cp: number) => inRanges(UNI_N, cp);
/** Unicode White_Space (Oniguruma \s). */
const isSpace = (cp: number) =>
  (cp >= 0x09 && cp <= 0x0d) || cp === 0x20 || cp === 0x85 || cp === 0xa0 || cp === 0x1680 ||
  (cp >= 0x2000 && cp <= 0x200a) || cp === 0x2028 || cp === 0x2029 || cp === 0x202f ||
  cp === 0x205f || cp === 0x3000;
const isP = (cp: number) => !isSpace(cp) && !inRanges(UNI_L, cp) && !isN(cp);

/** Split code points [0, n) into [start, end) pieces: each regex match is a piece, each maximal gap one. */
function pretokenize(cps: number[]): [number, number][] {
  const n = cps.length;
  const pieces: [number, number][] = [];
  const matchAt = (i: number): number => {
    // A1: \p{N}{1,3}
    if (isN(cps[i])) {
      let j = i;
      while (j < n && j - i < 3 && isN(cps[j])) j++;
      return j;
    }
    // A2: ' '?[^\s\p{L}\p{N}]{2,}[\r\n/]*
    {
      const j = i + (cps[i] === 0x20 ? 1 : 0);
      let k = j;
      while (k < n && isP(cps[k])) k++;
      if (k - j >= 2) {
        while (k < n && (cps[k] === 0x0d || cps[k] === 0x0a || cps[k] === 0x2f)) k++;
        return k;
      }
    }
    // A3: ' +(?!\S)'
    if (cps[i] === 0x20) {
      let e = i;
      while (e < n && cps[e] === 0x20) e++;
      if (e === n || isSpace(cps[e])) return e;
      if (e - i >= 2) return e - 1;
    }
    return i;
  };
  let gap = 0, i = 0;
  while (i < n) {
    const e = matchAt(i);
    if (e === i) { i++; continue; }
    if (gap < i) pieces.push([gap, i]);
    pieces.push([i, e]);
    i = gap = e;
  }
  if (gap < n) pieces.push([gap, n]);
  return pieces;
}

/** GPT-2 bytes_to_unicode: printable bytes map to themselves, the rest to U+0100 + k in byte order. */
const BYTE_SYMBOLS: string[] = (() => {
  const t: string[] = [];
  let extra = 0;
  for (let b = 0; b < 256; b++) {
    const keep = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
    t.push(String.fromCodePoint(keep ? b : 256 + extra++));
  }
  return t;
})();

const utf8 = new TextEncoder();

export class Tokenizer {
  private vocab: Map<string, number>;
  private ranks = new Map<string, number>();
  private added: { content: string; id: number }[];

  constructor(j: TokenizerJson, private offset = 4, private bos = 2, private eos = 3) {
    this.vocab = new Map(Object.entries(j.model.vocab));
    j.model.merges.forEach((m, r) => {
      if (!Array.isArray(m) || m.length !== 2) throw new Error('tokenizer: merges must be [a, b] arrays');
      this.ranks.set(m[0] + ' ' + m[1], r);
    });
    this.added = j.added_tokens.filter(a => a.content.length > 0).map(a => ({ content: a.content, id: a.id }));
  }

  encode(text: string): Int32Array {
    const out: number[] = [this.bos];
    let pos = 0;
    while (pos < text.length) {
      let best = -1, bestLen = 0, bestId = 0;
      for (const a of this.added) {
        const f = text.indexOf(a.content, pos);
        if (f >= 0 && (best < 0 || f < best || (f === best && a.content.length > bestLen))) {
          best = f; bestLen = a.content.length; bestId = a.id;
        }
      }
      if (best < 0) { this.segment(text.slice(pos), out); break; }
      this.segment(text.slice(pos, best), out);
      out.push(bestId + this.offset);
      pos = best + bestLen;
    }
    out.push(this.eos);
    return Int32Array.from(out);
  }

  private segment(seg: string, out: number[]): void {
    if (!seg) return;
    const chars: string[] = [];
    const cps: number[] = [];
    for (const ch of seg) { chars.push(ch); cps.push(ch.codePointAt(0)!); }
    for (const [b, e] of pretokenize(cps)) this.bpe(chars.slice(b, e).join(''), out);
  }

  private bpe(piece: string, out: number[]): void {
    const sym = Array.from(utf8.encode(piece), b => BYTE_SYMBOLS[b]);
    while (sym.length > 1) {
      let best = Infinity, at = 0;
      for (let i = 0; i + 1 < sym.length; i++) {
        const r = this.ranks.get(sym[i] + ' ' + sym[i + 1]);
        if (r !== undefined && r < best) { best = r; at = i; }
      }
      if (best === Infinity) break;
      sym.splice(at, 2, sym[at] + sym[at + 1]);
    }
    for (const s of sym) {
      const id = this.vocab.get(s);
      if (id !== undefined) out.push(id + this.offset);
    }
  }

  private bytesOf: Map<number, number> | null = null;

  /** UTF-8 bytes behind a model text id (generate.cpp token_bytes): a byte-level symbol holds one code point per byte;
   *  an added token its literal content (UTF-8 length). 0 for an unknown id. */
  tokenBytes(id: number): number {
    if (!this.bytesOf) {
      const t = new Map<number, number>();
      for (const [sym, v] of this.vocab) t.set(v + this.offset, Array.from(sym).length);
      for (const a of this.added) t.set(a.id + this.offset, utf8.encode(a.content).length);
      this.bytesOf = t;
    }
    return this.bytesOf.get(id) ?? 0;
  }
}

export function createTokenizer(): Tokenizer {
  return new Tokenizer(JSON.parse(raw) as TokenizerJson);
}
