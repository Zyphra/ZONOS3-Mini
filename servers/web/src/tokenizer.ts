// SuperBPE-8192 text tokenizer, a port of zonos3.cpp-superbpe8k/src/zonos2-tokenizer.cpp:
// added-token split, the Split pre-tokenizer
//   \p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S)   (behavior Isolated)
// as a hand-written matcher over Unicode 15.0 tables, GPT-2 byte-level symbols, rank-ordered BPE merges.
// The text markers of a checkpoint's extra text rows (zonos3.tts.superbpe): pronunciation groups (U+E000 IPA U+E001),
// the start marker (U+E002, first in the text) and pause markers (U+E003 + bucket), each encoded as its own ids with
// the text around them encoded stretch by stretch (lingua payload_ids).
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

export const PHONEME_OPEN = '\uE000', PHONEME_CLOSE = '\uE001';
export const START_MARKER = '\uE002';
/** First pause marker (bucket 0); buckets run to U+E00A. */
const PAUSE_BASE = 0xe003;
const PAUSE_RE = /[\uE003-\uE00A]/;
export const NO_PHONEME_TOKENS = 'This checkpoint has no phoneme tokens: remove the U+E000 / U+E001 pronunciation groups';
export const NO_START_TOKEN = 'This checkpoint has no start token: remove the U+E002 start marker';
export const NO_PAUSE_TOKENS = 'This checkpoint has no pause tokens: remove the U+E003-U+E00A pause markers';
export const START_NOT_FIRST = 'The start marker (U+E002) must be the first character of the text, once';
export const START_NEEDS_TOKEN = 'start_immediately needs a checkpoint with a start token';
export const EDIT_NO_MARKERS =
  'Speech editing takes no start or pause markers: remove U+E002 / U+E003-U+E00A from target_text';

/** The phoneme, start and pause tokens of a text_tokenizer_spec (each block optional): ids are model text ids. */
export interface TextMarkers {
  phonemes: { openId: number; closeId: number; open: string; close: string; symbols: string[] } | null;
  start: { id: number; marker: string } | null;
  pauses: { ids: number[]; markers: string[]; seconds: number[] } | null;
}

export const NO_TEXT_MARKERS: TextMarkers = { phonemes: null, start: null, pauses: null };

/** Text rows the markers take (superbpe.extra_token_count). */
export function markerRows(m: TextMarkers): number {
  return (m.phonemes ? 2 + m.phonemes.symbols.length : 0) + (m.start ? 1 : 0) + (m.pauses ? m.pauses.ids.length : 0);
}

const isObject = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);
const isInt = (v: unknown): v is number => typeof v === 'number' && Number.isInteger(v);

/** A text_tokenizer_spec's phoneme / start / pause blocks (superbpe.validate_spec and load_spec): contiguous, the last
 *  of the model's `extraTokens` extra text rows (after any edit marker rows). Throws on a spec the model cannot serve. */
export function parseTextMarkers(specJson: string, textVocab: number, extraTokens: number): TextMarkers {
  const spec: unknown = JSON.parse(specJson);
  const s = isObject(spec) ? spec : {};
  const m: TextMarkers = { phonemes: null, start: null, pauses: null };
  if (s.phonemes != null) {
    const p = s.phonemes;
    if (!isObject(p) || !isInt(p.open_id) || !isInt(p.close_id) || !Array.isArray(p.symbols) ||
        typeof p.open_marker !== 'string' || typeof p.close_marker !== 'string')
      throw new Error('Invalid phoneme symbols');
    if (p.open_id < textVocab + 1 || p.close_id !== p.open_id + 1)
      throw new Error('Phoneme token IDs must follow the text padding row (and any edit marker rows)');
    const symbols = p.symbols as unknown[];
    if (symbols.some(x => typeof x !== 'string' || Array.from(x).length !== 1) || new Set(symbols).size !== symbols.length ||
        symbols.includes(p.open_marker) || symbols.includes(p.close_marker))
      throw new Error('Invalid phoneme symbols');
    m.phonemes = { openId: p.open_id, closeId: p.close_id, open: p.open_marker, close: p.close_marker, symbols: symbols as string[] };
  }
  const first = m.phonemes ? m.phonemes.openId : isObject(s.start) && isInt(s.start.id) ? s.start.id
    : isObject(s.pauses) && Array.isArray(s.pauses.ids) && isInt(s.pauses.ids[0]) ? s.pauses.ids[0] : null;
  const nPh = m.phonemes ? 2 + m.phonemes.symbols.length : 0;
  if (s.start != null) {
    const st = s.start;
    if (!isObject(st) || st.marker !== START_MARKER) throw new Error('Invalid start marker');
    if (first === null || first < textVocab + 1 || st.id !== first + nPh) throw new Error('Start token ID must follow the phoneme rows');
    m.start = { id: st.id as number, marker: START_MARKER };
  }
  if (s.pauses != null) {
    const p = s.pauses;
    const ids = isObject(p) && Array.isArray(p.ids) ? p.ids as unknown[] : [];
    const markers = isObject(p) && Array.isArray(p.markers) ? p.markers as unknown[] : [];
    const seconds = isObject(p) && Array.isArray(p.seconds) ? p.seconds as unknown[] : [];
    if (ids.length < 1 || ids.length > 8 || markers.length !== ids.length || seconds.length !== ids.length ||
        markers.some((x, j) => x !== String.fromCharCode(PAUSE_BASE + j)) || seconds.some(x => typeof x !== 'number'))
      throw new Error('Invalid pause markers');
    const lo = (first ?? 0) + nPh + (m.start ? 1 : 0);
    if (first === null || first < textVocab + 1 || ids.some((x, j) => x !== lo + j))
      throw new Error('Pause token IDs must follow the start row');
    m.pauses = { ids: ids as number[], markers: markers as string[], seconds: seconds as number[] };
  }
  if (first !== null && first - textVocab - 1 + markerRows(m) !== extraTokens)
    throw new Error('Phoneme / start / pause tokens do not match text_extra_tokens (they must be the last extra rows)');
  return m;
}

/** Python repr() of one character, as the native servers write it (zonos3-tokenizer.cpp py_char_repr). */
function pyCharRepr(ch: string): string {
  const cp = ch.codePointAt(0)!;
  let body = ch;
  if (cp === 0x5c) body = '\\\\';
  else if (cp === 0x0a) body = '\\n';
  else if (cp === 0x0d) body = '\\r';
  else if (cp === 0x09) body = '\\t';
  else if (cp < 0x20 || (cp >= 0x7f && cp <= 0xa0) || cp === 0xad) body = '\\x' + cp.toString(16).padStart(2, '0');
  else if ((cp >= 0xe000 && cp <= 0xf8ff) || (cp >= 0x2000 && cp <= 0x200f) || (cp >= 0x2028 && cp <= 0x202f) ||
           (cp >= 0x205f && cp <= 0x206f) || cp === 0x1680 || cp === 0x3000 || cp === 0xfeff)
    body = '\\u' + cp.toString(16).padStart(4, '0');
  return cp === 0x27 ? `"'"` : `'${body}'`;
}

/** superbpe.payload_ids: the start id, then per pause marker the stretch before it and its id, then pronunciation
 *  groups (open id, one id per phone, close id) between SuperBPE stretches (`bpe`). Throws the Python message. */
function payload(text: string, m: TextMarkers, bpe: (s: string, out: number[]) => void, out: number[]): void {
  if (text.includes(START_MARKER)) {
    if (!m.start) throw new Error(NO_START_TOKEN);
    if (!text.startsWith(START_MARKER) || text.includes(START_MARKER, 1)) throw new Error(START_NOT_FIRST);
    out.push(m.start.id);
    payload(text.slice(START_MARKER.length), m, bpe, out);
    return;
  }
  if (PAUSE_RE.test(text)) {
    if (!m.pauses) throw new Error(NO_PAUSE_TOKENS);
    let pos = 0;
    for (let i = 0; i < text.length; i++) {
      const b = text.charCodeAt(i) - PAUSE_BASE;
      if (b < 0 || b > 7) continue;
      if (b >= m.pauses.ids.length)
        throw new Error(`Pause marker U+${(PAUSE_BASE + b).toString(16).toUpperCase()} outside the ${m.pauses.ids.length} pause buckets`);
      if (i > pos) payload(text.slice(pos, i), m, bpe, out);
      out.push(m.pauses.ids[b]);
      pos = i + 1;
    }
    if (pos < text.length) payload(text.slice(pos), m, bpe, out);
    return;
  }
  const ph = m.phonemes;
  if (!ph || (!text.includes(ph.open) && !text.includes(ph.close))) {
    bpe(text, out);
    return;
  }
  const phoneId = new Map(ph.symbols.map((s, i) => [s, ph.openId + 2 + i]));
  let pos = 0;
  while (pos < text.length) {
    const start = text.indexOf(ph.open, pos);
    const stretch = start < 0 ? text.slice(pos) : text.slice(pos, start);
    if (stretch.includes(ph.close)) throw new Error('Unbalanced phoneme group markers');
    if (stretch) bpe(stretch, out);
    if (start < 0) break;
    const g0 = start + ph.open.length;
    const end = text.indexOf(ph.close, g0);
    const group = end >= 0 ? text.slice(g0, end) : '';
    if (end < 0 || !group || group.includes(ph.open)) throw new Error('Unbalanced or empty phoneme group');
    const phones = Array.from(group);
    const unknown = [...new Set(phones.filter(s => !phoneId.has(s)))].sort((a, b) => a.codePointAt(0)! - b.codePointAt(0)!);
    if (unknown.length) throw new Error(`Phones outside the inventory: [${unknown.map(pyCharRepr).join(', ')}]`);
    out.push(ph.openId, ...phones.map(s => phoneId.get(s)!), ph.closeId);
    pos = end + ph.close.length;
  }
}

/** zonos3.tts.plan.validate_text_markers (R13): the 400 message for `text` on a checkpoint with markers `m`, or ''.
 *  Groups need phoneme tokens and must parse; the start marker needs a start token and must open the text, once;
 *  pause markers need pause tokens and a bucket the checkpoint has. `edit`: a speech-edit target text, which takes no
 *  start or pause markers. */
export function checkTextMarkers(m: TextMarkers, text: string, edit = false): string {
  const hasStart = text.includes(START_MARKER);
  const pause = PAUSE_RE.test(text);
  if (edit && (hasStart || pause)) return EDIT_NO_MARKERS;
  const open = m.phonemes?.open ?? PHONEME_OPEN, close = m.phonemes?.close ?? PHONEME_CLOSE;
  const hasGroups = text.includes(open) || text.includes(close);
  if (hasGroups && !m.phonemes) return NO_PHONEME_TOKENS;
  if (hasStart && !m.start) return NO_START_TOKEN;
  if (pause && !m.pauses) return NO_PAUSE_TOKENS;
  if (hasStart && (!text.startsWith(START_MARKER) || text.includes(START_MARKER, 1))) return START_NOT_FIRST;
  if (pause) {
    const n = m.pauses!.ids.length;
    for (let i = 0; i < text.length; i++) {
      const b = text.charCodeAt(i) - PAUSE_BASE;
      if (b >= n && b <= 7) return `Pause marker U+${(PAUSE_BASE + b).toString(16).toUpperCase()} outside the ${n} pause buckets`;
    }
  }
  if (hasGroups) {
    try {
      payload(text, m, () => {}, []);
    } catch (e) {
      return `Pronunciation groups: ${e instanceof Error ? e.message : String(e)}`;
    }
  }
  return '';
}

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

  /** Model text ids of `text`: BOS, the payload (superbpe.payload_ids with the checkpoint's `markers`), EOS. Throws the
   *  Python message on a marker the checkpoint cannot encode (callers check the text first: checkTextMarkers). */
  encode(text: string, markers: TextMarkers = NO_TEXT_MARKERS): Int32Array {
    const out: number[] = [this.bos];
    payload(text, markers, (s, o) => this.bpeText(s, o), out);
    out.push(this.eos);
    return Int32Array.from(out);
  }

  /** Code-point [start, end) of each SuperBPE token of a marker-free `piece` (the HF offsets
   *  zonos3.tts.span_follow.token_char_ranges reads): a token's characters run from the character of its first byte to
   *  the one of its last. */
  charOffsets(piece: string): [number, number][] {
    const out: [number, number][] = [];
    const bytes = utf8.encode(piece);
    const nChars = Array.from(piece).length;
    const charOf = new Array<number>(bytes.length + 1).fill(nChars);
    for (let b = 0, c = 0; b < bytes.length; b++) {
      if (b > 0 && (bytes[b] & 0xc0) !== 0x80) c++;
      charOf[b] = c;
    }
    const ids: number[] = [];
    this.bpeText(piece, ids);
    let at = 0;
    for (const id of ids) {
      const n = this.tokenBytes(id);
      if (n === 0) continue;
      out.push([charOf[at], charOf[at + n - 1] + 1]);
      at += n;
    }
    return out;
  }

  /** Plain SuperBPE of `text` (superbpe._bpe): added tokens split out first, then pre-tokenized byte-level BPE. */
  private bpeText(text: string, out: number[]): void {
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
