/**
 * Region planning for speech editing (pure). Ports the edit branch's `ui/js/regions.js`, itself a port of
 * `zonos3.tts.edit_align.diff_spans`: word diff of the aligned transcript against the target text, regions in
 * codec frames with a margin, merged when closer than MIN_GAP_FRAMES. `diffOpcodes` follows Python difflib's
 * SequenceMatcher (no junk) so repeated-token cases match the server.
 *
 * Pronunciation groups (phoneme-token checkpoints) are one target word each: the token `U+E000 IPA U+E001`, which
 * never equals a spelled word. Target text in the editor writes them "(X-SAMPA)" (`groups` arguments); the sent
 * form (`\ue000…\ue001`) is always one token.
 */

import { parsePron, xsampaToIpa } from '../lib/pron';
import { parseTags } from '../lib/tags';

/** An aligned word (`/v2/edit/align`, `/v2/edit/transcribe`): seconds, and its edit tokens. */
export interface Word {
  word: string;
  start: number;
  end: number;
  /** FuseAlign's presence probability; optional in the contract (null for caller-supplied timings). */
  confidence?: number | null;
  tokens: string[];
}

/** One non-equal diff opcode: original tokens [i1, i2) became target tokens [j1, j2). */
export interface Change {
  i1: number;
  i2: number;
  j1: number;
  j2: number;
}

/** A region to regenerate, in codec frames [start, end). */
export interface PlannedRegion {
  start: number;
  end: number;
  changes: Change[];
}

export const MIN_GAP_FRAMES = 5;

// The server's edit tokens (zonos3.tts.textmatch.tokens): curly quotes fold to ', '_' separates, everything but letters, digits and '
// is a separator, and each Han / kana run is cut into its words.
const QUOTES = /[\u2018\u2019\u201b\u02bc]/g;
const NON_TOKEN = /[^\p{L}\p{N}']+/gu;

// zonos3.tts.align_text: Han / kana runs are cut by ICU's word break (root locale; the server runs ICU too, with the
// same dictionaries as Intl.Segmenter). FuseAlign aligns unspaced text by these words, and the edit diffs them.
const CJK_RUN = /[\u3005\u3007\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\u31f0-\u31ff\uff66-\uff9f]+/gu;
const WORDS = new Intl.Segmenter('und', { granularity: 'word' });
const WORDY = /[\p{L}\p{N}']/u;

/** Offsets in `text` where a word of one of its Han / kana runs starts (align_text.cjk_words; non-words like ・ skipped). */
function cjkWordStarts(text: string): number[] {
  const out: number[] = [];
  for (const m of text.matchAll(CJK_RUN)) for (const s of WORDS.segment(m[0])) if (s.isWordLike) out.push(m.index + s.index);
  return out;
}

function plainTokens(text: string): string[] {
  const words = text.replace(QUOTES, "'").replace(/_/g, ' ').toUpperCase().replace(NON_TOKEN, ' ').trim().split(/\s+/).filter(Boolean);
  return words.flatMap((w) => {
    // align_text.split_cjk: cut where each run word starts and where a run ends.
    const cuts = new Set([0, w.length, ...cjkWordStarts(w)]);
    for (const m of w.matchAll(CJK_RUN)) cuts.add(m.index + m[0].length);
    const at = [...cuts].sort((a, b) => a - b);
    return at.slice(1).map((end, k) => w.slice(at[k], end));
  });
}

/**
 * A whitespace word as the server's alignment units (align_text.text_units): cut where a word of its Han / kana runs
 * starts and where letters resume after a run; punctuation stays with the word before it (leading punctuation with
 * the first). A word without Han / kana is one chunk.
 */
function wordChunks(word: string, from: number): Chunk[] {
  const cuts = new Set(cjkWordStarts(word));
  for (const m of word.matchAll(CJK_RUN)) {
    const end = m.index + m[0].length;
    const resume = word.slice(end).search(WORDY);
    if (resume >= 0) cuts.add(end + resume);
  }
  const at = [...new Set([0, ...cuts, word.length])].sort((a, b) => a - b);
  const out: Chunk[] = [];
  for (let k = 0; k + 1 < at.length; k++) {
    const piece: Chunk = { from: from + at[k], to: from + at[k + 1], tokens: plainTokens(word.slice(at[k], at[k + 1])) };
    const prev = out[out.length - 1];
    if (prev && !prev.tokens.length) Object.assign(prev, { to: piece.to, tokens: piece.tokens });
    else if (prev && !piece.tokens.length) prev.to = piece.to;
    else out.push(piece);
  }
  return out;
}

const OPEN = '\ue000';
const CLOSE = '\ue001';
const SENT_GROUP = /\ue000[^\ue000\ue001]*\ue001/g;

/** A pronunciation-group token (one target word, matched by any one recognized word). */
export const isGroupToken = (t: string) => t.startsWith(OPEN);

const TAG = '\ue002';
/**
 * A nonverbal-tag token: part of the diff (a tag inserted between unchanged words plans a region there, so it is
 * rendered), never heard by speech recognition (the server keeps markers out of Qwen3-ASR and FuseAlign).
 */
export const isTagToken = (t: string) => t.startsWith(TAG);

interface Chunk {
  /** [from, to): a whitespace-separated word of the text, or one word of a Han / kana run (a group may hold spaces). */
  from: number;
  to: number;
  tokens: string[];
}

/** The text's words and their tokens; groups and nonverbal tags are atomic, one token each. */
function chunks(text: string, groups: boolean): Chunk[] {
  const atoms = new Map<number, { end: number; token: string }>();
  for (const t of parseTags(text).tokens) atoms.set(t.start, { end: t.end, token: TAG + text.slice(t.start, t.end) });
  if (groups) for (const g of parsePron(text, null).groups) if (!g.empty) atoms.set(g.start, { end: g.end, token: OPEN + g.ipa + CLOSE });
  if (text.includes(OPEN)) for (const m of text.matchAll(SENT_GROUP)) atoms.set(m.index ?? 0, { end: (m.index ?? 0) + m[0].length, token: m[0] });
  if (!atoms.size) return [...text.matchAll(/\S+/g)].flatMap((m) => wordChunks(m[0], m.index ?? 0));
  const out: Chunk[] = [];
  let i = 0;
  while (i < text.length) {
    if (/\s/.test(text[i]) && !atoms.has(i)) {
      i++;
      continue;
    }
    const from = i;
    const tokens: string[] = [];
    let plainFrom = i;
    while (i < text.length && (atoms.has(i) || !/\s/.test(text[i]))) {
      const a = atoms.get(i);
      if (!a) {
        i++;
        continue;
      }
      tokens.push(...plainTokens(text.slice(plainFrom, i)), a.token);
      i = plainFrom = a.end;
    }
    tokens.push(...plainTokens(text.slice(plainFrom, i)));
    out.push({ from, to: i, tokens });
  }
  return out;
}

/** Edit tokens; with `groups`, each "(X-SAMPA)" group is one token (sent-form groups always are). */
export function normalizeTokens(text: string, groups = false): string[] {
  return chunks(text, groups).flatMap((c) => c.tokens);
}

/** The text's words: whitespace words, Han / kana runs cut into words (the units the server aligns). */
export function textWords(text: string): string[] {
  return chunks(text, false).map((c) => text.slice(c.from, c.to));
}

/** difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes(), non-equal ones, plus a → b matches. */
export function diffOpcodes(a: string[], b: string[]): { changes: Change[]; equalMap: Int32Array } {
  const b2j = new Map<string, number[]>();
  b.forEach((t, j) => {
    const list = b2j.get(t);
    if (list) list.push(j);
    else b2j.set(t, [j]);
  });
  const longest = (alo: number, ahi: number, blo: number, bhi: number): [number, number, number] => {
    let besti = alo;
    let bestj = blo;
    let bestsize = 0;
    let j2len = new Map<number, number>();
    for (let i = alo; i < ahi; i++) {
      const next = new Map<number, number>();
      for (const j of b2j.get(a[i]) ?? []) {
        if (j < blo) continue;
        if (j >= bhi) break;
        const k = (j2len.get(j - 1) ?? 0) + 1;
        next.set(j, k);
        if (k > bestsize) {
          besti = i - k + 1;
          bestj = j - k + 1;
          bestsize = k;
        }
      }
      j2len = next;
    }
    return [besti, bestj, bestsize];
  };
  const found: [number, number, number][] = [];
  const queue: [number, number, number, number][] = [[0, a.length, 0, b.length]];
  while (queue.length) {
    const [alo, ahi, blo, bhi] = queue.pop()!;
    const [i, j, k] = longest(alo, ahi, blo, bhi);
    if (!k) continue;
    found.push([i, j, k]);
    if (alo < i && blo < j) queue.push([alo, i, blo, j]);
    if (i + k < ahi && j + k < bhi) queue.push([i + k, ahi, j + k, bhi]);
  }
  found.sort((x, y) => x[0] - y[0] || x[1] - y[1] || x[2] - y[2]);
  const blocks: [number, number, number][] = [];
  let [i1, j1, k1] = [0, 0, 0];
  for (const [i2, j2, k2] of found) {
    if (i1 + k1 === i2 && j1 + k1 === j2) k1 += k2;
    else {
      if (k1) blocks.push([i1, j1, k1]);
      [i1, j1, k1] = [i2, j2, k2];
    }
  }
  if (k1) blocks.push([i1, j1, k1]);
  blocks.push([a.length, b.length, 0]);
  const equalMap = new Int32Array(a.length).fill(-1);
  const changes: Change[] = [];
  let i = 0;
  let j = 0;
  for (const [ai, bj, size] of blocks) {
    if (i < ai || j < bj) changes.push({ i1: i, i2: ai, j1: j, j2: bj });
    i = ai + size;
    j = bj + size;
    for (let k = 0; k < size; k++) equalMap[ai + k] = bj + k;
  }
  return { changes, equalMap };
}

/**
 * For each hypothesis token: true when no reference token matches it. A pronunciation group in the reference
 * matches whatever single word was recognized at its place in a changed stretch.
 */
export function wordDiff(refTokens: string[], hypTokens: string[]): boolean[] {
  const { changes, equalMap } = diffOpcodes(refTokens, hypTokens);
  const wrong = new Array<boolean>(hypTokens.length).fill(true);
  for (const j of equalMap) if (j >= 0) wrong[j] = false;
  for (const c of changes) for (let k = 0; c.i1 + k < c.i2 && c.j1 + k < c.j2; k++) if (isGroupToken(refTokens[c.i1 + k])) wrong[c.j1 + k] = false;
  return wrong;
}

/** [from, to) character range of the raw word that owns each normalized token of `text`. */
export function tokenOwners(text: string, groups = false): [number, number][] {
  return chunks(text, groups).flatMap((c) => c.tokens.map((): [number, number] => [c.from, c.to]));
}

/**
 * Frames [start, end) for the stretch t0…t1 (seconds) plus `margin` on each side, where the margin never reaches
 * into a neighbouring word: it stops halfway into the gap to it (contiguous words → the word boundary), and the
 * frame rounding of each edge never crosses that boundary either.
 */
export function marginFrames(t0: number, t1: number, prevEnd: number | null, nextStart: number | null, margin: number, fps: number, totalFrames: number): [number, number] {
  const lo = prevEnd === null ? 0 : (Math.min(prevEnd, t0) + t0) / 2;
  const hi = nextStart === null ? totalFrames / fps : (t1 + Math.max(nextStart, t1)) / 2;
  let start = Math.floor(Math.max(t0 - margin, lo) * fps + 1e-6);
  if (prevEnd !== null && start / fps < prevEnd - 1e-6) start = Math.min(Math.ceil(prevEnd * fps - 1e-6), Math.round(t0 * fps));
  let end = Math.ceil(Math.min(t1 + margin, hi) * fps - 1e-6);
  if (nextStart !== null && end / fps > nextStart + 1e-6) end = Math.max(Math.floor(nextStart * fps + 1e-6), Math.round(t1 * fps));
  start = Math.min(totalFrames - 1, Math.max(0, start));
  return [start, Math.max(start + 1, Math.min(totalFrames, end))];
}

export interface Plan {
  regions: PlannedRegion[];
  targetTokens: string[];
  owners: [number, number][];
  /** Character position in the target text of each pure deletion, in change order. */
  deletionPoints: number[];
  equalMap: Int32Array;
  totalFrames: number;
}

export function planRegions(words: Word[], targetText: string, o: { duration: number; frameRate: number; marginSeconds: number; minGapFrames?: number; groups?: boolean }): Plan {
  const minGap = o.minGapFrames ?? MIN_GAP_FRAMES;
  const expanded = words.flatMap((w) => w.tokens.map((t) => ({ t, w })));
  const original = expanded.map((e) => e.t);
  const targetTokens = normalizeTokens(targetText, o.groups);
  const totalFrames = Math.ceil(o.duration * o.frameRate - 1e-6);
  const { changes, equalMap } = diffOpcodes(original, targetTokens);
  const owners = tokenOwners(targetText, o.groups);
  const deletionPoints: number[] = [];
  const regions: PlannedRegion[] = [];
  for (const change of changes) {
    const { i1, i2, j1, j2 } = change;
    let t0: number;
    let t1: number;
    let before: Word | undefined;
    let after: Word | undefined;
    if (i1 !== i2) {
      t0 = expanded[i1].w.start;
      t1 = expanded[i2 - 1].w.end;
      // Neighbours: the nearest other words (a partly changed word is regenerated whole).
      for (let i = i1 - 1; i >= 0 && !before; i--) if (expanded[i].w !== expanded[i1].w) before = expanded[i].w;
      for (let i = i2; i < expanded.length && !after; i++) if (expanded[i].w !== expanded[i2 - 1].w) after = expanded[i].w;
    } else {
      before = i1 > 0 ? expanded[i1 - 1].w : undefined;
      after = i1 < expanded.length ? expanded[i1].w : undefined;
      t0 = before ? before.end : 0;
      t1 = after ? after.start : totalFrames / o.frameRate;
    }
    if (j1 === j2) deletionPoints.push(j1 < owners.length ? owners[j1][0] : targetText.length);
    const [start, end] = marginFrames(t0, t1, before ? before.end : null, after ? after.start : null, o.marginSeconds, o.frameRate, totalFrames);
    const prev = regions[regions.length - 1];
    if (prev && start - prev.end < minGap) {
      prev.end = Math.max(end, prev.end);
      prev.changes.push(change);
    } else regions.push({ start, end, changes: [change] });
  }
  return { regions, targetTokens, owners, deletionPoints, equalMap, totalFrames };
}

/** Map a time in audio with regions `from` to the audio where they became `to` (same length, sorted). */
export function mapTime(t: number, from: [number, number][], to: [number, number][]): number {
  let shift = 0;
  for (let k = 0; k < from.length && k < to.length; k++) {
    const [s, e] = from[k];
    const [es, ee] = to[k];
    if (t < s) return t + shift;
    if (t < e) return es + ((t - s) / (e - s || 1)) * (ee - es);
    shift = ee - e;
  }
  return t + shift;
}

const EDGE_PUNCT = /^[.,;:!?"“”()…]+|[.,;:!?"“”()…]+$/g;

function joinDistinct<T>(items: T[], key: (x: T) => unknown, text: (x: T) => string): string {
  const out: string[] = [];
  let last: unknown = undefined;
  for (const item of items) {
    const k = key(item);
    if (last !== undefined && k === last) continue;
    last = k;
    const t = text(item).replace(EDGE_PUNCT, '');
    if (t) out.push(t);
  }
  return out.join(' ');
}

/** "old → new · + added · − removed" for a set of changes. */
export function summarizeChanges(changes: Change[], words: Word[], targetText: string, groups = false, max = 64): string {
  const tokenWord: number[] = [];
  words.forEach((w, wi) => w.tokens.forEach(() => tokenWord.push(wi)));
  const owners = tokenOwners(targetText, groups);
  const parts: string[] = [];
  for (const c of changes) {
    const before = joinDistinct(tokenWord.slice(c.i1, c.i2), (wi) => wi, (wi) => words[wi].word);
    // A pronunciation group reads as its IPA: "Siobhan → /ʃəvɔːn/".
    const word = (o: [number, number]) => targetText.slice(o[0], o[1]);
    const after = joinDistinct(owners.slice(c.j1, c.j2), (o) => o[0], groups ? (o) => word(o).replace(/\(([^()]*)\)/g, (_, b: string) => `/${xsampaToIpa(b)}/`) : word);
    if (c.i1 !== c.i2 && c.j1 !== c.j2) parts.push(`${before} → ${after}`);
    else if (c.j1 !== c.j2) parts.push(`+ ${after}`);
    else if (c.i1 !== c.i2) parts.push(`− ${before}`);
  }
  const s = parts.join(' · ');
  return s.length > max ? s.slice(0, max - 1) + '…' : s;
}

/** Levenshtein similarity 1 − d / max(len): "MONAPOLIS" ~ "MINNEAPOLIS" (0.73). */
export function similarity(a: string, b: string): number {
  if (a === b) return 1;
  const n = a.length;
  const m = b.length;
  if (!n || !m) return 0;
  let prev = Array.from({ length: m + 1 }, (_, j) => j);
  for (let i = 1; i <= n; i++) {
    const cur = [i];
    for (let j = 1; j <= m; j++) cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    prev = cur;
  }
  return 1 - prev[m] / Math.max(n, m);
}

export const HEARD_SIMILARITY = 0.6;

/**
 * Target tokens (indices into `target`) that the transcript `asr` does not contain, order-aware: exact matches
 * from the difflib alignment anchor the search, and each remaining token may be matched by a similar transcript
 * token between the anchors around it (small ASR spelling slips count as heard). Tokens of ≤ 2 letters are
 * only checked when they are all the region holds (ASR drops "a", "to" freely). A pronunciation group is heard
 * when any recognized word is left between those anchors.
 */
export function unheard(target: string[], asr: string, want: number[]): number[] {
  want = want.filter((j) => !isTagToken(target[j]));
  if (!want.length) return [];
  const hyp = normalizeTokens(asr);
  const { equalMap } = diffOpcodes(target, hyp);
  const used = new Set<number>();
  for (const h of equalMap) if (h >= 0) used.add(h);
  const strict = want.every((j) => target[j].length <= 2);
  const missing: number[] = [];
  for (const j of want) {
    if (equalMap[j] >= 0) continue;
    if (!strict && target[j].length <= 2) continue;
    let lo = 0;
    for (let k = j - 1; k >= 0; k--) if (equalMap[k] >= 0) {
      lo = equalMap[k] + 1;
      break;
    }
    let hi = hyp.length;
    for (let k = j + 1; k < target.length; k++) if (equalMap[k] >= 0) {
      hi = equalMap[k];
      break;
    }
    let best = -1;
    let bestSim = HEARD_SIMILARITY;
    const wild = isGroupToken(target[j]);
    for (let h = lo; h < hi; h++) {
      if (used.has(h)) continue;
      if (wild) {
        best = h;
        break;
      }
      const s = similarity(target[j], hyp[h]);
      if (s >= bestSim) {
        bestSim = s;
        best = h;
      }
    }
    if (best >= 0) used.add(best);
    else missing.push(j);
  }
  return missing;
}

/**
 * The target tokens each region has to say: the target side of its changes, plus the unchanged words inside
 * it (a re-said stretch, the margins). `regions` are seconds in the original.
 */
export function regionTargets(words: Word[], targetText: string, regions: { start: number; end: number; changes: Change[] }[], groups = false): number[][] {
  const tokens = words.flatMap((w) => w.tokens.map(() => w));
  const { equalMap } = diffOpcodes(words.flatMap((w) => w.tokens), normalizeTokens(targetText, groups));
  return regions.map((r) => {
    const js = new Set<number>();
    for (const c of r.changes) for (let j = c.j1; j < c.j2; j++) js.add(j);
    tokens.forEach((w, i) => {
      const mid = (w.start + w.end) / 2;
      if (mid >= r.start && mid < r.end && equalMap[i] >= 0) js.add(equalMap[i]);
    });
    return [...js].sort((a, b) => a - b);
  });
}

/**
 * The server's context windows for a long clip: each region ± `ctx` seconds, each edge cut halfway between
 * words (an edge inside a word moves out past it), overlapping windows merged. Seconds, ascending.
 */
export function contextWindows(words: Word[], regions: [number, number][], ctx: number, duration: number): [number, number][] {
  const cut = (t: number, dir: -1 | 1): number => {
    if (t <= 0) return 0;
    if (t >= duration) return duration;
    const k = words.findIndex((w) => w.start < t && t < w.end);
    if (k >= 0) {
      const nb = words[k + dir];
      const edge = dir < 0 ? words[k].start : words[k].end;
      return nb ? (edge + (dir < 0 ? nb.end : nb.start)) / 2 : dir < 0 ? 0 : duration;
    }
    const before = words.filter((w) => w.end <= t).pop();
    const after = words.find((w) => w.start >= t);
    return before && after ? (before.end + after.start) / 2 : t;
  };
  const out: [number, number][] = [];
  for (const [s, e] of regions) {
    const a = cut(s - ctx, -1);
    const b = cut(e + ctx, 1);
    const last = out[out.length - 1];
    if (last && a <= last[1]) last[1] = Math.max(last[1], b);
    else out.push([a, b]);
  }
  return out;
}
