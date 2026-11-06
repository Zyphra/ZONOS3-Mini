// The word side of a speech edit (zonos3.tts.edit_align): aligned words with their edit tokens, caller-supplied word
// timings, the transcript diff's regions (difflib), the word-snapped context windows of a long clip and the expected
// fill lengths. Errors are Errors with the Python ValueError messages (HTTP 400). Times are seconds, spans
// [start_frame, length_frames].
import { pyRepr } from '../api/http';
import { pyFixed, pyRound, pySum, pySumFloats } from './py';
import { tokenChars } from './text';

/** An edit word: its interval, its confidence (FuseAlign presence) and its edit tokens (textmatch.tokens). */
export interface EditWord { word: string; start: number; end: number; confidence: number | null; tokens: string[] }

/** A FuseAlign word (one per align_tokens token). */
export interface FuseAlignWord { word: string; start: number; end: number; presence: number }

export type Span = [number, number];

/** aligned_words: FuseAlign's words with their edit tokens and presence as the confidence; tokenless words dropped and
 *  an overlap with the previous word split at its midpoint. */
export function alignedWords(words: FuseAlignWord[], wordTokens: string[][]): EditWord[] {
  if (words.length !== wordTokens.length) throw new Error('One token list is needed per aligned word.');
  const out: EditWord[] = [];
  words.forEach((w, k) => {
    if (wordTokens[k].length)
      out.push({ word: w.word, start: w.start, end: w.end, confidence: w.presence, tokens: [...wordTokens[k]] });
  });
  for (let k = 1; k < out.length; k++) {
    const previous = out[k - 1], word = out[k];
    if (word.start < previous.end) previous.end = word.start = (previous.end + word.start) / 2;
  }
  return out;
}

export interface WordIn { word: string; start: number; end: number; confidence?: number | null }

/** edit_words: caller-supplied word timings of the original, checked (finite, start < end, ascending and
 *  non-overlapping inside [0, duration], their tokens spelling `originalTokens`). */
export function editWords(words: WordIn[], wordTokens: string[][], originalTokens: string[], duration: number): EditWord[] {
  if (!words.length) throw new Error('original_words is empty.');
  if (words[0].start < 0) throw new Error(`original_words[0] (${pyRepr(words[0].word)}) starts before the clip.`);
  let previousEnd = 0;
  words.forEach((w, k) => {
    const { start, end } = w;
    const label = `original_words[${k}] (${pyRepr(w.word)})`;
    if (!Number.isFinite(start) || !Number.isFinite(end)) throw new Error(`${label} has a non-finite time.`);
    if (!(start < end)) throw new Error(`${label} must start before it ends (${pyFixed(start, 3)} >= ${pyFixed(end, 3)}).`);
    if (start < previousEnd - 1e-6)
      throw new Error(`${label} starts at ${pyFixed(start, 3)} s, before the previous word ends (${pyFixed(previousEnd, 3)} s).`);
    if (end > duration + 1e-3)
      throw new Error(`${label} ends at ${pyFixed(end, 3)} s, after the clip (${pyFixed(duration, 3)} s).`);
    previousEnd = end;
  });
  const spelled = wordTokens.flat();
  const same = spelled.length === originalTokens.length && spelled.every((t, i) => t === originalTokens[i]);
  if (!same) {
    const n = Math.min(spelled.length, originalTokens.length);
    let k = 0;
    while (k < n && spelled[k] === originalTokens[k]) k++;
    const around = (ts: string[]) => ts.slice(Math.max(0, k - 2), k + 3).join(' ') || '(nothing)';
    throw new Error(`original_words do not spell original_text: they read ${pyRepr(around(spelled))} where the text ` +
      `reads ${pyRepr(around(originalTokens))} (normalized).`);
  }
  return words.slice(0, wordTokens.length).map((w, k) => ({
    word: w.word, start: w.start, end: w.end, confidence: w.confidence ?? null, tokens: [...wordTokens[k]],
  }));
}

/** seconds_to_spans: [start, end] seconds -> (start_frame, length_frames), outward-rounded and clipped to the audio. */
export function secondsToSpans(pairs: number[][], totalFrames: number, frameRate = 15): Span[] {
  return pairs.map(([start, end]) => {
    const first = Math.max(0, Math.floor(start * frameRate + 1e-6));
    return [first, Math.min(totalFrames, Math.ceil(end * frameRate - 1e-6)) - first];
  });
}

export type Opcode = ['equal' | 'replace' | 'delete' | 'insert', number, number, number, number];

/** difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes(). */
export function opcodes(a: string[], b: string[]): Opcode[] {
  const b2j = new Map<string, number[]>();
  b.forEach((t, j) => {
    const list = b2j.get(t);
    if (list) list.push(j);
    else b2j.set(t, [j]);
  });
  const longest = (alo: number, ahi: number, blo: number, bhi: number): [number, number, number] => {
    let besti = alo, bestj = blo, bestsize = 0;
    let j2len = new Map<number, number>();
    for (let i = alo; i < ahi; i++) {
      const next = new Map<number, number>();
      for (const j of b2j.get(a[i]) ?? []) {
        if (j < blo) continue;
        if (j >= bhi) break;
        const k = (j2len.get(j - 1) ?? 0) + 1;
        next.set(j, k);
        if (k > bestsize) [besti, bestj, bestsize] = [i - k + 1, j - k + 1, k];
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
    if (i1 + k1 === i2 && j1 + k1 === j2) {
      k1 += k2;
    } else {
      if (k1) blocks.push([i1, j1, k1]);
      [i1, j1, k1] = [i2, j2, k2];
    }
  }
  if (k1) blocks.push([i1, j1, k1]);
  blocks.push([a.length, b.length, 0]);
  const out: Opcode[] = [];
  let i = 0, j = 0;
  for (const [ai, bj, size] of blocks) {
    const tag = i < ai && j < bj ? 'replace' : i < ai ? 'delete' : j < bj ? 'insert' : null;
    if (tag) out.push([tag, i, ai, j, bj]);
    i = ai + size;
    j = bj + size;
    if (size) out.push(['equal', ai, i, bj, j]);
  }
  return out;
}

/** diff_spans: the regions (frames) of the original words' tokens the target's tokens change, each widened by
 *  `marginSeconds`; regions closer than `minGapFrames` merge. */
export function diffSpans(words: EditWord[], targetTokens: string[], o: {
  totalFrames: number; frameRate?: number; marginSeconds?: number; minGapFrames?: number; maxSpans?: number;
}): { spans: Span[]; newTargetTokens: number[] } {
  const { totalFrames, frameRate = 15, marginSeconds = 0.08, minGapFrames = 5, maxSpans = 3 } = o;
  const expanded = words.flatMap(w => w.tokens.map(token => ({ token, w })));
  const spans: Span[] = [], newTokens: number[] = [];
  for (const [op, i1, i2, j1, j2] of opcodes(expanded.map(e => e.token), targetTokens)) {
    if (op === 'equal') continue;
    let t0: number, t1: number;
    if (i1 !== i2) {
      t0 = expanded[i1].w.start;
      t1 = expanded[i2 - 1].w.end;
    } else {
      t0 = i1 ? expanded[i1 - 1].w.end : 0;
      t1 = i1 < expanded.length ? expanded[i1].w.start : totalFrames / frameRate;
    }
    const start = Math.min(totalFrames - 1, Math.max(0, Math.floor((t0 - marginSeconds) * frameRate)));
    const end = Math.max(start + 1, Math.min(totalFrames, Math.ceil((t1 + marginSeconds) * frameRate)));
    const last = spans.at(-1);
    if (last && start - (last[0] + last[1]) < minGapFrames) last[1] = Math.max(end, last[0] + last[1]) - last[0];
    else spans.push([start, end - start]);
    for (let j = j1; j < j2; j++) newTokens.push(j);
  }
  if (!spans.length) throw new Error('target text equals the original transcript');
  if (spans.length > maxSpans) throw new Error(`edit touches ${spans.length} regions; at most ${maxSpans} are supported`);
  return { spans, newTargetTokens: newTokens };
}

/** WordDiff: the original's words (edit tokens) against the target's words and their edit tokens, diffed over the
 *  tokens: maps original word ranges onto target word ranges. */
export class WordDiff {
  readonly origStart: number[] = [0];
  readonly tgtStart: number[] = [0];
  readonly ops: Opcode[];

  constructor(readonly words: EditWord[], targetWords: string[], targetWordTokens: string[][]) {
    if (targetWordTokens.length !== targetWords.length) throw new Error('One token list is needed per target word.');
    for (const w of words) this.origStart.push(this.origStart.at(-1)! + w.tokens.length);
    for (const ts of targetWordTokens) this.tgtStart.push(this.tgtStart.at(-1)! + ts.length);
    this.ops = opcodes(words.flatMap(w => w.tokens), targetWordTokens.flat());
  }

  /** Target token position of a cut before original token i (an insertion at the cut joins the range), or null
   *  inside a changed block. */
  private targetCut(i: number, left: boolean): number | null {
    const ops = left ? this.ops : [...this.ops].reverse();
    for (const [tag, i1, i2, j1, j2] of ops) {
      if (i1 < i && i < i2) return tag === 'equal' ? j1 + i - i1 : null;
      if ((left ? i1 : i2) === i) return left ? j1 : j2;
    }
    return i === 0 ? 0 : this.tgtStart.at(-1)!;
  }

  /** Target word index of a cut before original word k (as the left or right edge of a range), or null when the cut
   *  falls inside a changed block or a target word. */
  targetWord(k: number, left: boolean): number | null {
    const j = this.targetCut(this.origStart[k], left);
    if (j === null) return null;
    const at: number[] = [];
    this.tgtStart.forEach((start, t) => { if (start === j) at.push(t); });
    if (!at.length) return null;
    return left ? at[0] : at.at(-1)!;
  }

  /** Original word range [a, b) widened until both edges map onto target word boundaries. */
  clean(a: number, b: number): [number, number] {
    while (a > 0 && this.targetWord(a, true) === null) a--;
    while (b < this.words.length && this.targetWord(b, false) === null) b++;
    return [a, b];
  }

  /** Target words aligned with the clean original word range [a, b) (null: an open slice end, as Python's). */
  targetRange(a: number, b: number): [number | null, number | null] {
    return [this.targetWord(a, true), this.targetWord(b, false)];
  }
}

/** Python's list[ja:jb] for non-negative or None bounds. */
function slice<T>(items: T[], ja: number | null, jb: number | null): T[] {
  return items.slice(ja ?? 0, jb ?? items.length);
}

/** The audio frames [start, end) of the clip the model sees for some regions (indices into the spans) and the target
 *  text of that stretch. */
export interface EditWindow { start: number; end: number; regions: number[]; text: string }

export function wholeClipWindow(spans: Span[], totalFrames: number, targetText: string): EditWindow {
  return { start: 0, end: totalFrames, regions: spans.map((_, k) => k), text: targetText };
}

/** EditWindow.spans: a window's regions as window-relative spans. */
export function windowSpans(window: EditWindow, spans: Span[]): Span[] {
  return window.regions.map(k => [spans[k][0] - window.start, spans[k][1]]);
}

/** plan_edit_windows: word-snapped context windows (words within `contextSeconds` of each region, edges moved out of
 *  changed diff blocks, overlapping windows merged), cut halfway between their edge words and the neighbours. A window
 *  over the whole clip takes `targetText` itself. */
export function planEditWindows(spans: Span[], words: EditWord[], targetWords: string[], targetWordTokens: string[][],
                                o: { totalFrames: number; frameRate?: number; contextSeconds?: number; targetText?: string | null }): EditWindow[] {
  const { totalFrames, frameRate = 15, contextSeconds = 4, targetText = null } = o;
  const n = words.length;
  if (!n) throw new Error('The aligner found no words in the audio, so the edit cannot be windowed.');
  const diff = new WordDiff(words, targetWords, targetWordTokens);
  const boundary = (k: number) => k === 0 ? 0 : k === n ? totalFrames
    : Math.min(totalFrames, pyRound((words[k - 1].end + words[k].start) / 2 * frameRate));
  type Win = [number, number, number[]];
  const windows: Win[] = spans.map(([start, length], index) => {
    const t0 = start / frameRate - contextSeconds, t1 = (start + length) / frameRate + contextSeconds;
    let a = words.findIndex(w => w.end > t0);
    if (a < 0) a = n;
    let b = 0;
    for (let k = n - 1; k >= 0; k--) if (words[k].start < t1) { b = k + 1; break; }
    return [...diff.clean(a, Math.max(a, b)), [index]];
  });
  const frames = ([a, b, regions]: Win): [number, number] => [
    Math.min(boundary(a), ...regions.map(k => spans[k][0])),
    Math.max(boundary(b), ...regions.map(k => spans[k][0] + spans[k][1])),
  ];
  windows.sort((x, y) => x[0] - y[0] || x[1] - y[1]);
  const merged: Win[] = [windows[0]];
  for (const window of windows.slice(1)) {
    const current = merged.at(-1)!;
    if (window[0] < current[1] || frames(window)[0] < frames(current)[1]) {
      current[0] = Math.min(current[0], window[0]);
      current[1] = Math.max(current[1], window[1]);
      current[2] = [...current[2], ...window[2]].sort((x, y) => x - y);
    } else {
      merged.push(window);
    }
  }
  return merged.map(([a, b, regions]) => {
    const [start, end] = frames([a, b, regions]);
    const text = a === 0 && b === n && targetText !== null ? targetText : slice(targetWords, ...diff.targetRange(a, b)).join(' ');
    return { start, end, regions, text };
  });
}

/** expected_fill_seconds: each region's seconds scaled by the target over the original text length (edit token
 *  characters) of the words in it; a region holding no original word takes the clip's characters per second. */
export function expectedFillSeconds(spans: Span[], words: EditWord[], targetWords: string[], targetWordTokens: string[][],
                                    frameRate = 15): number[] {
  const diff = new WordDiff(words, targetWords, targetWordTokens);
  const chars = (tokenLists: string[][]) => pySum(tokenLists.flatMap(ts => ts.map(tokenChars)));
  const spoken = pySumFloats(words.map(w => w.end - w.start));
  const rate = spoken > 0 ? chars(words.map(w => w.tokens)) / spoken : 0;
  const mids = words.map(w => (w.start + w.end) / 2);
  return spans.map(([start, length]) => {
    const r0 = start / frameRate, r1 = (start + length) / frameRate;
    const inside = mids.flatMap((mid, k) => (r0 <= mid && mid < r1 ? [k] : []));
    let a = inside.length ? inside[0] : mids.findIndex(mid => mid >= r0);
    if (a < 0) a = words.length;
    const [ca, cb] = diff.clean(a, inside.length ? inside.at(-1)! + 1 : a);
    const original = chars(words.slice(ca, cb).map(w => w.tokens));
    const target = chars(slice(targetWordTokens, ...diff.targetRange(ca, cb)));
    if (original) return (r1 - r0) * target / original;
    return rate > 0 ? target / rate : r1 - r0;
  });
}
