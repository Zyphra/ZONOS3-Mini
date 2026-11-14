// Span steering that follows the reading position (control_span_follow "reading"): port of
// servers/ggml/src/zonos3-span-follow.cpp (model/src/zonos3/tts/span_follow.py).
//
// Strings are handled as code-point arrays so offsets equal Python str indices. Types: FollowConfig {lead, lag,
// smooth, ramp}, Range = [start, end), Opcode {tag, i1, i2, j1, j2}, BpeOffsets (code-point offsets of the SuperBPE
// tokens of a string), MarkerSpec (the checkpoint's text markers: phoneme group open / close, whether it has start and
// pause tokens).
import { pyIsSpace } from '../edit/py';
import { STEER_RAMP_FRAMES } from './steer';

export interface FollowConfig { lead: number; lag: number; smooth: number; ramp: number }
export const defaultFollowConfig = (): FollowConfig => ({ lead: 1.0, lag: 0.0, smooth: 3, ramp: STEER_RAMP_FRAMES });
export type Range = [number, number];
export interface Opcode { tag: 'replace' | 'delete' | 'insert' | 'equal'; i1: number; i2: number; j1: number; j2: number }
export type BpeOffsets = (s: string) => Range[];
export interface MarkerSpec { phonemes: { open: string; close: string } | null; start: boolean; pauses: boolean }

// statistics.median
function median(v: number[]): number {
  const s = v.slice().sort((a, b) => a - b);
  const n = s.length, i = n >> 1;
  return n % 2 === 1 ? s[i] : (s[i - 1] + s[i]) / 2;
}

interface Match { i: number; j: number; k: number }

// difflib.SequenceMatcher(None, a, b, autojunk=False)
function matchingBlocks(a: number[], b: number[]): Match[] {
  const b2j = new Map<number, number[]>();
  b.forEach((c, i) => { const l = b2j.get(c); if (l) l.push(i); else b2j.set(c, [i]); });
  const longest = (alo: number, ahi: number, blo: number, bhi: number): Match => {
    let besti = alo, bestj = blo, bestsize = 0;
    let j2len = new Map<number, number>();
    for (let i = alo; i < ahi; i++) {
      const newj2len = new Map<number, number>();
      for (const j of b2j.get(a[i]) ?? []) {
        if (j < blo) continue;
        if (j >= bhi) break;
        const k = (j2len.get(j - 1) ?? 0) + 1;
        newj2len.set(j, k);
        if (k > bestsize) { besti = i - k + 1; bestj = j - k + 1; bestsize = k; }
      }
      j2len = newj2len;
    }
    while (besti > alo && bestj > blo && a[besti - 1] === b[bestj - 1]) { besti--; bestj--; bestsize++; }
    while (besti + bestsize < ahi && bestj + bestsize < bhi && a[besti + bestsize] === b[bestj + bestsize]) bestsize++;
    return { i: besti, j: bestj, k: bestsize };
  };
  const la = a.length, lb = b.length;
  const queue: [number, number, number, number][] = [[0, la, 0, lb]];
  const blocks: Match[] = [];
  while (queue.length) {
    const [alo, ahi, blo, bhi] = queue.pop()!;
    const x = longest(alo, ahi, blo, bhi);
    if (x.k) {
      blocks.push(x);
      if (alo < x.i && blo < x.j) queue.push([alo, x.i, blo, x.j]);
      if (x.i + x.k < ahi && x.j + x.k < bhi) queue.push([x.i + x.k, ahi, x.j + x.k, bhi]);
    }
  }
  blocks.sort((p, q) => p.i - q.i || p.j - q.j || p.k - q.k);
  let i1 = 0, j1 = 0, k1 = 0;
  const out: Match[] = [];
  for (const m of blocks) {
    if (i1 + k1 === m.i && j1 + k1 === m.j) k1 += m.k;
    else {
      if (k1) out.push({ i: i1, j: j1, k: k1 });
      i1 = m.i; j1 = m.j; k1 = m.k;
    }
  }
  if (k1) out.push({ i: i1, j: j1, k: k1 });
  out.push({ i: la, j: lb, k: 0 });
  return out;
}

const codePoints = (s: string): number[] => Array.from(s, (c) => c.codePointAt(0)!);

// difflib get_opcodes over code points.
export function diffOpcodes(a: string, b: string): Opcode[] {
  let i = 0, j = 0;
  const answer: Opcode[] = [];
  for (const m of matchingBlocks(codePoints(a), codePoints(b))) {
    const tag = i < m.i && j < m.j ? 'replace' : i < m.i ? 'delete' : j < m.j ? 'insert' : null;
    if (tag) answer.push({ tag, i1: i, i2: m.i, j1: j, j2: m.j });
    i = m.i + m.k;
    j = m.j + m.k;
    if (m.k) answer.push({ tag: 'equal', i1: m.i, i2: i, j1: m.j, j2: j });
  }
  return answer;
}

// Character offsets of `original` mapped into `final`: exact in unchanged stretches, proportional (rounded outward)
// in rewritten ones.
export function mapOffsets(original: string, final: string, offsets: number[], end = false): number[] {
  if (original === final) return offsets.slice();
  const olen = codePoints(original).length, flen = codePoints(final).length;
  const blocks = diffOpcodes(original, final).filter((op) => op.i2 > op.i1);
  return offsets.map((o) => {
    if (o >= olen) return flen;
    let k = 0;
    while (k < blocks.length && blocks[k].i1 <= o) k++;
    const op = blocks[k - 1];
    if (op.tag === 'equal') return op.j1 + o - op.i1;
    const x = op.j1 + ((o - op.i1) * (op.j2 - op.j1)) / (op.i2 - op.i1);
    return end ? Math.ceil(x) : Math.trunc(x);
  });
}

const utf8Len = (c: number) => (c < 0x80 ? 1 : c < 0x800 ? 2 : c < 0x10000 ? 3 : 4);

// Character range of `text` behind each id of its text span (BOS/EOS empty at 0 and len(text)): SuperBPE tokens their
// offsets, a phoneme group's open id its open marker, each phone id its code point, the close id the close marker, the
// start and pause ids their marker (superbpe.payload_ids validated the text). Without `bpe`: byte ids.
export function tokenCharRanges(text: string, bpe: BpeOffsets | null, markers: MarkerSpec | null): Range[] {
  const cps = Array.from(text);
  const n = cps.length;
  const ranges: Range[] = [[0, 0]];
  if (!bpe) {
    for (let i = 0; i < n; i++) for (let b = 0; b < utf8Len(cps[i].codePointAt(0)!); b++) ranges.push([i, i + 1]);
  } else {
    const stretch = (s: string[], base: number) => {
      if (!s.length) return;
      for (const [a, b] of bpe(s.join(''))) ranges.push([base + a, base + b]);
    };
    const find = (s: string[], needle: string, from: number): number => {
      const m = Array.from(needle);
      for (let i = from; i + m.length <= s.length; i++) if (m.every((c, k) => s[i + k] === c)) return i;
      return -1;
    };
    const isPause = (c: string) => c >= '\uE003' && c <= '\uE00A';
    // s: code points of a stretch of `text` starting at `base` (superbpe.payload_ids' recursion)
    const payload = (s: string[], base: number): void => {
      if (markers?.start && s.includes('\uE002')) {
        ranges.push([base, base + 1]);
        payload(s.slice(1), base + 1);
        return;
      }
      if (markers?.pauses && s.some(isPause)) {
        let pos = 0;
        s.forEach((c, i) => {
          if (!isPause(c)) return;
          if (i > pos) payload(s.slice(pos, i), base + pos);
          ranges.push([base + i, base + i + 1]);
          pos = i + 1;
        });
        if (pos < s.length) payload(s.slice(pos), base + pos);
        return;
      }
      const ph = markers?.phonemes;
      if (!ph || (find(s, ph.open, 0) < 0 && find(s, ph.close, 0) < 0)) {
        stretch(s, base);
        return;
      }
      let pos = 0;
      while (pos < s.length) {
        const start = find(s, ph.open, pos);
        stretch(s.slice(pos, start < 0 ? s.length : start), base + pos);
        if (start < 0) break;
        const end = find(s, ph.close, start + 1);
        for (let i = start; i <= end; i++) ranges.push([base + i, base + i + 1]);
        pos = end + 1;
      }
    };
    payload(cps, 0);
  }
  ranges.push([n, n]);
  return ranges;
}

// Text-span positions [a, b) of the tokens spelling each character span of `text` once it is `final`.
export function spanTokenRanges(text: string, final: string, spans: Range[], ranges: Range[]): Range[] {
  const tc = codePoints(text), fc = codePoints(final);
  const trimmed: Range[] = ranges.map(([a0, b0]) => {
    let a = a0, b = b0;
    while (a < b && pyIsSpace(fc[a])) a++;
    while (b > a && pyIsSpace(fc[b - 1])) b--;
    return [a, b];
  });
  const out: Range[] = [];
  for (let [start, end] of spans) {
    while (start < end && pyIsSpace(tc[start])) start++;
    while (end > start && pyIsSpace(tc[end - 1])) end--;
    const s = mapOffsets(text, final, [start])[0], e = mapOffsets(text, final, [end], true)[0];
    const hit: number[] = [];
    trimmed.forEach(([a, b], i) => {
      if (a < b && a < e && s < b) hit.push(i);
    });
    if (hit.length) out.push([hit[0], hit[hit.length - 1] + 1]);
    else {
      let first = ranges.length - 1;
      for (let i = 0; i < trimmed.length; i++)
        if (trimmed[i][0] < trimmed[i][1] && trimmed[i][1] > s) { first = i; break; }
      out.push([first, first]);
    }
  }
  return out;
}

// The span gate of one request (see zonos3-span-follow.h).
export class SpanFollower {
  private frames: number[] = [];
  private pos: number[] = [];
  private best = 0;
  private enter: ([number, number] | null)[];
  private exit: ([number, number] | null)[];
  private decided = new Map<number, number[]>();

  constructor(public ranges: Range[], public deltas: Float32Array[], public cfg: FollowConfig = defaultFollowConfig(),
              public origin = 0) {
    const flat = deltas.every((d) => d.length === deltas[0].length);
    if (!flat || deltas.length !== ranges.length)
      throw new Error(`span follower: ${ranges.length} token ranges for deltas of shape (${deltas.length}, ${deltas.length ? deltas[0].length : 0})`);
    if (cfg.smooth < 1 || cfg.ramp < 1) throw new Error('span follower: smooth and ramp must be >= 1');
    this.enter = ranges.map(() => null);
    this.exit = ranges.map(() => null);
  }

  onStep(step: number, pos: number): void { this.observe(this.origin + step, pos); }

  observe(frame: number, pos: number): void {
    if (this.frames.length && frame <= this.frames[this.frames.length - 1])
      throw new Error(`span follower: frames must increase, got ${frame} after ${this.frames[this.frames.length - 1]}`);
    this.frames.push(frame);
    this.pos.push(pos);
    this.advance(this.pos.length);
  }

  position(): number { return this.best; }

  private advance(n: number): void {
    const med = median(this.pos.slice(Math.max(0, n - this.cfg.smooth), n));
    this.best = Math.min(Math.max(this.best, med), this.best + 1);
  }

  weights(frame: number): number[] {
    const found = this.decided.get(frame);
    if (found) return found;
    const s = this.best, cfg = this.cfg;
    this.ranges.forEach(([a, b], k) => {
      if (a >= b) return;
      if (!this.enter[k] && s >= a - cfg.lead)
        this.enter[k] = [frame, this.frames.length === 0 ? frame - cfg.ramp + 1 : frame];
      if (this.enter[k] && !this.exit[k] && s >= b + cfg.lag) this.exit[k] = [frame, frame];
    });
    const w = this.ranges.map((_, k) => {
      const en = this.enter[k], ex = this.exit[k];
      if (!en) return 0;
      const up = (frame - en[1] + 1) / cfg.ramp;
      const down = !ex ? 1 : (ex[1] + cfg.ramp - 1 - frame) / cfg.ramp;
      return Math.min(1, Math.max(0, Math.min(up, down)));
    });
    this.decided.set(frame, w);
    return w;
  }

  // Σ w_k·deltas[k] (float32) for the frame generated now; null when no span is on.
  delta(frame: number): Float32Array | null {
    const w = this.weights(frame);
    if (!w.some((x) => x > 0)) return null;
    const dim = this.deltas.length ? this.deltas[0].length : 0;
    const out = new Float32Array(dim);
    w.forEach((wk0, k) => {
      const wk = Math.fround(wk0);
      for (let d = 0; d < dim; d++) out[d] += Math.fround(wk * this.deltas[k][d]);
    });
    return out;
  }

  truncate(frame: number): void {
    let keep = 0;
    while (keep < this.frames.length && this.frames[keep] < frame) keep++;
    this.frames.length = keep;
    this.pos.length = keep;
    this.best = 0;
    for (let n = 1; n <= keep; n++) this.advance(n);
    for (const events of [this.enter, this.exit])
      for (let k = 0; k < events.length; k++) if (events[k] && events[k]![0] >= frame) events[k] = null;
    for (const f of [...this.decided.keys()]) if (f >= frame) this.decided.delete(f);
  }
}
