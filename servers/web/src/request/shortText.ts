// Short texts: exact port of model/src/zonos3/tts/short_text.py (servers/ggml/src/zonos3-short-text.cpp). A text of
// one or two words is generated under LACI as the text, the 0.5 s pause marker and its language's filler sentence,
// and cut once the reading position reaches the filler: at the first frame a codes-only silence classifier calls
// silent, searched from `lookback` frames before the reading reached the filler, with `minVoiced` non-silent frames up
// to it. A take that reaches no such frame within `maxFrames` frames ran into the filler (LaciSession retakes it, then
// cuts where the reading reached it). The registry is model/src/zonos3/registries/short_text.json.
import doc from '../../../../model/src/zonos3/registries/short_text.json';
import { pyIsSpace } from '../edit/py';
import type { TextMarkers } from '../tokenizer';
import { tokenCharRanges, type BpeOffsets } from './spanFollow';

export interface ShortTextRegistry {
  pauseMarker: string;
  maxWords: number;
  maxRatio: [number, number];   // short: text tokens * maxRatio[1] <= filler tokens * maxRatio[0]
  lookback: number;
  minVoiced: number;
  maxFrames: number;
  maxRetries: number;
  nCodebooks: number;
  codebookSize: number;
  threshold: number;
  weights: Int8Array;           // [nCodebooks * codebookSize]
  fillers: Record<string, string>;  // per server language, with the space joining it to the marker
}

let cached: ShortTextRegistry | null = null;

export function shortTextRegistry(): ShortTextRegistry {
  if (cached) return cached;
  const s = doc.silence, cut = doc.cut;
  const bin = atob(s.weights);
  const weights = new Int8Array(bin.length);
  for (let i = 0; i < bin.length; i++) weights[i] = (bin.charCodeAt(i) << 24) >> 24;
  if (weights.length !== s.n_codebooks * s.codebook_size)
    throw new Error('short_text.json: silence.weights is not n_codebooks x codebook_size int8');
  cached = {
    pauseMarker: doc.pause_marker, maxWords: doc.max_words, maxRatio: [doc.max_ratio[0], doc.max_ratio[1]],
    lookback: cut.lookback, minVoiced: cut.min_voiced, maxFrames: cut.max_frames, maxRetries: cut.max_retries,
    nCodebooks: s.n_codebooks, codebookSize: s.codebook_size, threshold: s.threshold, weights,
    fillers: doc.fillers as Record<string, string>,
  };
  return cached;
}

const MARKER_RE = /[\uE000-\uE00A]/;   // pronunciation groups, the start marker and the pause markers

/** zonos3.tts.short_text.prompt: a qualifying normalized `text`'s prompt text (the text, the pause marker, the
 *  language's filler) and the text-span index (0 = BOS) of the filler's first token (the first past the marker that is
 *  not all whitespace); null when it does not qualify. `payloadLen(s)`: superbpe.payload_ids(s).length. */
export function shortTextPrompt(text: string, language: string, m: TextMarkers, bpe: BpeOffsets,
                                payloadLen: (s: string) => number): [string, number] | null {
  const reg = shortTextRegistry();
  const filler = reg.fillers[language];
  if (!m.pauses || !m.pauses.markers.includes(reg.pauseMarker) || filler === undefined || MARKER_RE.test(text)) return null;
  let words = 0, inWord = false;
  for (const ch of text) {
    const space = pyIsSpace(ch.codePointAt(0)!);
    if (!space && !inWord) words++;
    inWord = !space;
  }
  if (words > reg.maxWords) return null;
  const n = payloadLen(text);
  if (n === 0 || n * reg.maxRatio[1] > payloadLen(filler) * reg.maxRatio[0]) return null;
  const full = text + reg.pauseMarker + filler;
  const cps = Array.from(full);
  const textChars = Array.from(text).length;
  const ranges = tokenCharRanges(full, bpe, { phonemes: m.phonemes, start: m.start !== null, pauses: true });
  for (let i = 0; i < ranges.length; i++) {
    const [a, b] = ranges[i];
    if (a <= textChars || a >= b) continue;
    if (!cps.slice(a, b).every((c) => pyIsSpace(c.codePointAt(0)!))) return [full, i];
  }
  throw new Error('short text: the filler has no token');
}

/** The cut of a short-text take over its generated frames (frame f = codebook k of generated row f + k).
 *  observeReading(frame, pos) once per decode step, observeRow(row) per generated row; `decided` turns to
 *  ['cut', frames] (keep generated frames [0, frames)) or ['run_on', enter] once it can be told. truncate(n) keeps
 *  generated rows [0, n) (a LACI rollback). */
export class ShortCut {
  enter: number | null = null;
  decided: ['cut' | 'run_on', number] | null = null;
  readonly reg = shortTextRegistry();
  private rows: Int32Array[] = [];
  private silent: boolean[] = [];   // per complete frame

  constructor(private fillerToken: number, private nCodebooks: number) {}

  truncate(n: number): void {
    this.rows.length = Math.min(this.rows.length, n);
    this.silent.length = Math.min(this.silent.length, Math.max(0, n - this.nCodebooks + 1));
    if (this.enter !== null && this.enter >= n) this.enter = null;
    this.decided = null;
    this.decide();
  }

  observeReading(frame: number, pos: number): void {
    if (this.enter === null && pos >= this.fillerToken) {
      this.enter = frame;
      this.decide();
    }
  }

  observeRow(row: Int32Array): void {
    this.rows.push(row.slice(0, this.nCodebooks));
    const f = this.rows.length - this.nCodebooks;   // the frame this row completes
    if (f >= 0) {
      this.silent.push(this.isSilent(f));
      this.decide();
    }
  }

  private isSilent(f: number): boolean {
    const { codebookSize, weights, threshold } = this.reg;
    let score = 0;
    for (let k = 0; k < this.nCodebooks; k++) {
      const c = this.rows[f + k][k];
      if (c === undefined || c < 0 || c >= codebookSize) return false;
      score += weights[k * codebookSize + c];
    }
    return score > threshold;
  }

  private decide(): void {
    if (this.decided || this.enter === null) return;
    const { lookback, maxFrames, minVoiced } = this.reg;
    let voiced = 0;
    for (let f = 0; f < this.silent.length; f++) {
      voiced += this.silent[f] ? 0 : 1;
      if (f < this.enter - lookback) continue;
      if (f + 1 - this.enter > maxFrames) {
        this.decided = ['run_on', this.enter];
        return;
      }
      if (this.silent[f] && voiced >= minVoiced) {
        this.decided = ['cut', f + 1];
        return;
      }
    }
  }
}
