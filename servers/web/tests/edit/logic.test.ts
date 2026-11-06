// The speech-edit port (src/edit) against goldens of the Python implementation (tests/edit/make_goldens.py: textmatch,
// align_text, edit_align, edit, edit_runtime.region_doses).
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { regionDoses, type Doses } from '../../src/edit/doses';
import { editPromptTokens, extractFill, shearedBlock, spliceEdit, validateEditSpans, type EditLayout } from '../../src/edit/frames';
import { pyFixed, pyRound, pySumFloats } from '../../src/edit/py';
import { alignTokens, canonicalMarkup, targetTokens, tokenChars, tokens } from '../../src/edit/text';
import {
  alignedWords, diffSpans, editWords, expectedFillSeconds, opcodes, planEditWindows, secondsToSpans, WordDiff,
  type EditWord, type FuseAlignWord, type Span, type WordIn,
} from '../../src/edit/words';

type Outcome<T> = { out: T; error?: undefined } | { error: string; out?: undefined };
type Rows = number[][];
type Scenario = {
  words: EditWord[]; targetText: string; targetWords: string[]; targetTokens: string[][]; totalFrames: number;
  margin: number; diff: Outcome<{ spans: Span[]; newTokens: number[] }>;
  windows: ({ spans: Span[]; context: number } & Outcome<{ start: number; end: number; regions: number[]; text: string }[]>)[];
  expected: ({ spans: Span[] } & Outcome<number[]>)[];
  wordDiff: { k: number; left: number | null; right: number | null }[];
};
interface Goldens {
  tokens: { text: string; out: string[] }[];
  alignTokens: { text: string; lang: string; out: string[] }[];
  canonicalMarkup: { text: string; out: string }[];
  targetTokens: { text: string; out: string[][] }[];
  tokenChars: { token: string; out: number }[];
  alignedWords: ({ words: FuseAlignWord[]; wordTokens: string[][] } & Outcome<EditWord[]>)[];
  editWords: ({ words: WordIn[]; wordTokens: string[][]; originalTokens: string[]; duration: number } & Outcome<EditWord[]>)[];
  secondsToSpans: { pairs: number[][]; totalFrames: number; frameRate: number; out: Span[] }[];
  opcodes: { a: string[]; b: string[]; out: [string, number, number, number, number][] }[];
  scenarios: Scenario[];
  regionDoses: ({ controls: Doses | null; regionControls: (Doses | null)[] | null; spans: Span[] | null } & Outcome<Doses[] | null>)[];
  validateSpans: { spans: Span[]; totalFrames: number; maxSpans: number; error: string | null }[];
  shearedBlock: { frames: Rows; endId: number | null; out: Rows }[];
  editPrompt: ({ textRows: Rows; codes: Rows; spans: Span[]; fills: Rows[]; spanEndId: number } & Outcome<Rows>)[];
  extractFill: { rows: Rows; eosFrame: number | null; fill: Rows; ended: boolean }[];
  splice: ({ codes: Rows; spans: Span[]; fills: Rows[] } & Outcome<Rows>)[];
}

const g: Goldens = JSON.parse(readFileSync(new URL('./goldens.json', import.meta.url), 'utf8'));
const K = 10;
const LAYOUT: EditLayout = { nCodebooks: K, audioPadId: 4097, textVocab: 8196 };
const flat = (rows: Rows) => Int32Array.from(rows.flat());
const rowsOf = (a: Int32Array, w: number): Rows => Array.from({ length: a.length / w }, (_, r) => Array.from(a.subarray(r * w, (r + 1) * w)));

/** The Python outcome: the value, or the ValueError's message. */
function expectOutcome<T>(f: () => T, o: Outcome<unknown>, label: string): void {
  if (o.error !== undefined) {
    expect(f, label).toThrow(o.error);
    return;
  }
  expect(f(), label).toEqual(o.out);
}

describe('edit text (textmatch, align_text, markup)', () => {
  it('tokens', () => { for (const c of g.tokens) expect(tokens(c.text), c.text).toEqual(c.out); });
  it('alignTokens', () => {
    for (const c of g.alignTokens) expect(alignTokens(c.text, c.lang), `${c.lang} ${c.text}`).toEqual(c.out);
  });
  it('canonicalMarkup', () => { for (const c of g.canonicalMarkup) expect(canonicalMarkup(c.text), c.text).toBe(c.out); });
  it('targetTokens', () => { for (const c of g.targetTokens) expect(targetTokens(canonicalMarkup(c.text)), c.text).toEqual(c.out); });
  it('tokenChars', () => { for (const c of g.tokenChars) expect(tokenChars(c.token).value, c.token).toBe(c.out); });
});

describe('edit words (edit_align)', () => {
  it('alignedWords', () => {
    for (const c of g.alignedWords) expectOutcome(() => alignedWords(c.words, c.wordTokens), c, JSON.stringify(c.words[0]));
  });
  it('editWords', () => {
    g.editWords.forEach((c, k) => expectOutcome(() => editWords(c.words, c.wordTokens, c.originalTokens, c.duration), c, `case ${k}`));
  });
  it('secondsToSpans', () => {
    for (const c of g.secondsToSpans) expect(secondsToSpans(c.pairs, c.totalFrames, c.frameRate), JSON.stringify(c)).toEqual(c.out);
  });
  it('opcodes (difflib)', () => {
    for (const c of g.opcodes) expect(opcodes(c.a, c.b), `${c.a.join('')} / ${c.b.join('')}`).toEqual(c.out);
  });
  it.each(g.scenarios.map((s, k) => [k, s] as const))('scenario %i: diff, windows, expected lengths', (_, s) => {
    const tt = s.targetTokens;
    expect(targetTokens(s.targetText)).toEqual(tt);
    expectOutcome(() => {
      const d = diffSpans(s.words, tt[0], { totalFrames: s.totalFrames, marginSeconds: s.margin, maxSpans: 3 });
      return { spans: d.spans, newTokens: d.newTargetTokens };
    }, s.diff, 'diff');
    const diff = new WordDiff(s.words, s.targetWords, tt.slice(1));
    for (const c of s.wordDiff) {
      expect(diff.targetWord(c.k, true), `left ${c.k}`).toBe(c.left);
      expect(diff.targetWord(c.k, false), `right ${c.k}`).toBe(c.right);
    }
    for (const w of s.windows) {
      expectOutcome(() => planEditWindows(w.spans, s.words, s.targetWords, tt.slice(1), {
        totalFrames: s.totalFrames, contextSeconds: w.context, targetText: s.targetText,
      }), w, `windows ${JSON.stringify(w.spans)} ${w.context}`);
    }
    for (const e of s.expected)
      expectOutcome(() => expectedFillSeconds(e.spans, s.words, s.targetWords, tt.slice(1)), e, `expected ${JSON.stringify(e.spans)}`);
  });
  it('regionDoses', () => {
    g.regionDoses.forEach((c, k) => expectOutcome(() => regionDoses(c.controls, c.regionControls, c.spans), c, `case ${k}`));
  });
});

describe('edit frames (edit)', () => {
  it('validateEditSpans', () => {
    for (const c of g.validateSpans) {
      const f = () => validateEditSpans(c.spans, c.totalFrames, c.maxSpans);
      if (c.error === null) expect(f).not.toThrow();
      else expect(f, JSON.stringify(c.spans)).toThrow(c.error);
    }
  });
  it('shearedBlock', () => {
    for (const c of g.shearedBlock) expect(rowsOf(shearedBlock(flat(c.frames), LAYOUT, c.endId), K + 1)).toEqual(c.out);
  });
  it('editPromptTokens', () => {
    g.editPrompt.forEach((c, k) => expectOutcome(() => rowsOf(editPromptTokens(flat(c.textRows), flat(c.codes), c.spans,
      c.fills.map(flat), LAYOUT, c.spanEndId), K + 1), c, `case ${k}`));
  });
  it('extractFill', () => {
    g.extractFill.forEach((c, k) => {
      const { fill, ended } = extractFill(flat(c.rows), c.eosFrame, 4097, new Array(K).fill(4096));
      expect(rowsOf(fill, K), `case ${k}`).toEqual(c.fill);
      expect(ended, `case ${k}`).toBe(c.ended);
    });
  });
  it('spliceEdit', () => {
    g.splice.forEach((c, k) => expectOutcome(() => rowsOf(spliceEdit(flat(c.codes), c.spans, c.fills.map(flat), K), K), c, `case ${k}`));
  });
});

describe('python numerics', () => {
  it('round, format and sum', () => {
    expect([0.5, 1.5, 2.5, -0.5, -1.5, 2.4999, 3.5000001].map(pyRound)).toEqual([0, 2, 2, 0, -2, 2, 4]);
    expect(pyFixed(0.0625, 3)).toBe('0.062');   // an exact tie: to even
    expect(pyFixed(0.0375, 3)).toBe((0.0375).toFixed(3));   // not a tie in binary
    expect(pyFixed(2.5, 0)).toBe('2');
    expect(pyFixed(1.2, 3)).toBe('1.200');
    expect(pySumFloats([1e16, 1, 1, -1e16])).toBe(2);
  });
});
