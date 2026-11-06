// Host-side FuseAlign pieces vs the Python reference (tests/fixtures/aligner-host.json, scripts/make_aligner_goldens.py).
import { describe, expect, it } from 'vitest';
import {
  assembleWords, fillUnplaced, isWordy, jointPositions, resampleTo16k, transcriptBytes, transcriptIds, type Interval,
} from '../src/aligner/host';
import host from './fixtures/aligner-host.json';

describe('resampleTo16k', () => {
  for (const c of host.resample) {
    it(`matches torchaudio.functional.resample from ${c.sr} Hz`, () => {
      const y = resampleTo16k(Float32Array.from(c.x), c.sr);
      expect(y.length).toBe(c.y.length);
      let err = 0;
      for (let i = 0; i < y.length; i++) err = Math.max(err, Math.abs(y[i] - c.y[i]));
      expect(err).toBeLessThan(1e-5);
    });
  }

  it('returns a copy at 16 kHz', () => {
    const x = Float32Array.from([0.1, -0.2, 0.3]);
    const y = resampleTo16k(x, 16000);
    expect(Array.from(y)).toEqual(Array.from(x));
    expect(y).not.toBe(x);
  });
});

describe('fillUnplaced', () => {
  for (const c of host.fill_unplaced) {
    it(`fills ${JSON.stringify(c.intervals)}`, () => {
      const got = fillUnplaced(c.intervals.map((iv): Interval | null => (iv ? [iv[0], iv[1]] : null)), c.duration);
      expect(got.length).toBe(c.filled.length);
      got.forEach((iv, i) => {
        expect(iv[0]).toBeCloseTo(c.filled[i][0], 12);
        expect(iv[1]).toBeCloseTo(c.filled[i][1], 12);
      });
    });
  }
});

describe('tokenizer', () => {
  it('builds [1, 32] + bytes with byte spans excluding the prefix', () => {
    const { ids, spans } = transcriptIds(['Hi,', 'café']);
    expect(Array.from(ids)).toEqual([1, 32, 72, 105, 44, 32, 99, 97, 102, 0xc3, 0xa9]);
    expect(Array.from(spans)).toEqual([0, 3, 4, 9]);
    expect(transcriptBytes(['Hi,', 'café'])).toBe(9);
  });

  it('classifies wordy tokens by letters and digits', () => {
    expect(isWordy('—')).toBe(false);
    expect(isWordy('*')).toBe(false);
    expect(isWordy('42')).toBe(true);
    expect(isWordy('日本')).toBe(true);
  });

  it('places text bytes at fractional positions', () => {
    const pos = jointPositions(4, 3);
    expect(Array.from(pos)).toEqual([0, 1, 2, 3, 0, 2, 4]);
  });
});

describe('assembleWords', () => {
  it('makes punctuation zero-width at the previous end and fills unplaced words', () => {
    const tokens = ['—', 'one', '-', 'two', 'three'];
    const intervals: (Interval | null)[] = [[0.5, 0.8], null, [1.4, 1.6]];
    const words = assembleWords(tokens, [1, 3, 4], intervals, [0.9, 0.05, 0.8], 2);
    expect(words.map(w => [w.word, w.start, w.end, w.placed])).toEqual([
      ['—', 0.5, 0.5, false], ['one', 0.5, 0.8, true], ['-', 0.8, 0.8, false], ['two', 0.8, 1.4, false],
      ['three', 1.4, 1.6, true],
    ]);
    expect(words[0].presence).toBe(0);
    expect(words[3].presence).toBe(0.05);
  });
});
