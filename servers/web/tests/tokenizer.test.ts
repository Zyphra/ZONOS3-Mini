import { describe, expect, it } from 'vitest';
import { createTokenizer } from '../src/tokenizer';
import expected from './fixtures/tokenizer-expected.json';
import native from './fixtures/tokenizer-native.json';

const tok = createTokenizer();

describe.each([['reference', expected], ['native', native]])('SuperBPE tokenizer (%s fixture)', (_, fx) => {
  it.each(fx.cases.map(c => [c.text, c.ids] as const))('%j', (text, ids) => {
    expect(Array.from(tok.encode(text))).toEqual(ids);
  });
});
