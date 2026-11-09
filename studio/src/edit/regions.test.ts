import { describe, expect, it } from 'vitest';
// zonos3.tts.textmatch.tokens on fixed texts, written by servers/web/tests/edit/make_goldens.py.
import goldens from '../../../servers/web/tests/edit/goldens.json';
import { normalizeTokens, planRegions, textWords, type Word } from './regions';

describe('edit tokens', () => {
  it("equal the server's (textmatch.tokens), CJK words included", () => {
    // Markup and pronunciation groups are tokens of their own here, by design.
    const plain = goldens.tokens.filter((c) => !/[<\ue000]/.test(c.text));
    expect(plain.some((c) => /[\u4e00-\u9fff]/.test(c.text))).toBe(true);
    for (const c of plain) expect(normalizeTokens(c.text), c.text).toEqual(c.out);
  });
});

describe('CJK text', () => {
  it('is cut into words, not one word per line', () => {
    expect(textWords('我今天去学校，你呢？')).toEqual(['我', '今天', '去', '学校，', '你', '呢？']);
    expect(textWords('「今日は」学校に行きます。')).toEqual(['「今日', 'は」', '学校', 'に', '行き', 'ます。']);
  });

  it('plans a region around the changed word only', () => {
    const timed: [string, number, number][] = [['我', 0, 0.2], ['今天', 0.2, 0.6], ['去', 0.6, 0.8], ['学校，', 0.8, 1.2], ['你', 1.3, 1.5], ['呢？', 1.5, 1.8]];
    const words: Word[] = timed.map(([word, start, end]) => ({ word, start, end, tokens: normalizeTokens(word) }));
    const plan = planRegions(words, '我明天去学校，你呢？', { duration: 2, frameRate: 15, marginSeconds: 0.08 });
    expect(plan.regions).toEqual([{ start: 3, end: 9, changes: [{ i1: 1, i2: 2, j1: 1, j2: 2 }] }]);
  });
});
