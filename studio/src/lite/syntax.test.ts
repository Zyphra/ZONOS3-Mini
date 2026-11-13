import { describe, expect, it } from 'vitest';
import { parseScript, pieceAt, type Loc } from './syntax';

describe('lite script syntax', () => {
  it('gives every part of a line its exact range', () => {
    const t = '@voice A = ann {age 1, preset BBC newsreader}\nA {happy 2}: Hi [there]{emphasis 4}';
    const s = (l: Loc | null | undefined) => (l ? t.slice(l.from, l.to) : l);
    const [v, a] = parseScript(t, ['ann']);
    if (v.kind !== 'directive' || a.kind !== 'speech') throw new Error('line kinds');
    const d = v.voice!;
    expect([s(v.at), s(d.name), t[d.eq!], s(d.ref), t[d.items!.open!], t[d.items!.close!]]).toEqual(['@voice', 'A', '=', 'ann', '{', '}']);
    expect(d.items!.items.map((i) => s(i))).toEqual(['age 1', 'preset BBC newsreader']);
    expect(d.items!.items.map((i) => i.head.text)).toEqual(['age', 'preset']);
    const p = a.prefix!;
    expect([s(p.name), s(p.items!.items[0]), t[p.colon], s(a.text)]).toEqual(['A', 'happy 2', ':', 'Hi [there]{emphasis 4}']);
    expect([s(a.spans[0]), s(a.spans[0].words), s(a.spans[0].items.items[0])]).toEqual(['[there]{emphasis 4}', 'there', 'emphasis 4']);
  });

  it('reads an unclosed brace at a line’s start as a draft', () => {
    const l = parseScript('{hap', [])[0];
    if (l.kind !== 'speech') throw new Error('line kind');
    expect(l.draft!.items.from).toBe(1);
    expect(pieceAt('{hap', l.draft!.items, 4).words).toEqual([{ from: 1, to: 4, text: 'hap' }]);
  });
});
