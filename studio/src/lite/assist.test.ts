import { describe, expect, it } from 'vitest';
import type { ControlsSpec } from '../api';
import fixture from '../fixtures/controls.json';
import { FIXTURE_DECODERS } from '../fixtures/decoders';
import type { Speaker } from './api';
import { completeAt, highlight, type AssistEnv } from './assist';
import { NO_CONTROLS, NO_STEERING } from './script';

const spec = fixture as unknown as ControlsSpec;
const speaker = (id: string, label: string) =>
  ({ id, label, scope: 'default', is_default: true, source_type: 'audio', speaker_modes: ['embedding', 'reference'], has_preview: true, embedding_dim: 192, transcript: null, disabled_controls: [], original_name: null, created_at: 0 }) as Speaker;
const tagCfg = { available: false, trained: false, tags: [], min_scale: 1, max_scale: 4, default_scale: 1 };
const env: AssistEnv = {
  spec,
  steering: true,
  speakers: [speaker('ann', 'Ann en British F 1'), speaker('lie', 'Liesel de F 1')],
  caps: { decoders: FIXTURE_DECODERS, laci_languages: ['en_us'], tag_cfg: tagCfg, phoneme_tokens: null },
  languages: ['en_us', 'en_gb', 'de'],
};
const nv = { caps: { ...env.caps, tag_cfg: { ...tagCfg, available: true, trained: true } } };
const complete = (text: string, over: Partial<AssistEnv> = {}) => completeAt(text, text.length, { ...env, ...over });
const labels = (text: string, over: Partial<AssistEnv> = {}) => complete(text, over)?.items.map((i) => i.label);

describe('lite script completion', () => {
  it('offers directives not set yet, and their values', () => {
    expect(complete('@la')).toMatchObject({ from: 0, to: 3 });
    expect(labels('@la')).toEqual(['@language', '@laci']);
    expect(complete('@seed 1\n@se')).toBeNull();
    expect(complete('# @la')).toBeNull();
    expect(labels('@language ger')).toEqual(['de']);
    expect(complete('@language de')).toBeNull();
    expect(labels('@voice A = li')).toEqual(['Liesel de F 1']);
  });

  it('offers what each item list takes', () => {
    const pi = complete('@voice A = ann {pi')!;
    expect(pi).toMatchObject({ from: 16, to: 18 });
    expect(pi.items[0].label).toBe('pitch');
    const all = labels('@voice A = ann {')!;
    expect(all[0]).toBe('gender');
    expect(all).toContain('happy');
    expect(all).not.toContain('emphasis');
    expect(all.slice(-3)).toEqual(['preset', 'accent', 'mode']);
    expect(complete('{em')).toBeNull();
    expect(labels('Hi [there]{em')).toEqual(['emphasis']);
  });

  it('says what a control or accent takes', () => {
    expect(complete('{happy ')).toMatchObject({ items: [], hint: 'Happy 0…6 · safe 0…4' });
    expect(complete('@voice A = ann {age ')).toMatchObject({ items: [], hint: 'Age −3…3 · safe −2…2 · younger ↔ older' });
    const preset = complete('@voice A = ann {preset b')!.items;
    expect(preset.map((i) => [i.label, i.detail])).toEqual([['BBC newsreader', 'accent british 1.5, authority 3, room 2']]);
    const accent = complete('@voice A = ann {accent ir')!.items;
    expect(accent.map((i) => [i.label, i.detail])).toEqual([['irish', 'Ireland 0…2']]);
  });

  it('offers voice names at a line’s start, and tags the checkpoint reads', () => {
    const own = complete('@voice Bright {pitch 1}\nBr')!.items;
    expect(own.map((i) => [i.label, i.insert])).toEqual([['Bright', 'Bright: ']]);
    expect(labels('Li')).toEqual(['Liesel de F 1']);
    expect(complete('Hi <la')).toBeNull();
    expect(labels('Hi <la', nv)).toEqual(['<laugh/>']);
    const sh = complete('Hi <sh', nv)!.items;
    expect(sh.map((i) => i.label)).toEqual(['<shouting/>', '<shouting>…</shouting>']);
    expect(sh[1]).toMatchObject({ insert: '<shouting></shouting>', caret: 10 });
  });

  it('says why there is nothing to steer without a steering bundle', () => {
    const off = { steering: false, spec: NO_CONTROLS };
    expect(complete('{', off)).toMatchObject({ items: [], hint: NO_STEERING });
    expect(labels('@voice A = ann {', off)).toEqual(['mode']);
  });
});

describe('lite script coloring', () => {
  it('colors each part of the script', () => {
    const text = '@voice A = ann {age 1}\nA {happy 2}: Hi [there]{emphasis 4} <laugh/>\n# note\nNote: hi';
    expect(highlight(text, env).map((t) => [text.slice(t.from, t.to), t.cls])).toEqual([
      ['@voice', 'directive'],
      ['A', 'voice'],
      ['=', 'punct'],
      ['ann', 'ref'],
      ['{', 'punct'],
      ['age', 'key'],
      ['1', 'num'],
      ['}', 'punct'],
      ['A', 'voice'],
      ['{', 'punct'],
      ['happy', 'key'],
      ['2', 'num'],
      ['}', 'punct'],
      [':', 'punct'],
      ['[', 'span'],
      [']', 'span'],
      ['{', 'punct'],
      ['emphasis', 'key'],
      ['4', 'num'],
      ['}', 'punct'],
      ['<laugh/>', 'tag'],
      ['# note', 'comment'],
    ]);
  });
});
