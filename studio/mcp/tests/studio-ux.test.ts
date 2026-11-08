import { describe, expect, it } from 'vitest';
import type { GenerateRequest } from '../../src/api';
import { slug } from '../../src/lib/files';
import { matchLines, splitScript, staleReason, type Line } from '../../src/state/lines';

const line = (id: string, text: string, extra: Partial<Line> = {}): Line => ({ id, text, spans: [], ...extra });

describe('matchLines', () => {
  const old = [line('a', 'Alpha one.'), line('b', 'Beta two.', { voice: 'p1', spans: [{ id: 's', start: 0, end: 4, values: { pace: 0.5 } }] }), line('c', 'Gamma three.')];

  it('keeps an edited row at the same position: id, voice and spans survive', () => {
    const { lines, dropped } = matchLines(old, ['Alpha one.', 'Beta 2.', 'Gamma three.']);
    expect(lines.map((l) => l.id)).toEqual(['a', 'b', 'c']);
    expect(lines[1].voice).toBe('p1');
    expect(lines[1].text).toBe('Beta 2.');
    expect(lines[1].spans[0].values).toEqual({ pace: 0.5 });
    expect(dropped).toEqual([]);
  });

  it('reports a removed row as dropped and keeps the others by text', () => {
    const { lines, dropped } = matchLines(old, ['Alpha one.', 'Gamma three.']);
    expect(lines.map((l) => l.id)).toEqual(['a', 'c']);
    expect(dropped.map((l) => l.id)).toEqual(['b']);
  });

  it('gives two identical rows two distinct ids', () => {
    const { lines } = matchLines([line('a', 'Same.')], ['Same.', 'Same.']);
    expect(lines[0].id).toBe('a');
    expect(lines[1].id).not.toBe('a');
    expect(new Set(lines.map((l) => l.id)).size).toBe(2);
  });

  it('matches by text before position, so a moved row keeps its id', () => {
    const { lines, dropped } = matchLines(old, ['Gamma three.', 'Alpha one.', 'Beta two.']);
    expect(lines.map((l) => l.id)).toEqual(['c', 'a', 'b']);
    expect(dropped).toEqual([]);
  });
});

describe('splitScript', () => {
  it('splits at sentence ends unless told not to', () => {
    expect(splitScript('One. Two!\nThree')).toEqual(['One.', 'Two!', 'Three']);
    expect(splitScript('One. Two!\nThree', { sentences: false })).toEqual(['One. Two!', 'Three']);
  });
});

describe('staleReason', () => {
  const base: GenerateRequest = { text: 'Hi.', seed: 1, speaker_embedding_id: 'v1', speaker_mode: 'embedding', controls: { pace: 0.2 }, control_spans: [] };
  it('names the first difference', () => {
    expect(staleReason(base, { ...base, text: 'Hey.' })).toBe('text changed');
    expect(staleReason(base, { ...base, speaker_embedding_id: 'v2' })).toBe('voice changed');
    expect(staleReason(base, { ...base, controls: { pace: 0.4 } })).toBe('baseline changed');
    expect(staleReason(base, { ...base, control_spans: [{ start: 0, end: 2, controls: { pace: 1 } }] })).toBe('painting changed');
    expect(staleReason(base, { ...base, language: 'de' })).toBe('language changed');
    expect(staleReason(base, { ...base })).toBeNull();
  });
});

describe('slug', () => {
  it('keeps letters of any script', () => {
    expect(slug('Ébauche 日本')).toBe('ébauche-日本');
    expect(slug('  Hello, World!  ')).toBe('hello-world');
    expect(slug('***')).toBe('voice');
  });
});

describe('isTyping', () => {
  // The function reads only tagName / type / isContentEditable: plain objects stand in for elements.
  const el = (tagName: string, type?: string) => ({ tagName, type, isContentEditable: false }) as unknown as EventTarget;
  it('is false for inputs that take no text, true for text entry', async () => {
    const { isTyping } = await import('../../src/lib/dom');
    expect(isTyping(el('INPUT', 'checkbox'))).toBe(false);
    expect(isTyping(el('INPUT', 'range'))).toBe(false);
    expect(isTyping(el('INPUT', 'text'))).toBe(true);
    expect(isTyping(el('TEXTAREA'))).toBe(true);
    expect(isTyping(el('BUTTON'))).toBe(false);
    expect(isTyping(null)).toBe(false);
  });
});
