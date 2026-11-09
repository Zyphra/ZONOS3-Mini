import { describe, expect, it } from 'vitest';
import type { ControlsSpec } from '../api';
import fixture from '../fixtures/controls.json';
import { design } from './design';
import { toDoses } from './engine';
import { controlSpans, type Span } from './requests';

const spec = fixture as unknown as ControlsSpec;
const d = { ...design.value, mode: 'embedding' as const };
const text = 'I didn’t say she took the keys.';
const emphasis = { emphasis: 4.5 };
const span = (start: number, end: number, values: Span['values']): Span => ({ id: `${start}-${end}`, start, end, values });

describe('span-only controls (emphasis)', () => {
  it('go out on a word', () => {
    const start = text.indexOf('she');
    const spans = controlSpans(spec, [span(start, start + 3, emphasis)], d, text);
    expect(spans).toHaveLength(1);
    expect(spans[0].controls['emphasis_nv_nodur.emphasis']).toBeGreaterThan(0);
  });

  it('are dropped over the whole line, other paint on it kept', () => {
    const spans = controlSpans(spec, [span(0, text.length, { ...emphasis, pace: 1 })], d, text);
    expect(spans).toHaveLength(1);
    expect(Object.keys(spans[0].controls).some((k) => k.startsWith('emphasis_nv_nodur.'))).toBe(false);
    expect(controlSpans(spec, [span(1, text.length + 1, emphasis)], d, ` ${text} `)).toEqual([]);
  });

  it('stay on a one-word line', () => {
    expect(controlSpans(spec, [span(0, 4, emphasis)], d, 'Now!')).toHaveLength(1);
  });

  it('never apply to a whole voice', () => {
    expect(Object.keys(toDoses(spec, emphasis, {}, 'embedding')).some((k) => k.startsWith('emphasis_nv_nodur.'))).toBe(false);
  });
});
