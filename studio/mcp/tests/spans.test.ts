import { describe, expect, it } from 'vitest';
import type { ControlsSpec } from '../../src/api';
import fixture from '../../src/fixtures/controls.json';
import type { Span } from '../../src/state/requests';
import { anchorSpans, resolveValues } from '../src/design';

const spec = fixture as unknown as ControlsSpec;
const strip = (spans: Span[]) => spans.map(({ start, end, values }) => ({ start, end, values }));

describe('anchorSpans', () => {
  it('covers the line with the delivery and layers span values on top', () => {
    const spans = anchorSpans('Welcome to the late show', { calm: 2 }, [{ text: 'late show', values: { projection: 1.5 } }]);
    expect(strip(spans)).toEqual([
      { start: 0, end: 15, values: { calm: 2 } },
      { start: 15, end: 24, values: { calm: 2, projection: 1.5 } },
    ]);
  });

  it('finds the n-th occurrence', () => {
    const spans = anchorSpans('no, no, no', null, [{ text: 'no', occurrence: 3, values: { pace: -1 } }]);
    expect(strip(spans)).toEqual([{ start: 8, end: 10, values: { pace: -1 } }]);
  });

  it('throws when the text is not in the line', () => {
    expect(() => anchorSpans('Hello there', null, [{ text: 'goodbye', values: { calm: 1 } }])).toThrow('Span text "goodbye" (occurrence 1) not found in the line');
  });

  it('throws on overlapping spans', () => {
    expect(() => anchorSpans('the late show', null, [{ text: 'late show', values: {} }, { text: 'the late', values: {} }])).toThrow('Spans "the late" and "late show" overlap');
  });

  it('returns nothing for no input', () => {
    expect(anchorSpans('Hello', null, [])).toEqual([]);
  });

  it('refuses voice controls as delivery', () => {
    expect(() => resolveValues(spec, { gender: 1 }, 'delivery')).toThrow("'gender' is a voice control, not a delivery control: set it with zonos_design_voice.");
  });
});
