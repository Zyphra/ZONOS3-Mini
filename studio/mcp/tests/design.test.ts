import { describe, expect, it } from 'vitest';
import type { ControlsSpec } from '../../src/api';
import fixture from '../../src/fixtures/controls.json';
import { homePin, pinForMix } from '../../src/state/atlas';
import { presetToValues, toDoses } from '../../src/state/engine';
import { resolveAccents, resolveValues } from '../../src/state/resolve';
import { designView } from '../src/design';

const spec = fixture as unknown as ControlsSpec;

describe('resolveValues', () => {
  it('rejects an unknown control and lists the ids', () => {
    expect(() => resolveValues(spec, { growl: 1 }, 'design')).toThrow();
  });

  it('clamps to the range with a warning', () => {
    const r = resolveValues(spec, { age: 9 }, 'design');
    expect(r.values.age).toBe(3);
    expect(r.warnings.some((w) => w.includes('clamped to -3..3'))).toBe(true);
  });

  it('warns outside the tested-safe range', () => {
    const r = resolveValues(spec, { raspy: 5 }, 'design');
    expect(r.values.raspy).toBe(5);
    expect(r.warnings).toContain('Raspy 5 is outside the tested-safe range 0..4');
  });

  it('drops a zero value', () => {
    expect(resolveValues(spec, { age: 0 }, 'design').values).toEqual({});
  });
});

describe('designView', () => {
  it('drops embedding-only controls in reference mode', () => {
    const d = { baseId: 'spk', mode: 'reference' as const, values: { deep: 2, calm: 2 }, pin: homePin() };
    const v = designView(spec, d);
    expect(v.values.find((x) => x.id === 'deep')?.dropped).toBe(true);
    expect(v.warnings.some((w) => w.startsWith('Deep needs speaker_mode=embedding'))).toBe(true);
    expect(Object.keys(toDoses(spec, d.values, {}, 'reference')).some((k) => k.startsWith('texture.'))).toBe(false);
    expect(v.doses['emotions.serene']).toBeGreaterThan(0);
  });

  it('a preset with an override keeps the override', () => {
    const p = presetToValues(spec.ui.presets['BBC newsreader']);
    const values = { ...p.values, ...resolveValues(spec, { authority: 1 }, 'design').values };
    const v = designView(spec, { baseId: 'spk', mode: 'embedding', values, pin: pinForMix(p.accents, spec.ui.accents) });
    expect(v.values.find((x) => x.id === 'authority')?.value).toBe(1);
    expect(v.values.find((x) => x.id === 'room')?.value).toBe(2);
    expect(v.accents.find((a) => a.id === 'british')?.applied).toBeCloseTo(1.5, 1);
  });
});

describe('resolveAccents', () => {
  it('places the pin so the accent applies at the requested strength', () => {
    const r = resolveAccents(spec, { british: 1.5 });
    expect(Math.abs((r.applied.british ?? 0) - 1.5)).toBeLessThanOrEqual(0.05);
  });

  it('rejects an unknown accent', () => {
    expect(() => resolveAccents(spec, { martian: 1 })).toThrow();
  });

  it('warns that one pin is placed when several accents are asked', () => {
    const r = resolveAccents(spec, { british: 1.2, irish: 0.5 });
    expect(r.warnings.some((w) => w.startsWith('Studio places one accent pin: british'))).toBe(true);
  });
});
