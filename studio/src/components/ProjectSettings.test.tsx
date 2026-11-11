import { cleanup, fireEvent, render } from '@testing-library/preact';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { FIXTURE_DECODERS } from '../fixtures/decoders';
import { backend, projectSettings } from '../state/app';
import { DEFAULT_SETTINGS } from '../state/projectSchema';
import { offlineBackend } from '../test/backend';
import { DecoderPicker, LaciPicker } from './ProjectSettings';

const select = (c: Element) => c.querySelector('select');

describe('decoder picker', () => {
  beforeEach(() => {
    backend.value = offlineBackend();
    projectSettings.value = DEFAULT_SETTINGS;
  });
  afterEach(cleanup);

  it('shows the served decoders, the default first and marked, labels as tooltips', () => {
    const { container } = render(<DecoderPicker />);
    const options = [...container.querySelectorAll('option')];
    expect(options.map((o) => o.textContent)).toEqual(['Fast (keep2) (default)', 'Full-size']);
    expect(options.map((o) => o.title)).toEqual(FIXTURE_DECODERS.map((d) => d.label));
    expect(select(container)!.value).toBe('keep2_plain');
  });

  it('is hidden with one decoder or none', () => {
    backend.value = { ...offlineBackend(), decoders: FIXTURE_DECODERS.slice(0, 1) };
    expect(select(render(<DecoderPicker />).container)).toBeNull();
    backend.value = { ...offlineBackend(), decoders: [] };
    expect(select(render(<DecoderPicker />).container)).toBeNull();
  });

  it('stores the pick, and the default as null', () => {
    const { container } = render(<DecoderPicker />);
    fireEvent.change(select(container)!, { target: { value: 'full_mr' } });
    expect(projectSettings.value.decoder).toBe('full_mr');
    fireEvent.change(select(container)!, { target: { value: 'keep2_plain' } });
    expect(projectSettings.value.decoder).toBeNull();
  });

  it('shows a saved pick this server lacks as the default', () => {
    projectSettings.value = { ...DEFAULT_SETTINGS, decoder: 'gone_head' };
    expect(select(render(<DecoderPicker />).container)!.value).toBe('keep2_plain');
  });
});

describe('LACI picker', () => {
  beforeEach(() => {
    backend.value = offlineBackend();
    projectSettings.value = DEFAULT_SETTINGS;
  });
  afterEach(cleanup);

  it('shows only with alignment heads, and stores the pick', () => {
    expect(select(render(<LaciPicker />).container)).toBeNull();
    backend.value = { ...offlineBackend(), laciLanguages: ['en_us'] };
    const { container } = render(<LaciPicker />);
    expect([...container.querySelectorAll('option')].map((o) => o.textContent)).toEqual(['Off', 'On, streamed', 'On, whole take']);
    fireEvent.change(select(container)!, { target: { value: 'whole' } });
    expect(projectSettings.value.laci).toBe('whole');
  });
});
