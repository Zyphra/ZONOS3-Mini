import { describe, expect, it } from 'vitest';
import type { ControlsSpec } from '../api';
import fixture from '../fixtures/controls.json';
import { FIXTURE_DECODERS } from '../fixtures/decoders';
import { requestDecoder, unknownDecoder } from '../lib/decoders';
import { design } from './design';
import { toDoses } from './engine';
import { buildRequest, controlSpans, type RequestContext, type Span } from './requests';

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

describe('decoder', () => {
  const ctx: RequestContext = { phonemeTokens: null, tags: null, tagStrength: 1.5, decoder: null, laci: 'off', laciLanguages: [], startToken: false };

  it('is left out for the server default', () => {
    expect('decoder' in buildRequest(spec, ctx, d, text, 1)).toBe(false);
    expect(requestDecoder(FIXTURE_DECODERS, null)).toBeNull();
    expect(requestDecoder(FIXTURE_DECODERS, 'keep2_plain')).toBeNull();
  });

  it('goes out when another served one is picked, on both A/B sides', () => {
    const picked = { ...ctx, decoder: requestDecoder(FIXTURE_DECODERS, 'full_mr') };
    expect(buildRequest(spec, picked, d, text, 1).decoder).toBe('full_mr');
    expect(buildRequest(spec, picked, d, text, 1, { unsteered: true }).decoder).toBe('full_mr');
  });

  it('falls back to the default when the server does not serve the pick', () => {
    expect(requestDecoder(FIXTURE_DECODERS.slice(0, 1), 'full_mr')).toBeNull();
    expect(requestDecoder([], 'full_mr')).toBeNull();
    expect(requestDecoder(FIXTURE_DECODERS, 'gone_head')).toBeNull();
  });

  it('an unknown one gets the server’s 400 text', () => {
    expect(unknownDecoder('x', FIXTURE_DECODERS)).toBe("Unknown decoder 'x'; this server serves: keep2_plain, full_mr.");
  });
});

describe('laci', () => {
  const ctx: RequestContext = { phonemeTokens: null, tags: null, tagStrength: 1.5, decoder: null, laci: 'stream', laciLanguages: ['en_us'], startToken: false };

  it('streamed: laci on with the progressive decode of the request defaults, on both A/B sides', () => {
    for (const x of [{}, { unsteered: true }]) {
      const req = buildRequest(spec, ctx, d, text, 1, x);
      expect(req.laci).toBe(true);
      expect(req.stream_decode).toBe('prefix');
    }
  });

  it('whole take: the whole decode', () => {
    const req = buildRequest(spec, { ...ctx, laci: 'whole' }, d, text, 1);
    expect([req.laci, req.stream_decode]).toEqual([true, 'whole']);
  });

  it('is left out for a language without alignment heads', () => {
    expect('laci' in buildRequest(spec, ctx, d, text, 1, { language: 'fr_fr' })).toBe(false);
  });

  it('wins over tag guidance', () => {
    const tagged = { ...ctx, tags: { min_scale: 1, max_scale: 4 }, tagStrength: 2 };
    const line = 'Hi <laugh/> there.';
    expect('tag_cfg_scale' in buildRequest(spec, tagged, d, line, 1)).toBe(false);
    expect(buildRequest(spec, { ...tagged, laci: 'off' }, d, line, 1).tag_cfg_scale).toBe(2);
  });
});

describe('start immediately', () => {
  const ctx: RequestContext = { phonemeTokens: null, tags: null, tagStrength: 1.5, decoder: null, laci: 'off', laciLanguages: [], startToken: true };

  it('goes out on both A/B sides when the checkpoint has a start token', () => {
    expect(buildRequest(spec, ctx, d, text, 1, { startImmediately: true }).start_immediately).toBe(true);
    expect(buildRequest(spec, ctx, d, text, 1, { startImmediately: true, unsteered: true }).start_immediately).toBe(true);
  });

  it('is left out without the extra or without a start token', () => {
    expect('start_immediately' in buildRequest(spec, ctx, d, text, 1)).toBe(false);
    expect('start_immediately' in buildRequest(spec, { ...ctx, startToken: false }, d, text, 1, { startImmediately: true })).toBe(false);
  });
});
