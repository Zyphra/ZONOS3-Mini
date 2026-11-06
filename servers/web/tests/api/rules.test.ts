// The v2 request rules of the in-browser server against the cross-server contract: every refusal of
// model/tests/fixtures/request_errors.json, the frame budget at the slot's capacity edge (R5) and reference codes past
// the registry limit (R7).
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { planGenerate } from '../../src/api/generate';
import { Speakers } from '../../src/api/speakers';
import { CONTEXT_CAPACITY, DECODE_CONTEXT_FRAMES, MAX_FRAMES, REFERENCE_MAX_FRAMES } from '../../src/generation';
import { bareApi, call } from './stub';

interface RequestError { name: string; requires: string[]; request: Record<string, unknown>; error: string }

const fixtures: RequestError[] = JSON.parse(readFileSync(
  fileURLToPath(new URL('../../../../model/tests/fixtures/request_errors.json', import.meta.url)), 'utf8'));
/** The fixture `requires` tags this server meets: condition rows, and never phoneme tokens. */
const SERVES = new Set(['conditions', 'no_phoneme_tokens']);

/** [frames][10] reference codes; frame f holds f on every codebook. */
const referenceCodes = (frames: number) => Array.from({ length: frames }, (_, f) => new Array<number>(10).fill(f));

describe('request_errors.json', () => {
  for (const fx of fixtures) {
    it.skipIf(!fx.requires.every(t => SERVES.has(t)))(fx.name, async () => {
      const r = await call(bareApi().api, 'POST', '/v2/generate', { ...fx.request, stream: false });
      expect(r.status, JSON.stringify(r.json())).toBe(400);
      expect(r.json<{ detail: string }>().detail).toEqual(expect.any(String));
    });
  }
});

describe('request rules', () => {
  const { api, engine } = bareApi();
  const spk = new Speakers(engine);
  const plan = (body: Record<string, unknown>) =>
    planGenerate(engine, spk, { stream: false, silence_prefill_frames: 0, ...body }, null, undefined);

  it('R5: the frame budget shrinks to the slot rows the prompt leaves; a full slot is refused', async () => {
    expect((await plan({ text: 'Hello there.' })).maxFrames).toBe(MAX_FRAMES);
    // A text whose prompt with a one-frame reference leaves 10..REFERENCE_MAX_FRAMES rows of the slot (short lines:
    // long sentences make the tokenizer slow).
    const unit = 'word\n';
    const perUnit = engine.tokenizer.encode(unit.repeat(100)).length / 100;
    const text = unit.repeat(Math.round((CONTEXT_CAPACITY - REFERENCE_MAX_FRAMES / 2) / perUnit));
    const oneFrame = (await plan({ text, speaker_reference_codes: referenceCodes(1) })).prompt.nRows;
    expect(CONTEXT_CAPACITY - oneFrame).toBeGreaterThan(10);
    expect(CONTEXT_CAPACITY - oneFrame).toBeLessThan(REFERENCE_MAX_FRAMES);
    // Every reference frame adds one prompt row: the reference that leaves exactly `left` rows.
    const leaving = (left: number) => referenceCodes(CONTEXT_CAPACITY - left - oneFrame + 1);
    expect((await plan({ text, speaker_reference_codes: leaving(10) })).maxFrames).toBe(10);
    expect((await plan({ text, speaker_reference_codes: leaving(10), max_tokens: 3 })).maxFrames).toBe(3);
    const edge = await plan({ text, speaker_reference_codes: leaving(1) });
    expect(edge.prompt.nRows).toBe(CONTEXT_CAPACITY - 1);
    expect(edge.maxFrames).toBe(1);
    const full = await call(api, 'POST', '/v2/generate', {
      text, speaker_reference_codes: leaving(0), silence_prefill_frames: 0, stream: false,
    });
    expect(full.status).toBe(400);
    expect(full.json()).toEqual({ detail: `TTS input len ${CONTEXT_CAPACITY} exceeds ${CONTEXT_CAPACITY}` });
  }, 60_000);

  it('R7: reference codes past REFERENCE_MAX_FRAMES keep their first frames', async () => {
    const codes = referenceCodes(REFERENCE_MAX_FRAMES + 50);
    const p = await plan({ text: 'Hello there.', speaker_reference_codes: codes, decode_context: true, decode_context_frames: 0 });
    expect(p.referenceCodes).toEqual(Int32Array.from(codes.slice(0, REFERENCE_MAX_FRAMES).flat()));
    expect(p.prompt.speakerRows).toBe(REFERENCE_MAX_FRAMES + 9);
    // decode_context_frames 0 = the registry length, taken from the kept frames' tail (R8)
    expect(p.decodeContext).toEqual(
      Int32Array.from(codes.slice(REFERENCE_MAX_FRAMES - DECODE_CONTEXT_FRAMES, REFERENCE_MAX_FRAMES).flat()));
    const r = await call(api, 'POST', '/v2/generate', { text: 'Hello there.', speaker_reference_codes: codes, stream: false });
    expect(r.status).toBe(200);
    // the dropped frames are validated too
    codes[codes.length - 1][0] = 4096;
    const bad = await call(api, 'POST', '/v2/generate', { text: 'Hello there.', speaker_reference_codes: codes, stream: false });
    expect(bad.json()).toEqual({ detail: 'speaker_reference_codes must be codec ids in [0, 4095]' });
  });

  it('R3: each condition row has its own bucket count (speaking_rate: 30)', async () => {
    const buckets = async (body: Record<string, unknown>) => (await plan({ text: 'Hello there.', ...body })).prompt.cond.buckets;
    expect(await buckets({})).toEqual([10, 10, 30]);
    expect(await buckets({ condition_buckets: { speaking_rate: 29 } })).toEqual([10, 10, 29]);
    expect(await buckets({ condition_buckets: { speaking_rate: 30 } })).toEqual([10, 10, 30]);
    expect(await buckets({ condition_values: { speaking_rate: 29.25 } })).toEqual([10, 10, 29]);
    expect(await buckets({ condition_values: { speaking_rate: 6.0 } })).toEqual([10, 10, 0]);
    const bad = await call(api, 'POST', '/v2/generate', { text: 'Hello there.', condition_buckets: { speaking_rate: 31 }, stream: false });
    expect(bad.status).toBe(400);
    expect(bad.json()).toEqual({ detail: 'condition_buckets[speaking_rate] must be in [0, 29] or 30 (= unknown)' });
  });
});
