import { describe, expect, it } from 'vitest';
import type { ControlsSpec } from '../api';
import fixture from '../fixtures/controls.json';
import { FIXTURE_DECODERS } from '../fixtures/decoders';
import { homePin } from '../state/atlas';
import { buildRequest, capsContext } from '../state/requests';
import { layerSpans } from '../state/resolve';
import type { Speaker } from './api';
import { compileScript, NO_CONTROLS, type ScriptEnv } from './script';

const spec = fixture as unknown as ControlsSpec;
const speaker = (id: string, label: string, speaker_modes: Speaker['speaker_modes'] = ['embedding', 'reference']) =>
  ({ id, label, scope: 'default', is_default: true, source_type: 'audio', speaker_modes, has_preview: true, embedding_dim: 192, transcript: null, disabled_controls: [], original_name: null, created_at: 0 }) as Speaker;
const ann = speaker('ann', 'Ann en British F 1');
const lie = speaker('lie', 'Liesel de F 1');
const ref = speaker('ref', 'Ref', ['reference']);
const caps: ScriptEnv['caps'] = {
  decoders: FIXTURE_DECODERS,
  laci_languages: ['en_us'],
  tag_cfg: { available: false, trained: false, tags: [], min_scale: 1, max_scale: 4, default_scale: 1 },
  phoneme_tokens: null,
};
const env: ScriptEnv = {
  spec,
  steering: true,
  speakers: [ann, lie, ref],
  caps,
  languages: ['en_us', 'en_gb', 'de'],
  selected: ann,
  selectedWhy: null,
  language: 'en_us',
  seed: 7,
  laci: 'off',
};
const compile = (text: string, over: Partial<ScriptEnv> = {}) => compileScript(text, { ...env, ...over }, () => 0.5);
const messages = (text: string, over: Partial<ScriptEnv> = {}) => compile(text, over).errors.map((e) => `${e.line}: ${e.message}`);

describe('lite script → requests', () => {
  it('sends what Studio sends for the same design, baseline and painting', () => {
    const r = compile('@baseline pace -0.5\n@voice A = ann {age 1}\nA: Welcome to [the late show]{projection 1.5}.');
    expect(r.errors).toEqual([]);
    const ctx = capsContext(caps, { tagStrength: 1.5, decoder: null, laci: 'off' });
    const spans = layerSpans(25, null, [{ start: 11, end: 24, values: { projection: 1.5 } }]);
    const studio = buildRequest(spec, ctx, { baseId: 'ann', mode: 'embedding', values: { age: 1 }, pin: homePin() }, 'Welcome to the late show.', 7, { extra: { pace: -0.5 }, spans, language: 'en_us' });
    expect(r.render!.parts).toHaveLength(1);
    expect(r.render!.parts[0]).toMatchObject({ line: 3, label: 'A', text: 'Welcome to the late show.' });
    expect(r.render!.parts[0].request).toEqual({ ...studio, stream: true, format: 'pcm' });
    expect(r.render!.parts[0].request.control_spans![0]).toMatchObject({ start: 11, end: 24 });
  });

  it('waits for step 1’s clip to be cloned, and says why step 1 has no voice', () => {
    const waiting = compile('Hello there.', { selected: null });
    expect(waiting).toMatchObject({ errors: [], usesSelected: true, render: null });
    expect(messages('# intro\nHello there.', { selected: null, selectedWhy: 'Pick a voice first' })).toEqual(['2: This line speaks with the voice of step 1: Pick a voice first.']);
  });

  it('puts each error on its line', () => {
    expect(messages('Ann {gender 1}: Hi')).toEqual(["1: 'gender' is a voice control, not a delivery control: set it on the voice."]);
    expect(messages('{emphasis 4}: Hi')).toEqual(['1: Emphasis goes on words, not a whole line: [words]{emphasis 4}.']);
    expect(messages('@voice R = Ref {deep 2}\nR: Hi')).toEqual(['1: Deep needs mode embedding (mode reference keeps the recording: only delivery and mood apply).']);
    expect(messages('@voice B = ann {mode reference, preset BBC newsreader}\nB: Hi')).toEqual(['1: A preset sets mode embedding.']);
    expect(messages('@seed -1\nHi')).toEqual(['1: @seed takes a whole number from 0 to 4294967295.']);
    expect(messages('Hi\n@gap 2')).toEqual(['2: @gap takes seconds from 0 to 1.5.']);
    expect(messages('@foo\nHi')).toEqual(['1: Unknown directive @foo. Directives: @language, @mode, @seed, @laci, @decoder, @tags, @gap, @baseline, @voice.']);
    expect(messages('@language en_us\n@language de\nHi')).toEqual(['2: @language is set twice (first on line 1).']);
  });

  it('takes @seed over Advanced’s seed over a random one, and a default voice’s own language', () => {
    expect(compile('@seed 3\nHi', { seed: 9 }).render!.seed).toBe(3);
    expect(compile('Hi', { seed: 9 }).render!.seed).toBe(9);
    expect(compile('Hi', { seed: null }).render!.seed).toBe(500000);
    const r = compile('@language en_gb\nHi.\nLiesel de F 1: Hallo.');
    expect(r.render!.parts.map((p) => p.request.language)).toEqual(['en_gb', 'de']);
    expect(r.render!.parts.map((p) => p.request.seed)).toEqual([7, 7]);
  });

  it('turns LACI on where the server has heads, and says where it has none', () => {
    const r = compile('@laci whole\nHi.\nlie: Hallo.');
    expect(r.render!.parts[0].request).toMatchObject({ laci: true, stream_decode: 'whole' });
    expect(r.render!.parts[1].request.laci).toBeUndefined();
    expect(r.warnings).toEqual([{ line: 3, message: 'No alignment heads for German: its lines run without LACI.' }]);
  });

  it('renders plain lines without a steering bundle and refuses controls', () => {
    const off = { steering: false, spec: NO_CONTROLS };
    const r = compile('Hi there.', off);
    expect(r.errors).toEqual([]);
    expect(r.render!.parts[0].request.controls).toBeUndefined();
    expect(messages('{happy 2}: hi', off)).toEqual(['1: This server has no steering bundle (GET /v2/controls answers 404): controls, presets and accents are off.']);
  });

  it('reads a backslash line as text, and warns about a name that is no voice', () => {
    const r = compile('\\@home\n\\Ann: hi\nNote: hi');
    expect(r.errors).toEqual([]);
    expect(r.render!.parts.map((p) => [p.text, p.label])).toEqual([
      ['@home', 'Ann en British F 1'],
      ['Ann: hi', 'Ann en British F 1'],
      ['Note: hi', 'Ann en British F 1'],
    ]);
    expect(r.warnings).toEqual([{ line: 3, message: '“Note:” is spoken: no voice is named Note.' }]);
    expect(compile('Ann: hi').render!.parts[0].text).toBe('hi');
  });
});
