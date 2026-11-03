// Steering, directions, markup, LACI detector and span follower against the ggml fixtures and shipped companions.
import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { parseGguf } from '../../src/gguf';
import { accentSteering, addHiddenDelta, applyEmotion, calibrate, combine, hiddenDelta, loadDirections } from '../../src/request/directions';
import { defaultLaciConfig, Detector } from '../../src/request/laci';
import { hasNonverbalMarkup, nonverbalMarkerSpans, stripNonverbalMarkup } from '../../src/request/markup';
import { SpanFollower } from '../../src/request/spanFollow';
import { controlsDocument, loadSteer, resolveControls, steerIndex, Steering, textChars, type SteerBundle } from '../../src/request/steer';

const FIX = new URL('../../../ggml/tests/fixtures/', import.meta.url);
const MODELS = new URL('../../../ggml/models/', import.meta.url);
const gguf = (u: URL) => { const b = readFileSync(u); return parseGguf(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };
const norm = (v: Float32Array) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));

describe('steer', () => {
  it('loads dummy-steer.gguf like steering.cpp test_loader', () => {
    const g = gguf(new URL('dummy-steer.gguf', FIX));
    const b = loadSteer(g, { nEmbd: 32, nLayer: 4, speakerDim: 192 });
    expect(b.sources).toEqual([3, 5]);
    expect(b.dirs.map((d) => d.id)).toEqual(['who.gender', 'who.age', 'style.whisper', 'style.tempo', 'mood.valence']);
    expect(b.dirs[0].vec.length).toBe(192);
    expect(b.dirs[2].vec.length).toBe(32);
    expect(b.timing.get('style.tempo')?.length).toBe(3);
    const doc = controlsDocument(b, 192) as Record<string, any>;
    expect(Object.keys(doc)).toEqual(['bundle', 'embedding_dim', 'sets', 'directions', 'timing', 'ui']);
    expect(Object.keys(doc.sets)).toEqual(['who', 'style', 'mood']);
    expect(doc.sets.style).toEqual({ space: 'attnres_source', modes: ['embedding', 'reference'], source_index: 3 });
    expect(doc.sets.who).toEqual({ space: 'raw', modes: ['embedding'] });
    expect(Object.keys(doc.directions)).toEqual(g.kv.get('zonos3_steer.direction.ids'));
    expect(Object.keys(doc.directions['who.gender'])).toEqual(['set', 'space', 'kind', 'norm', 'vector']);
    expect(Object.keys(doc.directions['style.whisper'])).toEqual(['set', 'space', 'kind', 'norm']);
    expect(doc.directions['style.whisper'].kind).toBe('named');
    expect(doc.directions['who.gender'].norm).toBeCloseTo(norm(b.dirs[0].vec), 10);
    expect(doc.timing).toEqual(JSON.parse(g.kv.get('zonos3_steer.timing') as string));
    expect(() => loadSteer(g, { nEmbd: 64, nLayer: 4, speakerDim: 192 }))
      .toThrow('steering hidden directions have 32 values; the backbone hidden size is 64');
    expect(() => loadSteer(gguf(new URL('dummy-emotion.gguf', FIX)), { nEmbd: 32, nLayer: 4, speakerDim: 192 }))
      .toThrow('not a valid zonos3 steering GGUF');
  });

  it.skipIf(!existsSync(new URL('steer.gguf', MODELS)))('loads the shipped steer.gguf', () => {
    const b = loadSteer(gguf(new URL('steer.gguf', MODELS)), { nEmbd: 1024, nLayer: 11, speakerDim: 192 });
    expect(b.sets.map((s) => (s.hidden ? `${s.name}(${s.source})` : s.name))).toEqual([
      'identity', 'pitch', 'accents_en', 'accents_l1', 'mandarin', 'mandarin_hidden(15)', 'texture', 'room',
      'delivery(19)', 'expressive(19)', 'disfluency(19)', 'mood(19)', 'emotions(19)']);
    expect(b.sources).toEqual([15, 19]);
    expect(b.rateExtension?.direction).toBe('delivery.tempo');
    const doc = controlsDocument(b, 192) as Record<string, any>;
    expect(doc.ui.request_defaults).toEqual({ speaker_mode: 'embedding', speaker_cfg_scale: 1.5, topk: 40, stream_decode: 'prefix' });
  });

  // steering.cpp test_resolve on the same in-memory bundle
  const mem = (): SteerBundle => {
    const dir = (set: string, name: string, hidden: boolean, source: number, v: number[]) =>
      ({ id: `${set}.${name}`, set, name, axis: false, hidden, source, vec: Float32Array.from(v) });
    const b: SteerBundle = {
      name: 'mem', speakerDim: 3, hiddenSize: 2,
      sets: [{ name: 'who', hidden: false, source: -1 }, { name: 'accent', hidden: false, source: -1 },
        { name: 'deep', hidden: true, source: 15 }, { name: 'style', hidden: true, source: 19 }],
      dirs: [dir('who', 'gender', false, -1, [0, 2, 0]), dir('accent', 'british', false, -1, [0, 0, 1]),
        dir('deep', 'cmn', true, 15, [0, 3]), dir('style', 'whisper', true, 19, [1, 0]), dir('style', 'tempo', true, 19, [-1, 1])],
      timing: new Map([['style.tempo', [[-1, 2], [0, 1], [1, 0.5]]]]), timingJson: '{}', uiJson: '{}',
      index: new Map(), sources: [], rateExtension: null,
    };
    steerIndex(b);
    return b;
  };

  it('resolves raw and hidden controls', () => {
    const b = mem();
    expect(b.sources).toEqual([15, 19]);
    const emb = Float32Array.of(4, 0, 0);
    const st = resolveControls(b, { 'who.gender': 1.5, 'accent.british': 4 }, [], 10, 0, emb);
    const k = 4 / Math.sqrt(41);
    expect(emb[0]).toBeCloseTo(4 * k, 6);
    expect(emb[1]).toBeCloseTo(3 * k, 6);
    expect(emb[2]).toBeCloseTo(4 * k, 6);
    expect(norm(emb)).toBeCloseTo(4, 5);
    expect(st.hiddenActive()).toBe(false);
    const h = resolveControls(b, { 'style.whisper': 2, 'deep.cmn': 1 }, [], 10, 0, null);
    expect(Array.from(h.hidden!)).toEqual([0, 3, 2, 0]);
    const sp = resolveControls(b, { 'style.tempo': 1 }, [{ start: 4, end: 8, controls: { 'style.tempo': -1 } }], 10, 2, null);
    expect(sp.spans.map((s) => [s.start, s.end, s.ramp])).toEqual([[4, 12, 3]]);
    expect(Array.from(sp.spans[0].delta)).toEqual([0, 0, 1, -1]);
    const half = resolveControls(b, {}, [{ start: 5, end: 7, controls: { 'style.whisper': 1 } }], 10, 0.5, null);
    expect([half.spans[0].start, half.spans[0].end]).toEqual([2, 4]);
    expect(() => resolveControls(b, { 'nope.x': 1 }, [], 10, 0, null)).toThrow("Unknown control 'nope.x'");
    expect(() => resolveControls(b, { 'who.gender': 1, 'accent.british': 4 }, [], 10, 0, null))
      .toThrow('Controls accent.british, who.gender move the speaker embedding');
    expect(() => resolveControls(b, {}, [{ start: 0, end: 2, controls: { 'who.gender': 1 } }], 10, 0, null)).toThrow('cannot vary inside a request');
    expect(() => resolveControls(b, {}, [{ start: 0, end: 4, controls: { 'style.whisper': 1 } }, { start: 3, end: 6, controls: { 'style.whisper': 1 } }], 10, 0, null)).toThrow('overlap');
    expect(() => resolveControls(b, {}, [{ start: 8, end: 12, controls: { 'style.whisper': 1 } }], 10, 0, null)).toThrow('outside the text (0..10)');
  });

  it('deltaAt ramps', () => {
    const r = new Steering();
    r.sources = [15, 19];
    r.hiddenSize = 2;
    r.spans.push({ start: 10, end: 20, ramp: 3, delta: Float32Array.of(0, 0, 1, 0), charStart: 0, charEnd: 0 });
    const o: { v: Float32Array | null } = { v: null };
    r.deltaAt(7, o); expect(o.v).toBeNull();
    r.deltaAt(8, o); expect(o.v![2]).toBeCloseTo(1 / 3, 6);
    r.deltaAt(10, o); expect(o.v![2]).toBeCloseTo(1, 6);
    r.deltaAt(21, o); expect(o.v![2]).toBeCloseTo(2 / 3, 6);
    r.deltaAt(23, o); expect(o.v).toBeNull();
    expect(textChars('héllo😀')).toBe(6);
  });
});

describe('directions', () => {
  const d = loadDirections(gguf(new URL('dummy-emotion.gguf', FIX)));
  it('loads dummy-emotion.gguf', () => {
    expect(d.space).toBe('raw');
    expect(d.names).toEqual(['happy', 'sad', 'angry']);
    expect(d.axes).toEqual(['valence', 'arousal']);
    expect(d.calibrated).toBe(true);
    expect(d.refBaseNorm).toBe(9.5);
  });
  it('combine / applyEmotion keep the base norm', () => {
    const base = Float32Array.from({ length: 192 }, (_, i) => Math.sin(i + 1));
    const { delta, requested } = combine(d, [['sad', 0.7]], 0.3, 0, true);
    expect(requested).toBe(true);
    expect(delta[5]).toBeCloseTo(0.7 * d.named[1][5] + 0.3 * d.axis[0][5], 5);
    const out = applyEmotion(d, base, [['sad', 0.7]], 0.3, 0, 1, true, true);
    expect(Math.abs(norm(out) - norm(base))).toBeLessThan(1e-6 * norm(base) + 1e-6);
    expect(out).not.toEqual(base);
    expect(() => combine(d, [['nope', 1]], 0, 0, true)).toThrow("Unknown emotion direction 'nope'. Available: ['happy', 'sad', 'angry', 'valence', 'arousal'].");
    expect(hiddenDelta(d, [['sad', 1]], 0, 0, 1, true)).toBeNull();
  });
  it('calibration', () => {
    const s: [string, number][] = [['happy', 1], ['sad', 1]];
    const va = { valence: -1, arousal: 0 };
    calibrate(d, 'default_0123456789abcdef', s, va);
    expect(s).toEqual([['happy', 0], ['sad', 1.5]]);
    expect(va.valence).toBe(-0.25);
  });
  it.skipIf(!existsSync(new URL('emotion.gguf', MODELS)))('shipped emotion / accent directions are hidden', () => {
    const e = loadDirections(gguf(new URL('emotion.gguf', MODELS)));
    expect([e.space, e.sourceIndex, e.axes.length]).toEqual(['attnres_source', 19, 7]);
    const h = hiddenDelta(e, [], 1, 0, 2, true)!;
    expect(h[0]).toBe(Math.fround(2 * e.axis[0][0]));
    const a = loadDirections(gguf(new URL('accent-directions.gguf', MODELS)));
    expect([a.sourceIndex, a.names]).toEqual([15, ['hin']]);
    const hv = accentSteering(a, 'hin', 1, '', true, null)!;
    const layout = addHiddenDelta([15, 19], 1024, 15, hv, null)!;
    expect(layout[3]).toBe(hv[3]);
    expect(layout[1024 + 3]).toBe(0);
    expect(addHiddenDelta([19], 1024, 15, hv, null)).toBeNull();
  });
});

describe('markup', () => {
  it('finds and strips markers', () => {
    const t = 'Hi <laugh/> there <shouting >loud</shouting> é';
    expect(hasNonverbalMarkup(t)).toBe(true);
    expect(nonverbalMarkerSpans(t)).toEqual([[3, 11], [18, 29], [33, 44]]);
    expect(stripNonverbalMarkup(t)).toBe('Hi there loud é');
    expect(hasNonverbalMarkup('<laughter>')).toBe(false);
  });
});

describe('laci detector', () => {
  it('monotone readout passes, a stall is a hallucination', () => {
    const cfg = defaultLaciConfig();
    const ok = new Detector(40, 4, cfg, 0);
    for (let row = 0; row < 160; row++) expect(ok.observe(row, Math.min(39, row >> 2))).toBeNull();
    expect(ok.doneRow()).not.toBeNull();
    const bad = new Detector(40, 4, { ...cfg, minStallFrames: 20, stallFactor: 0.5 }, 0);
    let f = null;
    for (let row = 0; row < 40 && !f; row++) f = bad.observe(row, row < 10 ? row >> 2 : 2);
    expect(f?.kind).toBe('hallucination');
  });
});

describe('span follower', () => {
  it('ramps in at s >= a - lead over 3 frames and out after b', () => {
    const f = new SpanFollower([[3, 5]], [Float32Array.of(1)]);
    const w: number[] = [];
    for (let frame = 0; frame < 12; frame++) {
      f.observe(frame, frame);
      w.push(f.weights(frame)[0]);
    }
    // the smoothed position at frame f is f - 1: enters at s >= a - lead = 2 (frame 3), exits at s >= b = 5 (frame 6)
    expect(w.map((x) => +x.toFixed(4))).toEqual([0, 0, 0, 0.3333, 0.6667, 1, 0.6667, 0.3333, 0, 0, 0, 0]);
  });
});
