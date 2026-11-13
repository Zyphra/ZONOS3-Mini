// Steering, LACI detector and span follower against the ggml fixtures and the shipped steer.gguf.
import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { GGUF_TYPE, parseGguf } from '../../src/gguf';
import { defaultLaciConfig, Detector } from '../../src/request/laci';
import { SpanFollower } from '../../src/request/spanFollow';
import {
  Calibration, controlsDocument, disabledControls, loadSteer, NO_STEERING, releaseDocument, requestRelease,
  resolveControls, steerIndex, Steering, textChars, type SteerBundle,
} from '../../src/request/steer';

const FIX = new URL('../../../ggml/tests/fixtures/', import.meta.url);
const MODELS = new URL('../../../ggml/gguf/', import.meta.url);
const gguf = (u: URL) => { const b = readFileSync(u); return parseGguf(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };
const norm = (v: Float32Array) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));
/** The float32 bit patterns of a delta (null stays null): bit-identity checks. */
const bits = (v: Float32Array | null) => (v ? Array.from(new Uint32Array(v.buffer, v.byteOffset, v.length)) : null);

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
    expect(Object.keys(doc)).toEqual(['bundle', 'embedding_dim', 'sets', 'directions', 'timing', 'ui', 'steer_release']);
    expect(b.steerRelease).toBeNull();
    expect(doc.steer_release).toBeNull();
    expect(Object.keys(doc.sets)).toEqual(['who', 'style', 'mood']);
    expect(doc.sets.style).toEqual({
      space: 'attnres_source', modes: ['embedding', 'reference'], source_index: 3,
      calibration: { default: { tempo: 0.5 }, by_speaker: { default_19139adcad9351f1: { whisper: 0 } }, global_default: 1 },
    });
    expect(doc.sets.who).toEqual({ space: 'raw', modes: ['embedding'] });
    expect(doc.sets.mood.calibration).toEqual({
      default: { valence: 1.5 }, by_speaker: { default_19139adcad9351f1: { 'valence-': 0 } }, global_default: 1.25,
    });
    expect(disabledControls(b, 'default_19139adcad9351f1')).toEqual(['style.whisper', 'mood.valence-']);
    expect(disabledControls(b, 'someone_else')).toEqual([]);
    expect(Object.keys(doc.directions)).toEqual(g.kv.get('zonos3_steer.direction.ids'));
    expect(Object.keys(doc.directions['who.gender'])).toEqual(['set', 'space', 'kind', 'norm', 'vector']);
    expect(Object.keys(doc.directions['style.whisper'])).toEqual(['set', 'space', 'kind', 'norm']);
    expect(doc.directions['style.whisper'].kind).toBe('named');
    expect(doc.directions['who.gender'].norm).toBeCloseTo(norm(b.dirs[0].vec), 10);
    expect(doc.timing).toEqual(JSON.parse(g.kv.get('zonos3_steer.timing') as string));
    expect(() => loadSteer(g, { nEmbd: 64, nLayer: 4, speakerDim: 192 }))
      .toThrow();
    const noCalibrations = { ...g, kv: new Map(g.kv), types: new Map(g.types) };
    noCalibrations.kv.delete('zonos3_steer.set.calibrations');
    noCalibrations.types.delete('zonos3_steer.set.calibrations');
    expect(() => loadSteer(noCalibrations, { nEmbd: 32, nLayer: 4, speakerDim: 192 }))
      .toThrow();
    const notSteer = { ...g, kv: new Map(g.kv), types: new Map(g.types) };
    notSteer.kv.set('general.architecture', 'zonos3_directions');
    expect(() => loadSteer(notSteer, { nEmbd: 32, nLayer: 4, speakerDim: 192 })).toThrow();
  });

  it.skipIf(!existsSync(new URL('steer.gguf', MODELS)))('loads the shipped steer.gguf', () => {
    const b = loadSteer(gguf(new URL('steer.gguf', MODELS)), { nEmbd: 1024, nLayer: 11, speakerDim: 192 });
    expect(b.sets.map((s) => (s.hidden ? `${s.name}(${s.source})` : s.name))).toEqual([
      'identity', 'pitch', 'accents_en', 'accents_l1', 'mandarin', 'mandarin_hidden(15)', 'texture', 'room',
      'delivery(19)', 'expressive(19)', 'disfluency(19)', 'mood(19)', 'emotions(19)', 'emphasis_nv_nodur(19)']);
    expect(b.sources).toEqual([15, 19]);
    expect(b.rateExtension?.direction).toBe('delivery.tempo');
    const doc = controlsDocument(b, 192) as Record<string, any>;
    expect(doc.ui.request_defaults).toEqual({ speaker_mode: 'embedding', stream_decode: 'prefix' });
    expect(Object.entries(doc.sets).filter(([, s]: [string, any]) => s.calibration).map(([k]) => k)).toEqual(['mood']);
    expect(doc.sets.mood.calibration).toMatchObject({ by_speaker: {}, global_default: 4 });
    expect(disabledControls(b, 'default_0123456789abcdef')).toEqual([]);
  });

  // steering.cpp test_resolve on the same in-memory bundle
  const mem = (): SteerBundle => {
    const dir = (set: string, name: string, hidden: boolean, source: number, v: number[]) =>
      ({ id: `${set}.${name}`, set, name, axis: false, hidden, source, vec: Float32Array.from(v) });
    const b: SteerBundle = {
      name: 'mem', speakerDim: 3, hiddenSize: 2,
      sets: [{ name: 'who', hidden: false, source: -1, calibration: null }, { name: 'accent', hidden: false, source: -1, calibration: null },
        { name: 'deep', hidden: true, source: 15, calibration: null }, { name: 'style', hidden: true, source: 19, calibration: null }],
      dirs: [dir('who', 'gender', false, -1, [0, 2, 0]), dir('accent', 'british', false, -1, [0, 0, 1]),
        dir('deep', 'cmn', true, 15, [0, 3]), dir('style', 'whisper', true, 19, [1, 0]), dir('style', 'tempo', true, 19, [-1, 1])],
      timing: new Map([['style.tempo', [[-1, 2], [0, 1], [1, 0.5]]]]), timingJson: '{}', uiJson: '{}',
      index: new Map(), sources: [], rateExtension: null, steerRelease: null,
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
    expect(() => resolveControls(b, { 'nope.x': 1 }, [], 10, 0, null)).toThrow();
    expect(() => resolveControls(b, { 'who.gender': 1, 'accent.british': 4 }, [], 10, 0, null))
      .toThrow();
    expect(() => resolveControls(b, {}, [{ start: 0, end: 2, controls: { 'who.gender': 1 } }], 10, 0, null)).toThrow();
    expect(() => resolveControls(b, {}, [{ start: 0, end: 4, controls: { 'style.whisper': 1 } }, { start: 3, end: 6, controls: { 'style.whisper': 1 } }], 10, 0, null)).toThrow();
    expect(() => resolveControls(b, {}, [{ start: 8, end: 12, controls: { 'style.whisper': 1 } }], 10, 0, null)).toThrow();
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

  it('calibration: document and disabled controls', () => {
    const c = Calibration.parse(JSON.stringify({
      default: { whisper: 2, 'tempo-': 0.5 }, by_speaker: { spk: { whisper: 0, 'tempo-': 0 } }, global_default: 1.5,
    }));
    expect(c.strength('spk', 'whisper')).toBe(0);
    expect(c.strength('other', 'whisper')).toBe(2);
    expect(c.strength('', 'tempo')).toBe(1.5);
    expect(c.document()).toEqual({
      default: { whisper: 2, 'tempo-': 0.5 }, by_speaker: { spk: { whisper: 0, 'tempo-': 0 } }, global_default: 1.5,
    });
    expect(() => Calibration.parse('{"default": {"x": "1"}}')).toThrow();
    const b = mem();
    b.sets[3].calibration = c;
    expect(disabledControls(b, 'spk')).toEqual(['style.whisper', 'style.tempo-']);
    expect(disabledControls(b, 'other')).toEqual([]);
    expect(disabledControls(null, 'spk')).toEqual([]);
    const doc = controlsDocument(b, 3) as Record<string, any>;
    expect(Object.keys(doc.sets.style)).toEqual(['space', 'modes', 'source_index', 'calibration']);
    expect(doc.sets.deep.calibration).toBeUndefined();
  });

  // steering.py ReleaseSchedule / request_release / release_weight / resolve_controls(release_exempt=...)
  const withRelease = (json: string): SteerBundle => {
    const g = gguf(new URL('dummy-steer.gguf', FIX));
    const keyed = { ...g, kv: new Map(g.kv), types: new Map(g.types) };
    keyed.kv.set('zonos3_steer.steer_release', json);
    keyed.types.set('zonos3_steer.steer_release', { type: GGUF_TYPE.STRING, arrType: null });
    return loadSteer(keyed, { nEmbd: 32, nLayer: 4, speakerDim: 192 });
  };

  it('steer_release: loaded from the GGUF, checked, reported in the controls document', () => {
    const b = withRelease(JSON.stringify({ onset_seconds: 4, ramp_seconds: 1, hold: 0.25, exempt: ['style.tempo'] }));
    expect(b.steerRelease).toEqual({ onsetSeconds: 4, rampSeconds: 1, hold: 0.25, exempt: ['style.tempo'] });
    const doc = { onset_seconds: 4, ramp_seconds: 1, hold: 0.25, exempt: ['style.tempo'] };
    expect((controlsDocument(b, 192) as Record<string, unknown>).steer_release).toEqual(doc);
    expect(releaseDocument(b)).toEqual(doc);
    expect(releaseDocument(null)).toBeNull();
    expect(withRelease('').steerRelease).toBeNull();
    expect(withRelease(JSON.stringify({ onset_seconds: 4, ramp_seconds: 0, hold: 0 })).steerRelease)
      .toEqual({ onsetSeconds: 4, rampSeconds: 0, hold: 0, exempt: [] });
    const base = { onset_seconds: 4, ramp_seconds: 1, hold: 0.25 };
    const cases: [unknown, string][] = [
      [{ onset_seconds: 4, ramp_seconds: 1 }, 'expected {onset_seconds, ramp_seconds, hold, exempt?}'],
      [{ ...base, extra: 1 }, 'expected {onset_seconds, ramp_seconds, hold, exempt?}'],
      [[1, 2, 3], 'expected {onset_seconds, ramp_seconds, hold, exempt?}'],
      [{ ...base, hold: 1.5 }, 'steer_hold must be in [0, 1], got 1.5.'],
      [{ ...base, onset_seconds: -1 }, 'steer_onset_seconds must be finite and >= 0, got -1.0.'],
      [{ ...base, ramp_seconds: '1' }, 'onset_seconds, ramp_seconds and hold must be numbers'],
      [{ ...base, exempt: 'style.tempo' }, 'exempt is not a string array'],
      [{ ...base, exempt: ['who.gender'] }, "steer_release exempts 'who.gender', which is not a hidden direction of the bundle"],
      [{ ...base, exempt: ['style.nope'] }, "steer_release exempts 'style.nope', which is not a hidden direction of the bundle"],
    ];
    for (const [bad, why] of cases) expect(() => withRelease(JSON.stringify(bad)), why).toThrow(why);
    expect(() => withRelease('{')).toThrow('bad steer_release JSON');
  });

  it('release weight: full dose before onset, a linear ramp to hold, hold after; ramp 0 is a step', () => {
    const s = new Steering();
    s.releaseFrames = { onset: 60, ramp: 15, hold: 0.25 };
    const want = [[0, 1], [59, 1], [60, 1], [61, 0.95], [65, 0.75], [74, 0.3], [75, 0.25], [500, 0.25]];
    for (const [f, w] of want) expect(s.releaseWeight(f), `frame ${f}`).toBeCloseTo(w, 12);
    s.releaseFrames = { onset: 3, ramp: 0, hold: 0 };
    expect([0, 2, 3, 4].map(f => s.releaseWeight(f))).toEqual([1, 1, 0, 0]);
    s.releaseFrames = { onset: 0, ramp: 4, hold: 0.5 };
    expect([0, 1, 2, 4].map(f => s.releaseWeight(f))).toEqual([1, 0.875, 0.75, 0.5]);
  });

  it('request fields: bundle defaults, overrides in frames and the refusals', () => {
    const b = mem();
    expect(requestRelease(null, null, null, null, 15)).toBeNull();
    expect(requestRelease(null, 1, 2, 3, 15)).toBeNull();
    expect(() => requestRelease(null, 0.5, null, null, 15))
      .toThrow(`${NO_STEERING}; steer_hold below 1 needs its steer_release.`);
    expect(requestRelease(b, 1, null, null, 15)).toBeNull();
    expect(() => requestRelease(b, 0.25, null, null, 15))
      .toThrow("steer_hold below 1 needs a steering bundle with steer_release; this server's bundle has none.");
    b.steerRelease = { onsetSeconds: 4, rampSeconds: 1, hold: 0.25, exempt: [] };
    expect(requestRelease(b, null, null, null, 15)).toEqual({ onset: 60, ramp: 15, hold: 0.25 });
    expect(requestRelease(b, 1, null, null, 15)).toBeNull();
    // Python round: 7.5 -> 8, 10.5 -> 10
    expect(requestRelease(b, 0, 0.5, 0.7, 15)).toEqual({ onset: 8, ramp: 10, hold: 0 });
    expect(() => requestRelease(b, 1.5, null, null, 15)).toThrow('steer_hold must be in [0, 1], got 1.5.');
    expect(() => requestRelease(b, null, -1, null, 15)).toThrow('steer_onset_seconds must be finite and >= 0, got -1.0.');
    expect(() => requestRelease(b, null, null, Infinity, 15)).toThrow('steer_ramp_seconds must be finite and >= 0, got inf.');
    b.steerRelease.hold = 1;
    expect(requestRelease(b, null, null, null, 15)).toBeNull();
  });

  it('splits the request-level hidden controls: exempt ones constant, the others released; spans and raw unaffected', () => {
    const b = mem();
    b.steerRelease = { onsetSeconds: 4, rampSeconds: 1, hold: 0.25, exempt: ['style.whisper'] };
    const frames = requestRelease(b, null, null, null, 15)!;
    const emb = Float32Array.of(4, 0, 0);
    const st = resolveControls(b, { 'style.whisper': 2, 'deep.cmn': 1, 'style.tempo': 1, 'who.gender': 1.5 }, [], 10, 0, emb, frames);
    expect(Array.from(st.hidden!)).toEqual([0, 0, 2, 0]);
    expect(Array.from(st.release!)).toEqual([0, 3, -1, 1]);
    expect(st.releaseFrames).toEqual(frames);
    expect(st.hiddenActive()).toBe(true);
    expect(emb[1]).toBeCloseTo(3 * 4 / 5, 6);
    const kept = resolveControls(b, { 'style.whisper': 2 }, [], 10, 0, null, frames);
    expect(Array.from(kept.hidden!)).toEqual([0, 0, 2, 0]);
    expect(kept.release).toBeNull();
    const all = resolveControls(b, { 'deep.cmn': 1 }, [], 10, 0, null, frames);
    expect(all.hidden).toBeNull();
    expect(Array.from(all.release!)).toEqual([0, 3, 0, 0]);
    const sp = resolveControls(b, { 'deep.cmn': 1 }, [{ start: 0, end: 4, controls: { 'style.tempo': 1 } }], 10, 0, null, frames);
    expect(sp.spans.map(s => Array.from(s.delta))).toEqual([[0, 0, -1, 1]]);
    expect(Array.from(sp.release!)).toEqual([0, 3, 0, 0]);
    const raw = resolveControls(b, { 'who.gender': 1 }, [], 10, 0, Float32Array.of(4, 0, 0), frames);
    expect(raw.hiddenActive()).toBe(false);
  });

  it('the delta at a frame: constant + w * released + spans, before onset, in the ramp and after', () => {
    const b = mem();
    b.steerRelease = { onsetSeconds: 0.2, rampSeconds: 0.2, hold: 0.25, exempt: ['style.whisper'] };
    const frames = requestRelease(b, null, null, null, 15)!;
    expect(frames).toEqual({ onset: 3, ramp: 3, hold: 0.25 });
    const st = resolveControls(b, { 'style.whisper': 2, 'deep.cmn': 1 }, [], 10, 1, null, frames);
    const at = (f: number) => Array.from(st.at(f)!);
    for (const f of [0, 1, 2, 3]) expect(at(f), `frame ${f}`).toEqual([0, 3, 2, 0]);
    expect(at(4)).toEqual([0, 2.25, 2, 0]);
    expect(at(5)).toEqual([0, 1.5, 2, 0]);
    for (const f of [6, 7, 100]) expect(at(f), `frame ${f}`).toEqual([0, 0.75, 2, 0]);
    // a span (style.tempo halves its characters' pace: chars [6, 8) -> frames [6, 7)) adds on top
    const sp = resolveControls(b, { 'style.whisper': 2, 'deep.cmn': 1 }, [{ start: 6, end: 8, controls: { 'style.tempo': 1 } }],
      10, 1, null, frames);
    expect(sp.spans.map(s => [s.start, s.end])).toEqual([[6, 7]]);
    expect(Array.from(sp.at(1)!)).toEqual([0, 3, 2, 0]);
    expect(Array.from(sp.at(6)!)).toEqual([0, 0.75, 1, 1]);
    const out: { v: Float32Array | null } = { v: null };
    sp.deltaAt(6, out);
    expect(Array.from(out.v!)).toEqual([0, 0.75, 1, 1]);
    // hold 0 without a constant part: a zero delta after the ramp, still applied (as the reference's)
    const zero = resolveControls(b, { 'deep.cmn': 1 }, [], 10, 1, null, requestRelease(b, 0, null, null, 15));
    expect(Array.from(zero.at(1)!)).toEqual([0, 3, 0, 0]);
    expect(Array.from(zero.at(6)!)).toEqual([0, 0, 0, 0]);
  });

  it('hold 1 (or a bundle without steer_release) gives bit-identical per-frame deltas', () => {
    const plain = mem(), keyed = mem();
    keyed.steerRelease = { onsetSeconds: 0.2, rampSeconds: 0.2, hold: 0.25, exempt: ['deep.cmn'] };
    const controls = { 'style.whisper': 0.3, 'deep.cmn': 0.7, 'style.tempo': 1.1 };
    const spans = [{ start: 2, end: 5, controls: { 'style.whisper': 0.9 } }];
    const ref = resolveControls(plain, controls, spans, 10, 1, null, requestRelease(plain, null, null, null, 15));
    const one = resolveControls(keyed, controls, spans, 10, 1, null, requestRelease(keyed, 1, null, null, 15));
    const unset = resolveControls(plain, controls, spans, 10, 1, null, requestRelease(plain, 1, 3, 0, 15));
    expect(ref.release).toBeNull();
    expect(one.release).toBeNull();
    const held = resolveControls(keyed, controls, spans, 10, 1, null, requestRelease(keyed, null, null, null, 15));
    expect(held.release).not.toBeNull();
    for (let f = 0; f < 40; f++) {
      expect(bits(one.at(f)), `frame ${f}`).toEqual(bits(ref.at(f)));
      expect(bits(unset.at(f)), `frame ${f}`).toEqual(bits(ref.at(f)));
      const r = ref.at(f)!, h = held.at(f)!;
      if (f < 3) for (let i = 0; i < r.length; i++) expect(h[i]).toBeCloseTo(r[i], 6);
    }
    expect(bits(held.at(10))).not.toEqual(bits(ref.at(10)));
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
