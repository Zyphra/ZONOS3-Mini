// /v2/edit/transcribe, /v2/edit/align and /v2/edit of the in-browser engine over the stub engines (tests/api/stub.ts):
// the speech-edit backbone emits code 7 then span-end, the aligner spreads the words evenly over the clip.
import { describe, expect, it } from 'vitest';
import { validate } from '../../src/api/validate';
import { resolveControls } from '../../src/request/steer';
import { bareApi, call, haveEditModels, stubApi } from './stub';

const SR = 44100;
const b64 = (s: string) => Buffer.from(s).toString('base64');
const clip = (seconds: number) => new Float32Array(Math.round(seconds * SR)).fill(0.1);
const ASR = 'Speech recognition is not available on this server: ';

describe.skipIf(!haveEditModels)('speech editing without speech recognition', () => {
  it('transcribe, verify and an edit that needs the words of the clip answer 503 before any model work', async () => {
    const { api, codec, edit } = stubApi();
    const audio = { audio_base64: b64('wav') };
    const t = await call(api, 'POST', '/v2/edit/transcribe', audio, {}, clip(2));
    expect(t.status).toBe(503);
    expect(t.json()).toEqual({ detail: ASR + 'send the transcript to /v2/edit/align instead' });
    const cases: [Record<string, unknown>, string][] = [
      [{ target_text: 'Hi there.', original_text: 'Hello there.', verify: true }, 'set verify to false'],
      [{ target_text: 'Hi there.' }, "send the clip's transcript as original_text"],
      // explicit spans on a clip to window: its words are needed
      [{ target_text: 'Hi there.', spans: [[0.1, 0.5]], max_context_seconds: 1 }, "send the clip's transcript as original_text"],
    ];
    for (const [fields, fix] of cases) {
      const r = await call(api, 'POST', '/v2/edit', { ...audio, ...fields }, {}, clip(2));
      expect(r.status, JSON.stringify(fields)).toBe(503);
      expect(r.json()).toEqual({ detail: ASR + fix });
    }
    expect(edit!.backbone.prefills).toEqual([]);
    expect(edit!.aligner.calls).toEqual([]);
    expect(codec.frames).toBe(0);
  });

  it('align returns FuseAlign words with their edit tokens', async () => {
    const { api, edit } = stubApi();
    const r = await call(api, 'POST', '/v2/edit/align', { audio_base64: b64('wav'), text: '  Hello, there -- friend!  ' }, {}, clip(3));
    expect(r.status).toBe(200);
    expect(validate('EditTranscript', r.json())).toEqual([]);
    const body = r.json<{ text: string; duration: number; words: { word: string; tokens: string[]; confidence: number }[] }>();
    expect(body.text).toBe('Hello, there -- friend!');
    expect(body.duration).toBeCloseTo(3, 9);
    // the punctuation-only token aligns but carries no edit tokens, so it is dropped
    expect(edit!.aligner.calls[0].tokens).toEqual(['Hello,', 'there', '--', 'friend!']);
    expect(body.words.map(w => [w.word, w.tokens, w.confidence])).toEqual([
      ['Hello,', ['HELLO'], 0.9], ['there', ['THERE'], 0.9], ['friend!', ['FRIEND'], 0.9],
    ]);
    // ja: one unit per ICU word of each Han / kana run, punctuation attached
    const ja = await call(api, 'POST', '/v2/edit/align', { audio_base64: b64('wav'), text: '今日は学校。', language: 'ja' }, {}, clip(1));
    expect(ja.json<{ words: { word: string; tokens: string[] }[] }>().words.map(w => [w.word, w.tokens])).toEqual([
      ['今日', ['今日']], ['は', ['は']], ['学校。', ['学校']],
    ]);
  });

  it('align and edit limits', async () => {
    const { api } = stubApi();
    const cases: [string, Record<string, unknown>, Float32Array | undefined, number | undefined, number, string][] = [
      ['/v2/edit/align', { text: 'x' }, clip(31), undefined, 400,
        'Audio is 31.0 s long; this server aligns at most 30 s (edit_align_max_seconds).'],
      ['/v2/edit/align', { text: '   ' }, clip(1), undefined, 400, 'text is empty.'],
      ['/v2/edit/align', { text: 'x', language: 'xx' }, clip(1), undefined, 400,
        "Unsupported language code: 'xx'. Supported: en_us, en_gb, fr_fr, de, es, it, pt_br, ja, cmn, ko, ar, bn, bg, cs, da, nl, fi, ka, el, gu, he, hi, hu, id, kn, ms, ml, mr, no, or, pl, pa, ro, ru, sk, sv, tl, ta, te, th, tr, uk, ur, vi."],
      ['/v2/edit/align', { text: 'x' }, clip(150), 151.25, 400, 'Audio is 151.2 s long; at most 150 s are supported here.'],
      ['/v2/edit/align', { text: 'x' }, undefined, undefined, 400, 'could not decode the audio (any container this browser decodes)'],
      ['/v2/edit', { target_text: 'x', original_words: [{ word: 'y', start: 0, end: 0.5 }] }, clip(1), undefined, 400,
        'original_words needs the original_text they spell.'],
      ['/v2/edit', { target_text: 'Hi there', original_text: 'Hello there' }, clip(31), undefined, 400,
        'Audio is 31.0 s long; this server aligns at most 30 s (edit_align_max_seconds); send original_words for a longer clip.'],
      ['/v2/edit', { target_text: 'Hello there', original_text: 'Hello there' }, clip(2), undefined, 400,
        'target text equals the original transcript'],
      ['/v2/edit', { target_text: 'a', spans: [[0, 0.5]], region_controls: [{}, {}] }, clip(2), undefined, 400,
        'region_controls has 2 entries for 1 spans; send one per span (null for none).'],
      ['/v2/edit', { target_text: 'Say \ue000\u00ff\ue001', spans: [[0, 0.5]] }, clip(2), undefined, 400,
        "Pronunciation groups: Phones outside the inventory: ['\u00ff']"],
      ['/v2/edit', { target_text: 'Say,\ue004 hi', spans: [[0, 0.5]] }, clip(2), undefined, 400,
        'Speech editing takes no start or pause markers: remove U+E002 / U+E003-U+E00A from target_text'],
    ];
    for (const [path, fields, pcm, seconds, status, detail] of cases) {
      const r = await call(api, 'POST', path, { audio_base64: b64('wav'), ...fields }, {}, pcm, seconds);
      expect(r.status, `${path} ${JSON.stringify(fields)}`).toBe(status);
      expect(r.json().detail, `${path} ${JSON.stringify(fields)}`).toBe(detail);
    }
    expect((await call(api, 'POST', '/v2/edit/align', { audio_base64: '!!', text: 'x' }, {}, clip(1))).json().detail)
      .toBe('audio_base64 must be valid base64-encoded data.');
  });

  it('an edit of the diffed region regenerates it on the speech-edit backbone and keeps the other frames', async () => {
    const { api, engine, backbone, edit, codec } = stubApi({ editFrames: 6 });
    const r = await call(api, 'POST', '/v2/edit', {
      audio_base64: b64('wav'), original_text: 'Hello there my friend.', target_text: 'Hello there my dear friend.', seed: 5,
    }, {}, clip(3));
    expect(r.status).toBe(200);
    const body = r.json<{
      spans: number[][]; contexts: null; expected_seconds: number[]; words: unknown[];
      takes: { seed: number; audio_base64: string; duration: number; edited_regions: number[][]; fill_seconds: number[]; stopped: boolean[] }[];
    }>();
    expect(validate('EditResponse', body)).toEqual([]);
    expect(backbone.prefills).toEqual([]);                  // the release backbone never runs
    expect(edit!.backbone.prefills).toHaveLength(1);
    // the words of the four aligned tokens (0.75 s each); DEAR is inserted at 2.25 s: frames 32..35 (margin 0.08 s)
    expect(body.words).toHaveLength(4);
    expect(body.spans).toEqual([[32 / 15, 35 / 15]]);
    expect(body.contexts).toBeNull();
    // DEAR over the original's empty stretch: the clip's characters per second (18 token characters over 3 s)
    expect(body.expected_seconds[0]).toBeCloseTo(4 / (18 / 3), 9);
    // the prompt: [speaker row][3 condition rows] BOS text EOS, the context with span 0's marker, EOA, then marker 0
    const ids = edit!.backbone.prefills[0].ids, W = 11, rows = ids.length / W;
    expect(edit!.backbone.prefills[0].cond.speakerRow).toBe(0);
    expect(ids[(rows - 1) * W + 10]).toBe(engine.hp.textVocab + 1);
    expect(Array.from(ids.subarray((rows - 1) * W, (rows - 1) * W + 10))).toEqual(new Array(10).fill(engine.hp.audioPad));
    const take = body.takes[0];
    expect(take.seed).toBe(5);
    expect(take.stopped).toEqual([true]);
    // min_frames = 0.7 * expected * 15 = 7 frames hold the span-end back past the stub's 6 code frames
    expect(take.fill_seconds[0]).toBeCloseTo(Math.trunc(0.7 * body.expected_seconds[0] * 15) / 15, 9);
    const fillFrames = Math.round(take.fill_seconds[0] * 15);
    const total = Math.ceil(3 * SR / 2940) - 3 + fillFrames;
    expect(codec.frames).toBe(total);
    expect(take.duration).toBeCloseTo(total * 2940 / SR, 9);
    expect(take.edited_regions).toEqual([[32 / 15, (32 + fillFrames) / 15]]);
    const wav = Buffer.from(take.audio_base64, 'base64');
    expect(wav.subarray(0, 4).toString()).toBe('RIFF');
    expect(wav.length).toBe(44 + total * 2940 * 2);
  });

  it('explicit spans on a short clip need no transcript; each take reseeds', async () => {
    const { api, edit } = stubApi();
    const r = await call(api, 'POST', '/v2/edit', {
      audio_base64: b64('wav'), target_text: 'Something else.', spans: [[0.2, 0.6], [1.2, 1.5]], num_takes: 2, seed: 9,
      speaker_embedding: false,
    }, {}, clip(2));
    expect(r.status).toBe(200);
    const body = r.json<{ words: null; original_text: null; expected_seconds: number[]; takes: { seed: number; stopped: boolean[] }[] }>();
    expect(validate('EditResponse', body)).toEqual([]);
    expect(edit!.aligner.calls).toEqual([]);
    expect(body.words).toBeNull();
    expect(body.original_text).toBeNull();
    expect(body.expected_seconds).toEqual([6 / 15, 5 / 15]);   // the regions' own lengths: frames 3..9 and 18..23
    expect(body.takes.map(t => t.seed)).toEqual([9, 10]);
    expect(body.takes.map(t => t.stopped)).toEqual([[true, true], [true, true]]);
    // two regions per take, the second prompt carrying the first fill (ended by span-end) and marker 1
    expect(edit!.backbone.prefills).toHaveLength(4);
    const ids = edit!.backbone.prefills[1].ids, W = 11, rows = ids.length / W;
    expect(edit!.backbone.prefills[1].cond.speakerRow).toBe(-1);
    expect(ids[(rows - 1) * W + 10]).toBe(8196 + 2);
    expect(Array.from(ids).filter((v, i) => i % W < 10 && v === 4098).length).toBeGreaterThan(0);
  });

  it('steer then hold counts each fill from its first frame; hold 1 keeps the constant delta', async () => {
    const { api, engine, edit } = stubApi();
    const steer = engine.steer!;
    const kept = 'delivery.tempo', released = steer.dirs.find(d => d.hidden && d.id !== kept)!.id;
    const body = {
      audio_base64: b64('wav'), target_text: 'Something else.', spans: [[0.2, 0.6], [1.2, 1.5]], seed: 9,
      speaker_embedding: false, controls: { [kept]: 1, [released]: 1 },
      steer_hold: 0.5, steer_onset_seconds: 0.2, steer_ramp_seconds: 0,
    };
    const none = await call(api, 'POST', '/v2/edit', body, {}, clip(2));
    expect(none.status).toBe(400);
    expect(none.json()).toEqual({ detail: "steer_hold below 1 needs a steering bundle with steer_release; this server's bundle has none." });
    expect(edit!.backbone.prefills).toEqual([]);

    steer.steerRelease = { onsetSeconds: 4, rampSeconds: 1, hold: 0.25, exempt: [kept] };
    expect((await call(api, 'POST', '/v2/edit', body, {}, clip(2))).status).toBe(200);
    expect(edit!.backbone.prefills).toHaveLength(2);
    const c = resolveControls(steer, { [kept]: 1 }, [], 0, 0, null).hidden!;
    const r = resolveControls(steer, { [released]: 1 }, [], 0, 0, null).hidden!;
    let i = 0;
    for (let k = 1; k < r.length; k++) if (Math.abs(r[k]) > Math.abs(r[i])) i = k;
    // onset 3 frames, a step to 0.5: frames 1 and 2 of each fill at full dose, the rest held
    const ws = edit!.backbone.steered.map(s => +((s![i] - c[i]) / r[i]).toFixed(3));
    const second = ws.indexOf(1, 2);
    expect(second).toBeGreaterThan(3);
    expect(ws).toEqual([1, 1, ...new Array(second - 2).fill(0.5), 1, 1, ...new Array(ws.length - second - 2).fill(0.5)]);

    edit!.backbone.steered.length = 0;
    expect((await call(api, 'POST', '/v2/edit', { ...body, steer_hold: 1 }, {}, clip(2))).status).toBe(200);
    const constant = resolveControls(steer, body.controls, [], 0, 0, null).hidden!;
    expect(edit!.backbone.steered.length).toBeGreaterThan(0);
    expect(edit!.backbone.steered.every(s => s !== null && s.every((v, j) => Object.is(v, constant[j])))).toBe(true);
  });

  it('decoder picks the decoder of the edited clip; an unknown id is a 400 before any model work', async () => {
    const { api, codec, full, edit } = stubApi();
    const req = { audio_base64: b64('wav'), target_text: 'Something else.', spans: [[0.2, 0.6]], seed: 9, speaker_embedding: false };
    const bad = await call(api, 'POST', '/v2/edit', { ...req, decoder: 'nope' }, {}, clip(2));
    expect(bad.status).toBe(400);
    expect(bad.json()).toEqual({ detail: "Unknown decoder 'nope'; this server serves: keep2_plain, full_mr." });
    expect(edit!.backbone.prefills).toEqual([]);
    expect(full!.loads).toBe(0);
    const r = await call(api, 'POST', '/v2/edit', { ...req, decoder: 'full_mr' }, {}, clip(2));
    expect(r.status).toBe(200);
    expect(validate('EditResponse', r.json())).toEqual([]);
    expect(full!.frames).toBeGreaterThan(0);
    expect(codec.frames).toBe(0);
    // the default when unset or null
    expect((await call(api, 'POST', '/v2/edit', { ...req, decoder: null }, {}, clip(2))).status).toBe(200);
    expect(codec.frames).toBeGreaterThan(0);
  });
});

describe('speech editing unavailable', () => {
  it('a server without the speech-edit models answers 503', async () => {
    const { api } = bareApi();
    const caps = (await call(api, 'GET', '/v2/capabilities')).json();
    expect(caps).toMatchObject({ speech_editing: false, edit_transcribe: false, edit_max_spans: 0, edit_frame_rate: null });
    for (const [path, body] of [['/v2/edit/align', { audio_base64: 'AA==', text: 'x' }], ['/v2/edit', { audio_base64: 'AA==', target_text: 'x' }],
      ['/v2/edit/transcribe', { audio_base64: 'AA==' }]] as const) {
      const r = await call(api, 'POST', path, body, {}, clip(1));
      expect(r.status, path).toBe(503);
      expect(r.json().detail, path).toEqual(expect.any(String));
    }
    expect((await call(api, 'POST', '/v2/edit', { target_text: 'x' })).status).toBe(422);
  });
});
