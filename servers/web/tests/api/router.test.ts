import { describe, expect, it } from 'vitest';
import { validate } from '../../src/api/validate';
import { call, haveEditModels, haveModels, stubApi } from './stub';

const SESSION = { 'x-tts-session-id': 's1' };
type Detail = { detail: Record<string, unknown>[] };
type SpeakerList = { speakers: { id: string }[] };
const b64 = (s: string) => Buffer.from(s).toString('base64');

describe.skipIf(!haveModels)('engine api', () => {
  it('capabilities validate against the spec', async () => {
    const { api } = stubApi();
    const caps = await call(api, 'GET', '/v2/capabilities');
    expect(caps.status).toBe(200);
    expect(validate('Capabilities', caps.json())).toEqual([]);
    expect(caps.json()).toMatchObject({
      speaker_modes: ['reference', 'embedding'], speaker_cfg_supported: true, plugins: [], speech_editing: haveEditModels,
      edit_transcribe: false, edit_max_spans: haveEditModels ? 3 : 0, edit_frame_rate: haveEditModels ? 15 : null,
      edit_max_seconds: 150, edit_align_max_seconds: 30,
      projects: { enabled: false }, laci_languages: ['cmn', 'de', 'en_gb', 'en_us', 'es', 'fr_fr', 'it', 'ja', 'ko', 'pt_br'],
      accents: [],
    });
  });

  it('controls document validates and carries set calibrations', async () => {
    const r = await call(stubApi().api, 'GET', '/v2/controls');
    expect(r.status).toBe(200);
    expect(validate('ControlsDocument', r.json())).toEqual([]);
    const sets = r.json<{ sets: Record<string, { calibration?: unknown; source_index?: number }> }>().sets;
    expect(sets.accent_hidden).toMatchObject({ space: 'attnres_source', source_index: 15 });
    expect(sets.accent_hidden.calibration).toBeUndefined();
    expect(sets.mood.calibration).toMatchObject({ by_speaker: {}, global_default: 4 });
  });

  it('unknown paths and unknown fields', async () => {
    const { api } = stubApi();
    for (const [m, p] of [['GET', '/v2/nope'], ['POST', '/v2/nope'], ['GET', '/v2/'], ['GET', '/v2/edit'], ['GET', '/v2/projects']]) {
      const r = await call(api, m, p, m === 'POST' ? {} : undefined);
      expect(r.status, `${m} ${p}`).toBe(404);
      expect(r.json()).toEqual({ detail: 'Not Found' });
    }
    const bad = await call(api, 'POST', '/v2/generate', { text: 'x', no_such_field: 1 });
    expect(bad.status).toBe(422);
    expect(bad.json<Detail>().detail[0]).toMatchObject({ loc: ['body', 'no_such_field'], type: 'value_error' });
    const json = await call(api, 'POST', '/v2/generate', '{"text":');
    expect(json.status).toBe(422);
    expect(json.json<Detail>().detail[0]).toMatchObject({ msg: 'JSON decode error', type: 'json_invalid' });
    for (const fields of [{ headroom_db: 25 }, { seed: -1 }, { seed: 2 ** 32 }, { format: 'mp3' }, { text: 5 }, { speaker_blend_t: 2 },
      { emotion_cfg_scale: 1.5 }, { emotion_enabled: true }]) {
      const r = await call(api, 'POST', '/v2/generate', { text: 'x', ...fields });
      expect(r.status, JSON.stringify(fields)).toBe(422);
    }
  });

  it('buffered pcm, wav and seeds', async () => {
    const { api } = stubApi();
    const r = await call(api, 'POST', '/v2/generate', { text: 'Hello there.', stream: false, seed: 1 });
    expect(r.status).toBe(200);
    expect(r.headers).toMatchObject({ 'x-audio-sample-rate': '44100', 'x-audio-channels': '1', 'x-audio-format': 'float32', 'x-seed': '1' });
    // 14 generated frames + 3 prefill rows: eos at stream row 17 - 9 ... decoded frames minus the 3 prefill frames
    const audio = new Float32Array(r.body.buffer);
    expect(audio.length % 2940).toBe(0);
    expect(audio.length).toBeGreaterThan(0);
    expect(audio[0]).toBeCloseTo(0.5 * 10 ** (-3 / 20), 6);
    const wav = await call(api, 'POST', '/v2/generate', { text: 'Hello there.', stream: false, format: 'wav', seed: 4294967295 });
    expect(wav.status).toBe(200);
    expect(new TextDecoder().decode(wav.body.subarray(0, 4))).toBe('RIFF');
    expect(wav.headers['x-seed']).toBe('4294967295');
    expect((await call(api, 'POST', '/v2/generate', { text: 'x', format: 'wav' })).status).toBe(400);
    const drawn = await call(api, 'POST', '/v2/generate', { text: 'x', stream: false });
    expect(Number(drawn.headers['x-seed'])).toBeLessThan(2 ** 31);
  });

  it('speech voice rules', async () => {
    const { api, engine } = stubApi();
    const voice = engine.voices[0].id;
    const ok = await call(api, 'POST', '/v1/audio/speech', { model: 'zonos3', input: 'Hi.', voice });
    expect(ok.status).toBe(200);
    expect(ok.headers['x-audio-format']).toBe('float32');
    const unknown = await call(api, 'POST', '/v1/audio/speech', { model: 'zonos3', input: 'Hi.', voice: 'alloy' });
    expect(unknown.status).toBe(400);
    expect(unknown.json().detail).toEqual(expect.any(String));
    const both = await call(api, 'POST', '/v1/audio/speech', { model: 'zonos3', input: 'Hi.', voice, speaker_audio_base64: b64('x') });
    expect(both.json()).toEqual({ detail: 'voice cannot be combined with speaker_* fields' });
    expect((await call(api, 'POST', '/v1/audio/speech', { model: 'zonos3', input: 'Hi.', speed: 1 })).status).toBe(422);
  });

  it('speaker resolution errors', async () => {
    const { api, engine } = stubApi();
    const id = engine.voices[0].id;
    const cases: [Record<string, unknown>, string][] = [
      [{ speaker_embedding_id: 'nope' }, 'Cached speaker operations require the X-TTS-Session-ID header.'],
      [{ speaker_cfg_scale: 1.5 }, 'Speaker CFG needs a speaker prompt: speaker_mode=reference with reference audio or codes, or speaker_mode=embedding with a clip or an embedding'],
      [{ speaker_mode: 'reference', speaker_blend_embedding_id_a: id }, 'This checkpoint conditions on raw reference audio and does not accept speaker embedding files or embedding blends.'],
      [{ speaker_mode: 'embedding', speaker_blend_embedding_id_a: id }, 'Provide both cached speaker ids to blend between two speakers.'],
      [{ speaker_mode: 'embedding', speaker_embedding_id: id, speaker_audio_base64: b64('x') }, 'Provide speaker conditioning via upload, one cached speaker id, or two cached ids for blending.'],
      [{ speaker_mode: 'embedding', speaker_reference_codes: [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]] }, 'speaker_reference_codes need speaker_mode=reference.'],
      [{ speaker_mode: 'reference', speaker_reference_codes: [[1, 2, 3]] }, 'speaker_reference_codes must be a non-empty [frames, codebooks] list'],
      [{ speaker_mode: 'embedding', speaker_audio_base64: '!!!' }, 'speaker_audio_base64 must be valid base64-encoded data.'],
      [{ speaker_mode: 'embedding', speaker_audio_base64: b64('abc') }, 'could not decode the reference audio (any container ffmpeg reads, at most 60 seconds)'],
      [{ speaker_mode: 'embedding', speaker_embedding_base64: b64('not npy') }, 'Speaker embedding file must be a valid .npy or .npz file saved without pickle.'],
      [{ speaker_mode: 'x' }, ''],
      [{ tag_cfg_scale: 2 }, 'tag_cfg_scale above 1.0 requires recognized nonverbal markup in the text: point tags <laugh/>, <breath/>, <sigh/>, <cough/>, <panting/>, <shouting/>, <crying/> or the same names as span tags (<shouting>...</shouting>).'],
      [{ laci: true, language: 'en_us', speaker_embedding_id: id, speaker_cfg_scale: 1.5 }, 'LACI needs speaker_cfg_scale = controls_cfg_scale = 1'],
      [{ tag_cfg_scale: 2, text: 'Hi <laugh/>.', controls_cfg_scale: 1.5 }, 'tag_cfg_scale above 1.0 cannot be combined with controls_cfg_scale; set controls_cfg_scale to 1.0.'],
      [{ speaker_mode: 'embedding', speaker_embedding_id: id, speaker_cfg_scale: 1.5, controls_cfg_scale: 1.5 }, 'Speaker and controls CFG cannot be combined'],
      [{ language: 'xx' }, "Unsupported language code: 'xx'. Supported: en_us, en_gb, fr_fr, de, es, it, pt_br, ja, cmn, ko."],
      [{ controls: { 'nope.nothing': 1 } }, "Unknown control 'nope.nothing'; see /v2/controls."],
      [{ accent: 'zzz' }, "Unknown accent 'zzz'; loaded accents: none (load accent adapters)."],
      [{ max_tokens: 0 }, 'max_tokens must be positive.'],
    ];
    for (const [fields, detail] of cases) {
      const r = await call(api, 'POST', '/v2/generate', { text: 'Hello.', stream: false, ...fields });
      if (detail === '') { expect(r.status, JSON.stringify(fields)).toBe(422); continue; }
      expect(r.status, JSON.stringify(fields)).toBe(400);
      expect(r.json().detail, JSON.stringify(fields)).toBe(detail);
    }
  });

  it('session uploads, previews and embeddings', async () => {
    const { api, engine } = stubApi();
    const audio = { label: 'mine', speaker_audio_base64: b64('fake-mp3'), speaker_audio_name: 'voice.mp3' };
    const noSession = await call(api, 'POST', '/v2/speakers', audio);
    expect(noSession.status).toBe(400);
    expect(noSession.json().detail).toEqual(expect.any(String));
    const noAudio = await call(api, 'POST', '/v2/speakers', {}, SESSION);
    expect(noAudio.status).toBe(400);
    expect(noAudio.json().detail).toEqual(expect.any(String));
    expect((await call(api, 'POST', '/v2/speakers', { speaker_wav_base64: 'x' }, SESSION)).status).toBe(422);
    const created = await call(api, 'POST', '/v2/speakers', audio, SESSION, new Float32Array(44100).fill(0.2));
    expect(created.status).toBe(200);
    const sp = created.json<{ id: string }>();
    expect(validate('Speaker', sp)).toEqual([]);
    expect(sp).toMatchObject({ label: 'mine', scope: 'session', speaker_modes: ['reference', 'embedding'], has_preview: true, disabled_controls: [] });
    const mine = (await call(api, 'GET', '/v2/speakers', undefined, SESSION)).json<SpeakerList>();
    expect(validate('SpeakerList', mine)).toEqual([]);
    expect(mine.speakers.map((s: { id: string }) => s.id)).toContain(sp.id);
    expect((await call(api, 'GET', '/v2/speakers')).json<SpeakerList>().speakers.map((s: { id: string }) => s.id)).not.toContain(sp.id);
    const preview = await call(api, 'GET', `/v2/speakers/${engine.voices[0].id}/preview`);
    expect(preview.status).toBe(200);
    expect(preview.headers['content-type']).toBe('audio/wav');
    const emb = await call(api, 'GET', `/v2/speakers/${engine.voices[0].id}/embedding`);
    expect(emb.status).toBe(200);
    expect(validate('SpeakerEmbedding', emb.json())).toEqual([]);
    // a default voice answers its shipped embedding (0.2), not the speaker encoder's (the clip's first sample, 0.1)
    expect(emb.json<{ embedding: number[] }>().embedding).toEqual(new Array(192).fill(0.2));
    expect((await call(api, 'GET', '/v2/speakers/nope/embedding')).status).toBe(400);
    const gen = await call(api, 'POST', '/v2/generate', { text: 'Hi.', stream: false, speaker_embedding_id: sp.id }, SESSION);
    expect(gen.status).toBe(200);
  });

  it('speaker CFG prefills the twin without the speaker row', async () => {
    const { api, engine, backbone } = stubApi();
    const r = await call(api, 'POST', '/v2/generate', {
      text: 'Hi.', stream: false, speaker_mode: 'embedding', speaker_embedding_id: engine.voices[0].id, speaker_cfg_scale: 1.5,
    });
    expect(r.status).toBe(200);
    const [cond, twin] = backbone.prefills;
    expect(cond.slot).toBe(0);
    expect(cond.cond.speakerRow).toBe(0);
    expect(twin.slot).toBe(1);
    expect(twin.rows).toBe(cond.rows - 1);
    expect(twin.cond.buckets).toEqual(engine.hp.condBucketCounts);
  });

  it('controls CFG runs the same prompt as an unsteered twin', async () => {
    const { api, engine, backbone } = stubApi();
    const doc = (await call(api, 'GET', '/v2/controls')).json<{ directions: Record<string, { vector?: unknown }> }>();
    const hidden = Object.entries(doc.directions).filter(([, v]) => !('vector' in v)).map(([k]) => k);
    const r = await call(api, 'POST', '/v2/generate', {
      text: 'Hi.', stream: false, speaker_mode: 'embedding', speaker_embedding_id: engine.voices[0].id,
      controls: { [hidden[0]]: 0.5 }, controls_cfg_scale: 1.5,
    });
    expect(r.status).toBe(200);
    const [cond, twin] = backbone.prefills;
    expect(twin.slot).toBe(1);
    expect(twin.rows).toBe(cond.rows);
    expect(twin.cond.speakerRow).toBe(0);
    expect(backbone.steered.every(s => s !== null)).toBe(true);
    expect(backbone.twinSteered.length).toBeGreaterThan(0);
    expect(backbone.twinSteered.every(s => s === null)).toBe(true);
  });

  it('hidden controls steer decode steps; spans follow the rate schedule', async () => {
    const { api, backbone } = stubApi();
    const doc = (await call(api, 'GET', '/v2/controls')).json<{ directions: Record<string, { vector?: unknown }> }>();
    const hidden = Object.entries(doc.directions as Record<string, { vector?: unknown }>).filter(([, v]) => !('vector' in v)).map(([k]) => k);
    const r = await call(api, 'POST', '/v2/generate', {
      text: 'Hello there.', stream: false, controls: { [hidden[0]]: 0.5 },
      control_spans: [{ start: 0, end: 5, controls: { [hidden[0]]: 1.0 } }],
    });
    expect(r.status).toBe(200);
    expect(r.headers['x-control-span-follow']).toBe('rate');
    expect(backbone.steered.every(s => s !== null && s.length === 2 * 1024)).toBe(true);
    const span = await call(api, 'POST', '/v2/generate', {
      text: 'Hello there.', stream: false, control_spans: [{ start: 0, end: 50, controls: { [hidden[0]]: 1.0 } }],
    });
    expect(span.status).toBe(400);
  });

  it('LACI and reading follow attach the probe', async () => {
    const { api, backbone } = stubApi({ frames: 40 });
    const r = await call(api, 'POST', '/v2/generate', { text: 'Hello there, this is a short test.', stream: false, laci: true });
    expect(r.status).toBe(200);
    const probe = backbone.probes.find(p => p !== null)!;
    expect(probe.heads.length).toBeGreaterThan(0);
    expect(probe.textStart).toBe(3);   // no speaker row: [loudness][fmax][speaking_rate] then BOS at row 3
  });

  it('streamed pcm sends audio in chunks', async () => {
    const { api } = stubApi({ frames: 60 });
    const r = await call(api, 'POST', '/v2/generate', { text: 'Hello there.', seed: 2, stream_decode: 'prefix' });
    expect(r.status).toBe(200);
    expect(r.headers['content-type']).toBe('audio/pcm');
    expect(r.body.length % 4).toBe(0);
    expect(r.body.length).toBeGreaterThan(0);
  });
});
