// Keepalive wire mode of POST /v2/edit[/...] (model/src/zonos3/server/keepalive.py): 200 and a space at once, then the
// JSON; an error status travels in the body as {"detail", "status_code"}.
import { describe, expect, it } from 'vitest';
import { validate } from '../../src/api/validate';
import { call, haveEditModels, stubApi } from './stub';

const SR = 44100;
const clip = (seconds: number) => new Float32Array(Math.round(seconds * SR)).fill(0.1);
const KA = { 'x-keepalive': '1' };

describe.skipIf(!haveEditModels)('keepalive', () => {
  it('streams a space, then the route body under 200 with x-keepalive', async () => {
    const { api } = stubApi();
    const r = await call(api, 'POST', '/v2/edit/align', { audio_base64: 'AAAA', text: 'Hello there.' }, KA, clip(2));
    expect(r.status).toBe(200);
    expect(r.headers['x-keepalive']).toBe('1');
    expect(r.headers['content-type']).toBe('application/json');
    expect(r.body[0]).toBe(0x20);
    expect(validate('EditTranscript', r.json())).toEqual([]);
  });

  it('puts an error status into the 200 body', async () => {
    const { api } = stubApi();
    const r = await call(api, 'POST', '/v2/edit/align', { audio_base64: 'AAAA', text: '' }, KA, clip(2));
    expect(r.status).toBe(200);
    expect(r.body[0]).toBe(0x20);
    expect(r.json()).toEqual({ detail: 'text is empty.', status_code: 400 });
  });

  it('leaves requests without the header unchanged', async () => {
    const { api } = stubApi();
    const r = await call(api, 'POST', '/v2/edit/align', { audio_base64: 'AAAA', text: '' }, {}, clip(2));
    expect(r.status).toBe(400);
    expect(r.headers['x-keepalive']).toBeUndefined();
    expect(r.json()).toEqual({ detail: 'text is empty.' });
  });
});
