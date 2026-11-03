import { describe, expect, it } from 'vitest';
import { parseBody, ZonosError } from '../src/zonos';

describe('parseBody', () => {
  it('turns a keep-alive error body into the error it reports', () => {
    let err: unknown;
    try {
      parseBody(200, '   {"detail":"bad","status_code":400}');
    } catch (e) {
      err = e;
    }
    if (!(err instanceof ZonosError)) throw new Error('expected a ZonosError');
    expect(err.status).toBe(400);
    expect(err.detail).toBe('bad');
  });

  it('reports a proxy page', () => {
    expect(() => parseBody(200, '<html><body>Bad gateway</body></html>')).toThrow('The server sent a web page instead of an answer (a proxy error?)');
  });

  it('returns the JSON of a success', () => {
    expect(parseBody(200, '  {"ok":true}')).toEqual({ ok: true });
  });
});
