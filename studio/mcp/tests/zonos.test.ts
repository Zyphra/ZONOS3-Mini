import { describe, expect, it } from 'vitest';
import { ApiError, bodyOf } from '../../src/lib/http';

describe('bodyOf', () => {
  it('turns a keep-alive error body into the error it reports', () => {
    let err: unknown;
    try {
      bodyOf(200, '   {"detail":"bad","status_code":400}');
    } catch (e) {
      err = e;
    }
    if (!(err instanceof ApiError)) throw new Error('expected a ApiError');
    expect(err.status).toBe(400);
    expect(err.detail).toBe('bad');
  });

  it('reports a proxy page', () => {
    expect(() => bodyOf(200, '<html><body>Bad gateway</body></html>')).toThrow(ApiError);
  });

  it('returns the JSON of a success', () => {
    expect(bodyOf(200, '  {"ok":true}')).toEqual({ ok: true });
  });
});
