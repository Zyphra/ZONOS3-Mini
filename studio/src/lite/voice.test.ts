import { describe, expect, it } from 'vitest';
import type { Speaker } from './api';
import { clipLengthProblem, readiness, selectedVoice, type VoiceState } from './voice';

const voice = (id: string): Speaker => ({ id, label: id, scope: 'session', speaker_modes: ['embedding'], has_preview: false, source_type: 'audio' }) as Speaker;
const base: VoiceState = { source: 'default', library: voice('lib'), clip: null, fileVoice: null, text: 'Hello', generating: false };

describe('lite Generate gating', () => {
  it('speaks only with a voice of the shown source', () => {
    expect(selectedVoice(base)?.id).toBe('lib');
    expect(selectedVoice({ ...base, source: 'clip' })).toBeNull();
    expect(selectedVoice({ ...base, source: 'embedding' })).toBeNull();
    expect(selectedVoice({ ...base, source: 'embedding', fileVoice: voice('file') })?.id).toBe('file');
  });

  it('is disabled in Clip mode without a clip, and clones first with one', () => {
    expect(readiness({ ...base, source: 'clip' })).toEqual({ ok: false, why: 'Record or upload a clip first' });
    const clip = { blob: new Blob(), name: 'x.wav', samples: 44100 * 5, speaker: null };
    expect(readiness({ ...base, source: 'clip', clip })).toEqual({ ok: true, cloneFirst: true });
    expect(readiness({ ...base, source: 'clip', clip: { ...clip, speaker: voice('c') } })).toEqual({ ok: true, cloneFirst: false });
  });

  it('needs text and is off while generating', () => {
    expect(readiness({ ...base, text: '  ' })).toEqual({ ok: false, why: 'Type something to say' });
    expect(readiness({ ...base, generating: true }).ok).toBe(false);
  });

  it('checks a decoded clip length against 2–30 s', () => {
    expect(clipLengthProblem(44100, 44100)).toMatch(/at least 2 s/);
    expect(clipLengthProblem(44100 * 31, 44100)).toMatch(/up to 30 s/);
    expect(clipLengthProblem(44100 * 10, 44100)).toBeNull();
    expect(clipLengthProblem(0, 44100)).toBeNull();
  });
});
