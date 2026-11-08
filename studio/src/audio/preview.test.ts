import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { GenerateRequest, Rendered, Speaker } from '../api';
import * as api from '../api';
import { autoplay, autoRender, phrase, preview, previewText, previewTouched, side, startPreview } from './preview';
import { design, designs } from '../state/design';
import { speakers } from '../state/library';
import { useOfflineBackend } from '../test/backend';

// vi.mock is hoisted above the imports: `api.generate` is the stub everywhere.
vi.mock('../api', async (importOriginal) => {
  const mod = await importOriginal<typeof api>();
  const rendered = (req: GenerateRequest): Rendered => ({ pcm: new Float32Array(441), sr: 44100, ttfbMs: 1, request: req, words: null }) as unknown as Rendered;
  return { ...mod, generate: vi.fn(async (req: GenerateRequest) => rendered(req)) };
});

describe('preview phrase', () => {
  beforeEach(() => {
    useOfflineBackend();
    previewTouched.value = false;
    previewText.value = '';
  });

  it('falls back to the bundle text while nothing was typed', () => {
    expect(phrase.value).toBe(useOfflineBackend().spec.ui.preview_text);
  });

  it('stays empty once the reader cleared it', () => {
    previewTouched.value = true;
    previewText.value = '';
    expect(phrase.value).toBe('');
    previewText.value = 'Hello';
    expect(phrase.value).toBe('Hello');
  });

  it('a project load puts the fallback back', () => {
    previewTouched.value = true;
    previewTouched.value = false;
    previewText.value = '';
    expect(phrase.value).toBe(useOfflineBackend().spec.ui.preview_text);
  });
});

describe('preview follower', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('shows A again when a new request renders after B was heard', async () => {
    useOfflineBackend();
    const voice: Speaker = { id: 'v1', label: 'Voice one', is_default: true, speaker_modes: ['embedding', 'reference'], has_preview: false, source_type: 'audio' } as Speaker;
    speakers.value = [voice];
    autoplay.value = false;
    autoRender.value = true;
    previewTouched.value = true;
    previewText.value = 'First line';
    designs.reset({ ...design.value, baseId: 'v1', mode: 'embedding' });
    startPreview();
    await vi.advanceTimersByTimeAsync(300);
    side.value = 'B';
    previewText.value = 'Second line';
    await vi.advanceTimersByTimeAsync(300);
    expect(side.value).toBe('A');
    expect(preview.value.status).toBe('ready');
    expect(preview.value.a).not.toBeNull();
    expect(vi.mocked(api.generate)).toHaveBeenCalled();
  });
});
