/** An offline backend for state tests: the bundled fixture spec and decoders, no server features. */
import type { Backend, ControlsSpec } from '../api';
import fixtureSpec from '../fixtures/controls.json';
import { FIXTURE_DECODERS } from '../fixtures/decoders';
import { backend } from '../state/app';

// The fixture JSON's `ui` block is the studio's own contract (same cast api.ts makes).
const spec = fixtureSpec as unknown as ControlsSpec;

export function offlineBackend(): Backend {
  return {
    kind: 'offline',
    server: '',
    spec,
    steering: true,
    edit: { speech_editing: false, edit_transcribe: false, edit_max_spans: 3, edit_frame_rate: 15, edit_max_seconds: 150, edit_align_max_seconds: 150, fixture: false },
    pron: { phoneme_tokens: null, phonemize_enabled: false, phonemize_languages: [], phonemize_guess_languages: [], fixture: false },
    projects: null,
    decoders: FIXTURE_DECODERS,
    laciLanguages: [],
    startToken: false,
  };
}

/** Install the offline backend into app state (idempotent). */
export function useOfflineBackend(): Backend {
  if (!backend.value) backend.value = offlineBackend();
  return backend.value;
}
