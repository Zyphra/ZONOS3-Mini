/**
 * Which voice Generate speaks with on the lite page, and why it can't yet: pure decisions over the page's state
 * (the DOM wiring is in `main.ts`). Generate never speaks with a voice from another source than the one shown.
 */
import type { Speaker } from './api';

export type Source = 'default' | 'clip' | 'embedding';

export interface Clip {
  blob: Blob;
  name: string;
  /** Decoded length in samples (0 when the browser couldn't decode it). */
  samples: number;
  /** The voice this clip was cloned into, once it was. */
  speaker: Speaker | null;
}

export interface VoiceState {
  source: Source;
  /** The library voice picked in the select. */
  library: Speaker | null;
  clip: Clip | null;
  /** The voice the embedding file was opened into. */
  fileVoice: Speaker | null;
  text: string;
  generating: boolean;
}

/** Clip length limits (seconds): the server refuses shorter; longer than 30 s clones no better. */
export const CLIP_MIN_S = 2;
export const CLIP_MAX_S = 30;

/** The voice the chosen source has right now (null: nothing to speak with yet). */
export function selectedVoice(s: VoiceState): Speaker | null {
  switch (s.source) {
    case 'default':
      return s.library;
    case 'clip':
      return s.clip?.speaker ?? null;
    case 'embedding':
      return s.fileVoice;
  }
}

export type Readiness = { ok: true; cloneFirst: boolean } | { ok: false; why: string };

/** Whether Generate may run: with a voice, or (Clip mode) with a clip that it will clone first. */
export function readiness(s: VoiceState): Readiness {
  if (s.generating) return { ok: false, why: '' };
  if (!s.text.trim()) return { ok: false, why: 'Type something to say' };
  if (selectedVoice(s)) return { ok: true, cloneFirst: false };
  if (s.source === 'clip') return s.clip ? { ok: true, cloneFirst: true } : { ok: false, why: 'Record or upload a clip first' };
  if (s.source === 'embedding') return { ok: false, why: 'Open a voice file first' };
  return { ok: false, why: 'Pick a voice first' };
}

/** A decoded clip's length problem, or null when it clones fine. */
export function clipLengthProblem(samples: number, sampleRate: number): string | null {
  if (!samples) return null;
  const s = samples / sampleRate;
  if (s < CLIP_MIN_S) return `Too short: record at least ${CLIP_MIN_S} s`;
  if (s > CLIP_MAX_S) return `Clip is ${Math.round(s)} s; use up to ${CLIP_MAX_S} s`;
  return null;
}
