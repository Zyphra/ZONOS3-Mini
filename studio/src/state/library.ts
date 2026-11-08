/** Voices: the server's speaker list, their embeddings, reference-clip previews, uploads and the saved cast. */
import { signal } from '@preact/signals';
import * as api from '../api';
import type { Speaker } from '../api';
import * as player from '../audio/player';
import { wavBytes } from '../lib/wav';
import { notify } from './app';
import type { Design } from './design';
import { errorText } from '../audio/render';

export const speakers = signal<Speaker[]>([]);
export const speakersError = signal<string | null>(null);
export const speakersLoading = signal(false);

/** Embeddings by speaker id; bumps `embeddingsVersion` whenever one lands. */
const embeddings = new Map<string, Float32Array>();
const loading = new Map<string, Promise<Float32Array | null>>();
export const embeddingsVersion = signal(0);
/** True when any embedding had to be synthesized (server without /v2/speakers/{id}/embedding). */
export const embeddingsSynthetic = signal(false);
export const embeddingErrors = signal<Record<string, string>>({});

export async function refreshSpeakers() {
  speakersLoading.value = true;
  try {
    speakers.value = await api.getSpeakers();
    speakersError.value = null;
  } catch (e) {
    speakersError.value = errorText(e);
  } finally {
    speakersLoading.value = false;
  }
}

export const speakerById = (id: string | null | undefined) => speakers.value.find((s) => s.id === id);

export function embeddingOf(id: string | null | undefined): Float32Array | null {
  void embeddingsVersion.value; // subscribe
  return (id && embeddings.get(id)) || null;
}

/** The speaker's embedding (fetched once; `retry` forgets a failed fetch and asks again). */
export function ensureEmbedding(s: Speaker, retry = false): Promise<Float32Array | null> {
  const have = embeddings.get(s.id);
  if (have) return Promise.resolve(have);
  if (retry && embeddingErrors.value[s.id]) {
    const rest = { ...embeddingErrors.value };
    delete rest[s.id];
    embeddingErrors.value = rest;
  }
  let p = loading.get(s.id);
  if (!p) {
    p = api
      .getEmbedding(s)
      .then(({ embedding, synthetic }) => {
        embeddings.set(s.id, embedding);
        if (synthetic) embeddingsSynthetic.value = true;
        embeddingsVersion.value++;
        return embedding;
      })
      .catch((e) => {
        embeddingErrors.value = { ...embeddingErrors.value, [s.id]: errorText(e) };
        return null;
      })
      .finally(() => loading.delete(s.id));
    loading.set(s.id, p);
  }
  return p;
}

/** Landmarks for the Who map: every voice's embedding, in parallel. */
export function loadAllEmbeddings() {
  for (const s of speakers.value) void ensureEmbedding(s);
}

// ── Reference-clip previews ───────────────────────────────────────────────────────────────────────

const clips = new Map<string, { pcm: Float32Array; sr: number }>();
export const clipLoading = signal<string | null>(null);

export async function toggleClip(s: Speaker) {
  const id = `clip:${s.id}`;
  if (player.isPlaying(id)) return player.stop();
  let clip = clips.get(s.id);
  if (!clip) {
    clipLoading.value = s.id;
    try {
      clip = await player.decode(await api.getPreview(s));
      clips.set(s.id, clip);
    } catch (e) {
      notify(`Couldn't play ${s.label}: ${errorText(e)}`, 'error');
      return;
    } finally {
      clipLoading.value = null;
    }
  }
  player.play(id, clip.pcm, clip.sr);
}

// ── Uploads and recordings ────────────────────────────────────────────────────────────────────────

export const uploading = signal(false);

/** Longest reference clip the servers take (`model/src/zonos3/registries/generation.json` audio_max_seconds.reference). */
export const REFERENCE_MAX_S = 60;
/** A recording stops itself here. */
export const RECORD_MAX_S = 30;

/** A clip (dropped, chosen or recorded) becomes a voice; `makeBase` says the toast should mention it became the base. */
export async function uploadClip(blob: Blob, label: string, fileName: string, makeBase = false): Promise<Speaker | null> {
  uploading.value = true;
  try {
    // Normalise whatever the browser recorded / the user dropped to a WAV the server surely reads.
    let pcm: Float32Array;
    let sr: number;
    try {
      ({ pcm, sr } = await player.decode(await blob.arrayBuffer()));
    } catch {
      notify('That file isn’t audio the browser can read (try WAV or MP3)', 'error');
      return null;
    }
    const seconds = pcm.length / sr;
    if (seconds < 2) {
      notify('Too short: record at least 5 s', 'error');
      return null;
    }
    if (seconds > REFERENCE_MAX_S) {
      notify(`Clip is ${Math.round(seconds)} s; the limit is ${REFERENCE_MAX_S} s`, 'error');
      return null;
    }
    const s = await saveClipVoice(label, wavBytes(pcm, sr).buffer, fileName.replace(/\.[^.]+$/, '') + '.wav');
    await refreshSpeakers();
    notify(makeBase ? `Added ${s.label} and made it the base voice` : `Added “${s.label}” to your voices`, 'ok');
    if (seconds < 5 || seconds > 30) notify(`Clip is ${Math.round(seconds)} s; 5–30 s of clean speech clones best`);
    return s;
  } catch (e) {
    notify(`Upload failed: ${errorText(e)}`, 'error');
    return null;
  } finally {
    uploading.value = false;
  }
}

// ── Voices the project made ───────────────────────────────────────────────────────────────────────

/**
 * A voice this project created on the server, with what recreates it: server voices live in the session and are
 * gone after a restart, so opening the project registers them again (new ids) from the clip or the embedding.
 */
export interface OwnVoice {
  id: string;
  label: string;
  /** The reference clip as uploaded (WAV bytes). */
  clip?: ArrayBuffer;
  /** A designed voice: the steered embedding. */
  embedding?: Float32Array;
}

export const ownVoices = signal<OwnVoice[]>([]);

/** Cache a clip (WAV bytes) as a voice of the project. */
export async function saveClipVoice(label: string, wav: ArrayBuffer, fileName: string): Promise<Speaker> {
  const s = await api.uploadVoice(label, new Blob([wav], { type: 'audio/wav' }), fileName);
  ownVoices.value = [...ownVoices.value, { id: s.id, label: s.label, clip: wav }];
  return s;
}

/** Cache a designed voice (its steered embedding) as a voice of the project. */
export async function saveEmbeddingVoice(label: string, embedding: Float32Array): Promise<Speaker> {
  const s = await api.saveEmbeddingVoice(label, embedding);
  ownVoices.value = [...ownVoices.value, { id: s.id, label: s.label, embedding: embedding.slice() }];
  return s;
}

export const recording = signal<{ started: number; stop: () => void } | null>(null);

export async function startRecording(onDone: (s: Speaker | null) => void) {
  if (!navigator.mediaDevices?.getUserMedia) {
    notify('Recording needs HTTPS (or localhost)', 'error');
    return;
  }
  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    const denied = e instanceof DOMException && (e.name === 'NotAllowedError' || e.name === 'SecurityError');
    notify(denied ? 'Allow the microphone in the browser’s site settings' : `Microphone unavailable: ${errorText(e)}`, 'error');
    return;
  }
  const rec = new MediaRecorder(stream);
  const parts: Blob[] = [];
  rec.ondataavailable = (ev) => parts.push(ev.data);
  rec.onstop = async () => {
    stream.getTracks().forEach((t) => t.stop());
    recording.value = null;
    const blob = new Blob(parts, { type: rec.mimeType });
    const when = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    onDone(await uploadClip(blob, `Recording ${when}`, 'recording.webm', true));
  };
  rec.start();
  const auto = setTimeout(() => rec.state === 'recording' && rec.stop(), RECORD_MAX_S * 1000);
  recording.value = {
    started: Date.now(),
    stop: () => {
      clearTimeout(auto);
      if (rec.state === 'recording') rec.stop();
    },
  };
}

// ── Saved cast (personas) ─────────────────────────────────────────────────────────────────────────

export interface Persona {
  id: string;
  name: string;
  design: Design;
  signature: string;
  createdAt: number;
}

export const cast = signal<Persona[]>([]);

export function savePersona(p: Persona) {
  cast.value = [...cast.value.filter((x) => x.id !== p.id), p];
}

/** Remove a persona; returns it (and its index) so the caller can offer Undo. */
export function removePersona(id: string): { persona: Persona; index: number } | null {
  const index = cast.value.findIndex((x) => x.id === id);
  if (index < 0) return null;
  const persona = cast.value[index];
  cast.value = cast.value.filter((x) => x.id !== id);
  return { persona, index };
}

/** Put a removed persona back where it was. */
export function restorePersona(p: Persona, index: number) {
  const rest = cast.value.filter((x) => x.id !== p.id);
  cast.value = [...rest.slice(0, index), p, ...rest.slice(index)];
}

export function renamePersona(id: string, name: string) {
  const n = name.trim();
  if (!n) return;
  cast.value = cast.value.map((x) => (x.id === id ? { ...x, name: n } : x));
}
