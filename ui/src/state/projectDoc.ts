/**
 * The project document (`schema` 1): what every act holds, as JSON, with audio / video / embeddings referred to by
 * content hash (`media.ts`). `snapshot` reads the signals into a document, `restore` puts one back. Undo history,
 * drafts and in-flight work are per session; server ids that die with the server session (dub ref / dub ids) are
 * not kept, and voices the project created carry their clip or embedding so they can be registered again.
 */
import { batch } from '@preact/signals';
import type { Rendered } from '../api';
import * as player from '../audio/player';
import { previewText } from '../audio/preview';
import { base64ToBytes, bytesToBase64, parseWav } from '../lib/files';
import { DEFAULT_SETTINGS, projectSettings, type ProjectSettings } from './app';
import { designs, design, newDesign, type Design } from './design';
import * as dub from './dub';
import * as edit from './edit';
import { cast, ownVoices, type OwnVoice, type Persona } from './library';
import { MediaSink, prime, type MediaInfo } from './media';
import { load } from './persist';
import { newScene, renders, scene, scenes, type LineRender, type Scene } from './scene';
import { lineBusy, lineError, newScript, script, scripts, takes, type Script, type Take } from './script';
import { voiceOrigins, type Spot } from './who';

export const SCHEMA = 1;

/** A server voice the project created and the media that recreates it (WAV clip, or float32 embedding). */
export interface VoiceRecord {
  id: string;
  label: string;
  kind: 'clip' | 'embedding';
  media: string;
}

type StoredAudio = Omit<Rendered, 'pcm' | 'sr'> & { media: string };
type StoredTake = Omit<Take, 'audio'> & { audio: StoredAudio };
type StoredVersion = Omit<edit.Version, 'pcm' | 'sr' | 'b64'> & { media: string };
type StoredEditDoc = Omit<edit.Doc, 'versions'> & { versions: StoredVersion[] };
type StoredSource = Omit<dub.DubSource, 'pcm' | 'sr' | 'video'> & { media: string; video: { media: string; name: string; type: string } | null };
/** `dubId` is always '': the server's dub (its takes) is gone with its session. */
type StoredTrack = Omit<dub.DubTrack, 'pcm' | 'sr'> & { media: string };
type StoredDub = Omit<dub.DubDoc, 'source' | 'refId' | 'track' | 'tracks'> & { source: StoredSource | null; trackId: string | null; tracks: StoredTrack[] };

/** What a project holds, for lists (the server's `GET /projects` copies it from the document). */
export interface ProjectSummary {
  voices: number;
  cast: number;
  lines: number;
  takes: number;
  sceneLines: number;
  versions: number;
  dubTracks: number;
  video: boolean;
}

export function summaryOf(doc: ProjectDoc): ProjectSummary {
  return {
    voices: doc.voices.length,
    cast: doc.cast.personas.length,
    lines: doc.direct.script.lines.length,
    takes: Object.values(doc.direct.takes).reduce((n, t) => n + t.length, 0),
    sceneLines: doc.scene.scene.lines.length,
    versions: doc.edit.doc?.versions.length ?? 0,
    dubTracks: doc.dub.tracks.length,
    video: !!doc.dub.source?.video,
  };
}

export interface ProjectDoc {
  schema: number;
  id: string;
  name: string;
  createdAt: number;
  updatedAt: number;
  settings: ProjectSettings;
  voices: VoiceRecord[];
  cast: { design: Design; personas: Persona[]; origins: Record<string, Spot>; previewText: string };
  /** Takes by line id (lines of the script only). */
  direct: { script: Script; takes: Record<string, StoredTake[]> };
  /** The last render of each scene line, by line id. */
  scene: { scene: Scene; renders: Record<string, { key: string; audio: StoredAudio }> };
  edit: { doc: StoredEditDoc | null; clipLanguages: Record<string, string> };
  dub: StoredDub;
  /** Every media hash the document refers to (SHA-256 hex; the server checks they exist before it stores the document). */
  media: Record<string, MediaInfo>;
  /** Written on save for project lists. */
  summary?: ProjectSummary;
}

export type ProjectHead = Pick<ProjectDoc, 'id' | 'name' | 'createdAt' | 'updatedAt'>;

/** The dub without its source (stored by the caller) and server ids. */
function storedDub(d: dub.DubDoc, sink: MediaSink): StoredDub {
  const { source: _source, refId: _ref, track, tracks, ...rest } = d;
  return {
    ...rest,
    source: null,
    trackId: track?.id ?? null,
    tracks: tracks.map(({ pcm, sr, ...t }) => ({ ...t, dubId: '', media: sink.audio(pcm, sr) })),
  };
}

export function emptyDoc(head: Pick<ProjectDoc, 'id' | 'name'>, settings: ProjectSettings): ProjectDoc {
  const now = Date.now();
  return {
    schema: SCHEMA,
    id: head.id,
    name: head.name,
    createdAt: now,
    updatedAt: now,
    settings,
    voices: [],
    cast: { design: newDesign(settings.voiceMode), personas: [], origins: {}, previewText: '' },
    direct: { script: newScript(), takes: {} },
    scene: { scene: newScene(), renders: {} },
    edit: { doc: null, clipLanguages: {} },
    dub: storedDub(dub.EMPTY_DUB, new MediaSink()),
    media: {},
  };
}

/** What the studio kept in localStorage before projects existed (`zvs.<key>`), read into a first project. */
export const LEGACY_KEYS = ['design', 'cast', 'voiceOrigins', 'previewText', 'script', 'scene', 'edit.languages', 'pron.language'];

export function legacyDoc(head: Pick<ProjectDoc, 'id' | 'name'>): ProjectDoc {
  const d = emptyDoc(head, { ...DEFAULT_SETTINGS, language: load('pron.language', 'en_us') });
  return {
    ...d,
    cast: { design: load('design', d.cast.design), personas: load('cast', []), origins: load('voiceOrigins', {}), previewText: load('previewText', '') },
    direct: { script: load('script', d.direct.script), takes: {} },
    scene: { scene: load('scene', d.scene.scene), renders: {} },
    edit: { doc: null, clipLanguages: load('edit.languages', {}) },
  };
}

/** The open project as a document; media not yet stored land in `sink.pending`. */
export async function snapshot(head: ProjectHead, sink: MediaSink): Promise<ProjectDoc> {
  const audio = ({ pcm, sr, ...rest }: Rendered): StoredAudio => ({ ...rest, media: sink.audio(pcm, sr) });
  // Edit versions first: their exact WAV (what the server got) is the copy kept when a take shares their audio.
  const e = edit.doc.value;
  const editDoc: StoredEditDoc | null = e && { ...e, versions: e.versions.map(({ pcm, sr, b64, ...v }) => ({ ...v, media: sink.audio(pcm, sr, () => base64ToBytes(b64)) })) };
  const s = script.value;
  const t = takes.value;
  const sc = scene.value;
  const r = renders.value;
  const d = dub.doc.value;
  const src = d.source;
  let source: StoredSource | null = null;
  if (src) {
    const { pcm, sr, video, ...rest } = src;
    source = { ...rest, media: sink.audio(pcm, sr), video: video && { media: await sink.file(video.file), name: video.file.name, type: video.file.type } };
  }
  const doc: ProjectDoc = {
    schema: SCHEMA,
    // Field by field: the caller's head is usually a whole ProjectMeta.
    id: head.id,
    name: head.name,
    createdAt: head.createdAt,
    updatedAt: head.updatedAt,
    settings: projectSettings.value,
    voices: ownVoices.value.map((v) => ({ id: v.id, label: v.label, kind: v.clip ? 'clip' : 'embedding', media: v.clip ? sink.wav(v.clip) : sink.embedding(v.embedding!) })),
    cast: { design: design.value, personas: cast.value, origins: voiceOrigins.value, previewText: previewText.value },
    direct: { script: s, takes: Object.fromEntries(s.lines.filter((l) => t[l.id]?.length).map((l) => [l.id, t[l.id].map((x) => ({ ...x, audio: audio(x.audio) }))])) },
    scene: { scene: sc, renders: Object.fromEntries(sc.lines.flatMap((l) => (r[l.id]?.audio ? [[l.id, { key: r[l.id].key, audio: audio(r[l.id].audio!) }]] : []))) },
    edit: { doc: editDoc, clipLanguages: edit.clipLanguages.value },
    dub: { ...storedDub(d, sink), source },
    // Filled while the fields above were built.
    media: sink.info,
  };
  return { ...doc, summary: summaryOf(doc) };
}

/**
 * Read a document's media (`read` gives a stored blob by hash) and return the step that swaps it in: one batch
 * that stops playback, aborts what runs and resets every act (undo history included) to the project.
 */
export async function restore(doc: ProjectDoc, read: (hash: string) => Promise<Blob>): Promise<() => void> {
  const wavs = new Map<string, Promise<{ pcm: Float32Array; sr: number; bytes: ArrayBuffer }>>();
  const wav = (h: string) => {
    let p = wavs.get(h);
    if (!p) {
      p = read(h)
        .then((b) => b.arrayBuffer())
        .then((bytes) => {
          const a = parseWav(bytes);
          prime(a.pcm, h);
          return { ...a, bytes };
        });
      wavs.set(h, p);
    }
    return p;
  };
  const audio = async ({ media, ...rest }: StoredAudio): Promise<Rendered> => {
    const { pcm, sr } = await wav(media);
    return { ...rest, pcm, sr };
  };

  const voices = await Promise.all(
    doc.voices.map(async (v): Promise<OwnVoice> => {
      const buf = await (await read(v.media)).arrayBuffer();
      if (v.kind === 'clip') {
        prime(buf, v.media);
        return { id: v.id, label: v.label, clip: buf };
      }
      const embedding = new Float32Array(buf);
      prime(embedding, v.media);
      return { id: v.id, label: v.label, embedding };
    }),
  );
  const takesIn: Record<string, Take[]> = Object.fromEntries(
    await Promise.all(Object.entries(doc.direct.takes).map(async ([id, ts]) => [id, await Promise.all(ts.map(async (x) => ({ ...x, audio: await audio(x.audio) })))] as const)),
  );
  const rendersIn: Record<string, LineRender> = Object.fromEntries(
    await Promise.all(Object.entries(doc.scene.renders).map(async ([id, x]) => [id, { key: x.key, audio: await audio(x.audio), busy: false, error: null }] as const)),
  );
  const e = doc.edit.doc;
  const editDoc: edit.Doc | null = e && {
    ...e,
    versions: await Promise.all(
      e.versions.map(async ({ media, ...v }) => {
        const { pcm, sr, bytes } = await wav(media);
        return { ...v, pcm, sr, b64: bytesToBase64(bytes) };
      }),
    ),
  };
  const sd = doc.dub;
  const { source: ss, trackId, tracks: st, ...dubRest } = sd;
  const tracks: dub.DubTrack[] = await Promise.all(
    st.map(async ({ media, ...x }) => {
      const { pcm, sr } = await wav(media);
      return { ...x, pcm, sr };
    }),
  );
  let source: Omit<dub.DubSource, 'video'> | null = null;
  let videoFile: File | null = null;
  if (ss) {
    const { media, video, ...rest } = ss;
    const { pcm, sr } = await wav(media);
    source = { ...rest, pcm, sr };
    if (video) {
      videoFile = new File([await read(video.media)], video.name, { type: video.type });
      prime(videoFile, video.media);
    }
  }

  return () =>
    batch(() => {
      player.stop();
      projectSettings.value = { ...DEFAULT_SETTINGS, ...doc.settings };
      ownVoices.value = voices;
      designs.reset(doc.cast.design);
      cast.value = doc.cast.personas;
      voiceOrigins.value = doc.cast.origins;
      previewText.value = doc.cast.previewText;
      scripts.reset(doc.direct.script);
      takes.value = takesIn;
      lineBusy.value = {};
      lineError.value = {};
      scenes.reset(doc.scene.scene);
      renders.value = rendersIn;
      edit.clipLanguages.value = doc.edit.clipLanguages;
      edit.loadDoc(editDoc);
      dub.loadDoc({
        ...dubRest,
        refId: null,
        source: source && { ...source, video: videoFile && { url: URL.createObjectURL(videoFile), file: videoFile } },
        tracks,
        track: tracks.find((x) => x.id === trackId) ?? null,
      });
    });
}

/** Where each server voice id is used (Cast bench, cast members, dub voices): what a missing voice breaks. */
export function speakerUses(doc: ProjectDoc): Map<string, Set<string>> {
  const uses = new Map<string, Set<string>>();
  const use = (id: string, where: string) => id && uses.set(id, (uses.get(id) ?? new Set()).add(where));
  const voice = (v: dub.VoiceChoice | null, where: string) => v && v.kind !== 'source' && use(v.kind === 'speaker' ? v.id : v.design.baseId, where);
  use(doc.cast.design.baseId, 'the Cast bench');
  for (const p of doc.cast.personas) use(p.design.baseId, `cast member “${p.name}”`);
  voice(doc.dub.voice, 'the dub');
  doc.dub.segments.forEach((s) => voice(s.voice, 'dub segments'));
  doc.dub.source?.lines.forEach((l) => voice(l.voice, 'the dub'));
  return uses;
}

/**
 * Replace server voice ids everywhere in a document's JSON: ids are random tokens (`spk_…`, offline `local_…`),
 * so a whole-token match only hits them — as values, map keys (voice origins) and inside derived strings (the
 * requests of takes and render keys, dub voice keys), which keeps every take and render matching its line.
 */
export function remapIds(json: string, map: Map<string, string>): string {
  if (!map.size) return json;
  const esc = [...map.keys()].map((k) => k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
  return json.replace(new RegExp(`(?<![\\w-])(${esc.join('|')})(?![\\w-])`, 'g'), (id) => map.get(id)!);
}
