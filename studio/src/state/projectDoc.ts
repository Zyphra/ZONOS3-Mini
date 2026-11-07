/**
 * The project document (`schema` 1): what every act holds, as JSON, with audio / embeddings referred to by content
 * hash (`media.ts`). `snapshot` reads the signals into a document, `restore` puts one back. Undo history, drafts and
 * in-flight work are per session; voices the project created carry their clip or embedding so they can be
 * registered again. Keys older documents carry that the studio no longer has (a retired act's block) are ignored on
 * restore and dropped on the next save; the retired Scene act's lines join the script.
 */
import { batch } from '@preact/signals';
import type { Rendered } from '../api';
import * as player from '../audio/player';
import { holdPreview, previewText } from '../audio/preview';
import { base64ToBytes, bytesToBase64 } from '../lib/files';
import { parseWav } from '../lib/wav';
import { uid } from '../lib/math';
import { projectSettings } from './app';
import { designs, design, newDesign } from './design';
import * as edit from './edit';
import { cast, ownVoices, type OwnVoice } from './library';
import { MediaSink, prime } from './media';
import { load } from './persist';
import { lineBusy, lineError, lineRequest, newScript, script, scripts, takes, type Script, type Take } from './script';
import { voiceOrigins } from './who';
import { DEFAULT_SETTINGS, SCHEMA, summaryOf, type ProjectDoc, type ProjectHead, type ProjectSettings, type StoredAudio, type StoredEditDoc } from './projectSchema';

type OldScene = NonNullable<ProjectDoc['scene']>['scene'];

/**
 * A script as the studio holds it: lines carry only their own fields, a missing gap is filled (an old Scene's, else
 * 0.35), and an old Scene's lines are appended, each in its character's voice.
 */
function withScene(s: Script, scene: OldScene | undefined): Script {
  const lines = s.lines.map(({ id, text, spans, voice }) => ({ id, text, spans, ...(voice ? { voice } : {}) }));
  const sceneLines = (scene?.lines ?? []).map(({ id, characterId, text }) => ({ id, text, spans: [], voice: characterId }));
  return { ...s, lines: [...lines, ...sceneLines], gap: s.gap ?? scene?.gap ?? 0.35 };
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
    edit: { doc: null, clipLanguages: {} },
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
    // Scene lines (the retired act) join the script; they had no saved renders.
    direct: { script: withScene(load('script', d.direct.script), load<OldScene | undefined>('scene', undefined)), takes: {} },
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
    edit: { doc: editDoc, clipLanguages: edit.clipLanguages.value },
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
  // The retired Scene act's renders: each becomes its line's starred take, marked edited (its request was a
  // Scene request), so it isn't flagged stale while the text is unchanged.
  const sceneRenders: [string, Rendered][] = await Promise.all(Object.entries(doc.scene?.renders ?? {}).map(async ([id, x]) => [id, await audio(x.audio)] as [string, Rendered]));
  const e = doc.edit.doc;
  const KINDS: readonly string[] = ['take', 'voice', 'upload', 'recording'] satisfies edit.SourceKind[];
  const editDoc: edit.Doc | null = e && {
    ...e,
    // A clip from a retired act (older documents) opens as an upload.
    source: KINDS.includes(e.source.kind) ? e.source : { kind: 'upload', label: e.source.label },
    versions: await Promise.all(
      e.versions.map(async ({ media, ...v }) => {
        const { pcm, sr, bytes } = await wav(media);
        return { ...v, pcm, sr, b64: bytesToBase64(bytes) };
      }),
    ),
  };
  const scriptIn = withScene(doc.direct.script, doc.scene?.scene);

  return () =>
    batch(() => {
      player.stop();
      // Opening a project shows its Cast line without rendering it (pickBase, right after, is part of the load too).
      holdPreview();
      projectSettings.value = { ...DEFAULT_SETTINGS, ...doc.settings };
      ownVoices.value = voices;
      designs.reset(doc.cast.design);
      cast.value = doc.cast.personas;
      voiceOrigins.value = doc.cast.origins;
      previewText.value = doc.cast.previewText;
      scripts.reset(scriptIn);
      // After the cast, settings and backend are in (a migrated take's request is its line's).
      const migrated = sceneRenders.flatMap(([id, audio]): [string, Take[]][] => {
        const line = scriptIn.lines.find((l) => l.id === id);
        return line ? [[id, [{ id: uid(), text: line.text, seed: 0, request: lineRequest(line, 0), audio, createdAt: Date.now(), starred: true, edited: true }]]] : [];
      });
      takes.value = { ...takesIn, ...Object.fromEntries(migrated) };
      lineBusy.value = {};
      lineError.value = {};
      edit.clipLanguages.value = doc.edit.clipLanguages;
      edit.loadDoc(editDoc);
    });
}

/** Where each server voice id is used (Cast bench, cast members): what a missing voice breaks. */
export function speakerUses(doc: ProjectDoc): Map<string, Set<string>> {
  const uses = new Map<string, Set<string>>();
  const use = (id: string, where: string) => id && uses.set(id, (uses.get(id) ?? new Set()).add(where));
  use(doc.cast.design.baseId, 'the Cast bench');
  for (const p of doc.cast.personas) use(p.design.baseId, `cast member “${p.name}”`);
  return uses;
}
