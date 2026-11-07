/**
 * Studio projects on the server (`/v2/projects`, `/v2/media`): load (re-registering voices the server lost, as the
 * studio's `reconcile` does), patch, save against the loaded revision, retry on conflicts.
 */
import { ApiError, MissingMedia, RevisionConflict } from '../../src/lib/http';
import { createHash } from 'node:crypto';
import { encodeNpy, slug } from '../../src/lib/files';
import { uid } from '../../src/lib/math';
import { homePin } from '../../src/state/atlas';
import type { Act } from '../../src/state/app';
import { DEFAULT_SETTINGS, remapIds, SCHEMA, summaryOf, type ProjectDoc } from '../../src/state/projectSchema';
import { caps } from './backend';
import { config } from './config';
import { state } from './state';
import * as zonos from './zonos';

export const NOT_SAVED = 'Not saved: the server keeps no projects (start it with ZONOS3_STUDIO_PROJECTS=<dir>).';
export const RELOAD_HINT = 'If this project is open in Studio, choose Reload when it says it changed elsewhere.';

export const sha256 = (bytes: Uint8Array) => createHash('sha256').update(bytes).digest('hex');
export const b64 = (bytes: Uint8Array | ArrayBuffer) => Buffer.from(bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes)).toString('base64');

/** A new project as the studio's `emptyDoc` builds it, minus the demo script. */
export function newProjectDoc(id: string, name: string, language: string): ProjectDoc {
  const now = Date.now();
  return {
    schema: SCHEMA,
    id,
    name,
    createdAt: now,
    updatedAt: now,
    settings: { ...DEFAULT_SETTINGS, language },
    voices: [],
    cast: { design: { baseId: '', mode: 'embedding', values: {}, pin: homePin() }, personas: [], origins: {}, previewText: '' },
    direct: { script: { lines: [], baseline: {}, voice: null, gap: 0.35 }, takes: {} },
    edit: { doc: null, clipLanguages: {} },
    media: {},
  };
}

/** Media a patch adds: recorded in `doc.media` and uploaded before the document is saved. */
export class MediaBag {
  readonly pending = new Map<string, { bytes: Uint8Array; type: string }>();
  constructor(private doc: ProjectDoc) {}
  add(bytes: Uint8Array, type: string): string {
    const sha = sha256(bytes);
    this.pending.set(sha, { bytes, type });
    this.doc.media[sha] ??= { type, size: bytes.length };
    return sha;
  }
}

export interface ProjectRef {
  id: string;
  name: string;
  revision: number;
  url: string;
}

export interface Session {
  /** null: the server keeps no projects (the document lives in this process only). */
  id: string | null;
  doc: ProjectDoc;
  revision: number | null;
  /** Saved voice id → live id. */
  map: Map<string, string>;
  warnings: string[];
}

/** The document while the server keeps no projects (per process, never saved). */
let memoryDoc: ProjectDoc | null = null;

export const sessionName = () => {
  const d = new Date();
  const p = (n: number) => String(n).padStart(2, '0');
  return `MCP session ${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
};

/** Register the project's voices the server no longer has; returns saved → live. */
async function reconcile(projectId: string, doc: ProjectDoc, warnings: string[]): Promise<Map<string, string>> {
  if (!doc.voices.length) return new Map();
  const live = new Set((await zonos.speakers()).map((s) => s.id));
  const known = state.voiceMaps.get(projectId) ?? new Map<string, string>();
  const map = new Map<string, string>();
  for (const v of doc.voices) {
    if (live.has(v.id)) continue;
    const mapped = known.get(v.id);
    if (mapped && live.has(mapped)) {
      map.set(v.id, mapped);
      continue;
    }
    try {
      const bytes = await zonos.getMedia(v.media);
      const s =
        v.kind === 'clip'
          ? await zonos.createSpeaker({ label: v.label, speaker_audio_base64: `data:${doc.media[v.media]?.type ?? 'audio/wav'};base64,${b64(bytes)}`, speaker_audio_name: `${slug(v.label)}.wav` })
          : await zonos.createSpeaker({
              label: v.label,
              speaker_embedding_base64: `data:application/octet-stream;base64,${b64(encodeNpy(new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength))))}`,
              speaker_embedding_name: `${slug(v.label)}.npy`,
            });
      map.set(v.id, s.id);
    } catch (e) {
      warnings.push(`Voice "${v.label}" could not be registered again: ${e instanceof Error ? e.message : String(e)}`);
    }
  }
  state.voiceMaps.set(projectId, new Map([...known, ...map]));
  return map;
}

/** Load a project (the active one, else a new session project); `create: false` throws when there is none. */
export async function openProject(projectId?: string, opts: { create?: boolean } = {}): Promise<Session> {
  const c = await caps();
  if (!c.projects.enabled) {
    memoryDoc ??= newProjectDoc(uid(), sessionName(), DEFAULT_SETTINGS.language);
    return { id: null, doc: memoryDoc, revision: null, map: new Map(), warnings: [] };
  }
  const id = projectId ?? state.activeProjectId;
  if (!id) {
    if (opts.create === false) throw new Error('No project is open: create or open one with zonos_project.');
    // Becomes the active project once saved (saveProject).
    const nid = uid();
    return { id: nid, doc: newProjectDoc(nid, sessionName(), DEFAULT_SETTINGS.language), revision: null, map: new Map(), warnings: [] };
  }
  let p;
  try {
    p = await zonos.getProject(id);
  } catch (e) {
    if (e instanceof ApiError && e.status === 404) throw new Error(`Unknown project '${id}' (see zonos_project action=list)`);
    throw e;
  }
  state.activeProjectId = id;
  const warnings: string[] = [];
  const saved = p.data as unknown as ProjectDoc;
  const map = await reconcile(id, saved, warnings);
  const doc = map.size ? (JSON.parse(remapIds(JSON.stringify(saved), map)) as ProjectDoc) : saved;
  return { id, doc, revision: p.revision, map, warnings };
}

/** Run `patch` on the session's document and save it (conflicts: reload and patch again, up to 3 times). */
export async function saveProject<T>(s: Session, act: Act, patch: (doc: ProjectDoc, media: MediaBag) => T): Promise<{ result: T; project: ProjectRef | null; session: Session }> {
  for (let attempt = 1; ; attempt++) {
    const media = new MediaBag(s.doc);
    const result = patch(s.doc, media);
    if (s.id === null) return { result, project: null, session: s };
    s.doc.updatedAt = Date.now();
    s.doc.summary = summaryOf(s.doc);
    const unmap = new Map([...s.map].map(([saved, live]) => [live, saved]));
    const data = JSON.parse(remapIds(JSON.stringify(s.doc), unmap)) as ProjectDoc;
    const upload = async (shas: Iterable<string>) => {
      for (const sha of shas) {
        const m = media.pending.get(sha);
        if (m && !(await zonos.hasMedia(sha))) await zonos.putMedia(sha, m.bytes, m.type);
      }
    };
    await upload(media.pending.keys());
    const body = { name: s.doc.name, data: data as unknown as Record<string, unknown>, base_revision: s.revision };
    try {
      let saved;
      try {
        saved = await zonos.putProject(s.id, body);
      } catch (e) {
        if (!(e instanceof MissingMedia)) throw e;
        await upload(e.missing);
        saved = await zonos.putProject(s.id, body);
      }
      s.revision = saved.revision;
      state.activeProjectId = s.id;
      return { result, project: { id: s.id, name: s.doc.name, revision: saved.revision, url: `${config.server}/studio?project=${encodeURIComponent(s.id)}#${act}` }, session: s };
    } catch (e) {
      if (!(e instanceof RevisionConflict)) throw e;
      if (attempt >= 3) throw new Error('Project changed elsewhere 3 times in a row; try again');
      const fresh = await openProject(s.id);
      s = { ...fresh, warnings: [...s.warnings, ...fresh.warnings] };
    }
  }
}

/** Open (or create) and patch in one go. */
export async function withProject<T>(projectId: string | undefined, act: Act, patch: (doc: ProjectDoc, media: MediaBag) => T) {
  return saveProject(await openProject(projectId), act, patch);
}

/** The project line every saving tool ends with. */
export function projectLine(project: ProjectRef | null): string {
  return project ? `Saved to “${project.name}” rev ${project.revision} → ${project.url}\n${RELOAD_HINT}` : NOT_SAVED;
}
