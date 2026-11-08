/**
 * Where projects live: the server (`/v2/projects` + `/v2/media`, shared by every browser with the studio link) when it
 * offers them, else this browser (IndexedDB `zvs-projects`). Both keep a revision per project, so a save from a
 * stale copy (another tab, another browser) is refused with RevisionConflict instead of overwriting.
 */
import * as api from '../api';
import { MissingMedia, RevisionConflict } from '../lib/http';
import { type Revision, type ServerProjectEntry } from '../api';
import { hash128, isSha256, sha256Async } from '../lib/hash';
import { committed, openDb, request } from '../lib/idb';
import { stored } from './media';
import { summaryOf, type ProjectDoc, type ProjectSummary } from './projectSchema';

export interface ProjectMeta {
  id: string;
  name: string;
  createdAt: number;
  updatedAt: number;
  /** When this browser last opened it (browser store); the server list orders by updatedAt. */
  openedAt: number;
  /** Bytes: the document plus every media object it refers to (shared media count in each project). */
  size: number;
  summary: ProjectSummary | null;
  /** The revision this copy was read or saved at (null: never saved). */
  revision: Revision | null;
}

export interface Store {
  readonly kind: 'server' | 'browser';
  list(): Promise<ProjectMeta[]>;
  read(id: string): Promise<{ json: string; revision: Revision | null }>;
  media(hash: string): Promise<Blob>;
  /** Store `doc` (as `json`) with the media it needs that the store may lack, if the stored revision is still `base`. */
  write(doc: ProjectDoc, json: string, media: Map<string, Blob>, base: Revision | null): Promise<ProjectMeta>;
  remove(id: string): Promise<void>;
  /** Remember the project was opened now (orders the browser list). */
  opened(meta: ProjectMeta): Promise<void>;
}

export function metaOf(doc: ProjectDoc, json: string, revision: Revision | null, openedAt = Date.now()): ProjectMeta {
  const media = Object.values(doc.media);
  return {
    id: doc.id,
    name: doc.name,
    createdAt: doc.createdAt,
    updatedAt: doc.updatedAt,
    openedAt,
    size: json.length + media.reduce((n, m) => n + m.size, 0),
    summary: doc.summary ?? summaryOf(doc),
    revision,
  };
}

// ── Browser (IndexedDB) ───────────────────────────────────────────────────────────────────────────

interface MediaRecord {
  hash: string;
  type: string;
  size: number;
  blob: Blob;
}

/** What the browser store keeps per project besides the document (its media list drives the sweep). */
type BrowserMeta = ProjectMeta & { media: string[] };

export class BrowserStore implements Store {
  readonly kind = 'browser';
  private constructor(private readonly db: IDBDatabase) {}

  /** `track`: this is the open store, so the media it holds count as stored for snapshots. */
  static async open(track: boolean): Promise<BrowserStore> {
    const db = await openDb('zvs-projects', 1, (d) => {
      d.createObjectStore('meta', { keyPath: 'id' });
      d.createObjectStore('docs');
      d.createObjectStore('media', { keyPath: 'hash' });
    });
    if (track) {
      const media = await request<MediaRecord[]>(db.transaction('media').objectStore('media').getAll());
      for (const m of media) stored.set(m.hash, m.size);
    }
    return new BrowserStore(db);
  }

  /** How many projects the browser holds, without creating its database. */
  static async count(): Promise<number> {
    try {
      if (indexedDB.databases && !(await indexedDB.databases()).some((d) => d.name === 'zvs-projects')) return 0;
      const s = await BrowserStore.open(false);
      const n = (await s.list()).length;
      s.db.close();
      return n;
    } catch {
      return 0;
    }
  }

  private tx(stores: string[], mode: IDBTransactionMode = 'readonly') {
    return this.db.transaction(stores, mode);
  }

  async list(): Promise<ProjectMeta[]> {
    // Projects saved before revisions existed have none: null, like a project never saved.
    return (await request<BrowserMeta[]>(this.tx(['meta']).objectStore('meta').getAll())).map((m) => ({ ...m, revision: m.revision ?? null, summary: m.summary ?? null }));
  }

  async read(id: string) {
    const t = this.tx(['docs', 'meta']);
    const [json, meta] = await Promise.all([request<string | undefined>(t.objectStore('docs').get(id)), request<BrowserMeta | undefined>(t.objectStore('meta').get(id))]);
    if (json === undefined) throw new Error('the project is missing from this browser’s storage');
    return { json, revision: meta?.revision ?? null };
  }

  async media(hash: string): Promise<Blob> {
    const rec = await request<MediaRecord | undefined>(this.tx(['media']).objectStore('media').get(hash));
    if (!rec) throw new Error(`An audio file of this project is missing from this browser (${hash.slice(0, 8)}…)`);
    return rec.blob;
  }

  /** One transaction: the revision check, new media, the document, its meta. */
  async write(doc: ProjectDoc, json: string, media: Map<string, Blob>, base: Revision | null): Promise<ProjectMeta> {
    const t = this.tx(['media', 'docs', 'meta'], 'readwrite');
    const old = await request<BrowserMeta | undefined>(t.objectStore('meta').get(doc.id));
    const now = old?.revision ?? null;
    if (now !== base) {
      t.abort();
      throw new RevisionConflict(now);
    }
    const meta: BrowserMeta = { ...metaOf(doc, json, (typeof now === 'number' ? now : 0) + 1, old?.openedAt), media: Object.keys(doc.media) };
    for (const [hash, blob] of media) if (!stored.has(hash)) t.objectStore('media').put({ hash, type: blob.type, size: blob.size, blob } satisfies MediaRecord);
    t.objectStore('docs').put(json, doc.id);
    t.objectStore('meta').put(meta);
    await committed(t);
    for (const [hash, blob] of media) stored.set(hash, blob.size);
    const { media: _m, ...rest } = meta;
    if (old?.media.some((h) => !(h in doc.media))) await this.sweep();
    return rest;
  }

  async remove(id: string) {
    const t = this.tx(['docs', 'meta'], 'readwrite');
    t.objectStore('docs').delete(id);
    t.objectStore('meta').delete(id);
    await committed(t);
    await this.sweep();
  }

  async opened(meta: ProjectMeta) {
    const t = this.tx(['meta'], 'readwrite');
    const s = t.objectStore('meta');
    const old = await request<BrowserMeta | undefined>(s.get(meta.id));
    if (old) s.put({ ...old, openedAt: meta.openedAt });
    await committed(t);
  }

  /** Drop media no project refers to any more. */
  private async sweep() {
    const t = this.tx(['meta', 'media'], 'readwrite');
    const [metas, hashes] = await Promise.all([request<BrowserMeta[]>(t.objectStore('meta').getAll()), request<IDBValidKey[]>(t.objectStore('media').getAllKeys())]);
    const live = new Set(metas.flatMap((m) => m.media));
    const dead = (hashes as string[]).filter((h) => !live.has(h));
    for (const h of dead) t.objectStore('media').delete(h);
    await committed(t);
    for (const h of dead) stored.delete(h);
  }
}

// ── Server ────────────────────────────────────────────────────────────────────────────────────────

const time = (t: number | string | null): number => (typeof t === 'number' ? (t < 1e12 ? t * 1000 : t) : (t && Date.parse(t)) || Date.now());

const isSummary = (s: unknown): s is ProjectSummary => !!s && typeof s === 'object' && typeof (s as ProjectSummary).takes === 'number';

function entryMeta(e: ServerProjectEntry): ProjectMeta {
  const t = time(e.updated_at);
  return { id: e.id, name: e.name ?? e.id, createdAt: t, updatedAt: t, openedAt: t, size: (e.bytes ?? 0) + (e.media_bytes ?? 0), summary: isSummary(e.summary) ? e.summary : null, revision: e.revision };
}

export class ServerStore implements Store {
  readonly kind = 'server';

  async list() {
    return (await api.listProjects()).map(entryMeta);
  }

  async read(id: string) {
    const p = await api.getProject(id);
    const json = JSON.stringify(p.data);
    // Everything a stored document names is on the server.
    for (const [h, m] of Object.entries((p.data as unknown as ProjectDoc).media ?? {})) stored.set(h, m.size);
    return { json, revision: p.revision };
  }

  media(hash: string) {
    return api.getMedia(hash);
  }

  /** Upload what the server lacks (HEAD, then PUT), then the document against `base`. */
  async write(doc: ProjectDoc, json: string, media: Map<string, Blob>, base: Revision | null): Promise<ProjectMeta> {
    for (const [hash, blob] of media) {
      if (!(await api.hasMedia(hash))) await api.putMedia(hash, blob);
      stored.set(hash, blob.size);
    }
    const put = () => api.putProject(doc.id, { name: doc.name, data: doc, base_revision: base });
    let r: Awaited<ReturnType<typeof put>>;
    try {
      r = await put();
    } catch (e) {
      if (!(e instanceof MissingMedia)) throw e;
      // Media taken for stored are gone (collected): upload what this save holds and retry; the rest is forgotten,
      // so the next snapshot encodes and uploads it again.
      for (const h of e.missing) stored.delete(h);
      if (!e.missing.every((h) => media.has(h))) throw e;
      for (const h of e.missing) await api.putMedia(h, media.get(h)!);
      r = await put();
    }
    return { ...metaOf(doc, json, r.revision), updatedAt: time(r.updated_at) };
  }

  remove(id: string) {
    return api.deleteProject(id);
  }

  async opened() {}
}

// ── Moving between stores ─────────────────────────────────────────────────────────────────────────

/**
 * A document and its media addressed by SHA-256: media named by an older hash (browser projects and zips from
 * before) get their SHA-256 and every reference in the JSON follows.
 */
export async function readdress(json: string, media: Map<string, Blob>): Promise<{ json: string; media: Map<string, Blob> }> {
  const out = new Map<string, Blob>();
  for (const [h, blob] of media) {
    if (isSha256(h)) {
      out.set(h, blob);
      continue;
    }
    const next = await sha256Async(new Uint8Array(await blob.arrayBuffer()));
    json = json.split(`"${h}"`).join(`"${next}"`);
    out.set(next, blob);
  }
  return { json, media: out };
}

/** Whether `bytes` are what a media file name's hash (SHA-256, or the older 128-bit one) says. */
export async function matchesHash(hash: string, bytes: Uint8Array<ArrayBuffer>): Promise<boolean> {
  return isSha256(hash) ? (await sha256Async(bytes)) === hash : hash128(bytes) === hash;
}
