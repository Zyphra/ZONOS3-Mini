/**
 * The project server (`/projects`, `/media`) as a client-side fixture: the same contract — revisions with 409 on a
 * stale `base_revision`, 400 for a document naming media the store lacks, SHA-256 checked on upload — kept in its
 * own IndexedDB database, so two tabs share it like two browsers share a server. `?projects=fixture` selects it.
 */
import { sha256Async } from '../lib/hash';
import { committed, openDb, request } from '../lib/idb';

export interface Entry {
  id: string;
  name: string;
  updated_at: number;
  revision: number;
  bytes: number;
  media_bytes: number;
  summary: unknown;
}

interface Row {
  id: string;
  name: string;
  revision: number;
  updated_at: number;
  data: unknown;
  json: string;
}

export class FixtureError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string,
    readonly body: Record<string, unknown> = {},
  ) {
    super(detail);
  }
}

let db: Promise<IDBDatabase> | null = null;
const open = () =>
  (db ??= openDb('zvs-fixture-server', 1, (d) => {
    d.createObjectStore('projects', { keyPath: 'id' });
    d.createObjectStore('media', { keyPath: 'hash' });
  }));

const ID = /^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/;
const HASH = /^[0-9a-f]{64}$/;

function checkId(id: string) {
  if (!ID.test(id)) throw new FixtureError(400, 'invalid project id');
}

/** Hashes a document refers to: the keys of its `media` map. */
const mediaOf = (data: unknown): string[] => Object.keys((data as { media?: Record<string, unknown> } | null)?.media ?? {});

export async function list(): Promise<Entry[]> {
  const d = await open();
  const t = d.transaction(['projects', 'media']);
  const [rows, media] = await Promise.all([request<Row[]>(t.objectStore('projects').getAll()), request<{ hash: string; size: number }[]>(t.objectStore('media').getAll())]);
  const size = new Map(media.map((m) => [m.hash, m.size]));
  return rows
    .map((r) => ({
      id: r.id,
      name: r.name,
      updated_at: r.updated_at,
      revision: r.revision,
      bytes: r.json.length,
      media_bytes: mediaOf(r.data).reduce((n, h) => n + (size.get(h) ?? 0), 0),
      summary: (r.data as { summary?: unknown }).summary ?? null,
    }))
    .sort((a, b) => b.updated_at - a.updated_at);
}

export async function get(id: string) {
  checkId(id);
  const r = await request<Row | undefined>((await open()).transaction('projects').objectStore('projects').get(id));
  if (!r) throw new FixtureError(404, 'no such project');
  return { id: r.id, name: r.name, revision: r.revision, updated_at: r.updated_at, data: JSON.parse(r.json) as unknown };
}

export async function put(id: string, body: { name: string; data: unknown; base_revision: number | string | null }) {
  checkId(id);
  const json = JSON.stringify(body.data);
  const d = await open();
  const t = d.transaction(['projects', 'media'], 'readwrite');
  const [row, missing] = await Promise.all([
    request<Row | undefined>(t.objectStore('projects').get(id)),
    Promise.all(mediaOf(body.data).map(async (h) => ((await request(t.objectStore('media').getKey(h))) === undefined ? h : null))),
  ]);
  const stored = row?.revision ?? null;
  if (stored !== body.base_revision) {
    t.abort();
    throw new FixtureError(409, 'the project changed since base_revision', { revision: stored });
  }
  const gone = missing.filter((h): h is string => !!h);
  if (gone.length) {
    t.abort();
    throw new FixtureError(400, `missing media: ${gone.join(', ')}`, { missing: gone });
  }
  const next: Row = { id, name: body.name, revision: (stored ?? 0) + 1, updated_at: Date.now() / 1000, data: null, json };
  t.objectStore('projects').put({ ...next, data: body.data });
  await committed(t);
  return { revision: next.revision, updated_at: next.updated_at };
}

export async function remove(id: string) {
  checkId(id);
  const t = (await open()).transaction('projects', 'readwrite');
  t.objectStore('projects').delete(id);
  await committed(t);
}

export async function hasMedia(hash: string): Promise<boolean> {
  if (!HASH.test(hash)) throw new FixtureError(400, 'invalid hash');
  return (await request((await open()).transaction('media').objectStore('media').getKey(hash))) !== undefined;
}

export async function getMedia(hash: string): Promise<Blob> {
  if (!HASH.test(hash)) throw new FixtureError(400, 'invalid hash');
  const r = await request<{ blob: Blob } | undefined>((await open()).transaction('media').objectStore('media').get(hash));
  if (!r) throw new FixtureError(404, 'no such media');
  return r.blob;
}

export async function putMedia(hash: string, blob: Blob) {
  if (!HASH.test(hash)) throw new FixtureError(400, 'invalid hash');
  const bytes = new Uint8Array(await blob.arrayBuffer());
  if ((await sha256Async(bytes)) !== hash) throw new FixtureError(400, 'content does not match its hash');
  const t = (await open()).transaction('media', 'readwrite');
  t.objectStore('media').put({ hash, size: bytes.byteLength, blob: new Blob([bytes], { type: blob.type }) });
  await committed(t);
}
