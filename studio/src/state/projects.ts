/**
 * Projects: every project is a document (`projectDoc.ts`) plus media by SHA-256, kept by a store
 * (`projectStore.ts`): the server's, shared by every browser with the studio link, when it offers projects, else
 * this browser's. The open project autosaves against the revision it was opened at; a save refused because the
 * project changed elsewhere stops autosave and offers reload / keep mine as a copy. Opening one swaps every act's
 * state, registers the voices the server has lost again (servers restart, session voices go with them) and remaps
 * their ids. Export / import move a project as one zip (project.json + media/<hash>.<ext>).
 */
import { batch, effect, signal, untracked } from '@preact/signals';
import { strFromU8, strToU8, unzipSync, zipSync, type Unzipped, type Zippable } from 'fflate';
import * as api from '../api';
import { embeddingOnly, type Backend } from '../api';
import { MissingMedia, RevisionConflict } from '../lib/http';
import { previewText } from '../audio/preview';
import { errorText } from '../audio/render';
import { download, slug } from '../lib/files';
import { uid } from '../lib/math';
import { backend, notify, projectSettings } from './app';
import { baseSpeaker, design, designs, setMode } from './design';
import * as edit from './edit';
import { cast, ownVoices, refreshSpeakers, speakers, uploading, type OwnVoice } from './library';
import { EMBEDDING_TYPE, MediaSink } from './media';
import { emptyDoc, LEGACY_KEYS, legacyDoc, restore, snapshot, speakerUses } from './projectDoc';
import { remapIds, SCHEMA, summaryOf, type ProjectDoc, type ProjectSettings } from './projectSchema';
import { BrowserStore, matchesHash, metaOf, readdress, ServerStore, type ProjectMeta, type Store } from './projectStore';
import { script, takes } from './script';
import { voiceOrigins } from './who';

export type SaveState = 'saved' | 'saving' | 'error' | 'conflict' | 'off';

/** Every stored project, most recent first. */
export const projects = signal<ProjectMeta[]>([]);
export const current = signal<ProjectMeta | null>(null);
export const saveState = signal<SaveState>('saved');
export const saveError = signal<string | null>(null);
/** The save conflict is a delete elsewhere: there is nothing to reload, only a copy to keep. */
export const conflictGone = signal(false);
/** Where projects live: 'server' (shared) or 'browser'; null before boot. */
export const storeKind = signal<Store['kind'] | null>(null);
/** Projects still kept in this browser while the server holds projects: offered to move. */
export const browserProjects = signal(0);
export const moving = signal(false);
/** What opening the current project did about its voices. */
export const restoreReport = signal<{ reRegistered: string[]; issues: string[] }>({ reRegistered: [], issues: [] });
/** The browser's estimate for this origin (secure contexts only). */
export const storage = signal<{ usage: number; quota: number } | null>(null);
export const launcherOpen = signal(false);
export const switching = signal(false);

const LAST = 'zvs.project';
const AUTOSAVE_MS = 800;
const UNTITLED = 'Untitled project';
const CONFLICT = 'This project was changed elsewhere (another tab or browser) since you opened it. Reload it, or keep your version as a copy.';
const GONE = 'This project was deleted elsewhere (another tab or browser). Keep your version as a copy to go on with it.';

let store: Store | null = null;
/**
 * Voices registered again on open have new ids on this server (session voices die with a restart, and every browser
 * has its own session): the document keeps the ids it was saved with, and this tab maps them both ways, so
 * re-registering is no change to save (two browsers opening one project don't overwrite each other).
 */
let unmap = new Map<string, string>();

/** The open project as last saved, without its timestamp: an unchanged snapshot is not written. */
let savedBody = '';
let timer = 0;
/** Store writes run one after another (a save, a delete and a move never interleave). */
let queue: Promise<unknown> = Promise.resolve();

function serial<T>(fn: () => Promise<T>): Promise<T> {
  const p = queue.then(fn);
  queue = p.catch(() => undefined);
  return p;
}

export function fmtBytes(n: number): string {
  if (n < 1e6) return `${Math.max(1, Math.round(n / 1e3))} kB`;
  if (n < 1e9) return `${(n / 1e6).toFixed(n < 1e7 ? 1 : 0)} MB`;
  return `${(n / 1e9).toFixed(1)} GB`;
}

const where = () => (store?.kind === 'server' ? 'on the server' : 'in this browser');

// ── List ──────────────────────────────────────────────────────────────────────────────────────────

function setList(list: ProjectMeta[]) {
  projects.value = [...list].sort((a, b) => b.openedAt - a.openedAt);
}

const upsert = (meta: ProjectMeta) => setList([...projects.value.filter((p) => p.id !== meta.id), meta]);

/** The store's list now (another tab or browser may have changed it); the open project keeps its own entry. */
export async function refreshList() {
  if (!store) return;
  try {
    const list = await store.list();
    const open = current.value;
    setList(open ? [...list.filter((p) => p.id !== open.id), open] : list);
  } catch (e) {
    notify(`Couldn’t list the projects: ${errorText(e)}`, 'error');
  }
}

/** Store a new document as it is (an import keeps its JSON byte for byte, unless it predates list summaries). */
async function storeNew(doc: ProjectDoc, json = JSON.stringify(doc), media = new Map<string, Blob>()): Promise<string> {
  if (!doc.summary) {
    doc.summary = summaryOf(doc);
    json = JSON.stringify(doc);
  }
  const meta = await serial(() => store!.write(doc, json, media, null));
  upsert(meta);
  return doc.id;
}

export async function refreshStorage() {
  if (store?.kind !== 'browser') return void (storage.value = null);
  try {
    const e = await navigator.storage?.estimate();
    if (e) storage.value = { usage: e.usage ?? 0, quota: e.quota ?? 0 };
  } catch {
    /* no estimate outside secure contexts: the launcher shows the projects' own sizes */
  }
}

// ── Autosave ──────────────────────────────────────────────────────────────────────────────────────

function failed(e: unknown) {
  if (e instanceof RevisionConflict) {
    // The bar under the header says it and offers the two ways on (no toast on top of it).
    clearTimeout(timer);
    conflictGone.value = e.revision === null;
    saveState.value = 'conflict';
    saveError.value = conflictGone.value ? GONE : CONFLICT;
    return;
  }
  const s = storage.value;
  const msg =
    e instanceof DOMException && e.name === 'QuotaExceededError'
      ? `Browser storage is full${s ? ` (${fmtBytes(s.usage)} of ${fmtBytes(s.quota)})` : ''}: delete or export projects you don’t need. Your work stays open in this tab.`
      : `Couldn’t save the project ${where()} (${errorText(e)}). Your work stays open in this tab.`;
  // Once per problem: autosave retries on every change.
  if (saveError.value !== msg) notify(msg, 'error');
  saveState.value = 'error';
  saveError.value = msg;
}

async function write(retried = false): Promise<void> {
  const head = current.value;
  if (!head || !store || saveState.value === 'conflict') return;
  try {
    const sink = new MediaSink();
    const doc = await snap(head, sink);
    const body = JSON.stringify({ ...doc, updatedAt: 0 });
    if (body === savedBody) {
      if (saveState.value === 'saving') saveState.value = 'saved';
      return;
    }
    saveState.value = 'saving';
    doc.updatedAt = Date.now();
    const json = JSON.stringify(doc);
    const meta = { ...(await store.write(doc, json, sink.pending, head.revision)), openedAt: head.openedAt };
    savedBody = body;
    // A rename while saving made a new head: keep its name, take the new revision.
    current.value = current.value === head ? meta : current.value && { ...current.value, revision: meta.revision };
    upsert(current.value ?? meta);
    saveState.value = 'saved';
    saveError.value = null;
    void adopt(doc);
    void refreshStorage();
  } catch (e) {
    // The server collected media this tab took for stored: the next snapshot uploads them again.
    if (e instanceof MissingMedia && !retried) return write(true);
    failed(e);
  }
}

/** The open project as a document, with the ids it was saved with. */
async function snap(head: Parameters<typeof snapshot>[0], sink: MediaSink): Promise<ProjectDoc> {
  const doc = await snapshot(head, sink);
  return unmap.size ? (JSON.parse(remapIds(JSON.stringify(doc), unmap)) as ProjectDoc) : doc;
}

/** Write the open project now (no-op when nothing changed). */
export function save(): Promise<void> {
  clearTimeout(timer);
  timer = 0;
  return serial(() => write());
}

function schedule() {
  if (!current.value || !store || saveState.value === 'conflict') return;
  clearTimeout(timer);
  timer = window.setTimeout(() => void save(), AUTOSAVE_MS);
}

effect(() => {
  void [design.value, cast.value, voiceOrigins.value, previewText.value, script.value, takes.value, edit.doc.value, edit.clipLanguages.value, projectSettings.value, ownVoices.value];
  untracked(schedule);
});

addEventListener('visibilitychange', () => document.visibilityState === 'hidden' && timer && void save());
addEventListener('pagehide', () => timer && void save());

// ── Voices ────────────────────────────────────────────────────────────────────────────────────────

/**
 * Register the project's voices the server no longer has (from their clip or embedding) and remap every stored
 * id to the new ones; list what can't be restored (a failed upload, or a voice the project holds no media for).
 */
async function reconcile(doc: ProjectDoc): Promise<{ doc: ProjectDoc; map: Map<string, string>; reRegistered: string[]; issues: string[] }> {
  const live = new Set(speakers.value.map((s) => s.id));
  const map = new Map<string, string>();
  const reRegistered: string[] = [];
  const issues: string[] = [];
  for (const v of doc.voices) {
    if (live.has(v.id)) continue;
    try {
      const blob = await store!.media(v.media);
      const s = v.kind === 'clip' ? await api.uploadVoice(v.label, blob, `${slug(v.label)}.wav`) : await api.saveEmbeddingVoice(v.label, new Float32Array(await blob.arrayBuffer()));
      map.set(v.id, s.id);
      reRegistered.push(v.label);
    } catch (e) {
      issues.push(`“${v.label}” couldn’t be registered with the server again: ${errorText(e)}`);
    }
  }
  const owned = new Set(doc.voices.map((v) => v.id));
  for (const [id, uses] of speakerUses(doc)) {
    if (!live.has(id) && !owned.has(id)) issues.push(`A voice used by ${[...uses].join(', ')} is gone from the server, and the project holds no clip or embedding to recreate it.`);
  }
  if (!map.size) return { doc, map, reRegistered, issues };
  await refreshSpeakers();
  return { doc: JSON.parse(remapIds(JSON.stringify(doc), map)) as ProjectDoc, map, reRegistered, issues };
}

const adopted = new Set<string>();

/**
 * Session voices the project uses but did not create here (made before projects existed, or in another project):
 * fetch their clip or embedding while the server still has them, so they survive a restart too.
 */
async function adopt(doc: ProjectDoc) {
  const id0 = current.value?.id;
  const own = new Set(ownVoices.value.map((v) => v.id));
  for (const id of speakerUses(doc).keys()) {
    const s = speakers.value.find((x) => x.id === id);
    if (!s || s.is_default || own.has(id) || adopted.has(id)) continue;
    adopted.add(id);
    try {
      let v: OwnVoice;
      if (embeddingOnly(s)) {
        const { embedding, synthetic } = await api.getEmbedding(s);
        if (synthetic) continue;
        v = { id, label: s.label, embedding };
      } else {
        v = { id, label: s.label, clip: await api.getPreview(s) };
      }
      if (current.value?.id === id0) ownVoices.value = [...ownVoices.value, v];
    } catch {
      /* stays unowned: reported if the server loses it */
    }
  }
}

/** The Cast bench needs a voice the server has: the first default one when the project's is gone. */
function pickBase() {
  const list = speakers.value;
  if (list.some((s) => s.id === design.value.baseId)) return;
  const first = list.find((s) => s.is_default) ?? list[0];
  if (first) designs.reset({ ...design.value, baseId: first.id, mode: embeddingOnly(first) ? 'embedding' : projectSettings.value.voiceMode });
}

// ── Opening ───────────────────────────────────────────────────────────────────────────────────────

/** Open a stored project; `b` (boot) lands with the first project, so no screen renders the empty defaults. */
async function load(id: string, b?: Backend) {
  const { json, revision } = await store!.read(id);
  const doc = JSON.parse(json) as ProjectDoc;
  if (doc.schema > SCHEMA) throw new Error('it was saved by a newer version of the studio');
  await refreshSpeakers();
  const { doc: next, map, reRegistered, issues } = await reconcile(doc);
  const apply = await restore(next, (h) => store!.media(h));
  const meta = metaOf(next, json, revision);
  clearTimeout(timer);
  batch(() => {
    if (b) backend.value = b;
    apply();
    current.value = meta;
  });
  unmap = new Map([...map].map(([saved, live]) => [live, saved]));
  // The document as stored (saves map ids back): only a real change, such as a new bench voice, is saved.
  savedBody = JSON.stringify({ ...doc, updatedAt: 0 });
  saveState.value = 'saved';
  saveError.value = null;
  restoreReport.value = { reRegistered, issues };
  localStorage.setItem(LAST, id);
  pickBase();
  upsert(meta);
  await serial(() => store!.opened(meta));
  if (issues.length) notify(`${issues.length === 1 ? 'A voice' : `${issues.length} voices`} of “${doc.name}” couldn’t be restored: see Projects for details.`, 'error');
  void adopt(next);
  void refreshStorage();
}

/** No store (the browser blocks storage, or the server's projects don't answer): the session works, nothing is saved. */
async function memoryOnly(b: Backend, e: unknown) {
  store = null;
  saveState.value = 'off';
  saveError.value = `Projects can’t be stored ${b.projects ? 'on the server' : 'in this browser'} (${errorText(e)}): work in this tab is not saved.`;
  notify(saveError.value, 'error');
  const doc = legacyDoc({ id: uid(), name: 'Unsaved project' });
  const apply = await restore(doc, () => Promise.reject(new Error('no storage')));
  batch(() => {
    backend.value = b;
    apply();
    current.value = metaOf(doc, '', null);
  });
  pickBase();
}

/**
 * Boot: open the last project (`b` is the connected backend; the speaker list must be loaded). The first visit
 * makes a project of what the studio kept in localStorage before projects existed, then drops those keys. With
 * projects on the server, projects this browser still holds are offered to move there.
 */
export async function openLastProject(b: Backend) {
  try {
    store = b.projects ? new ServerStore() : await BrowserStore.open(true);
    storeKind.value = store.kind;
    setList(await store.list());
  } catch (e) {
    return memoryOnly(b, e);
  }
  if (store.kind === 'server') {
    browserProjects.value = await BrowserStore.count();
    if (browserProjects.value) notify(`${browserProjects.value === 1 ? 'A project is' : `${browserProjects.value} projects are`} still saved in this browser only: move ${browserProjects.value === 1 ? 'it' : 'them'} to the server from Projects.`);
  }
  const last = localStorage.getItem(LAST);
  try {
    // `?project=<id>` (links from the MCP server) wins; an unknown id falls through to the last project.
    const wanted = new URLSearchParams(location.search).get('project');
    let id = (projects.value.find((p) => p.id === wanted) ?? projects.value.find((p) => p.id === last) ?? projects.value[0])?.id;
    if (!id) {
      id = await storeNew(legacyDoc({ id: uid(), name: 'My project' }));
      for (const k of LEGACY_KEYS) localStorage.removeItem(`zvs.${k}`);
    }
    await load(id, b);
  } catch (e) {
    notify(`Couldn’t open the last project (${errorText(e)}); started a new one.`, 'error');
    try {
      await load(await storeNew(emptyDoc({ id: uid(), name: UNTITLED }, projectSettings.value)), b);
    } catch (e2) {
      await memoryOnly(b, e2);
    }
  } finally {
    if (!backend.value) backend.value = b;
  }
}

function busyWith(): string | null {
  if (edit.busy.value) return 'the Edit screen';
  if (uploading.value) return 'the voice upload';
  return null;
}

/** Leaving the open project: it must be idle and saved (a failed save keeps you in it, so nothing is lost). */
async function canLeave(): Promise<boolean> {
  if (!store) {
    notify(saveError.value ?? 'Projects are unavailable.', 'error');
    return false;
  }
  const why = busyWith();
  if (why) {
    notify(`Wait for ${why} to finish (or cancel it) before switching projects.`, 'error');
    return false;
  }
  await save();
  if (saveState.value === 'conflict') {
    notify('This project changed elsewhere: reload it or keep yours as a copy before switching.', 'error');
    return false;
  }
  if (saveState.value === 'error') {
    notify(`The open project couldn’t be saved, so it stays open: ${store.kind === 'server' ? 'check the connection to the server' : 'free some space (delete or export projects)'} and try again.`, 'error');
    return false;
  }
  return true;
}

async function switchTo(next: () => Promise<string>) {
  if (switching.value) return;
  switching.value = true;
  try {
    if (!(await canLeave())) return;
    await load(await next());
    launcherOpen.value = false;
  } catch (e) {
    notify(`Couldn’t open the project: ${errorText(e)}`, 'error');
  } finally {
    switching.value = false;
  }
}

// ── Actions ───────────────────────────────────────────────────────────────────────────────────────

export async function openProject(id: string) {
  if (id !== current.value?.id) return switchTo(async () => id);
  launcherOpen.value = false;
}

export function newProject(name = UNTITLED) {
  return switchTo(() => storeNew(emptyDoc({ id: uid(), name }, projectSettings.value)));
}

/** The stored document of a project that isn't open, renamed (against its revision). */
async function rewrite(id: string, change: (doc: ProjectDoc) => ProjectDoc) {
  const { json, revision } = await store!.read(id);
  const doc = change(JSON.parse(json) as ProjectDoc);
  return store!.write(doc, JSON.stringify(doc), new Map(), revision);
}

export async function renameProject(id: string, name: string) {
  name = name.trim();
  const meta = projects.value.find((p) => p.id === id);
  if (!name || !meta || name === meta.name || !store) return;
  if (id === current.value?.id) {
    current.value = { ...current.value, name };
    upsert(current.value);
    return schedule();
  }
  try {
    upsert({ ...(await serial(() => rewrite(id, (doc) => ({ ...doc, name })))), openedAt: meta.openedAt });
  } catch (e) {
    notify(`Couldn’t rename the project: ${errorText(e)}`, 'error');
  }
}

/** A copy sharing the media (stored once); the open project stays open. */
export async function duplicateProject(id: string) {
  if (!store) return;
  try {
    if (id === current.value?.id) await save();
    const doc = JSON.parse((await store.read(id)).json) as ProjectDoc;
    const now = Date.now();
    const copy: ProjectDoc = { ...doc, id: uid(), name: `${doc.name} copy`, createdAt: now, updatedAt: now };
    await storeNew(copy);
    notify(`Duplicated as “${copy.name}”`, 'ok');
  } catch (e) {
    notify(`Couldn’t duplicate the project: ${errorText(e)}`, 'error');
  }
}

/** Delete a project (and, in the browser, the media only it used); deleting the open one opens the most recent other (or a new one). */
export async function deleteProject(id: string) {
  if (!store) return;
  const open = id === current.value?.id;
  if (open && busyWith()) return notify(`Wait for ${busyWith()} to finish before deleting the open project.`, 'error');
  const name = projects.value.find((p) => p.id === id)?.name ?? 'the project';
  if (open) {
    clearTimeout(timer);
    current.value = null;
  }
  try {
    await serial(() => store!.remove(id));
    setList(projects.value.filter((p) => p.id !== id));
    notify(`Deleted “${name}”`);
    if (open) {
      const next = projects.value[0];
      await load(next ? next.id : await storeNew(emptyDoc({ id: uid(), name: UNTITLED }, projectSettings.value)));
    }
  } catch (e) {
    notify(`Couldn’t delete the project: ${errorText(e)}`, 'error');
  }
  void refreshStorage();
}

// ── Changed elsewhere ─────────────────────────────────────────────────────────────────────────────

/** Drop this tab's changes: open the project as it is stored now. */
export async function reloadProject() {
  const p = current.value;
  if (!p || !store || switching.value) return;
  switching.value = true;
  try {
    await load(p.id);
    notify(`Reloaded “${p.name}”`, 'ok');
  } catch (e) {
    notify(`Couldn’t reload the project: ${errorText(e)}`, 'error');
  } finally {
    switching.value = false;
  }
}

/** Keep this tab's version as a new project (the other one stays as it was saved elsewhere) and go on in it. */
export async function keepMineAsCopy() {
  const p = current.value;
  if (!p || !store) return;
  try {
    const now = Date.now();
    const head = { ...p, id: uid(), name: `${p.name} (my copy)`, createdAt: now, updatedAt: now, revision: null };
    const sink = new MediaSink();
    const doc = await snap(head, sink);
    const json = JSON.stringify(doc);
    const meta = await serial(() => store!.write(doc, json, sink.pending, null));
    current.value = meta;
    upsert(meta);
    savedBody = JSON.stringify({ ...doc, updatedAt: 0 });
    saveState.value = 'saved';
    saveError.value = null;
    localStorage.setItem(LAST, meta.id);
    notify(`Your version is now “${meta.name}”`, 'ok');
    void refreshList();
  } catch (e) {
    notify(`Couldn’t save your version as a copy: ${errorText(e)}`, 'error');
  }
}

// ── Export / import / move ────────────────────────────────────────────────────────────────────────

const EXT: Record<string, string> = { 'audio/wav': 'wav', [EMBEDDING_TYPE]: 'f32' };
const mediaPath = (hash: string, type: string) => `media/${hash}.${EXT[type] ?? (type.split('/')[1]?.replace(/[^a-z0-9]/g, '') || 'bin')}`;

/** One zip: project.json (the document exactly as stored) and media/<hash>.<ext>. */
export async function exportProject(id: string) {
  if (!store) return;
  try {
    if (id === current.value?.id) await save();
    const { json } = await store.read(id);
    const doc = JSON.parse(json) as ProjectDoc;
    const files: Zippable = { 'project.json': [strToU8(json), { level: 6 }] };
    for (const [h, info] of Object.entries(doc.media)) files[mediaPath(h, info.type)] = [new Uint8Array(await (await store.media(h)).arrayBuffer()), { level: 0 }];
    const zip = zipSync(files, { mtime: doc.updatedAt });
    download(zip.buffer, `${slug(doc.name)}.voice-studio.zip`, 'application/zip');
  } catch (e) {
    notify(`Couldn’t export the project: ${errorText(e)}`, 'error');
  }
}

/** Import an exported zip (media checked against their hashes) into the store and open it; an id already there gets a new one. */
export async function importProject(file: File) {
  if (!store) return notify(saveError.value ?? 'Projects are unavailable.', 'error');
  try {
    let files: Unzipped;
    try {
      files = unzipSync(new Uint8Array(await file.arrayBuffer()));
    } catch {
      throw new Error('it is not a zip file');
    }
    if (!files['project.json']) throw new Error('it has no project.json (not a Voice Studio project)');
    const doc0 = JSON.parse(strFromU8(files['project.json'])) as ProjectDoc;
    if (typeof doc0.schema !== 'number' || typeof doc0.id !== 'string' || !doc0.media) throw new Error('its project.json is not a Voice Studio project');
    if (doc0.schema > SCHEMA) throw new Error('it was saved by a newer version of the studio');
    const media = new Map<string, Blob>();
    for (const [path, bytes] of Object.entries(files)) {
      const h = /^media\/([0-9a-f]{32}|[0-9a-f]{64})\./.exec(path)?.[1];
      if (!h) continue;
      if (!(await matchesHash(h, bytes as Uint8Array<ArrayBuffer>))) throw new Error(`${path} is damaged`);
      media.set(h, new Blob([bytes as Uint8Array<ArrayBuffer>], { type: doc0.media[h]?.type ?? '' }));
    }
    const missing = Object.keys(doc0.media).filter((h) => !media.has(h)).length;
    if (missing) throw new Error(`${missing} media file${missing === 1 ? ' is' : 's are'} missing`);
    const r = await readdress(strFromU8(files['project.json']), media);
    let json = r.json;
    const doc = JSON.parse(json) as ProjectDoc;
    if (projects.value.some((p) => p.id === doc.id)) {
      doc.id = uid();
      doc.name = `${doc.name} (imported)`;
      json = JSON.stringify(doc);
    }
    await storeNew(doc, json, r.media);
    notify(`Imported “${doc.name}”`, 'ok');
    await openProject(doc.id);
  } catch (e) {
    notify(`Couldn’t import ${file.name}: ${errorText(e)}`, 'error');
  }
}

/** Move every project this browser holds to the server (media re-addressed by SHA-256), then drop the browser copies. */
export async function moveBrowserProjects() {
  if (store?.kind !== 'server' || moving.value) return;
  moving.value = true;
  let moved = 0;
  try {
    const local = await BrowserStore.open(false);
    for (const meta of await local.list()) {
      const { json: json0 } = await local.read(meta.id);
      const doc0 = JSON.parse(json0) as ProjectDoc;
      const media = new Map<string, Blob>();
      for (const h of Object.keys(doc0.media)) media.set(h, await local.media(h));
      const r = await readdress(json0, media);
      let json = r.json;
      const doc = JSON.parse(json) as ProjectDoc;
      if (projects.value.some((p) => p.id === doc.id)) {
        doc.id = uid();
        json = JSON.stringify(doc);
      }
      await storeNew(doc, json, r.media);
      await local.remove(meta.id);
      moved++;
      browserProjects.value--;
    }
    notify(`Moved ${moved} project${moved === 1 ? '' : 's'} to the server`, 'ok');
  } catch (e) {
    notify(`Moved ${moved}; couldn’t move the rest: ${errorText(e)}`, 'error');
  } finally {
    moving.value = false;
    void refreshList();
  }
}

export function setSettings(patch: Partial<ProjectSettings>) {
  projectSettings.value = { ...projectSettings.value, ...patch };
  if (patch.voiceMode && patch.voiceMode !== design.value.mode && !embeddingOnly(baseSpeaker.value)) setMode(patch.voiceMode);
}

/** Try the failed save again now. */
export function retrySave() {
  savedBody = '';
  return save();
}
