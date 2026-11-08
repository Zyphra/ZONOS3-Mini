/**
 * Project management UI: `ProjectSwitcher` (header: the open project, its save state, a menu of project actions)
 * and `ProjectLauncher` (overlay: every project with search, open / rename / duplicate / export / delete, import,
 * the open project's settings and voice report, storage use). State and storage live in `state/projects.ts`.
 */
import { createPortal } from 'preact';
import { useEffect, useLayoutEffect, useRef, useState } from 'preact/hooks';
import { LANG_LABEL } from '../lib/languages';
import { modalOpen, projectSettings } from '../state/app';
import type { ProjectSummary } from '../state/projectSchema';
import {
  browserProjects,
  conflictGone,
  current,
  deleteProject,
  duplicateProject,
  exportProject,
  fmtBytes,
  importProject,
  keepMineAsCopy,
  launcherOpen,
  moveBrowserProjects,
  moving,
  newProject,
  openProject,
  projects,
  refreshList,
  reloadProject,
  renameProject,
  restoreReport,
  retrySave,
  saveError,
  saveState,
  setSettings,
  storage,
  storeKind,
  switching,
} from '../state/projects';
import type { ProjectMeta } from '../state/projectStore';
import { trapTab } from './focusTrap';
import { Icon } from './Icon';
import '../styles/projects.css';

const SAVE_TEXT = { saved: 'All changes saved', saving: 'Saving…', error: 'Not saved', conflict: 'Changed elsewhere', off: 'Not saved' } as const;

function ago(t: number): string {
  const s = (Date.now() - t) / 1000;
  if (s < 60) return 'just now';
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  if (s < 86400 * 7) return `${Math.round(s / 86400)} d ago`;
  return new Date(t).toLocaleDateString();
}

function summaryText(s: ProjectSummary | null): string {
  if (!s) return 'project';
  const n = (k: number, one: string, many = `${one}s`) => (k ? [`${k} ${k === 1 ? one : many}`] : []);
  const parts = [...n(s.cast, 'cast member'), ...n(s.takes, 'take'), ...n(s.versions, 'edit version')];
  return parts.join(' · ') || 'empty';
}

/** Pick a zip and import it (the input is created on demand, so the components need no hidden markup). */
function pickImport() {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = '.zip,application/zip';
  input.onchange = () => input.files?.[0] && void importProject(input.files[0]);
  input.click();
}

function SaveDot() {
  const s = saveState.value;
  return <span class={`proj-dot is-${s}`} role="img" aria-label={SAVE_TEXT[s]} title={saveError.value ?? SAVE_TEXT[s]} />;
}

/** A name being edited: Enter or leaving commits, Escape cancels. */
export function NameInput({ value, label, onDone }: { value: string; label: string; onDone: (name: string | null) => void }) {
  const ref = useRef<HTMLInputElement>(null);
  const done = useRef(false);
  useLayoutEffect(() => {
    ref.current!.focus();
    ref.current!.select();
  }, []);
  const finish = (name: string | null) => {
    if (done.current) return;
    done.current = true;
    onDone(name);
  };
  return (
    <input
      ref={ref}
      class="input proj-name-input"
      value={value}
      aria-label={label}
      maxLength={80}
      onKeyDown={(e) => {
        e.stopPropagation();
        if (e.key === 'Enter') finish(e.currentTarget.value);
        else if (e.key === 'Escape') finish(null);
      }}
      onBlur={(e) => finish(e.currentTarget.value)}
    />
  );
}

// ── Header switcher ───────────────────────────────────────────────────────────────────────────────

/** The open project in the header: name, save state, and a menu (recent projects, new, rename, duplicate, export, import, all projects). */
export function ProjectSwitcher() {
  const [renaming, setRenaming] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const menu = useRef<HTMLDivElement>(null);
  /** Where the open menu sits (viewport px): it renders on <body>, above every screen's stacking context. */
  const [at, setAt] = useState<{ top: string; left: string } | null>(null);
  const open = !!at;
  const setOpen = (on: boolean) => {
    const r = trigger.current?.getBoundingClientRect();
    setAt(on && r ? { top: `${r.bottom + 6}px`, left: `${r.left}px` } : null);
  };
  const p = current.value;

  const close = (refocus = true) => {
    setOpen(false);
    if (refocus) trigger.current?.focus();
  };

  useEffect(() => {
    if (!open) return;
    menu.current?.querySelector<HTMLElement>('[role="menuitem"]')?.focus();
    const outside = (e: PointerEvent) => !root.current?.contains(e.target as Node) && !menu.current?.contains(e.target as Node) && setOpen(false);
    document.addEventListener('pointerdown', outside);
    return () => document.removeEventListener('pointerdown', outside);
  }, [open]);

  const run = (fn: () => unknown) => () => {
    close(false);
    fn();
  };

  // Keys stay in the menu (no act switching, undo or space-to-play underneath); arrows move, Esc / Tab close.
  const onMenuKey = (e: KeyboardEvent) => {
    e.stopPropagation();
    const items = [...menu.current!.querySelectorAll<HTMLElement>('[role="menuitem"]:not([disabled])')];
    const i = items.indexOf(document.activeElement as HTMLElement);
    const go = (j: number) => items[(j + items.length) % items.length]?.focus();
    if (e.key === 'ArrowDown') go(i + 1);
    else if (e.key === 'ArrowUp') go(i - 1);
    else if (e.key === 'Home') go(0);
    else if (e.key === 'End') go(items.length - 1);
    else if (e.key === 'Escape') close();
    else if (e.key === 'Tab') return close(false);
    else return;
    e.preventDefault();
  };

  const recent = projects.value.filter((x) => x.id !== p?.id).slice(0, 5);
  const off = saveState.value === 'off';
  return (
    <div ref={root} class="proj-switch">
      {renaming && p ? (
        <NameInput
          value={p.name}
          label="Project name"
          onDone={(name) => {
            setRenaming(false);
            if (name) void renameProject(p.id, name);
            trigger.current?.focus();
          }}
        />
      ) : (
        <button ref={trigger} class="proj-trigger" aria-haspopup="menu" aria-expanded={open} disabled={!p} onClick={() => setOpen(!open)} title={`${p?.name ?? 'Project'} · ${saveError.value ?? SAVE_TEXT[saveState.value]}`}>
          <SaveDot />
          <span class="proj-trigger-name">{p?.name ?? 'Opening…'}</span>
          <Icon name="down" size={12} />
        </button>
      )}
      {saveState.value === 'conflict' && <ConflictBar />}
      {at &&
        p &&
        createPortal(
          <div ref={menu} class="proj-menu" role="menu" aria-label="Project" style={at} onKeyDown={onMenuKey}>
            <p class={`proj-menu-status${saveState.value === 'error' || saveState.value === 'conflict' || off ? ' is-bad' : ''}`}>{saveError.value ?? SAVE_TEXT[saveState.value]}</p>
            {saveState.value === 'error' && (
              <button role="menuitem" class="proj-item" onClick={run(() => void retrySave())}>
                <Icon name="refresh" size={14} /> Try saving again
              </button>
            )}
            {saveState.value === 'conflict' && (
              <>
                {!conflictGone.value && (
                  <button role="menuitem" class="proj-item" onClick={run(() => void reloadProject())}>
                    <Icon name="refresh" size={14} /> Reload (drop my changes)
                  </button>
                )}
                <button role="menuitem" class="proj-item" onClick={run(() => void keepMineAsCopy())}>
                  <Icon name="copy" size={14} /> Keep mine as a copy
                </button>
              </>
            )}
            {recent.length > 0 && (
              <div role="group" aria-label="Recent projects" class="proj-menu-group">
                <span class="proj-menu-label">Recent</span>
                {recent.map((r) => (
                  <button key={r.id} role="menuitem" class="proj-item" disabled={switching.value} onClick={run(() => void openProject(r.id))}>
                    <span class="proj-item-name">{r.name}</span>
                    <span class="proj-item-meta">{ago(r.updatedAt)}</span>
                  </button>
                ))}
              </div>
            )}
            <div role="separator" class="proj-sep" />
            <button role="menuitem" class="proj-item" disabled={off || switching.value} onClick={run(() => void newProject())}>
              <Icon name="plus" size={14} /> New project
            </button>
            <button role="menuitem" class="proj-item" onClick={run(() => setRenaming(true))}>
              <Icon name="edit" size={14} /> Rename…
            </button>
            <button role="menuitem" class="proj-item" disabled={off} onClick={run(() => void duplicateProject(p.id))}>
              <Icon name="copy" size={14} /> Duplicate
            </button>
            <button role="menuitem" class="proj-item" disabled={off} onClick={run(() => void exportProject(p.id))}>
              <Icon name="download" size={14} /> Export…
            </button>
            <button role="menuitem" class="proj-item" disabled={off} onClick={run(pickImport)}>
              <Icon name="upload" size={14} /> Import…
            </button>
            <div role="separator" class="proj-sep" />
            <button role="menuitem" class="proj-item" onClick={run(() => (launcherOpen.value = true))}>
              All projects…
            </button>
          </div>,
          document.body,
        )}
    </div>
  );
}

/** Saving stopped because the project changed elsewhere: the two ways on, always in view. */
function ConflictBar() {
  return createPortal(
    <div class="proj-conflict" role="alert">
      <Icon name="alert" size={16} />
      <span>{saveError.value}</span>
      {!conflictGone.value && (
        <button class="btn sm" disabled={switching.value} onClick={() => void reloadProject()}>
          Reload
        </button>
      )}
      <button class="btn sm primary" onClick={() => void keepMineAsCopy()}>
        Keep mine as a copy
      </button>
    </div>,
    document.body,
  );
}

// ── Launcher ──────────────────────────────────────────────────────────────────────────────────────

/** Every project: search, open, rename, duplicate, export, delete (confirmed), new, import; the open one's settings. */
export function ProjectLauncher() {
  return launcherOpen.value ? <Launcher /> : null;
}

function Launcher() {
  const [query, setQuery] = useState('');
  const [renaming, setRenaming] = useState<string | null>(null);
  const [confirming, setConfirming] = useState<string | null>(null);
  const box = useRef<HTMLDivElement>(null);
  const downOnBackdrop = useRef(false);

  useLayoutEffect(() => {
    const back = document.activeElement as HTMLElement | null;
    modalOpen.value = true;
    void refreshList();
    box.current!.querySelector<HTMLInputElement>('.proj-search')?.focus();
    return () => {
      modalOpen.value = false;
      // The opener (a menu item) may be gone: then the project switcher.
      (back?.isConnected ? back : document.querySelector<HTMLElement>('.proj-trigger'))?.focus();
    };
  }, []);

  const close = () => (launcherOpen.value = false);
  const q = query.trim().toLowerCase();
  const list = q ? projects.value.filter((p) => p.name.toLowerCase().includes(q)) : projects.value;
  const total = projects.value.reduce((n, p) => n + p.size, 0);
  const s = storage.value;

  // Keys stay inside the dialog: no act switching, undo or space-to-play underneath.
  const onKey = (e: KeyboardEvent) => {
    e.stopPropagation();
    if (e.key === 'Escape') {
      e.preventDefault();
      if (confirming) setConfirming(null);
      else close();
    } else if (e.key === 'Tab') trapTab(e, box.current!);
  };

  return (
    <div
      class="proj-backdrop"
      onPointerDown={(e) => (downOnBackdrop.current = e.target === e.currentTarget)}
      onClick={(e) => {
        if (downOnBackdrop.current && e.target === e.currentTarget) close();
        downOnBackdrop.current = false;
      }}
    >
      <div ref={box} class="proj-launcher" role="dialog" aria-modal="true" aria-labelledby="proj-title" onKeyDown={onKey}>
        <header class="proj-head">
          <h2 id="proj-title">Projects</h2>
          <span class="proj-head-note">{storeKind.value === 'server' ? 'Shared on the server: everyone with the studio link sees these' : 'Saved in this browser · move them with Export / Import'}</span>
          <button class="icon-btn" onClick={close} aria-label="Close projects" title="Close (Esc)">
            <Icon name="x" />
          </button>
        </header>
        <div class="proj-tools">
          <input class="input proj-search" type="search" placeholder="Search projects" aria-label="Search projects" value={query} onInput={(e) => setQuery(e.currentTarget.value)} />
          <button class="btn sm" disabled={saveState.value === 'off'} onClick={pickImport}>
            <Icon name="upload" size={14} /> Import…
          </button>
          <button class="btn primary sm" disabled={saveState.value === 'off' || switching.value} onClick={() => void newProject()}>
            <Icon name="plus" size={14} /> New project
          </button>
        </div>
        {storeKind.value === 'server' && browserProjects.value > 0 && (
          <div class="proj-move">
            <span>
              {browserProjects.value === 1 ? 'A project is' : `${browserProjects.value} projects are`} still saved in this browser only.
            </span>
            <button class="btn sm primary" disabled={moving.value} onClick={() => void moveBrowserProjects()}>
              <Icon name="upload" size={14} /> {moving.value ? 'Moving…' : `Move ${browserProjects.value === 1 ? 'it' : `all ${browserProjects.value}`} to the server`}
            </button>
          </div>
        )}
        <ul class="proj-list" aria-label="Projects" aria-busy={switching.value}>
          {list.map((p) => (
            <Row key={p.id} p={p} renaming={renaming === p.id} confirming={confirming === p.id} onRename={setRenaming} onConfirm={setConfirming} />
          ))}
          {!list.length && <li class="proj-none">{q ? `No project matches “${query.trim()}”.` : 'No saved projects.'}</li>}
        </ul>
        <CurrentProject />
        <footer class="proj-foot">
          {s && s.quota ? (
            <>
              <span>
                Browser storage: {fmtBytes(s.usage)} of {fmtBytes(s.quota)} used
              </span>
              <span class="proj-meter" role="meter" aria-label="Browser storage used" aria-valuemin={0} aria-valuemax={s.quota} aria-valuenow={s.usage}>
                <span style={{ width: `${Math.min(100, (100 * s.usage) / s.quota)}%` }} />
              </span>
            </>
          ) : (
            <span>
              Projects use about {fmtBytes(total)}
              {storeKind.value === 'server' ? ' on the server' : ''} (shared audio counted in each)
            </span>
          )}
        </footer>
      </div>
    </div>
  );
}

function Row({ p, renaming, confirming, onRename, onConfirm }: { p: ProjectMeta; renaming: boolean; confirming: boolean; onRename: (id: string | null) => void; onConfirm: (id: string | null) => void }) {
  const isOpen = p.id === current.value?.id;
  const off = saveState.value === 'off';
  return (
    <li class={`proj-row${isOpen ? ' is-open' : ''}`}>
      {renaming ? (
        <div class="proj-row-main">
          <NameInput
            value={p.name}
            label={`Rename ${p.name}`}
            onDone={(name) => {
              onRename(null);
              if (name) void renameProject(p.id, name);
            }}
          />
        </div>
      ) : (
        <button class="proj-row-main" aria-current={isOpen ? 'true' : undefined} disabled={switching.value} onClick={() => void openProject(p.id)} title={isOpen ? 'The open project' : `Open ${p.name}`}>
          <span class="proj-row-name">
            {p.name}
            {isOpen && <span class="tag">open</span>}
          </span>
          <span class="proj-row-meta">
            {summaryText(p.summary)} · {fmtBytes(p.size)} · edited {ago(p.updatedAt)}
          </span>
        </button>
      )}
      <div class="proj-row-actions">
        <button class="icon-btn sm" onClick={() => onRename(p.id)} aria-label={`Rename ${p.name}`} title="Rename">
          <Icon name="edit" size={14} />
        </button>
        <button class="icon-btn sm" disabled={off} onClick={() => void duplicateProject(p.id)} aria-label={`Duplicate ${p.name}`} title="Duplicate">
          <Icon name="copy" size={14} />
        </button>
        <button class="icon-btn sm" disabled={off} onClick={() => void exportProject(p.id)} aria-label={`Export ${p.name}`} title="Export (zip)">
          <Icon name="download" size={14} />
        </button>
        <button class="icon-btn sm proj-del" disabled={off} onClick={() => onConfirm(p.id)} aria-label={`Delete ${p.name}`} title="Delete">
          <Icon name="trash" size={14} />
        </button>
      </div>
      {confirming && <Confirm p={p} onDone={() => onConfirm(null)} />}
    </li>
  );
}

function Confirm({ p, onDone }: { p: ProjectMeta; onDone: () => void }) {
  const cancel = useRef<HTMLButtonElement>(null);
  useLayoutEffect(() => cancel.current!.focus(), []);
  return (
    <div class="proj-confirm" role="alertdialog" aria-label={`Delete ${p.name}?`}>
      <span>
        Delete “{p.name}”{storeKind.value === 'server' ? ' for everyone' : ' and its audio (unless another project shares it)'}? This can’t be undone{p.id === current.value?.id ? '; another project opens' : ''}.
      </span>
      <button ref={cancel} class="btn sm ghost" onClick={onDone}>
        Cancel
      </button>
      <button
        class="btn sm proj-danger"
        onClick={() => {
          onDone();
          void deleteProject(p.id);
        }}
      >
        Delete
      </button>
    </div>
  );
}

/** The open project's settings, save problem and what happened to its voices on open. */
function CurrentProject() {
  const p = current.value;
  if (!p) return null;
  const st = projectSettings.value;
  const r = restoreReport.value;
  return (
    <section class="proj-current" aria-label={`${p.name}: settings`}>
      <div class="proj-current-head">
        <h3>{p.name}</h3>
        <span class="proj-current-state">
          <SaveDot /> {saveError.value ?? SAVE_TEXT[saveState.value]}
        </span>
        {saveState.value === 'error' && (
          <button class="btn sm" onClick={() => void retrySave()}>
            Try again
          </button>
        )}
        {saveState.value === 'conflict' && (
          <>
            {!conflictGone.value && (
              <button class="btn sm" disabled={switching.value} onClick={() => void reloadProject()}>
                Reload
              </button>
            )}
            <button class="btn sm" onClick={() => void keepMineAsCopy()}>
              Keep mine as a copy
            </button>
          </>
        )}
      </div>
      <div class="proj-settings">
        <label>
          <span>Language</span>
          <select class="input" value={st.language} onChange={(e) => setSettings({ language: e.currentTarget.value })} title="Direct lines render in this language; new Edit clips and pronunciation lookups start in it">
            {Object.entries(LANG_LABEL).map(([code, label]) => (
              <option key={code} value={code}>
                {label}
              </option>
            ))}
          </select>
        </label>
        <label>
          <span>Voice mode</span>
          <select class="input" value={st.voiceMode} onChange={(e) => setSettings({ voiceMode: e.currentTarget.value as 'embedding' | 'reference' })} title="How the Cast bench starts: designable (embedding) or an exact clone of the reference clip">
            <option value="embedding">Designable</option>
            <option value="reference">Exact clone</option>
          </select>
        </label>
      </div>
      {r.reRegistered.length > 0 && <p class="proj-note">Registered again with this server on open: {r.reRegistered.join(', ')}.</p>}
      {r.issues.length > 0 && (
        <ul class="proj-issues" aria-label="Voices that couldn’t be restored">
          {r.issues.map((x) => (
            <li key={x}>{x}</li>
          ))}
        </ul>
      )}
    </section>
  );
}
