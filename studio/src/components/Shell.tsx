/**
 * The shell's small surfaces: the backend pill and its capability panel, the phone top menu, the keyboard sheet,
 * the boot panel (stages, the in-browser consent card, failure) and the persistent backend banner.
 */
import { Fragment, createPortal } from 'preact';
import { useEffect, useLayoutEffect, useRef, useState } from 'preact/hooks';
import { signal } from '@preact/signals';
import * as api from '../api';
import { usingWebEngine } from '../lib/web-engine';
import { backend, bootError, bootProgress, bootStage, modalOpen } from '../state/app';
import { embeddingsSynthetic } from '../state/library';
import { theme, toggleTheme } from '../state/theme';
import { DownloadBar } from './DownloadBar';
import { dialogKeys, focusFirst } from './focusTrap';
import { Icon } from './Icon';

export const showKeys = signal(false);
export const backendPanelOpen = signal(false);
/** Boot asks whether to download the in-browser engine: resolves the pending `connect()`. */
export const webConsent = signal<((yes: boolean) => void) | null>(null);

// ── Backend pill and panel ────────────────────────────────────────────────────────────────────────

const KIND_LABEL: Record<api.BackendKind, string> = { live: 'Live', web: 'In this browser', fixture: 'Controls not applied', offline: 'Placeholder audio' };

function kindExplanation(b: api.Backend): string {
  switch (b.kind) {
    case 'web':
      return 'The voice engine runs in this tab on your graphics card; its models are cached in the browser.';
    case 'live':
      return 'A ZONOS server renders every voice; controls and painted words are applied there.';
    case 'fixture':
      return `This server can’t apply controls yet: renders are sent without them, so you hear the base voice as it is.${embeddingsSynthetic.value ? ' Its voice embeddings are stand-ins, so saving designed voices is off.' : ''}`;
    default:
      return 'No ZONOS server answered: voices and audio are placeholders made in this browser (a babbling synthesizer that follows every control).';
  }
}

/** The pill: a button opening the panel. Mounted in the top bar (desktop) or as a menu item (phone). */
export function BackendPill({ menuitem }: { menuitem?: boolean }) {
  const b = backend.value;
  if (!b) return null;
  return (
    <button
      class={`pill pill-${b.kind}${menuitem ? ' proj-item' : ''}`}
      role={menuitem ? 'menuitem' : undefined}
      aria-haspopup="dialog"
      aria-expanded={backendPanelOpen.value}
      onClick={() => (backendPanelOpen.value = !backendPanelOpen.value)}
      title="What this studio is connected to"
      id="backend-pill"
    >
      <span class="pill-dot" />
      <span class="pill-text">{KIND_LABEL[b.kind]}</span>
    </button>
  );
}

const yesNo = (on: boolean | null | undefined) => (on ? 'yes' : 'no');

export function BackendPanel() {
  const b = backend.value;
  const open = backendPanelOpen.value;
  const box = useRef<HTMLDivElement>(null);
  const [at, setAt] = useState<{ top: string; left?: string; right?: string } | null>(null);
  useLayoutEffect(() => {
    if (!open) return setAt(null);
    const r = document.getElementById('backend-pill')?.getBoundingClientRect();
    setAt(r ? { top: `${r.bottom + 6}px`, right: `${Math.max(8, window.innerWidth - r.right)}px` } : { top: '60px', right: '12px' });
  }, [open]);
  useEffect(() => {
    if (!open) return;
    const back = document.activeElement as HTMLElement | null;
    const outside = (e: PointerEvent) => {
      if (!box.current?.contains(e.target as Node) && !(e.target as HTMLElement).closest?.('#backend-pill')) backendPanelOpen.value = false;
    };
    document.addEventListener('pointerdown', outside);
    requestAnimationFrame(() => box.current && focusFirst(box.current));
    return () => {
      document.removeEventListener('pointerdown', outside);
      back?.focus({ preventScroll: true });
    };
  }, [open]);
  if (!b || !open || !at) return null;
  const rows: [string, string][] = [
    ['Server', b.kind === 'web' ? 'this browser' : b.kind === 'offline' ? 'none' : b.server || location.origin],
    ['Voice bundle', b.spec.bundle],
    ['Speech editing', yesNo(b.edit.speech_editing)],
    ['Speech recognition', yesNo(b.edit.edit_transcribe)],
    ['Pronunciation input', yesNo(!!b.pron.phoneme_tokens)],
    ['Nonverbal tags', yesNo(!!b.tags)],
    ['Projects on the server', yesNo(!!b.projects)],
  ];
  return createPortal(
    <div ref={box} class="backend-panel" role="dialog" aria-label="Connection" style={at} onKeyDown={dialogKeys(() => box.current, () => (backendPanelOpen.value = false))}>
      <header>
        <b>{KIND_LABEL[b.kind]}</b>
        <button class="icon-btn sm" onClick={() => (backendPanelOpen.value = false)} aria-label="Close" title="Close (Esc)">
          <Icon name="x" size={13} />
        </button>
      </header>
      <p>{kindExplanation(b)}</p>
      <dl>
        {rows.map(([k, v]) => (
          <Fragment key={k}>
            <dt>{k}</dt>
            <dd>{v}</dd>
          </Fragment>
        ))}
      </dl>
    </div>,
    document.body,
  );
}

/** Offline and fixture backends: a line under the top bar so nobody mistakes placeholder audio for the real thing. */
export function BackendBanner() {
  const b = backend.value;
  if (!b || (b.kind !== 'offline' && b.kind !== 'fixture')) return null;
  return (
    <div class="backend-banner" role="status">
      <Icon name="alert" size={14} />
      <span>{b.kind === 'offline' ? 'Placeholder audio: no ZONOS server answered.' : 'This server can’t apply controls yet: you hear the base voice as it is.'}</span>
      <button class="link" onClick={() => (backendPanelOpen.value = true)}>
        details
      </button>
    </div>
  );
}

// ── Phone top menu ────────────────────────────────────────────────────────────────────────────────

/** Phone: the top bar's secondary buttons (backend, theme, shortcuts, project settings) in one menu. */
export function TopMenu({ onSettings }: { onSettings: () => void }) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const menu = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!open) return;
    menu.current?.querySelector<HTMLElement>('[role="menuitem"]')?.focus();
    const outside = (e: PointerEvent) => !root.current?.contains(e.target as Node) && setOpen(false);
    document.addEventListener('pointerdown', outside);
    return () => document.removeEventListener('pointerdown', outside);
  }, [open]);
  const close = (refocus = true) => {
    setOpen(false);
    if (refocus) trigger.current?.focus();
  };
  const run = (f: () => void) => () => {
    close(false);
    f();
  };
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
  return (
    <div ref={root} class="top-menu">
      <button ref={trigger} class="icon-btn" aria-haspopup="menu" aria-expanded={open} onClick={() => setOpen(!open)} aria-label="More" title="More">
        <Icon name="more" />
      </button>
      {open && (
        <div ref={menu} class="top-menu-list" role="menu" aria-label="More" onKeyDown={onMenuKey}>
          <BackendPill menuitem />
          <button role="menuitem" class="proj-item" onClick={run(onSettings)}>
            <Icon name="gear" size={14} /> Project settings…
          </button>
          <button role="menuitem" class="proj-item" onClick={run(toggleTheme)}>
            <Icon name={theme.value === 'dark' ? 'sun' : 'moon'} size={14} /> {theme.value === 'dark' ? 'Light theme' : 'Dark theme'}
          </button>
          <button role="menuitem" class="proj-item" onClick={run(() => (showKeys.value = true))}>
            <span class="top-menu-key">?</span> Keyboard shortcuts
          </button>
        </div>
      )}
    </div>
  );
}

// ── Keyboard sheet ────────────────────────────────────────────────────────────────────────────────

const ROWS: [string, string][] = [
  ['Space', 'play / stop'],
  ['↵ · ⇧↵', 'Cast: speak the text · new line in it'],
  ['A / B', 'hear the design / the original voice (Cast)'],
  ['⌘Z / Ctrl+Z', 'undo'],
  ['⇧⌘Z / Ctrl+Y', 'redo'],
  ['1 2 3', 'Cast, Direct, Edit'],
  ['Arrows', 'nudge the focused control (⇧ for more)'],
  ['Home / End', 'a slider to its minimum / maximum'],
  ['0 / Delete', 'reset the focused control'],
  ['double-click', 'reset a slider or the mood pad'],
  ['scroll on the Who map', 'pitch (the map focused, or with ⌘ / Ctrl)'],
  ['+ − Delete', 'a mood landmark: stronger, weaker, off'],
  ['(S@vQn)', 'X-SAMPA pronunciation in a line (phoneme-token checkpoints); click its IPA chip to edit'],
  ['/ə/', 'look up the selected words’ pronunciations'],
  ['<', 'in a text field: nonverbal tag palette (point tag at the caret, span tag around the selected words; NV checkpoint)'],
  ['‹› Tag', 'Direct: tag the selected words from the paint bar'],
  ['?', 'this sheet'],
];
const EDIT_ROWS: [string, string][] = [
  ['Space', 'play / pause from the cursor'],
  ['A / B', 'the version / the selected take'],
  ['⌘↵ / Ctrl+Enter', 'generate takes (also while typing)'],
  ['K', 'keep the selected take as a new version'],
  ['⌘Z / ⇧⌘Z', 'step through versions'],
  ['[ / ] · 0', 'zoom the waveform out / in · fit'],
  ['⌘-wheel', 'zoom around the pointer'],
  ['← / →', 'move the cursor 1 s'],
  ['click a word', 'select it and hear it (pre-roll optional)'],
  ['← / → (word selected)', 'nudge its edge 10 ms · ⇧ 50 ms'],
  [', / .', 'start / end edge of the selected word'],
  ['Tab / ⇧Tab', 'next / previous word (from the transcript or a word)'],
  ['↵ · Esc', 'play the word · done'],
  ['⌥-drag an edge', 'open a gap at a shared boundary'],
];

export function Keys() {
  const open = showKeys.value;
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const back = document.activeElement as HTMLElement | null;
    modalOpen.value = true;
    requestAnimationFrame(() => box.current && focusFirst(box.current));
    return () => {
      modalOpen.value = false;
      (back?.isConnected ? back : document.querySelector<HTMLElement>('#keys-btn'))?.focus({ preventScroll: true });
    };
  }, [open]);
  if (!open) return null;
  const close = () => (showKeys.value = false);
  return (
    <div class="sheet-backdrop" onClick={close}>
      <div ref={box} class="sheet" role="dialog" aria-modal="true" aria-label="Keyboard shortcuts" onClick={(e) => e.stopPropagation()} onKeyDown={dialogKeys(() => box.current, close)}>
        <header class="sheet-head">
          <h2>Keyboard</h2>
          <button class="icon-btn" onClick={close} aria-label="Close" title="Close (Esc)">
            <Icon name="x" />
          </button>
        </header>
        <dl>
          {ROWS.map(([k, v]) => (
            <Fragment key={k}>
              <dt>
                <kbd>{k}</kbd>
              </dt>
              <dd>{v}</dd>
            </Fragment>
          ))}
        </dl>
        <h3 class="sheet-sub">Edit</h3>
        <dl>
          {EDIT_ROWS.map(([k, v]) => (
            <Fragment key={k}>
              <dt>
                <kbd>{k}</kbd>
              </dt>
              <dd>{v}</dd>
            </Fragment>
          ))}
        </dl>
      </div>
    </div>
  );
}

// ── Boot panel ────────────────────────────────────────────────────────────────────────────────────

const SLOW_MS = 10_000;

function reloadWith(params: Record<string, string>) {
  const u = new URL(location.href);
  for (const [k, v] of Object.entries(params)) u.searchParams.set(k, v);
  location.href = u.toString();
}

/** What the stage shows until a backend is set: the stage line, a download, the consent card, or a failure with ways on. */
export function BootPanel() {
  const [slow, setSlow] = useState(false);
  const [server, setServer] = useState('');
  const [canWeb, setCanWeb] = useState(false);
  const consent = webConsent.value;
  const err = bootError.value;
  useEffect(() => {
    const t = setTimeout(() => setSlow(true), SLOW_MS);
    return () => clearTimeout(t);
  }, []);
  useEffect(() => {
    if (err || consent) void api.canRunWeb().then(setCanWeb);
  }, [!!err, !!consent]);
  if (consent) {
    return (
      <div class="empty boot-consent" role="dialog" aria-labelledby="boot-consent-title">
        <h2 id="boot-consent-title">No server answered</h2>
        <p>Run the voice engine in this browser? One-time download ≈ 1.9 GB, kept in the browser’s cache.</p>
        <div class="row gap-s">
          <button class="btn primary" onClick={() => consent(true)}>
            Run here
          </button>
          <button class="btn" onClick={() => location.reload()}>
            Retry server
          </button>
          <button class="btn ghost" onClick={() => consent(false)}>
            Work offline (placeholder audio)
          </button>
        </div>
      </div>
    );
  }
  if (err) {
    const web = usingWebEngine();
    return (
      <div class="empty boot-error">
        <h2>{web ? 'The in-browser engine couldn’t start' : 'Can’t reach the studio server'}</h2>
        <p class="error-line">
          {web ? '' : `Tried ${api.serverLabel()}: `}
          {err}
        </p>
        <div class="row gap-s">
          <button class="btn primary" onClick={() => location.reload()}>
            Try again
          </button>
          <button class="btn" onClick={() => reloadWith({ api: 'offline' })}>
            Work offline
          </button>
          {canWeb && !web && (
            <button class="btn" onClick={() => reloadWith({ api: 'web' })}>
              Run in this browser
            </button>
          )}
        </div>
        <form
          class="boot-server"
          onSubmit={(e) => {
            e.preventDefault();
            if (server.trim()) reloadWith({ api: 'live', server: server.trim() });
          }}
        >
          <label>
            Server URL
            <input class="input" type="url" placeholder="https://host:port" value={server} onInput={(e) => setServer((e.currentTarget as HTMLInputElement).value)} />
          </label>
          <button class="btn sm" type="submit" disabled={!server.trim()}>
            Connect
          </button>
        </form>
      </div>
    );
  }
  const d = bootProgress.value;
  return (
    <div class="empty">
      {d ? <DownloadBar d={d} /> : <div class="loader" />}
      <p>{d && d.fraction >= 1 ? 'Starting the voice engine…' : bootStage.value}</p>
      {slow && !d && (
        <p class="muted small">
          Still waiting for {api.serverLabel()}.{' '}
          <button class="link" onClick={() => location.reload()}>
            Retry
          </button>
        </p>
      )}
    </div>
  );
}
