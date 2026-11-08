import { Fragment } from 'preact';
import { useEffect, useRef, useState } from 'preact/hooks';
import { signal } from '@preact/signals';
import * as api from './api';
import { startPreview, playSide } from './audio/preview';
import * as player from './audio/player';
import { DownloadBar } from './components/DownloadBar';
import { Icon } from './components/Icon';
import { act, activeHistory, backend, bootError, bootProgress, dismissToast, encoderDownload, modalOpen, primaryAction, screenKeys, toasts, type Act } from './state/app';
import { embeddingsSynthetic, loadAllEmbeddings, refreshSpeakers } from './state/library';
import { openLastProject } from './state/projects';
import { CastScreen } from './screens/cast/CastScreen';
import { DirectScreen } from './screens/direct/DirectScreen';
import { EditScreen } from './screens/edit/EditScreen';
import { PronEditor } from './components/Pron';
import { TagPalette } from './components/Tags';
import { ZyphraMark, ZyphraWord } from './components/Brand';
import { ProjectLauncher, ProjectSwitcher } from './components/Projects';
import { ConfirmDialog } from './components/ConfirmDialog';
import { theme, toggleTheme } from './state/theme';
import { phone } from './lib/media';
import { isTyping } from './lib/dom';

const showKeys = signal(false);

async function boot() {
  let b: api.Backend;
  try {
    b = await api.connect();
  } catch (e) {
    bootError.value = e instanceof Error ? e.message : String(e);
    return;
  }
  await refreshSpeakers();
  // Sets `backend` together with the project's state (re-registering its voices first).
  await openLastProject(b);
  loadAllEmbeddings();
  startPreview();
}

const ACTS: { id: Act; label: string; hint: string }[] = [
  { id: 'cast', label: 'Cast', hint: 'find or design a voice' },
  { id: 'direct', label: 'Direct', hint: 'perform a script with one or more voices' },
  { id: 'edit', label: 'Edit', hint: 'change words in recorded speech' },
];

function onKey(e: KeyboardEvent) {
  if (!modalOpen.value && !showKeys.value && screenKeys.value?.(e)) return;
  const mod = e.metaKey || e.ctrlKey;
  if (mod && e.key.toLowerCase() === 'z' && !isTyping(e.target)) {
    e.preventDefault();
    if (e.shiftKey) activeHistory.value?.redo();
    else activeHistory.value?.undo();
    return;
  }
  if (mod && e.key.toLowerCase() === 'y' && !isTyping(e.target)) {
    e.preventDefault();
    activeHistory.value?.redo();
    return;
  }
  if (mod || e.altKey || isTyping(e.target)) return;
  const role = (e.target as HTMLElement)?.getAttribute?.('role');
  if (e.key === ' ' && !(e.target instanceof HTMLButtonElement) && role !== 'slider' && role !== 'button') {
    e.preventDefault();
    if (player.nowPlaying.value) player.stop();
    else primaryAction.value?.();
  } else if (act.value === 'cast' && (e.key === 'a' || e.key === 'b')) {
    playSide(e.key === 'a' ? 'A' : 'B');
  } else if (e.key === '?' && !modalOpen.value) {
    showKeys.value = !showKeys.value;
  } else if (e.key === 'Escape') {
    if (e.defaultPrevented || !showKeys.value) return;
    e.preventDefault();
    showKeys.value = false;
  } else if (/^[1-3]$/.test(e.key) && !modalOpen.value) {
    act.value = ACTS[Number(e.key) - 1].id;
  }
}

function BackendPill() {
  const b = backend.value;
  if (!b) return null;
  const text =
    b.kind === 'web'
      ? `In-browser · ${b.spec.bundle}`
      : b.kind === 'live'
        ? `Live · ${b.spec.bundle}`
        : b.kind === 'fixture'
          ? 'Fixture controls · server can’t steer yet'
          : 'Offline · synthesized audio';
  const detail =
    b.kind === 'web'
      ? 'ZONOS3 runs in this tab on WebGPU; models are cached in the browser (Cache Storage).'
      : b.kind === 'live'
        ? `Connected to ${b.server || location.origin}; controls and spans are applied by the server.`
        : b.kind === 'fixture'
          ? 'This server has no /v2/controls: controls come from the bundled fixture and renders are sent without them (you hear the base voice).' +
            (embeddingsSynthetic.value ? ' Embeddings are synthetic, so saving designed voices is off.' : '')
          : 'No server answered: voices, embeddings and audio come from the client-side fixture (a babbling synthesizer that follows every control).';
  return (
    <span class={`pill pill-${b.kind}`} title={detail}>
      <span class="pill-dot" />
      <span class="pill-text">{text}</span>
    </span>
  );
}

/** Phone: the top bar's secondary buttons (backend, theme, shortcuts) in one menu. */
function TopMenu() {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const close = (e: Event) => {
      if (e instanceof KeyboardEvent ? e.key === 'Escape' : !root.current?.contains(e.target as Node)) setOpen(false);
    };
    window.addEventListener('pointerdown', close);
    window.addEventListener('keydown', close);
    return () => {
      window.removeEventListener('pointerdown', close);
      window.removeEventListener('keydown', close);
    };
  }, [open]);
  const run = (f: () => void) => () => {
    setOpen(false);
    f();
  };
  return (
    <div ref={root} class="top-menu">
      <button class="icon-btn" aria-haspopup="menu" aria-expanded={open} onClick={() => setOpen(!open)} aria-label="More" title="More">
        <Icon name="more" />
      </button>
      {open && (
        <div class="top-menu-list" role="menu" aria-label="More">
          <BackendPill />
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

function Keys() {
  if (!showKeys.value) return null;
  const rows: [string, string][] = [
    ['Space', 'play / stop'],
    ['A / B', 'hear the design / the original voice (Cast)'],
    ['⌘Z / Ctrl+Z', 'undo'],
    ['⇧⌘Z / Ctrl+Y', 'redo'],
    ['1 2 3', 'Cast, Direct, Edit'],
    ['Arrows', 'nudge the focused control (⇧ for more)'],
    ['0 / Delete', 'reset the focused control'],
    ['(S@vQn)', 'X-SAMPA pronunciation in a line (phoneme-token checkpoints); click its IPA chip to edit'],
    ['/ə/', 'look up the selected words’ pronunciations'],
    ['<', 'in a text field: nonverbal tag palette (point tag at the caret, span tag around the selected words; NV checkpoint)'],
    ['‹› Tag', 'Direct: tag the selected words from the paint bar'],
    ['?', 'this sheet'],
  ];
  const edit: [string, string][] = [
    ['Space', 'play / pause from the cursor'],
    ['A / B', 'compare the version and the selected take'],
    ['⌘↵ / Ctrl+Enter', 'generate takes (also while typing)'],
    ['K', 'keep the selected take as a new version'],
    ['⌘Z / ⇧⌘Z', 'step through versions'],
    ['[ / ] · 0', 'zoom the waveform out / in · fit'],
    ['← / →', 'move the cursor 1 s'],
    ['click a word', 'select it and hear it (pre-roll optional)'],
    ['← / → (word selected)', 'nudge its edge 10 ms · ⇧ 50 ms'],
    [', / .', 'start / end edge of the selected word'],
    ['Tab / ⇧Tab', 'next / previous word'],
    ['↵ · Esc', 'play the word · done'],
    ['Alt-drag an edge', 'open a gap at a shared boundary'],
  ];
  return (
    <div class="sheet-backdrop" onClick={() => (showKeys.value = false)}>
      <div class="sheet" role="dialog" aria-label="Keyboard shortcuts" onClick={(e) => e.stopPropagation()}>
        <h2>Keyboard</h2>
        <dl>
          {rows.map(([k, v]) => (
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
          {edit.map(([k, v]) => (
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

export function App() {
  useEffect(() => {
    void boot();
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
  useEffect(() => {
    const sync = () => {
      const hash = location.hash.slice(1);
      // The Scene act merged into Direct: old links open Direct.
      const a = (hash === 'scene' ? 'direct' : hash) as Act;
      if (ACTS.some((x) => x.id === a)) {
        act.value = a;
        if (hash !== a) history.replaceState(null, '', `#${a}`);
      } else history.replaceState(null, '', `#${act.value}`);
    };
    window.addEventListener('hashchange', sync);
    return () => window.removeEventListener('hashchange', sync);
  }, []);
  useEffect(() => {
    if (location.hash.slice(1) !== act.value) history.replaceState(null, '', `#${act.value}`);
  }, [act.value]);

  const h = activeHistory.value;
  return (
    <div class="app">
      <header class="topbar">
        <div class="brand" aria-label="Zyphra Voice Studio">
          <ZyphraMark size={22} />
          <ZyphraWord height={11} />
          <span class="brand-sep" aria-hidden="true" />
          <span class="brand-product">Voice Studio</span>
        </div>
        <ProjectSwitcher />
        <nav class="acts" role="tablist" aria-label="Acts">
          {ACTS.map((a, i) => (
            <button role="tab" aria-selected={act.value === a.id} class={`act${act.value === a.id ? ' is-on' : ''}`} onClick={() => (act.value = a.id)} title={`${a.hint} (${i + 1})`}>
              <span class="act-n">{i + 1}</span>
              {a.label}
            </button>
          ))}
        </nav>
        <div class="topbar-right">
          {!phone.value && <BackendPill />}
          <button class="icon-btn" disabled={!h?.canUndo.value} onClick={() => h?.undo()} title="Undo (⌘Z)" aria-label="Undo">
            <Icon name="undo" />
          </button>
          <button class="icon-btn" disabled={!h?.canRedo.value} onClick={() => h?.redo()} title="Redo (⇧⌘Z)" aria-label="Redo">
            <Icon name="redo" />
          </button>
          {phone.value ? (
            <TopMenu />
          ) : (
            <>
              <button class="icon-btn" onClick={toggleTheme} title={theme.value === 'dark' ? 'Light theme' : 'Dark theme'} aria-label={theme.value === 'dark' ? 'Switch to the light theme' : 'Switch to the dark theme'}>
                <Icon name={theme.value === 'dark' ? 'sun' : 'moon'} />
              </button>
              <button class="icon-btn" onClick={() => (showKeys.value = !showKeys.value)} title="Keyboard shortcuts (?)" aria-label="Keyboard shortcuts">
                ?
              </button>
            </>
          )}
        </div>
      </header>
      <main class="stage">
        {bootError.value ? (
          <div class="empty">
            <h2>Can’t reach the studio server</h2>
            <p class="error-line">{bootError.value}</p>
            <button class="btn" onClick={() => location.reload()}>
              Try again
            </button>
          </div>
        ) : !backend.value ? (
          <div class="empty">
            {bootProgress.value ? <DownloadBar d={bootProgress.value} /> : <div class="loader" />}
            {!bootProgress.value && <p>Opening the studio…</p>}
          </div>
        ) : act.value === 'cast' ? (
          <CastScreen />
        ) : act.value === 'direct' ? (
          <DirectScreen />
        ) : (
          <EditScreen />
        )}
      </main>
      <div class="toasts">
        {encoderDownload.value && (
          <div class="toast toast-dl">
            <DownloadBar d={encoderDownload.value} />
          </div>
        )}
        <div class="toasts-live" aria-live="polite">
          {toasts.value.map((t) => (
          <div key={t.id} class={`toast toast-${t.kind}`} role={t.kind === 'error' ? 'alert' : undefined}>
            <span class="toast-text">{t.text}</span>
            {t.action && (
              <button
                class="link toast-action"
                onClick={() => {
                  dismissToast(t.id);
                  t.action!.run();
                }}
              >
                {t.action.label}
              </button>
            )}
            <button class="icon-btn sm toast-x" onClick={() => dismissToast(t.id)} aria-label="Dismiss">
              <Icon name="x" size={12} />
            </button>
          </div>
          ))}
        </div>
      </div>
      <Keys />
      <PronEditor />
      <TagPalette />
      <ProjectLauncher />
      <ConfirmDialog />
    </div>
  );
}
