import { useEffect } from 'preact/hooks';
import * as api from './api';
import { startPreview, playSide } from './audio/preview';
import * as player from './audio/player';
import { DownloadBar } from './components/DownloadBar';
import { Icon } from './components/Icon';
import { act, activeHistory, backend, bootError, bootStage, dismissToast, encoderDownload, modalOpen, notify, pauseToasts, primaryAction, screenKeys, settingsOpen, toasts, type Act, type Toast } from './state/app';
import { loadAllEmbeddings, refreshSpeakers } from './state/library';
import { openLastProject } from './state/projects';
import { redoVersion, undoVersion } from './state/edit';
import { CastScreen } from './screens/cast/CastScreen';
import { DirectScreen } from './screens/direct/DirectScreen';
import { EditScreen } from './screens/edit/EditScreen';
import { PronEditor } from './components/Pron';
import { ZyphraMark, ZyphraWord } from './components/Brand';
import { ProjectLauncher, ProjectSwitcher } from './components/Projects';
import { ConfirmDialog } from './components/ConfirmDialog';
import { ProjectSettings } from './components/ProjectSettings';
import { BackendBanner, BackendPanel, BackendPill, BootPanel, Keys, showKeys, TopMenu, backendPanelOpen, webConsent } from './components/Shell';
import { theme, toggleTheme } from './state/theme';
import { phone } from './lib/media';
import { isTyping } from './lib/dom';

async function boot() {
  let b: api.Backend;
  try {
    b = await api.connect({
      confirmWeb: () => {
        const { promise, resolve } = Promise.withResolvers<boolean>();
        webConsent.value = (yes) => {
          webConsent.value = null;
          resolve(yes);
        };
        return promise;
      },
    });
  } catch (e) {
    bootError.value = e instanceof Error ? e.message : String(e);
    return;
  }
  bootStage.value = 'Loading voices…';
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
    // Edit asks before a version step drops the draft.
    if (act.value === 'edit') void (e.shiftKey ? redoVersion() : undoVersion());
    else if (e.shiftKey) activeHistory.value?.redo();
    else activeHistory.value?.undo();
    return;
  }
  if (mod && e.key.toLowerCase() === 'y' && !isTyping(e.target)) {
    e.preventDefault();
    if (act.value === 'edit') void redoVersion();
    else activeHistory.value?.redo();
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
    if (e.defaultPrevented) return;
    if (backendPanelOpen.value) {
      e.preventDefault();
      backendPanelOpen.value = false;
    }
  } else if (/^[1-3]$/.test(e.key) && !modalOpen.value) {
    act.value = ACTS[Number(e.key) - 1].id;
  }
}

/** ← / → move between the act tabs. */
function actKeys(e: KeyboardEvent) {
  if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
  e.preventDefault();
  const i = ACTS.findIndex((a) => a.id === act.value);
  const next = ACTS[(i + (e.key === 'ArrowRight' ? 1 : ACTS.length - 1)) % ACTS.length];
  act.value = next.id;
  (e.currentTarget as HTMLElement).querySelector<HTMLElement>(`[data-act="${next.id}"]`)?.focus();
}

/** The service worker installed a new build: offer the reload that brings it in. */
function watchUpdates() {
  if (!('serviceWorker' in navigator)) return;
  navigator.serviceWorker.addEventListener('message', (e: MessageEvent) => {
    if (e.data?.type === 'zvs-update') notify('A new version is ready', 'info', { label: 'Reload', run: () => location.reload() });
  });
}

function ToastRow({ t }: { t: Toast }) {
  return (
    <div class={`toast toast-${t.kind}`} role={t.kind === 'error' ? 'alert' : undefined}>
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
  );
}

export function App() {
  useEffect(() => {
    void boot();
    watchUpdates();
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
        <nav class="acts" role="tablist" aria-label="Acts" onKeyDown={actKeys}>
          {ACTS.map((a, i) => (
            <button
              role="tab"
              data-act={a.id}
              aria-selected={act.value === a.id}
              aria-controls="stage"
              tabIndex={act.value === a.id ? 0 : -1}
              class={`act${act.value === a.id ? ' is-on' : ''}`}
              onClick={() => (act.value = a.id)}
              title={`${a.hint} (${i + 1})`}
            >
              <span class="act-n">{i + 1}</span>
              {a.label}
            </button>
          ))}
        </nav>
        <div class="topbar-right">
          {!phone.value && <BackendPill />}
          <button class="icon-btn" disabled={!h?.canUndo.value} onClick={() => h?.undo()} title="Undo (⌘Z / Ctrl+Z)" aria-label="Undo">
            <Icon name="undo" />
          </button>
          <button class="icon-btn" disabled={!h?.canRedo.value} onClick={() => h?.redo()} title="Redo (⇧⌘Z / Ctrl+Y)" aria-label="Redo">
            <Icon name="redo" />
          </button>
          {phone.value ? (
            <TopMenu onSettings={() => (settingsOpen.value = true)} />
          ) : (
            <>
              <button class="icon-btn" onClick={toggleTheme} title={theme.value === 'dark' ? 'Light theme' : 'Dark theme'} aria-label={theme.value === 'dark' ? 'Switch to the light theme' : 'Switch to the dark theme'}>
                <Icon name={theme.value === 'dark' ? 'sun' : 'moon'} />
              </button>
              <button id="keys-btn" class="icon-btn" onClick={() => (showKeys.value = !showKeys.value)} title="Keyboard shortcuts (?)" aria-label="Keyboard shortcuts">
                ?
              </button>
            </>
          )}
        </div>
      </header>
      <BackendBanner />
      <main class="stage" id="stage">
        {!backend.value ? (
          <BootPanel />
        ) : act.value === 'cast' ? (
          <CastScreen />
        ) : act.value === 'direct' ? (
          <DirectScreen />
        ) : (
          <EditScreen />
        )}
      </main>
      <div class="toasts" onPointerEnter={() => pauseToasts(true)} onPointerLeave={() => pauseToasts(false)}>
        {encoderDownload.value && (
          <div class="toast toast-dl">
            <DownloadBar d={encoderDownload.value} />
          </div>
        )}
        <div class="toasts-live" aria-live="polite">
          {toasts.value.filter((t) => t.kind !== 'error').map((t) => <ToastRow key={t.id} t={t} />)}
        </div>
        <div class="toasts-live" aria-live="assertive">
          {toasts.value.filter((t) => t.kind === 'error').map((t) => <ToastRow key={t.id} t={t} />)}
        </div>
      </div>
      <Keys />
      <BackendPanel />
      <ProjectSettings />
      <PronEditor />
      <ProjectLauncher />
      <ConfirmDialog />
    </div>
  );
}
