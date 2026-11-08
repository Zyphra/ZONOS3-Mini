/** The open project's settings in one dialog: language, voice mode, tag strength, and what opening it did to its voices. */
import { useLayoutEffect, useRef } from 'preact/hooks';
import { LANG_LABEL } from '../lib/languages';
import { backend, modalOpen, projectSettings, settingsOpen } from '../state/app';
import { current, restoreReport, setSettings } from '../state/projects';
import { dialogKeys } from './focusTrap';
import { Icon } from './Icon';

export function ProjectSettings() {
  const open = settingsOpen.value;
  const box = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    if (!open) return;
    const back = document.activeElement as HTMLElement | null;
    modalOpen.value = true;
    box.current?.querySelector<HTMLElement>('select')?.focus({ preventScroll: true });
    return () => {
      modalOpen.value = false;
      (back?.isConnected ? back : document.querySelector<HTMLElement>('.proj-trigger'))?.focus({ preventScroll: true });
    };
  }, [open]);
  const p = current.value;
  if (!open || !p) return null;
  const close = () => (settingsOpen.value = false);
  const st = projectSettings.value;
  const tags = backend.value?.tags;
  const r = restoreReport.value;
  return (
    <div class="sheet-backdrop" onClick={close}>
      <div ref={box} class="sheet settings-sheet" role="dialog" aria-modal="true" aria-labelledby="settings-title" onClick={(e) => e.stopPropagation()} onKeyDown={dialogKeys(() => box.current, close)}>
        <header class="sheet-head">
          <h2 id="settings-title">Project settings</h2>
          <button class="icon-btn" onClick={close} aria-label="Close" title="Close (Esc)">
            <Icon name="x" />
          </button>
        </header>
        <p class="muted small">{p.name}</p>
        <div class="settings-grid">
          <label class="settings-row">
            <span>Language</span>
            <select class="input" value={st.language} onChange={(e) => setSettings({ language: e.currentTarget.value })} aria-describedby="settings-lang-hint">
              {Object.entries(LANG_LABEL).map(([code, label]) => (
                <option key={code} value={code}>
                  {label}
                </option>
              ))}
            </select>
            <span class="hint" id="settings-lang-hint">
              Direct lines and the Cast preview render in it: numbers and abbreviations are read in this language, and painted words follow the reading where the server can. New Edit clips and pronunciation lookups start in it.
            </span>
          </label>
          <label class="settings-row">
            <span>Voice mode</span>
            <select class="input" value={st.voiceMode} onChange={(e) => setSettings({ voiceMode: e.currentTarget.value as 'embedding' | 'reference' })} aria-describedby="settings-mode-hint">
              <option value="embedding">Designable</option>
              <option value="reference">Exact clone</option>
            </select>
            <span class="hint" id="settings-mode-hint">
              How the Cast bench starts when a recorded voice is picked: designable (its embedding, every control works) or an exact clone of the clip (delivery and mood only).
            </span>
          </label>
          {tags && (
            <label class="settings-row">
              <span>Tag strength</span>
              <span class="settings-range">
                <input
                  type="range"
                  min={Math.max(1, tags.min_scale)}
                  max={tags.max_scale}
                  step={0.1}
                  value={st.tagStrength}
                  onInput={(e) => setSettings({ tagStrength: Number(e.currentTarget.value) })}
                  aria-valuetext={st.tagStrength > 1 ? `${st.tagStrength.toFixed(1)}, tag guidance on` : '1, off'}
                  aria-describedby="settings-tags-hint"
                />
                <span class="mono">{st.tagStrength > 1 ? st.tagStrength.toFixed(1) : 'off'}</span>
              </span>
              <span class="hint" id="settings-tags-hint">
                Guidance for lines with nonverbal tags (laugh, sigh…): stronger makes the tags louder. On those lines it replaces voice guidance (the server allows one at a time); 1 turns it off.
              </span>
            </label>
          )}
        </div>
        {r.reRegistered.length > 0 && <p class="proj-note">Restored after a server restart: {r.reRegistered.join(', ')}.</p>}
        {r.issues.length > 0 && (
          <ul class="proj-issues" aria-label="Voices that couldn’t be restored">
            {r.issues.map((x) => (
              <li key={x}>Couldn’t restore {x}</li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
