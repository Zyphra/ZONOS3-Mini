/** The open project's settings in one dialog: language, voice mode, decoder, LACI, and what opening it did to its voices. */
import { useLayoutEffect, useRef } from 'preact/hooks';
import { decoderName } from '../lib/decoders';
import { LACI_LABEL, LACI_MODES, type LaciMode } from '../lib/laci';
import { LANG_LABEL, langLabel } from '../lib/languages';
import { backend, modalOpen, projectSettings, requestContext, settingsOpen } from '../state/app';
import { current, restoreReport, setSettings } from '../state/projects';
import { dialogKeys } from './focusTrap';
import { Icon } from './Icon';

/**
 * The audio decoder renders use, when the server serves a choice (two or more). It shows what requests send: a
 * saved choice this server lacks reads as the default. The default is stored as null (requests leave it out).
 */
export function DecoderPicker() {
  const decoders = backend.value?.decoders ?? [];
  if (decoders.length < 2) return null;
  const id = requestContext.value.decoder ?? decoders[0].id;
  return (
    <label class="settings-row">
      <span>Decoder</span>
      <select
        class="input"
        value={id}
        title={decoders.find((d) => d.id === id)?.label}
        onChange={(e) => setSettings({ decoder: e.currentTarget.value === decoders[0].id ? null : e.currentTarget.value })}
        aria-describedby="settings-decoder-hint"
      >
        {decoders.map((d, i) => (
          <option key={d.id} value={d.id} title={d.label}>
            {decoderName(d.id)}
            {i === 0 ? ' (default)' : ''}
          </option>
        ))}
      </select>
      <span class="hint" id="settings-decoder-hint">
        What turns the model’s codes into sound, for every render (Cast previews, Direct lines, Edit takes). Fast (keep2) runs a pruned decoder; Full-size runs all of its layers (heavier). Direct takes rendered with another decoder read as stale.
      </span>
    </label>
  );
}

/** LACI for every render, when the server has alignment heads (`/v2/capabilities.laci_languages`). */
export function LaciPicker() {
  const languages = backend.value?.laciLanguages ?? [];
  if (!languages.length) return null;
  const st = projectSettings.value;
  return (
    <label class="settings-row">
      <span>LACI</span>
      <select class="input" value={st.laci} onChange={(e) => setSettings({ laci: e.currentTarget.value as LaciMode })} aria-describedby="settings-laci-hint">
        {LACI_MODES.map((m) => (
          <option key={m} value={m}>
            {LACI_LABEL[m]}
          </option>
        ))}
      </select>
      <span class="hint" id="settings-laci-hint">
        Watches the model read the text and regenerates a take from where it skipped words, babbled or stopped early. Streamed: playback starts about a phrase later. Whole take: audio comes when the take is done, and every fix restarts exactly at its slip.
        {!languages.includes(st.language) && ` No alignment heads for ${langLabel(st.language)} on this server: its renders run without LACI.`}
      </span>
    </label>
  );
}

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
              Direct lines and the Cast preview render in it: numbers and abbreviations are read in this language, and painted words follow the reading where the server can. A library voice of another language (German, French…) speaks its own instead; English voices keep a UK English project. New Edit clips and pronunciation lookups start in it.
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
          <DecoderPicker />
          <LaciPicker />
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
