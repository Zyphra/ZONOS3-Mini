import { signal } from '@preact/signals';
import { useRef, useState } from 'preact/hooks';
import type { Rendered } from '../../api';
import { nowPlaying } from '../../audio/player';
import * as player from '../../audio/player';
import { autoplay, autoRender, ID_A, ID_B, phrase, playSide, preview, previewText, reference, retry, retryReference, side, speak } from '../../audio/preview';
import { errorText, isAbort, render } from '../../audio/render';
import { exportWav } from '../../audio/mix';
import { Icon } from '../../components/Icon';
import { isPronError, PronMarks, PronounceButton } from '../../components/Pron';
import { openRail } from '../../components/Rail';
import { TagButton, TagField, TagMarks } from '../../components/Tags';
import { Waveform } from '../../components/Waveform';
import { uid } from '../../lib/math';
import { backend, notify, projectSettings, requestContext, spec } from '../../state/app';
import { applyPreset, baseSpeaker, design, designs, neighbours, steered, type Design, type Variation } from '../../state/design';
import { presetToValues } from '../../state/engine';
import { pinForMix } from '../../state/atlas';
import { cast, embeddingErrors, embeddingsSynthetic, ensureEmbedding, refreshSpeakers, saveEmbeddingVoice, savePersona } from '../../state/library';
import { buildRequest, PREVIEW_SEED } from '../../state/requests';
import { rememberOrigin, who } from '../../state/who';

/** Scroll a freshly added library / cast row into view and flash it. */
export function revealRow(selector: string) {
  requestAnimationFrame(() => {
    const el = document.querySelector<HTMLElement>(selector);
    if (!el) return;
    el.scrollIntoView({ block: 'nearest' });
    el.classList.add('is-flash');
    setTimeout(() => el.classList.remove('is-flash'), 1200);
  });
}

/** A / B: the design vs the unsteered voice (keys A and B do the same). */
function ABButtons() {
  const which = side.value;
  const playingA = nowPlaying.value?.id === ID_A;
  const playingB = nowPlaying.value?.id === ID_B;
  return (
    <div class="ab" role="group" aria-label="A/B">
      <button class={`ab-btn${which === 'A' ? ' is-on' : ''}${playingA ? ' is-playing' : ''}`} onClick={() => (playingA ? player.stop() : playSide('A'))} title="Your design (A)">
        <b>A</b> design
      </button>
      <button class={`ab-btn${which === 'B' ? ' is-on' : ''}${playingB ? ' is-playing' : ''}`} onClick={() => (playingB ? player.stop() : playSide('B'))} title="The unsteered voice (B)">
        <b>B</b> original
      </button>
    </div>
  );
}

function statusText(): string {
  const p = preview.value;
  const a = p.a;
  if (p.status === 'waiting') return 'waiting…';
  if (p.status === 'rendering') return 'rendering…';
  if (p.status === 'idle') return a ? 'out of date' : design.value.baseId ? 'Press Speak to hear it' : '';
  if (p.status === 'error') return '';
  return a ? `${(a.pcm.length / a.sr).toFixed(1)} s · first audio ${Math.round(a.ttfbMs)} ms` : '';
}

/** The B side failed: the same error box as A, with its own Retry. */
function ReferenceError() {
  const err = reference.value.error;
  if (side.value !== 'B' || !err) return null;
  return (
    <div class="error-box" role="alert">
      <b>Couldn’t render the original voice</b>
      <span>{err}</span>
      <button class="btn sm" onClick={retryReference}>
        <Icon name="refresh" size={14} /> Retry
      </button>
    </div>
  );
}

/** Auto-generate / auto-play, as two toggles beside A/B. */
function PreviewToggles() {
  return (
    <div class="preview-toggles" role="group" aria-label="Preview settings">
      <button
        class={`icon-btn sm${autoRender.value ? ' is-on' : ''}`}
        aria-pressed={autoRender.value}
        onClick={() => (autoRender.value = !autoRender.value)}
        title="Auto-generate on every change (off: only on Speak)"
        aria-label="Auto-generate on every change"
      >
        <Icon name="wand" size={14} />
      </button>
      <button class={`icon-btn sm${autoplay.value ? ' is-on' : ''}`} aria-pressed={autoplay.value} onClick={() => (autoplay.value = !autoplay.value)} title="Auto-play new takes" aria-label="Auto-play new takes">
        <Icon name="speaker" size={14} />
      </button>
    </div>
  );
}

/** The waveform of the side being shown (A: the design, still coming while it renders; B: the unsteered voice). */
function PreviewWave() {
  const p = preview.value;
  const ref = reference.value;
  const isA = side.value === 'A';
  const id = isA ? ID_A : ID_B;
  const take = (isA ? p.a : ref.take) as Rendered | null;
  return (
    <Waveform
      pcm={isA ? (p.status === 'rendering' && p.partial ? p.partial : p.a?.pcm) : ref.take?.pcm}
      color={isA ? 'var(--o)' : 'var(--text-2)'}
      playId={id}
      busy={isA ? p.status === 'waiting' || p.status === 'rendering' : !ref.take && !ref.error}
      height={40}
      label={isA ? 'Designed voice waveform' : 'Original voice waveform'}
      onSeek={(f) => take && player.play(id, take.pcm, take.sr, f * (take.pcm.length / take.sr))}
    />
  );
}

/** One-line preview for dialogs: A/B, waveform, render status. */
export function PreviewStrip() {
  const p = preview.value;
  return (
    <div class="preview-strip" aria-label="Preview">
      <ABButtons />
      <PreviewWave />
      <span class={`metrics${p.status === 'rendering' || p.status === 'waiting' ? ' is-busy' : ''}`} aria-live="polite">
        {p.status === 'error' ? <span class="error-line">{isPronError(p.error) ? 'not rendered: fix the pronunciation' : p.error}</span> : statusText()}
      </span>
    </div>
  );
}

/** The project's preview line from what was typed ('' when it is the bundle's own). */
export function setPhrase(text: string) {
  previewText.value = text.trim() === spec.value.ui.preview_text ? '' : text;
}

/**
 * The first thing on Cast: type a line and hear the bench voice (picked in the library rail) say it. Every design
 * move below re-renders this line, so it is also the A/B monitor for the bench.
 */
export function SpeakPanel() {
  const p = preview.value;
  const a = p.a;
  const [draft, setDraft] = useState<string | null>(null);
  const area = useRef<HTMLTextAreaElement>(null);
  const noSteer = backend.value?.kind === 'fixture';
  const text = draft ?? phrase.value;
  const playingA = nowPlaying.value?.id === ID_A;
  const pending = p.status === 'waiting' || p.status === 'rendering';
  const go = () => {
    // The element's value, not `draft`: an edit made just before (Pronounce) is not in state yet.
    const typed = area.current?.value ?? text;
    if (typed !== phrase.value) setPhrase(typed);
    setDraft(null);
    speak();
  };
  const noVoice = !design.value.baseId;
  const stale = p.status === 'idle' && !!a;
  const isErr = p.status === 'error';
  return (
    <section class={`speak${stale || isErr ? ' is-stale' : ''}`} aria-label="Speak">
      <div class="speak-text">
        <TagField text={text} block>
          <textarea
            ref={area}
            class="speak-area"
            rows={2}
            value={text}
            placeholder="Type what the voice should say"
            aria-label="Text to speak"
            data-tags
            onInput={(e) => setDraft((e.currentTarget as HTMLTextAreaElement).value)}
            onKeyDown={(e) => {
              if (e.key !== 'Enter' || e.shiftKey) return;
              e.preventDefault();
              go();
            }}
            onBlur={(e) => {
              // The tag palette borrows focus and gives it back.
              if ((e.relatedTarget as HTMLElement | null)?.closest?.('.tag-palette, .speak-go')) return;
              const typed = (e.currentTarget as HTMLTextAreaElement).value;
              if (typed !== phrase.value) setPhrase(typed);
              setDraft(null);
            }}
          />
        </TagField>
        <div class="speak-tools">
          <PronounceButton input={() => area.current} />
          <TagButton input={() => area.current} />
        </div>
      </div>
      <PronMarks
        text={text}
        replace={(s, e, t) => {
          setDraft(null);
          setPhrase(text.slice(0, s) + t + text.slice(e));
        }}
      />
      <TagMarks
        text={text}
        replace={(next) => {
          setDraft(null);
          setPhrase(next);
        }}
      />
      <div class="speak-bar">
        <button
          class={`btn primary speak-go${pending ? ' is-busy' : ''}`}
          disabled={noVoice || !text.trim()}
          onClick={() => (playingA ? player.stop() : go())}
          title={noVoice ? 'Pick a voice in Voices first' : 'Speak this text (Enter in the text box)'}
        >
          <Icon name={playingA ? 'stop' : 'play'} size={14} /> {playingA ? 'Stop' : pending ? 'Rendering…' : stale && !autoRender.value ? 'Speak · out of date' : 'Speak'}
        </button>
        {noVoice ? (
          <span class="hint speak-hint">
            <button class="link" onClick={() => openRail('cast-library')}>
              Pick a voice in Voices
            </button>{' '}
            to start
          </span>
        ) : (
          <PreviewWave />
        )}
        <ABButtons />
        <PreviewToggles />
        <span class="metrics" title="clip length · time to first audio">
          {statusText()}
        </span>
        <button class="icon-btn" disabled={!a} onClick={() => a && exportWav(a.pcm, a.sr, `${baseSpeaker.value?.label ?? 'voice'}-preview`)} title="Export this take as WAV" aria-label="Export WAV">
          <Icon name="download" />
        </button>
      </div>
      {isErr && isPronError(p.error) && (
        <div class="error-box" role="alert">
          <b>Fix the pronunciation marked above</b>
          <span>{p.error}</span>
        </div>
      )}
      {isErr && !isPronError(p.error) && (
        <div class="error-box" role="alert">
          <b>Couldn’t render</b>
          <span>{p.error}</span>
          <button class="btn sm" onClick={retry}>
            <Icon name="refresh" size={14} /> Retry
          </button>
        </div>
      )}
      <ReferenceError />
      {noSteer && <p class="note">This server can’t apply controls yet: A and B both play the base voice.</p>}
    </section>
  );
}

export function Personas() {
  const presets = Object.keys(spec.value.ui.presets);
  const d = design.value;
  const isOn = (name: string) => {
    const { values, accents } = presetToValues(spec.value.ui.presets[name]);
    const pin = pinForMix(accents, spec.value.ui.accents, d.pin.tab);
    const keys = new Set([...Object.keys(values), ...Object.keys(d.values).filter((k) => d.values[k])]);
    return [...keys].every((k) => (values[k] ?? 0) === (d.values[k] ?? 0)) && Math.hypot(pin.x - d.pin.x, pin.y - d.pin.y) < 1 && pin.tab === d.pin.tab;
  };
  if (!presets.length) return null;
  return (
    <section class="monitor-card" aria-label="Personas">
      <header class="monitor-head">
        <h3>Personas</h3>
      </header>
      <div class="persona-chips">
        {presets.map((name) => (
          <button class={`persona-chip${isOn(name) ? ' is-on' : ''}`} onClick={() => applyPreset(name)} title={Object.entries(spec.value.ui.presets[name]).map(([k, v]) => `${k.replace('accent:', '')} ${v}`).join(' · ')}>
            {name}
          </button>
        ))}
      </div>
    </section>
  );
}

interface Tile extends Variation {
  id: string;
  audio: Rendered | null;
  error: string | null;
  busy: boolean;
}

const tiles = signal<Tile[]>([]);
const tilesFor = signal<Design | null>(null);
let tileCtrl: AbortController | null = null;

function findVariations() {
  tileCtrl?.abort();
  const ctrl = (tileCtrl = new AbortController());
  const around = design.value;
  tilesFor.value = around;
  tiles.value = neighbours(around).map((v) => ({ ...v, id: uid(), audio: null, error: null, busy: true }));
  for (const t of tiles.value) {
    const req = buildRequest(spec.value, requestContext.value, t.design, phrase.value, PREVIEW_SEED, { language: projectSettings.value.language });
    render(req, { signal: ctrl.signal })
      .then((audio) => (tiles.value = tiles.value.map((x) => (x.id === t.id ? { ...x, audio, busy: false } : x))))
      .catch((e) => !isAbort(e) && (tiles.value = tiles.value.map((x) => (x.id === t.id ? { ...x, error: errorText(e), busy: false } : x))));
  }
}

export function Variations() {
  const list = tiles.value;
  const d = design.value;
  // The tile that was stepped to is "current": moving onto it is not moving away from the set.
  const current = list.find((t) => t.design === d) ?? null;
  const stale = tilesFor.value !== null && tilesFor.value !== d && !current;
  const rendering = list.filter((t) => t.busy).length;
  const allFailed = list.length > 0 && list.every((t) => t.error);
  return (
    <section class="monitor-card" aria-label="Variations">
      <header class="monitor-head">
        <h3>Voices near this one</h3>
        <button class="btn sm" onClick={findVariations} disabled={!d.baseId || rendering > 0}>
          <Icon name="dice" size={14} /> {rendering ? `Rendering ${list.length - rendering}/${list.length}…` : list.length ? 'Five more' : 'Find five'}
        </button>
      </header>
      {!list.length ? (
        <p class="muted small">Five small, safe moves around this voice, rendered side by side. Click one to step there.</p>
      ) : allFailed ? (
        <div class="error-line">
          Couldn’t render: {list[0].error}{' '}
          <button class="link" onClick={findVariations}>
            Retry
          </button>
        </div>
      ) : (
        <ul class={`tiles${stale ? ' is-stale' : ''}`}>
          {list.map((t) => {
            const id = `var:${t.id}`;
            const playing = nowPlaying.value?.id === id;
            return (
              <li class={`tile${t === current ? ' is-current' : ''}`}>
                <button class={`play-mini${playing ? ' is-playing' : ''}${t.busy ? ' is-busy' : ''}`} disabled={!t.audio} onClick={() => t.audio && player.toggle(id, t.audio.pcm, t.audio.sr)} aria-label={`Play variation: ${t.changes.join(', ')}`}>
                  <Icon name={playing ? 'stop' : 'play'} size={12} />
                </button>
                <button
                  class="tile-body"
                  aria-current={t === current ? 'true' : undefined}
                  onClick={() => {
                    if (t === current) return;
                    designs.apply(t.design);
                    if (t.audio) player.play(id, t.audio.pcm, t.audio.sr);
                    notify(`Stepped to: ${t.changes.join(' · ')}`, 'info', { label: 'Undo', run: () => designs.undo() });
                  }}
                  title={t === current ? 'The voice on the bench' : 'Step to this voice (undo with ⌘Z)'}
                >
                  <span class="tile-changes">{t.changes.join(' · ')}</span>
                  {t.error ? <span class="error-line small">{t.error}</span> : <Waveform pcm={t.audio?.pcm} color="var(--o)" playId={id} height={22} busy={t.busy} />}
                </button>
              </li>
            );
          })}
        </ul>
      )}
      {stale && list.length > 0 && <p class="note">You moved since these were found; they still sit around the earlier voice.</p>}
    </section>
  );
}

/** `name`, or "name 2", "name 3"… when the cast already has it. */
function uniqueName(name: string): string {
  const taken = new Set(cast.value.map((p) => p.name));
  if (!taken.has(name)) return name;
  for (let i = 2; ; i++) if (!taken.has(`${name} ${i}`)) return `${name} ${i}`;
}

export function SavePanel() {
  const [name, setName] = useState('');
  const [signature, setSignature] = useState('');
  const [busy, setBusy] = useState(false);
  const d = design.value;
  const b = backend.value!;
  const embErr = embeddingErrors.value[d.baseId];
  const canServer = d.mode === 'embedding' && !!steered.value && (b.kind === 'offline' || !embeddingsSynthetic.value) && b.steering;
  const why = d.mode !== 'embedding'
    ? 'Exact clones are already library voices.'
    : !b.steering
      ? 'This server can’t apply controls yet, so there is no designed voice to save.'
      : embeddingsSynthetic.value
        ? 'Needs the server’s real embedding for this voice.'
        : embErr
          ? `Couldn't fetch this voice's embedding: ${embErr}`
          : !steered.value
            ? 'Waiting for the base voice’s embedding…'
            : null;
  const label = name.trim() || `${baseSpeaker.value?.label ?? 'Voice'} variant`;

  const toCast = () => {
    const final = uniqueName(label);
    const p = { id: uid(), name: final, design: structuredClone(d), signature: signature.trim() || phrase.value, createdAt: Date.now() };
    savePersona(p);
    notify(`“${final}” joined your cast`, 'ok');
    setName('');
    setSignature('');
    revealRow(`[data-persona="${p.id}"]`);
  };
  const toServer = async () => {
    if (!steered.value) return;
    setBusy(true);
    try {
      const s = await saveEmbeddingVoice(label, steered.value);
      const g = who.value;
      rememberOrigin(s, baseSpeaker.value, d.values[g.gender.control] ?? 0, d.values[g.age.control] ?? 0);
      await refreshSpeakers();
      notify(`Saved “${s.label}” as a voice`, 'ok');
      revealRow(`[data-voice="${s.id}"]`);
    } catch (e) {
      notify(`Couldn’t save the voice: ${errorText(e)}`, 'error');
    } finally {
      setBusy(false);
    }
  };

  return (
    <section class="monitor-card save" aria-label="Save">
      <header class="monitor-head">
        <h3>Keep it</h3>
      </header>
      <form
        class="col gap-s"
        onSubmit={(e) => {
          e.preventDefault();
          if (d.baseId) toCast();
        }}
      >
        <input class="input" placeholder={label} value={name} onInput={(e) => setName((e.currentTarget as HTMLInputElement).value)} aria-label="Name" />
        <input class="input" placeholder="Signature line: what this voice says when loaded (optional)" value={signature} onInput={(e) => setSignature((e.currentTarget as HTMLInputElement).value)} aria-label="Signature line" />
        <div class="row gap-s">
          <button type="submit" class="btn primary" disabled={!d.baseId} title="Save the whole design (voice, accent, delivery, mood) to this project's cast">
            <Icon name="plus" size={14} /> Save to cast
          </button>
          <button type="button" class="btn" onClick={() => void toServer()} disabled={!canServer || busy} title={why ?? 'Caches the steered embedding on the server as a new library voice (delivery and mood stay in the persona)'}>
            <Icon name="save" size={14} /> {busy ? 'Saving…' : 'Save as voice'}
          </button>
        </div>
      </form>
      {why && (
        <p class="hint">
          {why}
          {embErr && (
            <>
              {' · '}
              <button class="link" onClick={() => baseSpeaker.value && void ensureEmbedding(baseSpeaker.value, true)}>
                Retry
              </button>
            </>
          )}
        </p>
      )}
    </section>
  );
}
