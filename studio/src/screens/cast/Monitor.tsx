import { signal } from '@preact/signals';
import { useRef, useState } from 'preact/hooks';
import type { Rendered } from '../../api';
import { nowPlaying } from '../../audio/player';
import * as player from '../../audio/player';
import { autoplay, autoRender, ID_A, ID_B, phrase, playSide, preview, previewText, reference, retry, side, speak } from '../../audio/preview';
import { errorText, isAbort, render } from '../../audio/render';
import { exportWav } from '../../audio/mix';
import { Icon } from '../../components/Icon';
import { isPronError, PronMarks, PronounceButton } from '../../components/Pron';
import { TagButton, TagField, TagMarks } from '../../components/Tags';
import { Waveform } from '../../components/Waveform';
import { uid } from '../../lib/math';
import { backend, notify, requestContext, spec } from '../../state/app';
import { applyPreset, baseSpeaker, design, designs, neighbours, steered, type Design, type Variation } from '../../state/design';
import { presetToValues } from '../../state/engine';
import { pinForMix } from '../../state/atlas';
import { embeddingsSynthetic, refreshSpeakers, saveEmbeddingVoice, savePersona } from '../../state/library';
import { buildRequest, PREVIEW_SEED } from '../../state/requests';
import { rememberOrigin, who } from '../../state/who';

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
  if (p.status === 'waiting') return 'listening…';
  if (p.status === 'rendering') return 'rendering…';
  if (p.status === 'idle') return a ? 'out of date' : 'not generated yet';
  return a ? `${(a.pcm.length / a.sr).toFixed(1)} s · ${Math.round(a.ttfbMs)} ms` : '';
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
function setPhrase(text: string) {
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
  return (
    <section class={`speak${p.status === 'idle' && a ? ' is-stale' : ''}`} aria-label="Speak">
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
          disabled={!design.value.baseId || !text.trim()}
          onClick={() => (playingA ? player.stop() : go())}
          title="Speak this text (Enter in the text box)"
        >
          <Icon name={playingA ? 'stop' : 'play'} size={14} /> {playingA ? 'Stop' : pending ? 'Rendering…' : 'Speak'}
        </button>
        <PreviewWave />
        <ABButtons />
        <span class="metrics" title="clip length · time to first audio">
          {statusText()}
        </span>
        <button class="icon-btn" disabled={!a} onClick={() => a && exportWav(a.pcm, a.sr, `${baseSpeaker.value?.label ?? 'voice'}-preview`)} title="Export this take as WAV" aria-label="Export WAV">
          <Icon name="download" />
        </button>
      </div>
      {p.status === 'error' && !isPronError(p.error) && (
        <div class="error-box" role="alert">
          <b>The server refused this render.</b>
          <span>{p.error}</span>
          <button class="btn sm" onClick={retry}>
            <Icon name="refresh" size={14} /> Retry
          </button>
        </div>
      )}
      {noSteer && <p class="note">This server can’t apply controls yet: A and B both play the base voice.</p>}
    </section>
  );
}

/** Right column: when the Speak take renders and plays. */
export function PreviewSettings() {
  return (
    <section class="monitor-card preview-settings" aria-label="Preview settings">
      <header class="monitor-head">
        <h3>Preview</h3>
      </header>
      <label class="toggle" title="Generate again after every change to the text or the voice (off: only when you press Speak)">
        <input type="checkbox" checked={autoRender.value} onChange={(e) => (autoRender.value = (e.currentTarget as HTMLInputElement).checked)} />
        <span>Auto-generate on every change</span>
      </label>
      <label class="toggle" title="Play every new render automatically">
        <input type="checkbox" checked={autoplay.value} onChange={(e) => (autoplay.value = (e.currentTarget as HTMLInputElement).checked)} />
        <span>Auto-play new takes</span>
      </label>
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
    const req = buildRequest(spec.value, requestContext.value, t.design, phrase.value, PREVIEW_SEED);
    render(req, { signal: ctrl.signal })
      .then((audio) => (tiles.value = tiles.value.map((x) => (x.id === t.id ? { ...x, audio, busy: false } : x))))
      .catch((e) => !isAbort(e) && (tiles.value = tiles.value.map((x) => (x.id === t.id ? { ...x, error: errorText(e), busy: false } : x))));
  }
}

export function Variations() {
  const list = tiles.value;
  const stale = tilesFor.value !== null && tilesFor.value !== design.value;
  return (
    <section class="monitor-card" aria-label="Variations">
      <header class="monitor-head">
        <h3>Voices near this one</h3>
        <button class="btn sm" onClick={findVariations} disabled={!design.value.baseId}>
          <Icon name="dice" size={14} /> {list.length ? 'Five more' : 'Find five'}
        </button>
      </header>
      {!list.length ? (
        <p class="muted small">Five small, safe moves around this voice, rendered side by side. Click one to step there.</p>
      ) : (
        <ul class={`tiles${stale ? ' is-stale' : ''}`}>
          {list.map((t) => {
            const id = `var:${t.id}`;
            const playing = nowPlaying.value?.id === id;
            return (
              <li class="tile">
                <button class={`play-mini${playing ? ' is-playing' : ''}${t.busy ? ' is-busy' : ''}`} disabled={!t.audio} onClick={() => t.audio && player.toggle(id, t.audio.pcm, t.audio.sr)} aria-label={`Play variation: ${t.changes.join(', ')}`}>
                  <Icon name={playing ? 'stop' : 'play'} size={12} />
                </button>
                <button
                  class="tile-body"
                  onClick={() => {
                    designs.apply(t.design);
                    if (t.audio) player.play(id, t.audio.pcm, t.audio.sr);
                  }}
                  title="Step to this voice (undo with ⌘Z)"
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

export function SavePanel() {
  const [name, setName] = useState('');
  const [signature, setSignature] = useState('');
  const [busy, setBusy] = useState(false);
  const d = design.value;
  const b = backend.value!;
  const canServer = d.mode === 'embedding' && !!steered.value && (b.kind === 'offline' || !embeddingsSynthetic.value) && b.steering;
  const why = d.mode !== 'embedding'
    ? 'Exact clones are already library voices.'
    : !b.steering
      ? 'This server can’t apply controls yet, so there is no designed voice to save.'
      : embeddingsSynthetic.value
        ? 'Needs the server’s real embedding for this voice.'
      : !steered.value
        ? 'Waiting for the base voice’s embedding…'
        : 'Caches the steered embedding on the server as a new library voice (delivery and mood stay in the persona).';
  const label = name.trim() || `${baseSpeaker.value?.label ?? 'Voice'} variant`;

  const toCast = () => {
    savePersona({ id: uid(), name: label, design: structuredClone(d), signature: signature.trim() || phrase.value, createdAt: Date.now() });
    notify(`“${label}” joined your cast`, 'ok');
    setName('');
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
      <input class="input" placeholder={label} value={name} onInput={(e) => setName((e.currentTarget as HTMLInputElement).value)} aria-label="Name" />
      <input class="input" placeholder="Signature line (optional)" value={signature} onInput={(e) => setSignature((e.currentTarget as HTMLInputElement).value)} aria-label="Signature line" />
      <div class="row gap-s">
        <button class="btn primary" onClick={toCast} disabled={!d.baseId} title="Save the whole design (voice, accent, delivery, mood) to your cast in this browser">
          <Icon name="plus" size={14} /> Save to cast
        </button>
        <button class="btn" onClick={() => void toServer()} disabled={!canServer || busy} title={why}>
          <Icon name="save" size={14} /> {busy ? 'Saving…' : 'Save as voice'}
        </button>
      </div>
      <p class="hint">{why}</p>
    </section>
  );
}
