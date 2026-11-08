import { useEffect, useRef, useState } from 'preact/hooks';
import { embeddingOnly, type Speaker } from '../../api';
import { nowPlaying } from '../../audio/player';
import { FileDrop } from '../../components/FileDrop';
import { Icon } from '../../components/Icon';
import { confirm } from '../../components/ConfirmDialog';
import { notify } from '../../state/app';
import { design, designs, switchBase } from '../../state/design';
import {
  cast,
  clipLoading,
  ensureEmbedding,
  recording,
  removePersona,
  restorePersona,
  speakers,
  speakersError,
  speakersLoading,
  refreshSpeakers,
  startRecording,
  toggleClip,
  uploadClip,
  uploading,
  type Persona,
} from '../../state/library';
import { clearVoice, script, setLinesVoice } from '../../state/script';

/** Remove a persona from the cast; lines that spoke as it go back to the script voice (asked first), with Undo. */
async function removeFromCast(p: Persona) {
  const s = script.value;
  const n = s.lines.filter((l) => l.voice === p.id).length + (s.voice === p.id ? 1 : 0);
  if (n > 0) {
    const ok = await confirm(`${n} line${n === 1 ? ' speaks' : 's speak'} as ${p.name}. Remove it? They will fall back to the script voice.`, { ok: 'Remove', danger: true });
    if (!ok) return;
  }
  const wasScriptVoice = s.voice === p.id;
  const ids = n > 0 ? clearVoice(p.id) : [];
  const removed = removePersona(p.id);
  if (!removed) return;
  notify(`Removed “${p.name}”`, 'info', {
    label: 'Undo',
    run: () => {
      restorePersona(removed.persona, removed.index);
      if (ids.length || wasScriptVoice) setLinesVoice(ids, p.id, wasScriptVoice ? p.id : undefined);
    },
  });
}

function VoiceRow({ s }: { s: Speaker }) {
  const on = design.value.baseId === s.id;
  const playing = nowPlaying.value?.id === `clip:${s.id}`;
  const busy = clipLoading.value === s.id;
  return (
    <li class={`voice${on ? ' is-on' : ''}`}>
      <button
        class={`play-mini${playing ? ' is-playing' : ''}${busy ? ' is-busy' : ''}`}
        onClick={() => void toggleClip(s)}
        disabled={s.has_preview === false}
        aria-label={`${playing ? 'Stop' : 'Play'} ${s.label}'s reference clip`}
        title={s.has_preview === false ? 'No reference clip (designed voice)' : 'Reference clip'}
      >
        <Icon name={playing ? 'stop' : 'play'} size={12} />
      </button>
      <button class="voice-name" onClick={() => switchBase(s.id, 'library')} aria-pressed={on} title={on ? 'Current base voice' : 'Use as the base voice'}>
        <span class="voice-label">{s.label}</span>
        {embeddingOnly(s) && <span class="tag">designed</span>}
      </button>
    </li>
  );
}

function Recorder() {
  const rec = recording.value;
  const [, tick] = useState(0);
  useEffect(() => {
    if (!rec) return;
    const t = setInterval(() => tick((n) => n + 1), 250);
    return () => clearInterval(t);
  }, [rec]);
  if (rec) {
    const s = Math.floor((Date.now() - rec.started) / 1000);
    return (
      <button class="btn btn-rec is-recording" onClick={rec.stop}>
        <span class="rec-dot" /> Stop · 0:{String(s).padStart(2, '0')}
      </button>
    );
  }
  return (
    <button
      class="btn btn-rec"
      disabled={uploading.value}
      onClick={() => void startRecording((s) => s && (switchBase(s.id, 'clip'), void ensureEmbedding(s)))}
      title="Record 5-30 s of clean speech to clone a voice"
    >
      <Icon name="mic" /> Record
    </button>
  );
}

function PersonaRow({ p }: { p: Persona }) {
  const base = speakers.value.find((s) => s.id === p.design.baseId);
  return (
    <li class="voice persona">
      <button class="voice-name" onClick={() => designs.apply(structuredClone(p.design))} title={`Load ${p.name} onto the bench`}>
        {p.name}
        <span class="sub">{base?.label ?? 'missing base voice'}</span>
      </button>
      <button
        class="icon-btn sm"
        aria-label={`Duplicate ${p.name}`}
        title="Duplicate onto the bench"
        onClick={() => {
          designs.apply(structuredClone(p.design));
          notify(`Loaded a copy of ${p.name}; save it under a new name to keep both`);
        }}
      >
        <Icon name="copy" size={14} />
      </button>
      <button class="icon-btn sm" aria-label={`Remove ${p.name}`} title="Remove from cast" onClick={() => void removeFromCast(p)}>
        <Icon name="trash" size={14} />
      </button>
    </li>
  );
}

/** A chosen or dropped clip becomes a voice of the project and the bench's base. */
async function cloneFile(f: File) {
  const s = await uploadClip(f, f.name.replace(/\.[^.]+$/, ''), f.name);
  if (s) {
    switchBase(s.id, 'clip');
    void ensureEmbedding(s);
  }
}

export function LibraryRail() {
  const file = useRef<HTMLInputElement>(null);
  const defaults = speakers.value.filter((s) => s.is_default);
  const mine = speakers.value.filter((s) => !s.is_default);
  return (
    <aside class="rail" aria-label="Voice library">
      <div class="rail-section">
        <h4>Voices</h4>
        {speakersError.value ? (
          <div class="error-line">
            {speakersError.value}
            <button class="link" onClick={() => void refreshSpeakers()}>
              retry
            </button>
          </div>
        ) : speakersLoading.value && !defaults.length ? (
          <p class="muted">Loading voices…</p>
        ) : (
          <ul class="voices">{defaults.map((s) => <VoiceRow key={s.id} s={s} />)}</ul>
        )}
      </div>
      <div class="rail-section">
        <h4>Your voices</h4>
        {mine.length ? <ul class="voices">{mine.map((s) => <VoiceRow key={s.id} s={s} />)}</ul> : <p class="muted small">Upload, drop or record a clip to clone a voice. Designed voices you save appear here too.</p>}
        <div class="row gap-s">
          <button class="btn" disabled={uploading.value} onClick={() => file.current?.click()}>
            <Icon name="upload" /> {uploading.value ? 'Uploading…' : 'Upload'}
          </button>
          <Recorder />
          <input
            ref={file}
            type="file"
            accept="audio/*"
            hidden
            onChange={(e) => {
              const f = (e.currentTarget as HTMLInputElement).files?.[0];
              (e.currentTarget as HTMLInputElement).value = '';
              if (f) void cloneFile(f);
            }}
          />
        </div>
      </div>
      <div class="rail-section grow">
        <h4>Your cast</h4>
        {cast.value.length ? (
          <ul class="voices">{cast.value.map((p) => <PersonaRow key={p.id} p={p} />)}</ul>
        ) : (
          <p class="muted small">Design a voice, then “Save to cast” to keep it for Direct (each line can use its own voice).</p>
        )}
      </div>
      <FileDrop label="Drop to clone this voice" hint="An audio clip of 5–30 s of clean speech" busy={uploading.value ? 'Wait for the current upload…' : null} onFile={(f) => void cloneFile(f)} />
    </aside>
  );
}
