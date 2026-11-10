import { useEffect, useRef, useState } from 'preact/hooks';
import { embeddingOnly, type Speaker } from '../../api';
import { nowPlaying } from '../../audio/player';
import * as player from '../../audio/player';
import { phrase } from '../../audio/preview';
import { FileDrop } from '../../components/FileDrop';
import { Icon } from '../../components/Icon';
import { confirm } from '../../components/ConfirmDialog';
import { NameInput } from '../../components/Projects';
import { Waveform } from '../../components/Waveform';
import { notify } from '../../state/app';
import { design, designs, switchBase } from '../../state/design';
import { byLanguage } from '../../lib/voices';
import { revealRow, setPhrase } from './Monitor';
import {
  cast,
  clipLoading,
  ensureEmbedding,
  pendingClip,
  RECORD_MAX_S,
  recording,
  recordLevel,
  removePersona,
  removeVoice,
  renamePersona,
  renameVoice,
  restorePersona,
  speakers,
  speakersError,
  speakersLoading,
  refreshSpeakers,
  startRecording,
  toggleClip,
  uploadClip,
  uploading,
  usePendingClip,
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

/** Drop one of the project's voices (lines that spoke as it fall back to the default voice, asked first), with Undo. */
async function removeFromVoices(s: Speaker) {
  const sc = script.value;
  const n = sc.lines.filter((l) => l.voice === s.id).length + (sc.voice === s.id ? 1 : 0);
  if (n > 0) {
    const ok = await confirm(`${n} line${n === 1 ? ' speaks' : 's speak'} as ${s.label}. Remove it? They will fall back to the default voice.`, { ok: 'Remove', danger: true });
    if (!ok) return;
  }
  const wasScriptVoice = sc.voice === s.id;
  const ids = n > 0 ? clearVoice(s.id) : [];
  const wasBase = design.value.baseId === s.id;
  const restore = removeVoice(s.id);
  if (!restore) return;
  notify(`Removed “${s.label}” from this project`, 'info', {
    label: 'Undo',
    run: () => {
      restore();
      if (ids.length || wasScriptVoice) setLinesVoice(ids, s.id, wasScriptVoice ? s.id : undefined);
      if (wasBase) switchBase(s.id, 'library');
    },
  });
}

function VoiceRow({ s, own }: { s: Speaker; own?: boolean }) {
  const on = design.value.baseId === s.id;
  const playing = nowPlaying.value?.id === `clip:${s.id}`;
  const busy = clipLoading.value === s.id;
  const [renaming, setRenaming] = useState(false);
  return (
    <li class={`voice${on ? ' is-on' : ''}`} data-voice={s.id}>
      <button
        class={`play-mini${playing ? ' is-playing' : ''}${busy ? ' is-busy' : ''}`}
        onClick={() => void toggleClip(s)}
        disabled={s.has_preview === false}
        aria-label={`${playing ? 'Stop' : 'Play'} ${s.label}'s reference clip`}
        title={s.has_preview === false ? 'No reference clip (designed voice)' : 'Reference clip'}
      >
        <Icon name={playing ? 'stop' : 'play'} size={12} />
      </button>
      {renaming ? (
        <NameInput
          value={s.label}
          label={`Rename ${s.label}`}
          onDone={(name) => {
            setRenaming(false);
            if (name) renameVoice(s.id, name);
          }}
        />
      ) : (
        <button class="voice-name" onClick={() => switchBase(s.id, 'library')} aria-pressed={on} title={on ? 'Current base voice' : 'Use as the base voice'}>
          <span class="voice-label">{s.label}</span>
          {embeddingOnly(s) && <span class="tag">designed</span>}
        </button>
      )}
      {own && (
        <>
          <button class="icon-btn sm" aria-label={`Rename ${s.label}`} title="Rename (in this project)" onClick={() => setRenaming(true)}>
            <Icon name="edit" size={14} />
          </button>
          <button class="icon-btn sm" aria-label={`Remove ${s.label}`} title="Removes it from this project (the server keeps it for this session)" onClick={() => void removeFromVoices(s)}>
            <Icon name="trash" size={14} />
          </button>
        </>
      )}
    </li>
  );
}

/** A finished recording: listen, then Use (it becomes a voice and the base) or Discard. */
function PendingClip() {
  const clip = pendingClip.value;
  if (!clip) return null;
  const id = 'clip:pending';
  const playing = nowPlaying.value?.id === id;
  const use = async () => {
    const s = await usePendingClip();
    if (s) {
      switchBase(s.id, 'clip');
      void ensureEmbedding(s);
      revealRow(`[data-voice="${s.id}"]`);
    }
  };
  return (
    <div class="pending-clip" role="group" aria-label="Your recording">
      <div class="row gap-s">
        <button class={`play-mini${playing ? ' is-playing' : ''}`} onClick={() => player.toggle(id, clip.pcm, clip.sr)} aria-label={playing ? 'Stop' : 'Play the recording'}>
          <Icon name={playing ? 'stop' : 'play'} size={12} />
        </button>
        <Waveform pcm={clip.pcm} color="var(--o)" playId={id} height={28} label="Recording waveform" onSeek={(f) => player.play(id, clip.pcm, clip.sr, f * clip.seconds)} />
        <span class="metrics">{clip.seconds.toFixed(1)} s</span>
      </div>
      <div class="row gap-s">
        <button class="btn sm primary" disabled={uploading.value} onClick={() => void use()}>
          <Icon name="plus" size={12} /> Use
        </button>
        <button class="btn sm" onClick={() => (player.isPlaying(id) && player.stop(), (pendingClip.value = null))}>
          Discard
        </button>
        {clip.seconds < 5 && <span class="hint">Short: 5–30 s clones best</span>}
      </div>
    </div>
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
      <button class="btn btn-rec is-recording" onClick={rec.stop} title={`Recording stops by itself at ${RECORD_MAX_S} s`}>
        <span class="rec-dot" /> Stop · 0:{String(s).padStart(2, '0')} / 0:{RECORD_MAX_S}
        <span class="rec-level" aria-hidden="true">
          <span style={{ width: `${Math.round(recordLevel.value * 100)}%` }} />
        </span>
      </button>
    );
  }
  return (
    <button class="btn btn-rec" disabled={uploading.value || !!pendingClip.value} onClick={() => void startRecording()} title="Record 5-30 s of clean speech to clone a voice">
      <Icon name="mic" /> Record
    </button>
  );
}

/** Load a cast member onto the bench (one undo step). Its signature line is offered, never forced over a typed one. */
function loadPersona(p: Persona) {
  const base = speakers.value.find((s) => s.id === p.design.baseId);
  if (!base) notify(`${p.name}'s voice is gone from the server; pick a base voice`, 'error');
  designs.apply(structuredClone(p.design));
  const sig = p.signature.trim();
  const current = phrase.value.trim();
  if (sig && (!current || current === sig)) {
    if (current !== sig) setPhrase(sig);
    notify(`Loaded ${p.name}`, 'info', { label: 'Undo', run: () => designs.undo() });
    return;
  }
  notify(`Loaded ${p.name}`, 'info', sig ? { label: 'Use its line', run: () => setPhrase(sig) } : { label: 'Undo', run: () => designs.undo() });
}

function PersonaRow({ p }: { p: Persona }) {
  const base = speakers.value.find((s) => s.id === p.design.baseId);
  const [renaming, setRenaming] = useState(false);
  return (
    <li class={`voice persona${base ? '' : ' is-missing'}`} data-persona={p.id}>
      {renaming ? (
        <NameInput
          value={p.name}
          label={`Rename ${p.name}`}
          onDone={(name) => {
            setRenaming(false);
            if (name) renamePersona(p.id, name);
          }}
        />
      ) : (
        <button class="voice-name" onClick={() => loadPersona(p)} title={base ? `Load ${p.name} onto the bench` : `${p.name}'s base voice is gone from the server`}>
          {p.name}
          <span class="sub">{base?.label ?? 'voice unavailable'}</span>
        </button>
      )}
      <button class="icon-btn sm" aria-label={`Rename ${p.name}`} title="Rename" onClick={() => setRenaming(true)}>
        <Icon name="edit" size={14} />
      </button>
      <button class="icon-btn sm" aria-label={`Remove ${p.name}`} title="Remove from cast" onClick={() => void removeFromCast(p)}>
        <Icon name="trash" size={14} />
      </button>
    </li>
  );
}

/** A chosen or dropped clip becomes a voice of the project and the bench's base. */
async function cloneFile(f: File) {
  const s = await uploadClip(f, f.name.replace(/\.[^.]+$/, ''), f.name, true);
  if (s) {
    switchBase(s.id, 'clip');
    void ensureEmbedding(s);
    revealRow(`[data-voice="${s.id}"]`);
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
          <div class="error-box" role="alert">
            <b>Couldn’t load the voices</b>
            <span>{speakersError.value}</span>
            <button class="btn sm" onClick={() => void refreshSpeakers()}>
              <Icon name="refresh" size={14} /> Retry
            </button>
          </div>
        ) : speakersLoading.value && !defaults.length ? (
          <p class="muted">Loading voices…</p>
        ) : (
          byLanguage(defaults).map((g) => (
            <section key={g.label} aria-label={g.label ?? undefined}>
              {g.label && <h5 class="voice-group">{g.label}</h5>}
              <ul class="voices">{g.voices.map((s) => <VoiceRow key={s.id} s={s} />)}</ul>
            </section>
          ))
        )}
      </div>
      <div class="rail-section">
        <h4>Your voices</h4>
        {mine.length ? <ul class="voices">{mine.map((s) => <VoiceRow key={s.id} s={s} own />)}</ul> : <p class="muted small">Upload, drop or record a clip to clone a voice. Designed voices you save appear here too.</p>}
        <PendingClip />
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
