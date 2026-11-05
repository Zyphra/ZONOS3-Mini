/**
 * Dub: drop a video / audio file or hand over a Scene, Direct take or voice clip; transcribe, translate per segment,
 * pick voices and direction; generate (LACI timing, optional lip-sync); listen to source / dub / mix against the
 * picture; regenerate segments; hand the track to Edit or the lines to Scene.
 */
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'preact/hooks';
import type { DubSegmentOut } from '../../api';
import { exportWav } from '../../audio/mix';
import * as player from '../../audio/player';
import { nowPlaying, playhead } from '../../audio/player';
import { FileDrop } from '../../components/FileDrop';
import { Icon } from '../../components/Icon';
import { PronMarks, PronounceButton } from '../../components/Pron';
import { TagButton, TagField, TagMarks } from '../../components/Tags';
import { slug } from '../../lib/files';
import { DUB_TO_SERVER } from '../../lib/languages';
import { clamp } from '../../lib/math';
import { act, activeHistory, backend, notify, primaryAction, screenKeys } from '../../state/app';
import * as D from '../../state/dub';
import { canEdit, openClip } from '../../state/edit';
import type { History } from '../../state/history';
import { speakers } from '../../state/library';
import { lineState, scene } from '../../state/scene';
import { script, takes as scriptTakes } from '../../state/script';
import { Direction } from '../edit/Direction';
import { theme, token } from '../../state/theme';

const fmtT = (s: number) => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, '0')}`;
const ms = (s: number | null | undefined) => (s == null ? '–' : `${s >= 0 ? '+' : '−'}${Math.round(Math.abs(s) * 1000)} ms`);
const factor = (x: number) => (x >= 9.95 ? '≥10×' : `${x.toFixed(2)}×`);
/** A bucket for people: its steered label ("faster+1") or its row index. */
const bucketName = (i: number) => {
  const b = D.bucketScale.value.find((x) => x.index === i);
  return b?.steered ? b.label : String(i);
};
const syncFmt = (x: number | null) => (x === null ? '–' : x.toFixed(Math.abs(x) < 1 ? 3 : 1));
/** A metric in ms ('–' when the server had nothing to measure). */
const msFmt = (x: number | null) => (x === null ? '–' : String(Math.round(x)));

function isTyping(t: EventTarget | null) {
  const el = t as HTMLElement | null;
  return !!el && (el.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName));
}

/** Screen keys (see the ? sheet). */
function dubKeys(e: KeyboardEvent): boolean {
  const mod = e.metaKey || e.ctrlKey;
  if (mod && e.key === 'Enter') {
    e.preventDefault();
    void D.generate();
    return true;
  }
  if (mod || e.altKey || isTyping(e.target)) return false;
  const d = D.doc.value;
  const segs = d.segments;
  const sel = segs.findIndex((s) => s.id === D.selected.value);
  switch (e.key.toLowerCase()) {
    case 's':
    case 'd':
    case 'm':
      if (!d.track) return false;
      D.setListen(({ s: 'source', d: 'dub', m: 'mix' } as const)[e.key.toLowerCase() as 's' | 'd' | 'm']);
      return true;
    case 'l':
      if (!d.source?.video || !D.caps.value?.lipsync) return false;
      D.lipsyncOn.value = !D.lipsyncOn.value;
      notify(`Lip-sync ${D.lipsyncOn.value ? 'on' : 'off'} for the next generation`);
      return true;
    case 'j':
    case 'k': {
      if (!segs.length) return false;
      const k = clamp((sel < 0 ? -1 : sel) + (e.key.toLowerCase() === 'k' ? 1 : -1), 0, segs.length - 1);
      selectSeg(segs[k].id, true);
      return true;
    }
    case 'r':
      if (sel < 0 || !d.track) return false;
      void D.regenerate(segs[sel].id);
      return true;
    case 'arrowleft':
    case 'arrowright': {
      const role = (e.target as HTMLElement)?.getAttribute?.('role');
      if (role === 'slider' || !d.source) return false;
      e.preventDefault();
      const playing = !!nowPlaying.value?.id.startsWith('dub:');
      const at = Math.max(0, (playing ? playhead.value : D.cursor.value) + (e.key === 'ArrowLeft' ? -1 : 1));
      if (playing) D.togglePlay(at);
      else D.cursor.value = at;
      return true;
    }
  }
  return false;
}

/** Select a segment (scrolls its row into view); `play` hears it in the current listening mode. */
function selectSeg(id: string, play = false) {
  D.selected.value = id;
  const d = D.doc.value;
  const k = d.segments.findIndex((s) => s.id === id);
  document.getElementById(`dseg-${id}`)?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  if (!play || k < 0) return;
  const o = d.track?.segments[k];
  const s = d.segments[k];
  const mode = d.track ? D.listen.value : 'source';
  const from = mode === 'source' || !o || o.status !== 'ok' ? s.ref_start : (o.lip_start ?? o.speech_start ?? s.ref_start);
  const to = mode === 'source' || !o || o.status !== 'ok' ? s.ref_end : (o.lip_end ?? o.speech_end ?? s.ref_end);
  const l = D.listened();
  if (!l) return;
  D.cursor.value = Math.max(0, from - 0.15);
  player.play(l.id, l.pcm, l.sr, Math.max(0, from - 0.15), undefined, to + 0.2);
}

// ── Sources ───────────────────────────────────────────────────────────────────────────────────────

function DropZone({ big }: { big?: boolean }) {
  const file = useRef<HTMLInputElement>(null);
  const busy = !!D.busy.value;
  return (
    <div
      class={`dub-drop${big ? ' is-big' : ''}`}
      role="button"
      tabIndex={0}
      aria-label="Drop or choose a video or audio file to dub"
      onClick={() => !busy && file.current?.click()}
      onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), file.current?.click())}
    >
      <Icon name="film" size={big ? 28 : 18} />
      <div>
        <b>{big ? 'Drop a video or an audio file' : 'Drop or choose a file'}</b>
        <span class="muted small">MP4, WebM, MOV, WAV, MP3 · 0.5–{Math.round(D.caps.value?.max_ref_seconds ?? 180)} s{big && D.caps.value?.lipsync ? ' · video gets lip-sync' : ''}</span>
      </div>
      <input
        ref={file}
        type="file"
        accept="video/*,audio/*"
        hidden
        onChange={(e) => {
          const f = (e.currentTarget as HTMLInputElement).files?.[0];
          (e.currentTarget as HTMLInputElement).value = '';
          if (f) void D.openFile(f);
        }}
      />
    </div>
  );
}

function StudioSources() {
  const lineTakes = script.value.lines.flatMap((l, i) => (scriptTakes.value[l.id] ?? []).map((t, k) => ({ l, i, t, k })));
  const sceneLines = scene.value.lines.length;
  const sceneReady = scene.value.lines.filter((l) => lineState(l)?.audio).length;
  const busy = !!D.busy.value;
  return (
    <div class="rail-section dub-studio">
      <h4>From the studio</h4>
      <button class="src-item dub-scene-btn" disabled={busy || !sceneLines} onClick={() => void D.dubScene()} title={sceneLines ? 'Dub the Scene: one segment per line, each keeping its character’s voice and direction' : 'Write a scene first'}>
        <Icon name="globe" size={13} />
        <span class="src-text">Dub this scene</span>
        <span class="tag">{sceneLines ? `${sceneReady}/${sceneLines} lines` : 'empty'}</span>
      </button>
      <details class="src-group" open={!!lineTakes.length && !sceneLines}>
        <summary>Direct takes {lineTakes.length ? <b>{lineTakes.length}</b> : null}</summary>
        {lineTakes.length ? (
          <ul class="src-list">
            {lineTakes.map(({ i, t, k }) => (
              <li>
                <button class="src-item" disabled={busy} title={t.text} onClick={() => void D.dubTake(t, `Line ${i + 1} · take ${k + 1}`)}>
                  <span class="src-n">
                    {i + 1}.{k + 1}
                  </span>
                  <span class="src-text">{t.text}</span>
                </button>
              </li>
            ))}
          </ul>
        ) : (
          <p class="muted small">Render takes in Direct; each one can be dubbed with its text known.</p>
        )}
      </details>
      <details class="src-group">
        <summary>Voice clips</summary>
        <ul class="src-list">
          {speakers.value
            .filter((s) => s.has_preview !== false)
            .map((s) => (
              <li>
                <button class="src-item" disabled={busy} onClick={() => void D.dubVoiceClip(s.id)} title={s.transcript ? `Transcript known: ${s.transcript}` : 'No transcript: it will be transcribed'}>
                  <span class="src-text">{s.label}</span>
                  <span class="tag">{s.transcript ? 'text' : 'asr'}</span>
                </button>
              </li>
            ))}
        </ul>
      </details>
    </div>
  );
}

const KIND_LABEL: Record<D.SourceKind, string> = { file: 'File', scene: 'Scene', take: 'Direct take', voice: 'Voice clip' };

function LipStatus() {
  const d = D.doc.value;
  const l = D.lips.value;
  const c = D.caps.value!;
  if (!d.source?.video) return null;
  if (!c.lipsync) return <p class="dub-lips muted small">Lip-sync isn’t set up on this server (AV-sync); the dub follows the timing only.</p>;
  const text = {
    none: 'Lip-sync: waiting for the upload',
    uploading: `Lip-sync: uploading the video ${Math.round(l.progress * 100)}%`,
    tracking: 'Lip-sync: tracking the face…',
    ready: l.info ? `Face tracked ${fmtT(l.info.start)}–${fmtT(l.info.end)} (${l.info.track_frames} frames)` : 'Face tracked',
    failed: `Lip-sync unavailable: ${l.error ?? 'no face found'}`,
  }[l.state];
  return (
    <p class={`dub-lips small is-${l.state}`} role="status">
      {(l.state === 'uploading' || l.state === 'tracking') && <span class="spin" />}
      {text}
    </p>
  );
}

function SourceCard() {
  const d = D.doc.value;
  const s = d.source;
  if (!s) return null;
  return (
    <div class="rail-section">
      <h4>Source</h4>
      <div class="dub-source">
        <span class="tag">{KIND_LABEL[s.kind]}</span>
        <b title={s.name}>{s.name}</b>
        <span class="muted small">
          {s.duration.toFixed(1)} s{s.video ? ' · video' : ''}
          {s.lines.length ? ` · ${s.lines.length} lines` : ''}
        </span>
      </div>
      <LipStatus />
      <DropZone />
    </div>
  );
}

function VoiceSelect({ value, onChange, inherit, label }: { value: D.VoiceChoice | null; onChange: (v: D.VoiceChoice | null) => void; inherit?: string; label: string }) {
  const opts = D.voiceOptions.value;
  const key = D.voiceKey(value);
  const known = !value || opts.some((o) => D.voiceKey(o.voice) === key);
  const groups = [...new Set(opts.map((o) => o.group))];
  return (
    <select
      class="input dub-voice"
      aria-label={label}
      value={key}
      onChange={(e) => {
        const k = (e.currentTarget as HTMLSelectElement).value;
        if (!k) return onChange(null);
        const o = opts.find((x) => D.voiceKey(x.voice) === k);
        if (o) onChange(o.voice);
        else if (value && D.voiceKey(value) === k) onChange(value);
      }}
    >
      {inherit !== undefined && <option value="">{inherit}</option>}
      {!known && value && <option value={key}>{D.voiceLabel(value)}</option>}
      {groups.map((g) =>
        g ? (
          <optgroup label={g}>
            {opts
              .filter((o) => o.group === g)
              .map((o) => (
                <option value={D.voiceKey(o.voice)}>{D.voiceLabel(o.voice)}</option>
              ))}
          </optgroup>
        ) : (
          opts.filter((o) => !o.group).map((o) => <option value={D.voiceKey(o.voice)}>{D.voiceLabel(o.voice)}</option>)
        ),
      )}
    </select>
  );
}

function Settings() {
  const d = D.doc.value;
  const c = D.caps.value!;
  return (
    <>
      <div class="rail-section">
        <h4>Languages</h4>
        <label class="dub-field">
          <span>From</span>
          <select class="input" value={d.refLang} onChange={(e) => D.setRefLang((e.currentTarget as HTMLSelectElement).value)} aria-label="Source language">
            {c.ref_languages.map((l) => (
              <option value={l.code}>{l.name}</option>
            ))}
          </select>
        </label>
        <label class="dub-field">
          <span>Into</span>
          <select class="input" value={d.targetLang} onChange={(e) => D.setTargetLang((e.currentTarget as HTMLSelectElement).value)} aria-label="Target language">
            {c.target_languages.map((l) => (
              <option value={l.code}>{l.label}</option>
            ))}
          </select>
        </label>
      </div>
      <div class="rail-section">
        <h4>Voice</h4>
        <VoiceSelect value={d.voice} onChange={(v) => v && D.setVoice(v)} label="Voice of the dub" />
        <p class="hint">{d.voice.kind === 'source' ? 'Clones the speaker from the source audio itself.' : 'A studio voice in embedding mode, with its Cast steering.'} Each segment can override it.</p>
      </div>
      <div class="rail-section">
        <h4>Options</h4>
        <label class="dub-check" title={c.lipsync ? 'Warp the finished dub onto the speaker’s lips (AV-sync), video sources only' : 'AV-sync is not set up on this server'}>
          <input type="checkbox" checked={D.lipsyncOn.value} disabled={!c.lipsync || !d.source?.video} onChange={(e) => (D.lipsyncOn.value = (e.currentTarget as HTMLInputElement).checked)} />
          Lip-sync (L)
        </label>
        <label class="dub-field" title="Pauses at least this long split the transcript into segments">
          <span>Split at pauses ≥</span>
          <input type="range" min={c.min_pause_range[0]} max={c.min_pause_range[1]} step={0.05} value={D.minPause.value} onInput={(e) => (D.minPause.value = Number((e.currentTarget as HTMLInputElement).value))} aria-label="Minimum pause between segments (s)" />
          <b class="mono">{D.minPause.value.toFixed(2)} s</b>
        </label>
      </div>
    </>
  );
}

function Versions() {
  const d = D.doc.value;
  if (!d.tracks.length) return null;
  return (
    <div class="rail-section grow">
      <h4>Versions</h4>
      <ol class="versions">
        {d.tracks.map((t, i) => {
          const id = `dub:v:${t.id}`;
          const playing = nowPlaying.value?.id === id;
          return (
            <li class={`version-row${t === d.track ? ' is-current' : ''}`}>
              <button class="version-main" onClick={() => D.useTrack(t.id)} aria-current={t === d.track} title={t === d.track ? 'Current dub track' : 'Make this the dub track (undoable)'}>
                <span class="badge">v{i + 1}</span>
                <span class="version-label">{t.label}</span>
                <span class="mono muted">{(t.pcm.length / t.sr).toFixed(1)} s</span>
              </button>
              <button class={`play-mini${playing ? ' is-playing' : ''}`} onClick={() => player.toggle(id, t.pcm, t.sr)} aria-label={`Play version ${i + 1}`}>
                <Icon name={playing ? 'stop' : 'play'} size={11} />
              </button>
            </li>
          );
        })}
      </ol>
      <p class="hint">⌘Z / ⇧⌘Z step through generations, regenerations and edits.</p>
    </div>
  );
}

// ── Preview: picture + timeline ───────────────────────────────────────────────────────────────────

const isDubPlaying = () => !!nowPlaying.value?.id.startsWith('dub:');
const at = () => (isDubPlaying() ? playhead.value : D.cursor.value);

/** The muted video following the playhead, or a still stand-in with the source's and the dub's levels. */
function Picture() {
  const d = D.doc.value;
  const s = d.source!;
  const video = useRef<HTMLVideoElement>(null);
  const t = at();
  const playing = isDubPlaying();
  useEffect(() => {
    const v = video.current;
    if (!v) return;
    if (playing) {
      if (Math.abs(v.currentTime - t) > 0.15) v.currentTime = t;
      if (v.paused) void v.play().catch(() => {});
    } else {
      if (!v.paused) v.pause();
      if (Math.abs(v.currentTime - t) > 0.04) v.currentTime = t;
    }
  }, [t, playing]);
  if (s.video) return <video ref={video} class="dub-video" src={s.video.url} muted playsInline preload="auto" aria-label="Source video (muted; the dub plays over it)" />;
  const level = (pcm: Float32Array | undefined, sr: number) => {
    if (!pcm) return 0;
    const i0 = Math.max(0, Math.floor((t - 0.03) * sr));
    const i1 = Math.min(pcm.length, Math.floor((t + 0.03) * sr));
    let m = 0;
    for (let i = i0; i < i1; i++) m = Math.max(m, Math.abs(pcm[i]));
    return Math.min(1, m * 1.6);
  };
  const src = level(s.pcm, s.sr);
  const dub = d.track ? level(d.track.pcm, d.track.sr) : 0;
  const initials = s.name.replace(/[^\p{L}\s]/gu, '').split(/\s+/).filter(Boolean).slice(0, 2).map((w) => w[0]).join('').toUpperCase() || '♪';
  return (
    <div class="dub-still" aria-label="Audio source: no picture">
      <div class="dub-still-face">
        <span>{initials}</span>
        <i class="dub-still-mouth" style={{ transform: `scaleY(${0.15 + src * 0.85})` }} title="Mouth follows the source speech" />
      </div>
      <div class="dub-still-meters">
        <span>
          source <i style={{ width: `${src * 100}%` }} />
        </span>
        {d.track && (
          <span class="is-dub">
            dub <i style={{ width: `${dub * 100}%` }} />
          </span>
        )}
      </div>
    </div>
  );
}

function peaks(pcm: Float32Array, sr: number, span: number, width: number): Float32Array {
  const out = new Float32Array(width);
  const per = (span * sr) / width;
  for (let x = 0; x < width; x++) {
    const a = Math.floor(x * per);
    const b = Math.min(pcm.length, Math.floor((x + 1) * per));
    let m = 0;
    for (let i = a; i < b; i++) {
      const v = Math.abs(pcm[i]);
      if (v > m) m = v;
    }
    out[x] = m;
  }
  // Each lane shows its own clip's shape: quiet sources (a phone video) would otherwise draw a flat line.
  let top = 0;
  for (const v of out) if (v > top) top = v;
  if (top > 1e-4) for (let x = 0; x < width; x++) out[x] /= top;
  return out;
}

/** Warp colours of placed takes: the status tokens; neutral (the dub tint) when there is no warp. */
const KIND_TOKEN: Record<string, string> = { ok: '--ok', warn: '--warn', bad: '--bad' };

/** Source lane (reference segments) over dub lane (placed takes), joined by bands; click to seek / select. */
function Timeline() {
  const d = D.doc.value;
  const s = d.source!;
  const t = d.track;
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const ro = new ResizeObserver(([e]) => setWidth(Math.round(e.contentRect.width)));
    ro.observe(wrap.current!);
    return () => ro.disconnect();
  }, []);
  // Every reference span and every placed take fits, whatever ends last: the decoded audio, the dub, the server's
  // reference (its decoder can make the clip a little longer than the browser's).
  const span = Math.max(
    s.duration,
    t ? t.pcm.length / t.sr : 0,
    ...d.segments.map((x) => x.ref_end),
    ...(t?.segments ?? []).map((o) => Math.max(o.lip_end ?? 0, o.speech_end ?? 0)),
    0.5,
  );
  const H = 150;
  const lane = 48;
  const srcY = 6;
  const dubY = H - lane - 6;
  const srcPeaks = useMemo(() => (width ? peaks(s.pcm, s.sr, span, width) : null), [s, span, width]);
  const dubPeaks = useMemo(() => (width && t ? peaks(t.pcm, t.sr, span, width) : null), [t, span, width]);
  const stale = D.stale.value;
  const sel = D.selected.value;
  const mode = theme.value;
  useEffect(() => {
    const el = canvas.current;
    if (!el || !width) return;
    const dpr = window.devicePixelRatio || 1;
    el.width = width * dpr;
    el.height = H * dpr;
    const g = el.getContext('2d')!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, width, H);
    const X = (sec: number) => (sec / span) * width;
    const env = (p: Float32Array | null, y: number, color: string) => {
      if (!p) return;
      g.fillStyle = color;
      for (let x = 0; x < p.length; x++) {
        const h = Math.max(0.5, p[x] * (lane / 2 - 2));
        g.fillRect(x, y + lane / 2 - h, 1, 2 * h);
      }
    };
    // Source in the neutral ink, the dub in the accent: the dub is what this screen makes.
    g.fillStyle = token('--text', 0.03);
    g.fillRect(0, srcY, width, lane);
    g.fillRect(0, dubY, width, lane);
    d.segments.forEach((seg, k) => {
      const o = t?.segments[k];
      const x0 = X(seg.ref_start);
      const x1 = X(seg.ref_end);
      const on = seg.id === sel;
      // At least 3 px and outlined: a one-word segment stays visible, neighbours stay apart.
      const w0 = Math.max(3, x1 - x0);
      g.fillStyle = token('--text', on ? 0.14 : 0.05);
      g.fillRect(x0, srcY, w0, lane);
      g.strokeStyle = on ? token('--text', 0.9) : token('--text', 0.28);
      g.lineWidth = 1;
      g.strokeRect(Math.min(x0, width - w0) + 0.5, srcY + 0.5, w0 - 1, lane - 1);
      if (o && o.status === 'ok' && o.speech_start != null && o.speech_end != null) {
        const a = o.lip_start ?? o.speech_start;
        const b = o.lip_end ?? o.speech_end;
        const kind = D.warpKind(o);
        const col = kind ? KIND_TOKEN[kind] : '--o';
        g.fillStyle = token(col, on ? 0.25 : 0.13);
        g.beginPath();
        g.moveTo(x0, srcY + lane);
        g.lineTo(x1, srcY + lane);
        g.lineTo(X(b), dubY);
        g.lineTo(X(a), dubY);
        g.closePath();
        g.fill();
        g.fillStyle = token(col, on ? 0.33 : 0.18);
        const wd = Math.max(3, X(b) - X(a));
        g.fillRect(X(a), dubY, wd, lane);
        g.strokeStyle = token(col, on ? 0.93 : 0.53);
        g.strokeRect(Math.min(X(a), width - wd) + 0.5, dubY + 0.5, wd - 1, lane - 1);
        if (stale.has(seg.id)) {
          g.strokeStyle = token('--warn');
          g.setLineDash([3, 3]);
          g.strokeRect(X(a) + 0.5, dubY + 0.5, Math.max(1, X(b) - X(a)) - 1, lane - 1);
          g.setLineDash([]);
        }
      }
      g.fillStyle = on ? token('--text') : token('--text-2');
      g.font = '10px "Inter Variable", system-ui, sans-serif';
      const label = String(k + 1);
      g.fillText(label, Math.min(x0 + 3, width - g.measureText(label).width - 3), srcY + 11);
    });
    env(srcPeaks, srcY, token('--text-2'));
    env(dubPeaks, dubY, token('--o', 0.9));
    g.fillStyle = token('--text-3');
    g.font = '10px "Inter Variable", system-ui, sans-serif';
    g.fillText('source', 4, srcY + lane - 4);
    g.fillText(t ? 'dub' : 'dub · generate to hear it', 4, dubY + lane - 4);
  }, [width, srcPeaks, dubPeaks, d.segments, t, sel, stale, span, mode]);
  const x = (at() / span) * 100;
  const seek = (e: PointerEvent) => {
    const r = wrap.current!.getBoundingClientRect();
    const sec = clamp(((e.clientX - r.left) / r.width) * span, 0, span);
    const y = e.clientY - r.top;
    const hit = d.segments.findIndex((seg, k) => {
      if (y < dubY - 4) return sec >= seg.ref_start && sec <= seg.ref_end;
      const o = t?.segments[k];
      return !!o && o.speech_start != null && sec >= (o.lip_start ?? o.speech_start) && sec <= (o.lip_end ?? o.speech_end ?? 0);
    });
    if (hit >= 0) D.selected.value = d.segments[hit].id;
    if (isDubPlaying()) D.togglePlay(sec);
    else D.cursor.value = sec;
  };
  return (
    <div
      ref={wrap}
      class="dub-timeline"
      role="slider"
      tabIndex={0}
      aria-label="Dub timeline: source above, dub below"
      aria-valuemin={0}
      aria-valuemax={+span.toFixed(1)}
      aria-valuenow={+at().toFixed(1)}
      aria-valuetext={`${fmtT(at())} of ${fmtT(span)}`}
      onPointerDown={seek}
    >
      <canvas ref={canvas} style={{ height: `${H}px` }} />
      <div class="dub-playhead" style={{ left: `${x}%` }} />
    </div>
  );
}

function Transport() {
  const d = D.doc.value;
  const t = d.track;
  const playing = isDubPlaying();
  const mode = t ? D.listen.value : 'source';
  const dur = Math.max(d.source!.duration, t ? t.pcm.length / t.sr : 0);
  return (
    <div class="dub-transport">
      <button class={`play-mini dub-play${playing ? ' is-playing' : ''}`} onClick={() => D.togglePlay()} aria-label={playing ? 'Pause' : 'Play'} title="Play / pause (Space)">
        <Icon name={playing ? 'stop' : 'play'} size={16} />
      </button>
      <span class="mono small dub-time">
        {fmtT(at())} / {fmtT(dur)}
      </span>
      <div class="segctl" role="radiogroup" aria-label="Listen to">
        {(['source', 'dub', 'mix'] as const).map((m) => (
          <button role="radio" aria-checked={mode === m} class={`seg-btn${mode === m ? ' is-on' : ''}`} disabled={m !== 'source' && !t} onClick={() => D.setListen(m)} title={{ source: 'The original (S)', dub: 'The dub alone (D)', mix: 'The dub over the source, ducked (M)' }[m]}>
            {{ source: 'Source', dub: 'Dub', mix: 'Mix' }[m]}
          </button>
        ))}
      </div>
      {t?.lipsync && t.lipsync.applied === false && (
        <span class="small dub-lip-skipped" title={`The bounded warp did not bring the dub closer to the lips (${t.lipsync.reason ?? 'no sync gain'}), so the dub is kept as timed, unwarped`}>
          lip-sync skipped — no sync gain ({syncFmt(t.lipsync.sync_distance_before)} → {syncFmt(t.lipsync.sync_distance_after)})
        </span>
      )}
      {t?.lipsync && t.lipsync.applied !== false && (
        <span
          class="muted small"
          title={`AV-sync over ${fmtT(t.lipsync.start)}–${fmtT(t.lipsync.end)}: largest shift ${Math.round(t.lipsync.max_shift_ms)} ms (mean ${Math.round(t.lipsync.mean_abs_shift_ms)} ms), largest stretch ${factor(t.lipsync.max_stretch)} / squeeze ${factor(t.lipsync.max_squeeze)}; SyncNet distance ${syncFmt(t.lipsync.sync_distance_before)} → ${syncFmt(t.lipsync.sync_distance_after)} (lower is better)${t.lipsync.guard ? `. Bounded: every shift ≤ ${Math.round(t.lipsync.guard.max_shift_s * 1000)} ms, local rate ${(1 / t.lipsync.guard.max_rate).toFixed(2)}–${t.lipsync.guard.max_rate.toFixed(2)}×, no advance before the face track` : ''}`}
        >
          lip-synced{t.lipsync.guard ? ` · bounded ±${Math.round(t.lipsync.guard.max_shift_s * 1000)} ms, ≤${t.lipsync.guard.max_rate}×` : ''} · shift ≤{Math.round(t.lipsync.max_shift_ms)} ms · SyncNet {syncFmt(t.lipsync.sync_distance_before)} → {syncFmt(t.lipsync.sync_distance_after)}
        </span>
      )}
      {t?.edited && <span class="tag" title="This track was changed in the Edit screen">edited</span>}
    </div>
  );
}

function Metrics() {
  const t = D.doc.value.track;
  const m = t?.metrics;
  if (!t || !m) return null;
  return (
    <p class="dub-metrics small">
      <span title="Mean |speech start − reference start| / |end − end| over the placed segments">
        timing {msFmt(m.mean_abs_start_error_ms)} / {msFmt(m.mean_abs_end_error_ms)} ms
      </span>
      <span title="Largest push of a segment past its reference start (an overrun moves the next segment later)">max shift {msFmt(m.max_shift_ms)} ms</span>
      <span title="LACI retries of the kept takes, and over every candidate this request synthesized (early EOS = truncation, stall = hallucination)">
        LACI retries {m.laci_retries} kept{m.laci_retries ? ` (${D.retryText(m.retries_by_kind)})` : ''} · {m.all_retries} over {m.candidates} candidates
        {m.exhausted_takes ? ` · ${m.exhausted_takes} used all retries` : ''}
      </span>
      {D.lipApplied(t) && <span>warp max {factor(Math.max(t.lipsync!.max_stretch, t.lipsync!.max_squeeze))}</span>}
      <span class="muted">{m.generation_s.toFixed(1)} s</span>
    </p>
  );
}

// ── Script stage ──────────────────────────────────────────────────────────────────────────────────

function ScriptStage() {
  const d = D.doc.value;
  const c = D.caps.value!;
  const busy = !!D.busy.value;
  const ref = useRef<HTMLTextAreaElement>(null);
  const tgt = useRef<HTMLTextAreaElement>(null);
  const langName = c.ref_languages.find((l) => l.code === d.refLang)?.name ?? d.refLang;
  const target = c.target_languages.find((l) => l.code === d.targetLang)?.label ?? d.targetLang;
  return (
    <section class="dub-script" aria-label="Transcript and translation">
      <div class="dub-col">
        <header class="dub-col-head">
          <h3>Transcript · {langName}</h3>
          {d.source?.kind === 'scene' ? (
            <span class="tag" title="Word timings came with the source: no speech recognition runs">timings known</span>
          ) : (
            <button class="link" disabled={busy || !d.refId} onClick={() => void D.transcribe(d.refLang)} title="Transcribe again in the selected language">
              <Icon name="refresh" size={12} /> transcribe again
            </button>
          )}
        </header>
        <textarea ref={ref} class="input dub-area" value={d.refText} placeholder={busy ? 'Transcribing…' : 'What is said in the source'} onChange={(e) => D.setRefText((e.currentTarget as HTMLTextAreaElement).value)} aria-label="Transcript of the source" />
      </div>
      <div class="dub-col">
        <header class="dub-col-head">
          <h3>Translation · {target}</h3>
          <PronounceButton input={() => tgt.current} />
          <TagButton input={() => tgt.current} />
        </header>
        <textarea ref={tgt} class="input dub-area" data-tags value={d.targetText} placeholder={`The whole text in ${target}; it is split across the segments at the pauses`} onChange={(e) => D.setTargetText((e.currentTarget as HTMLTextAreaElement).value)} aria-label={`Translation (${target})`} />
        <PronMarks text={d.targetText} replace={(s, e, t) => D.setTargetText(d.targetText.slice(0, s) + t + d.targetText.slice(e))} />
        <TagMarks text={d.targetText} replace={(next) => D.setTargetText(next)} />
      </div>
      <div class="dub-script-go">
        <button class="btn primary" disabled={busy || !d.refId || !d.refText.trim() || !d.targetText.trim()} onClick={() => void D.analyze()}>
          Split into segments
        </button>
        <span class="muted small">Splits the source at pauses ≥ {D.minPause.value.toFixed(2)} s and spreads the translation over the pieces; every piece stays editable.</span>
      </div>
    </section>
  );
}

// ── Segments ──────────────────────────────────────────────────────────────────────────────────────

function Result({ o }: { o: DubSegmentOut | undefined }) {
  if (!o) return <span class="muted small">not generated</span>;
  if (o.status === 'silent') return <span class="muted small">silent</span>;
  const c = o.chosen != null ? o.candidates[o.chosen] : null;
  const kind = D.warpKind(o);
  const worst = Math.max(o.lip_stretch ?? 1, o.lip_squeeze ?? 1);
  const bound = D.doc.value.track?.lipsync?.guard?.max_rate;
  const atBound = !!bound && worst >= bound - 0.005;
  const allRetries = o.candidates.reduce((n, x) => n + (x.attempts ?? 0), 0);
  const fit = o.duration_error ?? 0;
  const fitKind = Math.abs(fit) < 0.12 ? 'ok' : Math.abs(fit) < 0.3 ? 'warn' : 'bad';
  return (
    <div class="dseg-result">
      <span class={`badge ${fitKind}`} title={`Speech ${ms(o.start_error)} at the start, ${ms(o.end_error)} at the end vs the reference; length ${ms(o.duration_error)}`}>
        fit {ms(o.duration_error)}
      </span>
      {(o.shift ?? 0) > 0.005 && (
        <span class="badge warn" title="The previous segment overran: this one starts later than its reference">
          shift {Math.round((o.shift ?? 0) * 1000)} ms
        </span>
      )}
      {c && (
        <span class="small muted" title={`Kept candidate: bucket ${bucketName(c.bucket)} (offset ${c.offset >= 0 ? '+' : ''}${c.offset})${c.final_bucket != null && c.final_bucket !== c.bucket ? ` → ${bucketName(c.final_bucket)} after a LACI retry` : ''}, seed ${c.seed}, ${c.end ?? ''}${c.capped ? ', capped' : ''}. ${allRetries} retries over all ${o.candidates.length} candidates.`}>
          {D.bucketTag(c.bucket)}
          {c.final_bucket != null && c.final_bucket !== c.bucket ? `→${D.bucketTag(c.final_bucket)}` : ''}
          {c.attempts ? ` · ${c.attempts} retr. (${D.retryText(c.retries)})` : ''}
        </span>
      )}
      {kind && (
        <span class={`badge ${kind}`} title={`Lip-sync time scale: length ${factor(o.lip_scale ?? 1)}, max stretch ${factor(o.lip_stretch ?? 1)}, max squeeze ${factor(o.lip_squeeze ?? 1)}, mean ${factor(o.lip_warp ?? 1)}. Green < 1.1× (clean), amber < 1.3× (audible on close listening), red up to the server's bound${atBound ? '; this segment hit the bound: the lips could not be followed more closely' : ''}`}>
          warp {factor(worst)}
          {atBound ? ' · at bound' : ''}
        </span>
      )}
    </div>
  );
}

/** Height to content: a one-line field for short text, more lines for long. */
function fit(el: HTMLTextAreaElement) {
  el.style.height = 'auto';
  el.style.height = `${el.scrollHeight + 2}px`;
}

function SegmentRow({ seg, k }: { seg: D.DubSeg; k: number }) {
  const d = D.doc.value;
  const o = d.track?.segments[k];
  const isStale = D.stale.value.has(seg.id);
  const sel = D.selected.value === seg.id;
  const input = useRef<HTMLTextAreaElement>(null);
  const [open, setOpen] = useState(false);
  // The whole target text stays visible: the field grows with it (and with the row's width).
  useLayoutEffect(() => {
    const el = input.current;
    if (!el) return;
    fit(el);
    const ro = new ResizeObserver(() => fit(el));
    ro.observe(el);
    return () => ro.disconnect();
  }, [seg.target_text]);
  const mode = D.modeOf(d, seg);
  const r = D.rateOf(seg.target_text, seg, mode);
  const scale = D.scaleOf(d.targetLang, mode);
  const bucketLabel = (b: D.ScaleBucket) =>
    b.steered
      ? `${b.label} · pace${b.dose != null ? ` ${b.dose >= 0 ? '+' : '−'}${Math.abs(b.dose)}` : ''} · ${b.rate != null ? `${b.rate.toFixed(1)} B/s` : ''}`
      : `${b.index} · ${rangeOf(b)}`;
  // The language's lowest bucket is open below and its highest open above (the server's rule).
  const rangeOf = (b: D.ScaleBucket) => {
    const lo = b === scale[0] || b.low == null ? null : b.low.toFixed(1);
    const hi = b === scale[scale.length - 1] || b.high == null ? null : b.high.toFixed(1);
    return lo == null ? `< ${hi}` : hi == null ? `≥ ${lo}` : `${lo}–${hi}`;
  };
  const dirCount = Object.keys(seg.values).length;
  const busy = !!D.busy.value;
  const flagged = D.flagged.value.has(seg.id);
  const cloned = (seg.voice ?? d.voice).kind === 'source';
  // A take far from its reference length is often the clone failing on the text.
  const poor = !!o && o.status === 'ok' && Math.abs(o.duration_error ?? 0) >= 0.3;
  const fallback = D.fallbackVoice.value;
  return (
    <li id={`dseg-${seg.id}`} class={`dseg${sel ? ' is-selected' : ''}${isStale ? ' is-stale' : ''}`} onFocusIn={() => (D.selected.value = seg.id)}>
      <button class="dseg-n" onClick={() => selectSeg(seg.id, true)} aria-label={`Play segment ${k + 1}`} title="Select and hear this segment (J / K: previous / next)">
        {k + 1}
      </button>
      <div class="dseg-main">
        <div class="dseg-ref">
          <span class="mono small">
            {fmtT(seg.ref_start)}–{fmtT(seg.ref_end)}
          </span>
          <span class="dseg-reftext" title={seg.ref_text}>
            {seg.ref_text}
          </span>
        </div>
        <div class="dseg-target">
          <TagField text={seg.target_text} block>
            <textarea
              ref={input}
              class="input dseg-input"
              rows={1}
              value={seg.target_text}
              placeholder="silent (no text)"
              aria-label={`Segment ${k + 1} target text`}
              data-tags
              onInput={(e) => fit(e.currentTarget as HTMLTextAreaElement)}
              onChange={(e) => D.updateSeg(seg.id, { target_text: (e.currentTarget as HTMLTextAreaElement).value })}
              onKeyDown={(e) => e.key === 'Enter' && (e.preventDefault(), (e.currentTarget as HTMLTextAreaElement).blur())}
            />
          </TagField>
          <PronounceButton input={() => input.current} onDone={() => input.current?.blur()} />
          <TagButton input={() => input.current} />
        </div>
        <PronMarks text={seg.target_text} replace={(s, e, t) => D.updateSeg(seg.id, { target_text: seg.target_text.slice(0, s) + t + seg.target_text.slice(e) })} />
        <TagMarks text={seg.target_text} replace={(next) => D.updateSeg(seg.id, { target_text: next })} />
        <div class="dseg-meta">
          <span class="small muted" title="UTF-8 bytes per second of the reference span: the speaking rate the text asks for">
            {r ? (
              <>
                {r.rate.toFixed(1)} B/s
                {r.fast && <span class="badge warn"> fast</span>}
                {r.slow && <span class="badge warn"> slow</span>}
              </>
            ) : (
              'silent'
            )}
          </span>
          <select class="input dseg-bucket" value={seg.bucket ?? ''} onChange={(e) => D.updateSeg(seg.id, { bucket: (e.currentTarget as HTMLSelectElement).value === '' ? null : Number((e.currentTarget as HTMLSelectElement).value) })} aria-label={`Segment ${k + 1} speaking-rate bucket`} title={`Speaking-rate bucket (UTF-8 bytes/s): auto plans candidates around the text’s rate; a set bucket puts every candidate in it. Buckets marked “pace” lie beyond the model’s rate row: they are reached by pace steering, only as far as measured safe for this language${mode === 'source' ? '. The slower pace buckets need a studio voice (a cloned source voice loses words there)' : ''}`}>
            <option value="">{r ? `auto (${bucketName(r.bucket)})` : 'auto'}</option>
            {scale.map((b) => (
              <option value={b.index}>{bucketLabel(b)}</option>
            ))}
          </select>
          <VoiceSelect value={seg.voice} inherit={`Project voice (${D.voiceLabel(d.voice)})`} onChange={(v) => D.updateSeg(seg.id, { voice: v })} label={`Segment ${k + 1} voice`} />
          <button class={`btn sm ghost${dirCount ? ' is-on' : ''}`} onClick={() => setOpen(!open)} aria-expanded={open} title="Direction for this segment: delivery and mood (spannable controls)">
            Direction{dirCount ? ` · ${dirCount}` : ''} {open ? '▴' : '▾'}
          </button>
        </div>
        {open && <Direction values={seg.values} set={(control, v) => D.setSegValue(seg.id, control, v)} clear={() => D.updateSeg(seg.id, { values: {} })} name={`segment ${k + 1}`} />}
      </div>
      <div class="dseg-side">
        <Result o={o} />
        {isStale && <span class="dseg-stale">changed · regenerate</span>}
        {o && o.status === 'ok' && (
          <div class="dseg-voicefix">
            <button class={`icon-btn sm dseg-flag${flagged ? ' is-on' : ''}`} onClick={() => D.toggleFlag(seg.id)} aria-pressed={flagged} aria-label={`Segment ${k + 1} sounds wrong`} title="Sounds wrong (garbled, unintelligible)? Flag it">
              ⚑
            </button>
            {cloned && (flagged || poor) && fallback && (
              <button class="btn sm" disabled={busy} onClick={() => void D.tryStudioVoice(seg.id)} title={`The source clone can garble a fast or noisy segment where a studio voice stays clear: regenerate this segment with ${D.voiceLabel(fallback)} (change it in the voice menu)`}>
                Try {D.voiceLabel(fallback).replace(/^Bench: /, '')}
              </button>
            )}
          </div>
        )}
        <button class={`btn sm${isStale ? ' primary' : ''}`} disabled={!d.track || busy} onClick={() => void D.regenerate(seg.id)} title={d.track ? 'New takes for this segment with its current text, bucket, voice and direction; the others are kept (R)' : 'Generate the dub first'}>
          <Icon name="refresh" size={12} /> Regenerate
        </button>
      </div>
    </li>
  );
}

function Segments() {
  const d = D.doc.value;
  const problems = D.segmentProblems.value;
  return (
    <section class="dub-segments" aria-label="Segments">
      <header class="ed-sub-head">
        <h3>Segments</h3>
        <span class="muted small">
          {d.segments.length} · {d.segments.filter((s) => s.target_text.trim()).length} with text
          {D.stale.value.size ? ` · ${D.stale.value.size} changed since the dub` : ''}
        </span>
        <span class="grow" />
        {d.source?.kind !== 'scene' && (
          <button class="link" onClick={D.clearSegments} title="Back to the transcript and the whole translation (undoable)">
            re-split
          </button>
        )}
      </header>
      {problems.length > 0 && (
        <p class="error-box" role="alert">
          {problems.join(' ')}
        </p>
      )}
      <ol class="dsegs">
        {d.segments.map((s, k) => (
          <SegmentRow key={s.id} seg={s} k={k} />
        ))}
      </ol>
    </section>
  );
}

// ── Screen ────────────────────────────────────────────────────────────────────────────────────────

function Unavailable() {
  const b = backend.value!;
  return (
    <div class="empty">
      <Icon name="globe" size={28} />
      <h2>This server can’t dub</h2>
      <p>Dubbing needs the speech setup next to the studio server: speech recognition (Qwen3-ASR), the word aligner (SimAlign) and, for lip-sync, AV-sync. Set it up once, then restart the server:</p>
      <pre class="mono">bash tools/setup_asr_env.sh</pre>
      <p class="muted small">{b.server || location.origin} does not serve dubbing (/v2/dub/capabilities: not enabled, or its workers are not installed). Without a server, add ?api=offline to try the whole flow on the built-in fixture.</p>
    </div>
  );
}

function Empty() {
  return (
    <div class="dub-empty">
      <h2>Dub a video, a scene or a take</h2>
      <p class="muted">Drop a file to transcribe and translate it, or hand over something you made in the studio: its text, timing and voices come with it.</p>
      <DropZone big />
      <div class="dub-empty-cards">
        <button class="dub-card" disabled={!scene.value.lines.length} onClick={() => void D.dubScene()}>
          <b>Dub this scene</b>
          <span class="muted small">{scene.value.lines.length ? `${scene.value.lines.length} lines, each keeps its character’s voice and direction` : 'Write a scene first'}</span>
        </button>
        <button class="dub-card" onClick={() => (act.value = 'direct')}>
          <b>A Direct take</b>
          <span class="muted small">Use the globe on any take, or pick one on the left</span>
        </button>
        <button class="dub-card" onClick={() => document.querySelector<HTMLElement>('.dub-studio details:last-of-type summary')?.click()}>
          <b>A voice clip</b>
          <span class="muted small">Any library voice’s reference clip</span>
        </button>
      </div>
    </div>
  );
}

function Header() {
  const d = D.doc.value;
  const t = d.track;
  const busy = D.busy.value;
  const [, tick] = useState(0);
  useEffect(() => {
    if (!busy) return;
    const i = setInterval(() => tick((n) => n + 1), 200);
    return () => clearInterval(i);
  }, [busy]);
  const generating = !!busy && /^Dubbing|^Regenerating/.test(busy.label);
  const editable = canEdit.value;
  const lang = D.caps.value?.target_languages.find((l) => l.code === d.targetLang)?.label ?? d.targetLang;
  return (
    <header class="screen-head">
      <div>
        <span class="eyebrow">Dub · {lang}</span>
        <h2 title={d.source?.name}>{d.source?.name}</h2>
      </div>
      <span class={`ed-status${busy ? ' is-busy' : ''}`} aria-live="polite">
        {busy ? `${busy.label}${busy.progress != null ? ` ${Math.round(busy.progress * 100)}%` : ''} ${((performance.now() - busy.started) / 1000).toFixed(1)} s` : ''}
      </span>
      <div class="row gap-s">
        <button class="btn" disabled={!t || !!busy || !editable} onClick={() => t && void openClip({ kind: 'dub', label: `Dub · ${lang}` }, t.pcm, t.sr, D.dubText(), DUB_TO_SERVER[t.targetLang] ?? 'en_us')} title={editable ? 'Open the dub track in Edit with its text known; a kept version comes back here' : 'This server can’t edit speech'}>
          <Icon name="edit" size={14} /> Open in Edit
        </button>
        <button class="btn" disabled={!t || !!busy} onClick={() => void D.sendToScene()} title="The dubbed lines as the Scene, each with its voice, text, place and direction">
          <Icon name="undo" size={14} /> Send to Scene
        </button>
        <button class="btn" disabled={!t} onClick={() => t && exportWav(t.pcm, t.sr, `${slug(d.source?.name.replace(/\.[^.]+$/, '') ?? 'dub')}-dub-${d.targetLang}`)} title="Download the dub track (WAV)">
          <Icon name="download" size={14} /> WAV
        </button>
        {d.source?.video && (
          <button class="btn" disabled={!t || !!busy} onClick={() => void D.downloadMp4()} title="The video with the dub as its audio (MP4, muxed on the server)">
            <Icon name="film" size={14} /> MP4
          </button>
        )}
        <button class={`btn primary${generating ? ' is-busy' : ''}`} disabled={!generating && !D.canGenerate.value} onClick={() => void D.generate()} title={generateHint()}>
          {generating ? (
            <>
              <Icon name="x" size={14} /> Cancel
            </>
          ) : (
            <>
              <Icon name="wand" size={14} /> {t ? 'Generate again' : 'Generate dub'}
            </>
          )}
        </button>
      </div>
    </header>
  );
}

function generateHint(): string {
  const d = D.doc.value;
  if (!d.segments.length) return 'Split the transcript into segments first';
  if (!d.segments.some((s) => s.target_text.trim())) return 'Write the target text of at least one segment';
  if (D.segmentProblems.value.length) return 'Fix the pronunciation first';
  if (d.source?.video && D.caps.value?.lipsync && D.lipsyncOn.value && (D.lips.value.state === 'uploading' || D.lips.value.state === 'tracking')) return 'Waiting for the face track (lip-sync)…';
  return 'Generate the whole dub (⌘↵ / Ctrl+Enter)';
}

export function DubScreen() {
  const c = backend.value!.dub;
  useEffect(() => {
    activeHistory.value = D.docs as History<unknown>;
    primaryAction.value = () => D.togglePlay();
    screenKeys.value = c ? dubKeys : null;
    return () => {
      activeHistory.value = null;
      primaryAction.value = null;
      screenKeys.value = null;
      if (isDubPlaying()) player.stop();
    };
  }, [!!c]);
  if (!c) return <Unavailable />;
  const d = D.doc.value;
  return (
    <div class="dub">
      <aside class="rail dub-rail" aria-label="Source, languages and voice">
        <SourceCard />
        <StudioSources />
        {d.source && <Settings />}
        <Versions />
      </aside>
      <section class="dub-main" aria-label="Dub">
        {!d.source ? (
          <Empty />
        ) : (
          <>
            <Header />
            {D.error.value && (
              <p class="error-box" role="alert">
                {D.error.value}
              </p>
            )}
            <div class="dub-stage">
              <div class="dub-picture">
                <Picture />
              </div>
              <div class="dub-tracks">
                <Transport />
                <Timeline />
                <Metrics />
              </div>
            </div>
            {d.segments.length ? <Segments /> : <ScriptStage />}
          </>
        )}
      </section>
      <FileDrop label="Drop to dub this file" hint="A video or an audio file" icon="film" busy={D.busy.value ? 'Wait for the current step to finish…' : null} onFile={(f) => void D.openFile(f)} />
    </div>
  );
}
