import { signal } from '@preact/signals';
import { useEffect, useMemo, useRef, useState } from 'preact/hooks';
import * as api from '../../api';
import { exportWav } from '../../audio/mix';
import * as player from '../../audio/player';
import { nowPlaying } from '../../audio/player';
import { FileDrop } from '../../components/FileDrop';
import { Icon } from '../../components/Icon';
import { closeRail, openRail, Rail, RailBar, RailToggle } from '../../components/Rail';
import { Waveform } from '../../components/Waveform';
import { isTagToken, normalizeTokens, planRegions, wordDiff } from '../../edit/regions';
import { isTyping } from '../../lib/dom';
import { slug } from '../../lib/files';
import { EDIT_LANGUAGES, langLabel } from '../../lib/languages';
import { parsePron } from '../../lib/pron';
import { PronMarks, PronounceButton } from '../../components/Pron';
import { TagButton, TagMarks, tagsOn } from '../../components/Tags';
import { clamp } from '../../lib/math';
import { activeHistory, backend, notify, primaryAction, screenKeys } from '../../state/app';
import * as E from '../../state/edit';
import type { History } from '../../state/history';
import { speakerLanguage, speakers } from '../../state/library';
import { script, takes as scriptTakes } from '../../state/script';
import { Direction } from './Direction';
import { EditWave, fmtTime } from './EditWave';

const zoom = signal(1);
const zoomTo = signal<{ s: number; e: number; n: number } | null>(null);
const focusRegion = signal<string | null>(null);

/** The word edge a pointer drag is moving (redirected once for Alt at a shared boundary). */
let wordDrag: { i: number; edge: 'start' | 'end'; t0: number; decided: boolean } | null = null;

function focusSelectedWord() {
  requestAnimationFrame(() => document.querySelector<HTMLElement>('.ew-wblock.is-sel .ew-wlabel')?.focus({ preventScroll: false }));
}

function editKeys(e: KeyboardEvent): boolean {
  const mod = e.metaKey || e.ctrlKey;
  if (mod && e.key === 'Enter') {
    e.preventDefault();
    void E.generate();
    return true;
  }
  const target = e.target as HTMLElement | null;
  const role = target?.getAttribute?.('role');
  // Tab cycles words only from a word label; in the transcript and elsewhere it is the browser's.
  const inWords = !!target?.closest?.('.ew-wlabel');
  if (mod || e.altKey || (isTyping(e.target) && !(e.key === 'Tab' && inWords)) || !E.version.value) return false;
  const d = E.displayed.value;
  // A selected word: the keys edit its timing.
  const sw = E.selectedWord.value;
  if (sw !== null && d?.kind === 'version' && role !== 'slider') {
    const n = E.version.value.words.length;
    switch (e.key) {
      case 'ArrowLeft':
      case 'ArrowRight':
        if (isTyping(e.target)) return false;
        e.preventDefault();
        E.nudgeWord((e.key === 'ArrowLeft' ? -1 : 1) * (e.shiftKey ? 0.05 : 0.01));
        return true;
      case 'Tab':
        if (!inWords) return false;
        e.preventDefault();
        E.selectedWord.value = clamp(sw + (e.shiftKey ? -1 : 1), 0, n - 1);
        focusSelectedWord();
        return true;
      case ',':
      case '.':
        if (isTyping(e.target)) return false;
        E.activeEdge.value = e.key === ',' ? 'start' : 'end';
        return true;
      case 'Enter':
        if (isTyping(e.target)) return false;
        e.preventDefault();
        E.playWord(sw);
        return true;
      case 'Escape':
        E.selectedWord.value = null;
        return true;
    }
  }
  if (isTyping(e.target)) return false;
  switch (e.key) {
    case ' ':
      if (e.target instanceof HTMLButtonElement || role === 'slider' || role === 'button') return false;
      e.preventDefault();
      E.togglePlay();
      return true;
    case 'a':
    case 'b':
      if (role === 'slider') return false;
      if (e.key === 'b' && !E.selectedTake.value) {
        notify('Generate takes first');
        return true;
      }
      E.toggleAB(e.key === 'a' ? 'version' : 'take');
      return true;
    case 'k':
      if (!E.selectedTake.value) {
        notify('Generate takes first');
        return true;
      }
      void E.keep();
      return true;
    case '[':
      if (role === 'slider') return false;
      zoom.value = clamp(zoom.value / 1.6, 1, 64);
      return true;
    case ']':
      if (role === 'slider') return false;
      zoom.value = clamp(zoom.value * 1.6, 1, 64);
      return true;
    case '0':
      if (role === 'slider') return false;
      zoom.value = 1;
      return true;
    case 'ArrowLeft':
    case 'ArrowRight': {
      if (role === 'slider' || !d) return false;
      e.preventDefault();
      const playing = nowPlaying.value?.id === d.id;
      const at = clamp((playing ? player.playhead.value : E.cursor.value) + (e.key === 'ArrowLeft' ? -1 : 1), 0, d.duration);
      E.cursor.value = at;
      if (playing) player.play(d.id, d.pcm, d.sr, at);
      return true;
    }
  }
  return false;
}

// ── Sources ───────────────────────────────────────────────────────────────────────────────────────

const loadingVoice = signal<string | null>(null);

async function openVoice(id: string) {
  const s = speakers.value.find((x) => x.id === id);
  if (!s) return;
  loadingVoice.value = id;
  try {
    const { pcm, sr } = await player.decode(await api.getPreview(s));
    if (await E.openClip({ kind: 'voice', label: s.label }, pcm, sr, s.transcript, speakerLanguage(s.id))) closeRail();
  } catch (e) {
    notify(`Couldn’t load ${s.label}: ${e instanceof Error ? e.message : e}`, 'error');
  } finally {
    loadingVoice.value = null;
  }
}

/** A Direct take into the editor; on a phone the sources drawer closes once it is open. */
async function openTake(lineId: string, lineIndex: number, takeIndex: number) {
  const t = scriptTakes.value[lineId]?.[takeIndex];
  if (!t) return;
  if (await E.openClip({ kind: 'take', label: `Line ${lineIndex + 1} · take ${takeIndex + 1}`, lineId, takeId: t.id }, t.audio.pcm, t.audio.sr, t.text, t.request.language ?? 'en_us')) closeRail();
}

function Sources() {
  const file = useRef<HTMLInputElement>(null);
  const rec = E.recording.value;
  const [, tick] = useState(0);
  useEffect(() => {
    if (!rec) return;
    const t = setInterval(() => tick((n) => n + 1), 250);
    return () => clearInterval(t);
  }, [rec]);
  const lineTakes = script.value.lines.flatMap((l, i) => (scriptTakes.value[l.id] ?? []).map((t, k) => ({ l, i, t, k })));
  const clips = speakers.value.filter((s) => s.has_preview !== false);
  const busy = !!E.busy.value;
  return (
    <div class="rail-section edit-sources">
      <h4>Open audio</h4>
      <div class="row gap-s">
        <button class="btn sm" disabled={busy} onClick={() => file.current?.click()}>
          <Icon name="upload" size={13} /> Upload
        </button>
        <button class={`btn sm btn-rec${rec ? ' is-recording' : ''}`} disabled={busy && !rec} onClick={() => void E.recordClip()}>
          {rec ? (
            <>
              <span class="rec-dot" /> Stop · {Math.floor((Date.now() - rec.started) / 1000)} / {Math.round(Math.min(120, E.maxSeconds.value))} s
              <span class="rec-level" aria-hidden="true">
                <span style={{ width: `${Math.round(E.recordLevel.value * 100)}%` }} />
              </span>
            </>
          ) : (
            <>
              <Icon name="mic" size={13} /> Record
            </>
          )}
        </button>
        <input
          ref={file}
          type="file"
          accept="audio/*,video/*"
          hidden
          onChange={(e) => {
            const f = (e.currentTarget as HTMLInputElement).files?.[0];
            (e.currentTarget as HTMLInputElement).value = '';
            if (f) void E.openFile(f);
          }}
        />
      </div>
      <p class="hint">Clips up to {Math.round(E.maxSeconds.value)} s.</p>
      <details class="src-group" open={!!lineTakes.length}>
        <summary>Direct takes {lineTakes.length ? <b>{lineTakes.length}</b> : null}</summary>
        {lineTakes.length ? (
          <ul class="src-list">
            {lineTakes.map(({ l, i, t, k }) => (
              <li>
                <button class="src-item" disabled={busy} title={t.text} onClick={() => void openTake(l.id, i, k)}>
                  <span class="src-n">
                    {i + 1}.{k + 1}
                  </span>
                  <span class="src-text">{t.text}</span>
                  {t.starred && <Icon name="star" size={11} filled />}
                </button>
              </li>
            ))}
          </ul>
        ) : (
          <p class="muted small">Render takes in Direct; each one can be opened here.</p>
        )}
      </details>
      <details class="src-group" open={clips.length > 0}>
        <summary>
          Voice clips {clips.length ? <b>{clips.length}</b> : null}
        </summary>
        <ul class="src-list">
          {clips.map((s) => (
            <li>
              <button
                class={`src-item${loadingVoice.value === s.id ? ' is-busy' : ''}`}
                disabled={busy}
                onClick={() => void openVoice(s.id)}
                title={s.transcript ? `Transcript: ${s.transcript}` : E.canTranscribe.value ? 'No transcript: it will be transcribed' : 'No transcript: this server can’t transcribe, so you type what the clip says'}
              >
                <span class="src-text">{s.label}</span>
                <span class="tag">{s.transcript ? 'text' : E.canTranscribe.value ? 'auto' : 'needs text'}</span>
              </button>
            </li>
          ))}
        </ul>
      </details>
    </div>
  );
}

function Versions() {
  const d = E.doc.value;
  if (!d) return null;
  return (
    <div class="rail-section grow">
      <h4>Versions</h4>
      <ol class="versions">
        {d.versions.map((v, i) => {
          const id = `edit:v:${v.id}`;
          const playing = nowPlaying.value?.id === id;
          return (
            <li class={`version-row${i === d.current ? ' is-current' : ''}${i > d.current ? ' is-future' : ''}`}>
              <button class="version-main" onClick={() => void E.jumpTo(i)} aria-current={i === d.current} title={i === d.current ? 'Current version' : 'Step to this version'}>
                <span class="badge">v{i}</span>
                <span class="version-label">{v.label}</span>
                <span class="mono muted">{(v.pcm.length / v.sr).toFixed(1)} s</span>
              </button>
              <button class={`play-mini${playing ? ' is-playing' : ''}`} onClick={() => player.toggle(id, v.pcm, v.sr)} aria-label={`Play version ${i}`}>
                <Icon name={playing ? 'stop' : 'play'} size={11} />
              </button>
            </li>
          );
        })}
      </ol>
      <p class="hint">Keep a take to add a version; ⌘Z / ⇧⌘Z step through them.</p>
    </div>
  );
}

// ── Transcript ────────────────────────────────────────────────────────────────────────────────────

/** Changed words highlighted behind the textarea (same metrics), deletions as a caret mark. */
function Mirror({ value }: { value: string }) {
  const v = E.version.value!;
  const groups = E.pronEdit.value;
  const plan = useMemo(() => planRegions(v.words, value, { duration: v.duration, frameRate: E.fps.value, marginSeconds: E.margin.value, groups }), [v, value, groups]);
  const cuts: { from: number; to: number; type: 'mark' | 'del' }[] = [];
  const marks = new Map<number, number>();
  let del = 0;
  for (const r of plan.regions) {
    for (const c of r.changes) {
      if (c.j1 === c.j2) {
        cuts.push({ from: plan.deletionPoints[del++], to: 0, type: 'del' });
        continue;
      }
      for (let j = c.j1; j < c.j2; j++) {
        const o = plan.owners[j];
        if (o) marks.set(o[0], o[1]);
      }
    }
  }
  for (const [from, to] of marks) cuts.push({ from, to, type: 'mark' });
  cuts.sort((a, b) => a.from - b.from || (a.type === 'del' ? -1 : 1));
  const out: preact.ComponentChildren[] = [];
  let pos = 0;
  for (const c of cuts) {
    if (c.from < pos) continue;
    out.push(value.slice(pos, c.from));
    if (c.type === 'del') {
      out.push(<span class="ed-del" />);
      pos = c.from;
    } else {
      out.push(<mark class="ed-mark">{value.slice(c.from, c.to)}</mark>);
      pos = c.to;
    }
  }
  out.push(value.slice(pos));
  if (value.endsWith('\n')) out.push('\u200b');
  return (
    <div class="ed-mirror" aria-hidden="true">
      {out}
    </div>
  );
}

function Transcript({ area }: { area: { current: HTMLTextAreaElement | null } }) {
  const v = E.version.value!;
  const fix = E.fixMode.value;
  // No transcript yet (a server without speech recognition): typing it is the first step.
  const first = E.needsTranscript.value;
  const initial = () => v.transcript || E.suggestedTranscript.value;
  const [draft, setDraft] = useState(initial);
  useEffect(() => setDraft(initial()), [v.id, fix]);
  const value = fix ? draft : E.text.value;
  const autosize = () => {
    const el = area.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.max(64, el.scrollHeight)}px`;
  };
  useEffect(autosize, [value, fix]);
  useEffect(() => {
    if (first) area.current?.focus();
  }, [v.id, first]);
  const busy = !!E.busy.value;
  return (
    <section class="ed-text-card" aria-label="Transcript">
      <header class="ed-text-head">
        {first || fix ? (
          <div class="segctl" role="tablist" aria-label="Transcript mode">
            <button role="tab" aria-selected={!fix} class={`seg-btn${!fix ? ' is-on' : ''}`} onClick={() => void E.setFix(false)} disabled={first} title={first ? 'Type what the clip says first: edits are planned against it' : undefined}>
              Edit the words
            </button>
            <button role="tab" aria-selected={fix} class={`seg-btn${fix ? ' is-on' : ''}`} onClick={() => void E.setFix(true)} disabled={busy} title={first ? 'Type what is said in the clip; the audio is aligned to it' : 'Correct what was actually said; the audio is re-aligned to it'}>
              {first ? 'Type what the clip says' : 'Fix the transcript'}
            </button>
          </div>
        ) : (
          <>
            <h3 class="ed-text-title">Edit the words</h3>
            {!v.automatic && (
              <button class="link" onClick={() => void E.setFix(true)} disabled={busy} title="Correct what was actually said; the audio is re-aligned to it">
                Fix the transcript
              </button>
            )}
          </>
        )}
        {!fix && E.text.value !== v.transcript && (
          <button
            class="link"
            onClick={() => {
              const before = E.text.value;
              E.setText(v.transcript);
              notify('Text reverted to the transcript', 'info', { label: 'Undo', run: () => E.setText(before) });
            }}
            title="Back to the transcript of this version"
          >
            <Icon name="undo" size={13} /> revert text
          </button>
        )}
      </header>
      {first && (
        <div class="ed-banner" role="note">
          <span>
            {E.canTranscribe.value ? 'No words were recognized in this clip. ' : 'This server can’t transcribe speech. '}
            <b>Type exactly what the clip says</b> (play it to check), then use it: edits are planned against it.
          </span>
        </div>
      )}
      {!fix && v.automatic && (
        <div class="ed-banner ed-check" role="note">
          <span>
            <b>Transcribed automatically</b> — check it: play the clip and read along.
          </span>
          <button class="btn sm primary" onClick={E.confirmTranscript} disabled={busy}>
            Looks right
          </button>
          <button class="btn sm" onClick={() => void E.setFix(true)} disabled={busy}>
            Fix it
          </button>
        </div>
      )}
      <div class={`ed-editor${fix ? ' is-fix' : ''}`}>
        {!fix && <Mirror value={value} />}
        <textarea
          ref={area}
          class="ed-area"
          value={value}
          spellcheck={false}
          data-tags={fix ? undefined : true}
          aria-label={first ? 'What the clip says' : fix ? 'Correct transcript of the audio' : 'Edited transcript: change the words to change the speech'}
          placeholder={first ? 'Type what is said in the clip…' : undefined}
          onInput={(e) => {
            const t = (e.currentTarget as HTMLTextAreaElement).value;
            if (fix) setDraft(t);
            else E.setText(t);
          }}
          onKeyDown={(e) => {
            if (e.key === 'Escape') (e.currentTarget as HTMLTextAreaElement).blur();
          }}
        />
      </div>
      {!fix && (E.pronEdit.value || tagsOn.value) && (
        <div class="ed-pron">
          {E.pronEdit.value && <PronounceButton input={() => area.current} />}
          <TagButton input={() => area.current} />
          {E.pronEdit.value && <PronMarks text={value} replace={(s, e, t) => E.setText(value.slice(0, s) + t + value.slice(e))} />}
          <TagMarks text={value} replace={(next) => E.setText(next)} />
        </div>
      )}
      {fix && (
        <div class="ed-fix-bar">
          <span class="hint">
            {first ? 'Type what is said in the audio, word for word.' : 'Type what is actually said in the audio. Low-confidence words are underlined in the waveform’s word lane.'}
          </span>
          {!first && (
            <button class="btn sm ghost" onClick={() => void E.setFix(false)}>
              Cancel
            </button>
          )}
          <button class="btn sm primary" disabled={busy || !draft.trim() || draft.trim() === v.transcript} onClick={() => void E.saveFix(draft)}>
            Use this transcript
          </button>
        </div>
      )}
    </section>
  );
}

// ── Regions ───────────────────────────────────────────────────────────────────────────────────────

/** The one region whose direction chips are open (the others show a summary row). */
const expandedRegion = signal<string | null>(null);

function Regions() {
  const rs = E.regions.value;
  const g = E.gen.value;
  const sel = E.selectedTake.value;
  const earlier = !!g && g.target !== E.text.value.trim();
  const missing = g && sel && !earlier && g.spans.length === rs.length ? E.takeUnheard(sel, g) : null;
  const err = E.regionError.value;
  const secs = rs.reduce((a, r) => a + E.seconds(r.end - r.start), 0);
  return (
    <section class="ed-regions" aria-label="Regions">
      <header class="ed-sub-head">
        <h3>Regions</h3>
        <span class={`ed-count${err ? ' is-error' : ''}`}>
          {rs.length}/{E.caps.value.edit_max_spans} · {secs.toFixed(1)} s to regenerate
        </span>
      </header>
      {err && (
        <p class="error-box" role="alert">
          {err}
        </p>
      )}
      {earlier && g && g.takes.length > 0 && <p class="note">The takes were made from earlier text; generate again for this one.</p>}
      {!rs.length ? (
        <p class="ed-regions-empty">
          <b>Change words</b>, or <b>drag across the waveform</b> to re-say a stretch as it is. Then pick how it should sound: whisper, slower, angry…
        </p>
      ) : (
        <ol class="region-list">
          {rs.map((r, k) => {
            const s = E.seconds(r.start);
            const e = E.seconds(r.end);
            const over = k >= E.caps.value.edit_max_spans;
            return (
              <li class={`region-row${r.manual ? ' is-manual' : ''}${over ? ' is-over' : ''}${focusRegion.value === r.id ? ' is-focus' : ''}`} onPointerEnter={() => (focusRegion.value = r.id)} onPointerLeave={() => (focusRegion.value = null)}>
                <div class="region-line">
                  <span class="badge region-n">{k + 1}</span>
                  <span class="mono region-time">
                    {fmtTime(s)}–{fmtTime(e)}
                  </span>
                  <span class="region-sum" title={E.summaryOf(r)}>
                    {E.summaryOf(r)}
                  </span>
                  <button class="icon-btn sm" onClick={() => E.playRegion(k)} aria-label={`Play around region ${k + 1}`} title="Play with 1 s of pre-roll">
                    <Icon name="play" size={12} />
                  </button>
                  <button class="icon-btn sm" onClick={() => (zoomTo.value = { s: s - 1, e: e + 1, n: Date.now() })} aria-label={`Zoom to region ${k + 1}`} title="Zoom to this region">
                    <Icon name="expand" size={12} />
                  </button>
                  {r.manual && (
                    <button class="icon-btn sm" onClick={() => E.removeRegion(r.id)} aria-label={`Remove region ${k + 1}`} title="Remove this region">
                      <Icon name="x" size={12} />
                    </button>
                  )}
                  {r.changes.length > 0 && (
                    <button class="link" onClick={() => E.revertRegion(r.id)} title="Put the original words back into the text">
                      revert these words
                    </button>
                  )}
                </div>
                {(r.edited || r.mergedFrom) && (
                  <p class="region-note">
                    {r.edited ? 'edges moved by hand' : ''}
                    {r.edited && r.mergedFrom ? ' · ' : ''}
                    {r.mergedFrom ? `merged from ${r.mergedFrom} regions` : ''}
                  </p>
                )}
                {missing?.[k]?.length ? (
                  <p class="region-unheard" role="note">
                    Not heard in {sel!.label.toLowerCase()}: <b>{missing[k].join(', ')}</b>
                  </p>
                ) : null}
                <div class="region-dir">
                  {expandedRegion.value === r.id ? (
                    <>
                      <div class="region-dir-head">
                        <span class="muted small">How region {k + 1} sounds</span>
                        <button class="link" onClick={() => (expandedRegion.value = null)}>
                          Done
                        </button>
                      </div>
                      <Direction name={`region ${k + 1}`} values={r.values} set={(c, v) => E.setRegionValue(r.id, c, v)} clear={() => E.clearRegionValues(r.id)} />
                    </>
                  ) : (
                    <div class="region-dir-sum">
                      <span class="muted small">How it sounds: {E.describeValues(r.values) || (E.directed.value && Object.keys(E.wholeValues.value).length ? 'as all changes' : 'as the voice says it')}</span>
                      <button class="link" onClick={() => (expandedRegion.value = r.id)}>
                        Edit
                      </button>
                    </div>
                  )}
                </div>
              </li>
            );
          })}
        </ol>
      )}
    </section>
  );
}

/** What the server is doing for this clip, with elapsed time and a Cancel: a sticky row under the header. */
function BusyRow() {
  const b = E.busy.value;
  const [, tick] = useState(0);
  useEffect(() => {
    if (!b) return;
    const t = setInterval(() => tick((n) => n + 1), 100);
    return () => clearInterval(t);
  }, [b]);
  if (!b) return null;
  const elapsed = (performance.now() - b.started) / 1000;
  return (
    <div class="ed-busy-row" role="status" aria-live="polite">
      <span class="loader sm" />
      <span class="ed-status is-busy">
        {b.label} {elapsed.toFixed(1)} s
      </span>
      <button class="btn sm" onClick={E.cancel}>
        <Icon name="x" size={13} /> Cancel
      </button>
    </div>
  );
}

function GenerateBar() {
  const b = E.busy.value;
  const n = E.takeCount.value;
  const generating = !!b && /Generating|Regenerating/.test(b.label);
  const long = (E.version.value?.duration ?? 0) > 16;
  return (
    <section class="ed-generate" aria-label="Generate">
      <div class="ed-all">
        <span class="ed-all-label">All changes sound:</span>
        <Direction name="all changes" values={E.wholeValues.value} set={E.setWholeValue} clear={() => (E.wholeValues.value = {})} />
      </div>
      <details class="pro ed-advanced">
        <summary>Advanced</summary>
        <div class="ed-pro-row">
          <label class="ed-seed" title="A fixed seed repeats a take's randomness; empty picks one each time">
            Seed
            <input
              type="number"
              class="input"
              placeholder="random"
              value={E.seedPin.value ?? ''}
              onChange={(e) => {
                const x = (e.currentTarget as HTMLInputElement).value;
                E.seedPin.value = x === '' ? null : Math.max(0, Math.floor(Number(x)));
              }}
            />
          </label>
          <label title="Each change is widened by this much on both sides so the join sounds smooth">
            Blend around the change
            <input
              type="number"
              class="input"
              min={0}
              max={0.4}
              step={0.02}
              value={E.margin.value}
              onChange={(e) => {
                E.margin.value = clamp(Number((e.currentTarget as HTMLInputElement).value) || 0, 0, 0.4);
                E.replan();
              }}
            />
            s
          </label>
          {long && (
            <label title="Long clips are edited on a window around the changes; this is how much audio the model listens to on each side">
              Audio the model listens to
              <input
                type="number"
                class="input"
                min={2}
                max={8}
                step={0.5}
                value={E.contextSeconds.value}
                onChange={(e) => (E.contextSeconds.value = clamp(Number((e.currentTarget as HTMLInputElement).value) || E.DEFAULT_CONTEXT, 2, 8))}
              />
              s per side
            </label>
          )}
        </div>
      </details>
      <div class="ed-gen-row">
        <div class="segctl" role="radiogroup" aria-label="Number of takes">
          {[1, 2, 3, 4].map((k) => (
            <button role="radio" aria-checked={n === k} class={`seg-btn${n === k ? ' is-on' : ''}`} onClick={() => (E.takeCount.value = k)}>
              {k}
            </button>
          ))}
        </div>
        <span class="muted small">take{n > 1 ? 's' : ''}</span>
        <span class="grow" />
        <button
          class={`btn primary${generating ? ' is-busy' : ''}`}
          disabled={generating || !E.canGenerate.value}
          onClick={() => void E.generate()}
          title={E.pronError.value ? `Fix the pronunciation first: ${E.pronError.value.replace(/^Pronunciation: /, '')}` : 'Generate (⌘↵ / Ctrl+Enter)'}
        >
          <Icon name="wand" size={14} /> {generating ? 'Generating…' : `Generate ${n} take${n > 1 ? 's' : ''}`}
          {!generating && E.directed.value && <span class="gen-directed"> · directed</span>}
        </button>
      </div>
      {E.genError.value && (
        <div class="error-box" role="alert">
          <b>{E.genRefused.value ? 'The server refused this edit' : 'Generation failed'}</b>
          <span>{E.genError.value}</span>
        </div>
      )}
    </section>
  );
}

// ── Takes ─────────────────────────────────────────────────────────────────────────────────────────

function Takes() {
  const g = E.gen.value;
  const v = E.version.value;
  if (!g || !v || g.versionId !== v.id) {
    return (
      <div class="takes-empty">
        <Icon name="wand" size={20} />
        <p>{E.canTranscribe.value ? 'Takes appear here. Each one is checked by speech recognition; the closest to your text is picked for you.' : 'Takes appear here. Listen to each one and keep the best (this server can’t check takes with speech recognition).'}</p>
      </div>
    );
  }
  // Tags are never heard: the transcript is compared with the words only.
  const target = normalizeTokens(g.target, g.groups).filter((t) => !isTagToken(t));
  const written = g.groups ? parsePron(g.target, null).groups.filter((x) => !x.empty) : [];
  const best = E.bestOf(g.takes, g);
  return (
    <ol class="take-cards">
      {g.takes.map((t, i) => {
        const id = `edit:take:${t.id}`;
        const playing = nowPlaying.value?.id === id;
        const sel = g.selected === i;
        const wer = t.verify?.wer ?? null;
        const hyp = t.verify ? t.verify.asr_text.split(/\s+/).filter(Boolean) : [];
        const hypTok = t.verify ? normalizeTokens(t.verify.asr_text) : [];
        const wrong = hyp.length === hypTok.length ? wordDiff(target, hypTok) : null;
        const warns = E.takeWarnings(t, g);
        // One verdict per take: every word heard, some missing, or not checked (no speech recognition).
        const unheard = t.verify ? E.takeUnheard(t, g).flat() : null;
        const verdict = !t.verify ? (['Not checked', 'muted', 'No speech recognition on this server: listen to it'] as const) : unheard!.length ? ([`Missing: ${unheard!.join(', ')}`, 'warn', 'Words of your text speech recognition did not hear in this take'] as const) : (['All words heard', 'ok', 'Speech recognition heard every word of your text'] as const);
        return (
          <li class={`take-card${sel ? ' is-selected' : ''}`}>
            <button class="take-card-head" onClick={() => E.selectTake(i)} aria-pressed={sel} title="Select (A/B compares it with the version)">
              <b>{t.label}</b>
              <span class={`badge verdict is-${verdict[1]}`} title={verdict[2]}>
                {verdict[0]}
              </span>
              {i === best && wer !== null && <span class="badge badge-best">best match</span>}
            </button>
            <div class="take-wave" onClick={() => E.selectTake(i)}>
              <Waveform pcm={t.pcm} color={sel ? 'var(--o)' : 'var(--text-2)'} playId={id} height={34} />
            </div>
            {t.verify && (
              <p class="take-asr" title="What speech recognition heard; differences from your text are marked">
                {wrong
                  ? hyp.map((w, k) => (
                      <>
                        <span class={wrong[k] ? 'asr-wrong' : ''}>{w}</span>{' '}
                      </>
                    ))
                  : t.verify.asr_text}
              </p>
            )}
            {written.length > 0 && t.verify && (
              <p class="take-heard" title="What speech recognition wrote where each pronunciation is (any word counts as said)">
                {written.map((w, k) => {
                  const h = t.verify!.groups?.[k]?.heard;
                  return (
                    <span class={`take-heard-item${h ? '' : ' is-missing'}`}>
                      <span class="mono">({w.body})</span> {h ? <>heard as ‘{h}’</> : 'nothing recognized there'}
                    </span>
                  );
                })}
              </p>
            )}
            <p class={`take-dir${t.direction ? '' : ' is-none'}`} title="How this take was directed">
              {t.direction || 'no direction: as the voice says it'}
            </p>
            {warns.map((w) => (
              <p class="warn-line">{w.text}</p>
            ))}
            <div class="take-actions">
              <button class={`play-mini${playing ? ' is-playing' : ''}`} onClick={() => (E.selectTake(i), player.toggle(id, t.pcm, t.sr))} aria-label={`Play ${t.label}`}>
                <Icon name={playing ? 'stop' : 'play'} size={11} />
              </button>
              {g.spans.map((_, k) => (
                <button class="btn sm ghost" onClick={() => (E.selectTake(i), E.toggleAB('take'), E.playRegion(k))} title={`Play change ${k + 1} of this take with pre-roll`}>
                  <Icon name="play" size={11} /> Play change {k + 1}
                </button>
              ))}
              <span class="grow" />
              {sel &&
                g.spans.map((_, k) => (
                  <button class="btn sm ghost" disabled={!!E.busy.value} onClick={() => void E.retryRegion(k)} title={`Regenerate only change ${k + 1} of this take`}>
                    <Icon name="refresh" size={12} /> Redo change {k + 1}
                  </button>
                ))}
              <button class="icon-btn sm" onClick={() => exportWav(t.pcm, t.sr, `${slug(E.doc.value?.name ?? 'edit')}-${slug(t.label)}`)} aria-label={`Export ${t.label} as WAV`} title={`Export ${t.label.replace('Take ', 'take ')}`}>
                <Icon name="download" size={13} />
              </button>
              <button class="btn sm primary" disabled={!!E.busy.value} onClick={() => (E.selectTake(i), void E.keep())} title="Keep: becomes the new version (K)">
                Keep
              </button>
            </div>
            <details class="pro take-details">
              <summary>Details</summary>
              <p class="mono small">
                {wer !== null ? `Word error ${Math.round(wer * 100)}% · ` : ''}seed {t.seed}
                {g.spans.map(([s, e], k) => ` · change ${k + 1}: ${(e - s).toFixed(2)} → ${(t.fill[k] ?? 0).toFixed(2)} s`).join('')}
              </p>
              <button class="link" onClick={() => (E.seedPin.value = t.seed)} title="Generate again with this take's seed (Advanced → Seed)">
                Reuse seed
              </button>
            </details>
          </li>
        );
      })}
    </ol>
  );
}

// ── Screen ────────────────────────────────────────────────────────────────────────────────────────

function Unavailable() {
  const b = backend.value!;
  return (
    <div class="empty">
      <Icon name="edit-speech" size={28} />
      <h2>This server can’t edit speech</h2>
      <p>Changing words in a recording needs a server with the speech-editing model. Cast and Direct work as usual.</p>
      <details class="pro">
        <summary>For whoever runs the server</summary>
        <p>
          {b.kind !== 'fixture' && b.kind !== 'offline' ? `${b.server || location.origin} reports speech_editing: false. ` : 'Fixture mode: started with ?edit=0. '}
          The studio README (“Edit”) lists what the server needs to edit speech.
        </p>

      </details>
    </div>
  );
}

/** The empty workspace: ways to get a clip in (the drawer on phones; the rail on desktop). */
function EmptyWorkspace() {
  const file = useRef<HTMLInputElement>(null);
  const busy = E.busy.value;
  if (busy) {
    return (
      <div class="empty">
        <BusyRow />
      </div>
    );
  }
  return (
    <div class="empty">
      <Icon name="edit-speech" size={26} />
      <h3>Open a clip to edit</h3>
      <p>Change the words of a recording; only the changed stretch is regenerated.</p>
      <div class="row gap-s">
        <button class="btn primary" onClick={() => file.current?.click()}>
          <Icon name="upload" size={14} /> Upload
        </button>
        <button class="btn" onClick={() => void E.recordClip()}>
          <Icon name="mic" size={14} /> Record
        </button>
        <button class="btn" onClick={() => openRail('edit-sources')}>
          <Icon name="film" size={14} /> Choose a take or voice
        </button>
        <input
          ref={file}
          type="file"
          accept="audio/*,video/*"
          hidden
          onChange={(e) => {
            const f = (e.currentTarget as HTMLInputElement).files?.[0];
            (e.currentTarget as HTMLInputElement).value = '';
            if (f) void E.openFile(f);
          }}
        />
      </div>
    </div>
  );
}

function Workspace() {
  const v = E.version.value;
  const d = E.doc.value;
  const disp = E.displayed.value;
  const area = useRef<HTMLTextAreaElement>(null);
  const isTake = disp?.kind === 'take';
  // After Generate the view switches to the take: bring the selected card into view.
  useEffect(() => {
    if (!isTake) return;
    requestAnimationFrame(() => document.querySelector<HTMLElement>('.take-card.is-selected')?.scrollIntoView({ block: 'nearest', behavior: 'smooth' }));
  }, [isTake, E.gen.value?.selected]);
  if (!v || !d || !disp) return <EmptyWorkspace />;
  const t = E.selectedTake.value;
  const shownLabel = isTake ? disp.label : d.current === 0 ? 'Original' : `v${d.current}`;
  const playing = nowPlaying.value?.id === disp.id;
  const timingEditable = !isTake && !E.fixMode.value;
  const onWord = (i: number) => {
    const w = disp.words[i];
    if (timingEditable) {
      E.selectedWord.value = i;
      E.playWord(i);
      focusSelectedWord();
      return;
    }
    if (w) player.play(disp.id, disp.pcm, disp.sr, Math.max(0, w.start - 0.05), undefined, w.end + 0.05);
    if (isTake || E.fixMode.value || !area.current) return;
    // Select the word in the editor: its matched token, else the change it belongs to.
    const plan = planRegions(v.words, E.text.value, { duration: v.duration, frameRate: E.fps.value, marginSeconds: E.margin.value, groups: E.pronEdit.value });
    let first = 0;
    for (let k = 0; k < i; k++) first += v.words[k].tokens.length;
    const j = plan.equalMap[first];
    const owner = j >= 0 ? plan.owners[j] : null;
    if (owner) {
      area.current.focus();
      area.current.setSelectionRange(owner[0], owner[1]);
    }
  };
  return (
    <div class="ed-work">
      <header class="screen-head ed-head">
        <div class="ed-title">
          <span class="eyebrow">{({ take: 'Direct take', voice: 'Voice clip', upload: 'Upload', recording: 'Recording' } as const)[d.source.kind]}</span>
          <h2 title={d.name}>{d.name}</h2>
        </div>
        <label class="ed-lang-wrap">
          <span class="muted small">clip language</span>
          <select
            class="input ed-lang"
            value={d.language}
            disabled={!!E.busy.value}
            onChange={(e) => {
              const el = e.currentTarget as HTMLSelectElement;
              const next = el.value;
              el.value = d.language;
              void E.setLanguage(next);
            }}
            aria-label="Language of the clip"
            title={E.canTranscribe.value ? "The clip's language: transcription, alignment and the take check use its speech model (the first use of a language loads it, about half a minute)" : "The clip's language: alignment uses it (switching aligns the transcript again)"}
          >
            {EDIT_LANGUAGES.map((l) => (
              <option value={l}>{langLabel(l)}</option>
            ))}
          </select>
        </label>
        <div class="ab" role="group" aria-label="Compare">
          <button class={`ab-btn${!isTake ? ' is-on' : ''}`} onClick={() => E.toggleAB('version')} title="The current version (A)">
            <b>A</b> {d.current === 0 ? 'Original' : `v${d.current}`}
          </button>
          <button class={`ab-btn${isTake ? ' is-on' : ''}`} disabled={!t} onClick={() => E.toggleAB('take')} title="The selected take (B)">
            <b>B</b> {t ? t.label : 'Take'}
          </button>
        </div>
        <button class="btn sm" onClick={E.togglePlay} title="Play / pause (Space)">
          <Icon name={playing ? 'stop' : 'play'} size={13} /> {playing ? 'Pause' : 'Play'}
        </button>
        <span class="mono ed-time">
          {fmtTime(playing ? player.playhead.value : E.cursor.value)} / {fmtTime(disp.duration)}
        </span>
        <div class="ed-zoom" role="group" aria-label="Zoom">
          <button class="icon-btn sm" onClick={() => (zoom.value = clamp(zoom.value / 1.6, 1, 64))} aria-label="Zoom out ([)" title="Zoom out ([)">
            −
          </button>
          <button class="icon-btn sm" onClick={() => (zoom.value = clamp(zoom.value * 1.6, 1, 64))} aria-label="Zoom in (])" title="Zoom in (]) · ⌘/Ctrl+wheel">
            +
          </button>
          <button class="btn sm ghost" onClick={() => (zoom.value = 1)} title="Fit (0)">
            fit
          </button>
        </div>
        <span class="grow" />
        {d.current > 0 && d.source.kind === 'take' && (
          <button class="btn sm" disabled={!E.canSendBack.value} onClick={() => void E.sendBack()} title={E.canSendBack.value ? 'Add this version to the Direct line as a new keeper take' : 'The original take is gone from Direct'}>
            <Icon name="undo" size={13} /> Send back to Direct
          </button>
        )}
        <button class="btn sm" onClick={() => exportWav(disp.pcm, disp.sr, `${slug(d.name)}-${isTake ? slug(disp.label) : `v${d.current}`}`)} aria-label={`Export ${shownLabel} as WAV`} title={`Save ${shownLabel} as a WAV file`}>
          <Icon name="download" size={13} /> Export {shownLabel}
        </button>
      </header>
      {isTake && t && (
        <div class="ed-compare-strip" role="status">
          <span>
            Comparing <b>{t.label}</b> with {d.current === 0 ? 'the original' : `v${d.current}`}
          </span>
          <button class="link" onClick={() => E.toggleAB('version')}>
            Back to {d.current === 0 ? 'the original' : `v${d.current}`}
          </button>
        </div>
      )}
      <BusyRow />
      <div class={`ed-wave-card${isTake ? ' is-take' : ''}`}>
        <div class="ed-wave-label">
          {isTake ? (
            <span>
              <b>{disp.label}</b> · green = regenerated
            </span>
          ) : (
            <span>
              <b>{d.current === 0 ? 'Original' : `v${d.current} ${v.label}`}</b> · orange = will be regenerated
            </span>
          )}
          <span class="ed-timing-tools">
            {!isTake && !E.fixMode.value && (
              <>
                {E.selectedWord.value !== null && disp.words[E.selectedWord.value] ? (
                  <span class="ed-readout" role="status">
                    Editing “{disp.words[E.selectedWord.value].word}” · ←/→ nudge · Esc done
                  </span>
                ) : (
                  <span class="muted">{E.adjustedCount.value ? `${E.adjustedCount.value} timing${E.adjustedCount.value === 1 ? '' : 's'} adjusted` : ''}</span>
                )}
                <label class="toggle" title="Start word playback 0.3 s early">
                  <input type="checkbox" checked={E.preRoll.value} onChange={(e) => (E.preRoll.value = (e.currentTarget as HTMLInputElement).checked)} />
                  <span>pre-roll</span>
                </label>
                {E.adjustedCount.value > 0 && (
                  <button class="link" onClick={E.resetTimings} title="Back to the alignment's word timings (undoable)">
                    Reset timings
                  </button>
                )}
              </>
            )}
            <span class="muted">{v.duration.toFixed(1)} s original</span>
          </span>
        </div>
        <p class="hint ed-word-hint">{timingEditable ? 'Drag on the waveform to re-say a stretch · drag a word’s edge to fix its timing' : isTake ? 'Word timings: switch to A to edit' : 'Word timings: finish the transcript first'}</p>
        <EditWave
          pcm={disp.pcm}
          sr={disp.sr}
          words={disp.words}
          regions={E.regions.value.map((r, k) => ({ id: r.id, start: E.seconds(r.start), end: E.seconds(r.end), n: k + 1, manual: r.manual, directed: Object.keys(r.values).length > 0, ghost: isTake }))}
          overlays={isTake && t ? t.edited : []}
          contexts={isTake && t ? t.contexts : E.gen.value?.versionId === v.id && E.gen.value.contexts.length ? E.gen.value.contexts : E.contextPreview.value}
          contextsEstimated={!isTake && !(E.gen.value?.versionId === v.id && E.gen.value.contexts.length)}
          timing={
            timingEditable
              ? {
                  selected: E.selectedWord.value,
                  edge: E.activeEdge.value,
                  adjusted: E.isAdjusted,
                  onEdge: (i, edge, x, phase, detach) => {
                    if (phase === 'begin') {
                      E.wordGesture.begin();
                      E.selectedWord.value = i;
                      E.activeEdge.value = edge;
                      wordDrag = { i, edge, t0: x, decided: false };
                      return;
                    }
                    const g = wordDrag ?? { i, edge, t0: x, decided: true };
                    // Alt at a shared boundary: the word on the side you drag towards keeps its edge.
                    if (!g.decided && detach && Math.abs(x - g.t0) > 0.002) {
                      const ws = v.words;
                      if (g.edge === 'start' && x < g.t0 && ws[g.i - 1] && Math.abs(ws[g.i - 1].end - ws[g.i].start) < 0.002) Object.assign(g, { i: g.i - 1, edge: 'end' });
                      else if (g.edge === 'end' && x > g.t0 && ws[g.i + 1] && Math.abs(ws[g.i + 1].start - ws[g.i].end) < 0.002) Object.assign(g, { i: g.i + 1, edge: 'start' });
                      g.decided = true;
                      E.selectedWord.value = g.i;
                      E.activeEdge.value = g.edge;
                    }
                    E.moveWordEdge(g.i, g.edge, x, detach);
                    if (phase === 'end') {
                      E.wordGesture.end();
                      wordDrag = null;
                    }
                  },
                }
              : undefined
          }
          editable={!isTake && !E.fixMode.value}
          playId={disp.id}
          cursor={E.cursor.value}
          zoom={zoom.value}
          setZoom={(z) => (zoom.value = z)}
          onSeek={(x) => {
            E.cursor.value = x;
            if (nowPlaying.value?.id === disp.id) player.play(disp.id, disp.pcm, disp.sr, x);
          }}
          onCreate={E.addManual}
          onEdges={E.moveEdges}
          onWord={onWord}
          focusRegion={focusRegion.value}
          zoomTo={zoomTo.value}
        />
      </div>
      <div class="ed-columns">
        <div class="ed-left">
          <Transcript area={area} />
          <Regions />
          <GenerateBar />
        </div>
        <aside class="ed-takes" aria-label="Takes">
          <header class="ed-sub-head">
            <h3>Takes</h3>
            {t && <span class="muted small">{E.ab.value === 'take' ? 'comparing (B)' : 'press B to compare'}</span>}
          </header>
          {E.gen.value && E.gen.value.versionId === v.id && <p class="hint">Takes aren’t saved until you keep one.</p>}
          <Takes />
        </aside>
      </div>
    </div>
  );
}

export function EditScreen() {
  const enabled = backend.value!.edit.speech_editing;
  useEffect(() => {
    activeHistory.value = E.docs as History<unknown>;
    primaryAction.value = E.togglePlay;
    screenKeys.value = enabled ? editKeys : null;
    // A selected word is a keyboard mode: a click outside the word lane, or typing in the transcript, leaves it.
    const outside = (e: PointerEvent) => {
      if (E.selectedWord.peek() !== null && !(e.target as HTMLElement).closest?.('.ew-words, .ed-timing-tools')) E.selectedWord.value = null;
    };
    const typing = (e: FocusEvent) => {
      if (E.selectedWord.peek() !== null && (e.target as HTMLElement).matches?.('textarea')) E.selectedWord.value = null;
    };
    document.addEventListener('pointerdown', outside);
    document.addEventListener('focusin', typing);
    return () => {
      document.removeEventListener('pointerdown', outside);
      document.removeEventListener('focusin', typing);
      activeHistory.value = null;
      primaryAction.value = null;
      screenKeys.value = null;
      player.stop();
    };
  }, [enabled]);
  if (!enabled) return <Unavailable />;
  return (
    <div class="edit">
      <Rail id="edit-sources" side="left" title="Sources">
        <aside class="rail edit-rail" aria-label="Sources and versions">
          <Sources />
          <Versions />
        </aside>
      </Rail>
      <RailBar>
        <RailToggle id="edit-sources" title="Sources" icon="film" />
      </RailBar>
      <Workspace />
      <FileDrop label="Drop to edit this clip" hint="An audio or video file; only the words you change are regenerated" busy={E.busy.value ? 'Wait for the current step…' : null} onFile={(f) => void E.openFile(f)} />
    </div>
  );
}
