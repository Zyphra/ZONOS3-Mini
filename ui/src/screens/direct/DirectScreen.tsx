import { signal } from '@preact/signals';
import { useEffect, useRef, useState } from 'preact/hooks';
import { exportWav, sequence } from '../../audio/mix';
import * as player from '../../audio/player';
import { nowPlaying, playhead } from '../../audio/player';
import { Icon } from '../../components/Icon';
import { guessed, isPronError, langLabel, lookupLanguage, openPronEditor, phonemizeOn, phonemizeRange, phonemizing, PronHelp, PronMarks, PronounceButton } from '../../components/Pron';
import { parsePron } from '../../lib/pron';
import { Slider } from '../../components/Slider';
import { Waveform } from '../../components/Waveform';
import { clamp } from '../../lib/math';
import { activeHistory, backend, controlById, hue, notify, primaryAction, pronTokens, spec } from '../../state/app';
import { dubTake } from '../../state/dub';
import { design } from '../../state/design';
import type { History } from '../../state/history';
import { cast, speakers } from '../../state/library';
import {
  addLine,
  deleteTake,
  keepers,
  lineBusy,
  lineError,
  moveLine,
  paint,
  removeLine,
  renderLine,
  replaceScript,
  script,
  scripts,
  setBaseline,
  setLineText,
  setVoice,
  starTake,
  takeIsStale,
  takes,
  valuesAt,
  voiceDesign,
  type Line,
} from '../../state/script';
import { MoodControls, type Binding } from '../cast/QuickControls';
import { canEdit, openClip } from '../../state/edit';
import { LanguageSelect } from '../../components/LanguageSelect';
import { guidanceNote, openTagPaletteAt, TagButton, TagMarks, tagsOn, TagStrength } from '../../components/Tags';
import { parseTags, TAG_INFO } from '../../lib/tags';
import type { SpanFollow } from '../../api';

const GAP = 0.35;
const SCRIPT_ID = 'script';

/** Baseline performance controls (whole script). */
const BASELINE = ['chatty', 'arousal', 'pace'];
const BASELINE_LABEL: Record<string, string> = { chatty: 'Read ↔ chatty', arousal: 'Energy', pace: 'Pace' };

interface Brush {
  lineId: string;
  start: number;
  end: number;
  x: number;
  y: number;
}
const brush = signal<Brush | null>(null);

function playScript() {
  const ready = keepers.value.filter((k) => k.take);
  if (!ready.length) return notify('Render some lines first: each line’s starred take plays in order.');
  const { pcm, sr } = sequence(
    ready.map((k) => k.take!.audio),
    GAP,
  );
  player.play(SCRIPT_ID, pcm, sr);
}

function exportScript() {
  const ready = keepers.value.filter((k) => k.take);
  if (!ready.length) return notify('Nothing rendered yet.');
  const missing = keepers.value.length - ready.length;
  const { pcm, sr } = sequence(
    ready.map((k) => k.take!.audio),
    GAP,
  );
  exportWav(pcm, sr, 'script');
  if (missing) notify(`Exported ${ready.length} lines; ${missing} without a take were skipped.`);
}

/** Which line is under the playhead while the script plays. */
function playingLine(): string | null {
  if (nowPlaying.value?.id !== SCRIPT_ID) return null;
  let t = 0;
  const at = playhead.value;
  for (const k of keepers.value) {
    if (!k.take) continue;
    const dur = k.take.audio.pcm.length / k.take.audio.sr;
    if (at < t + dur + GAP) return k.line.id;
    t += dur + GAP;
  }
  return null;
}

function underline(values: Record<string, number>): string | undefined {
  const fams = new Set<string>();
  for (const [id, v] of Object.entries(values)) {
    const c = controlById.value.get(id);
    if (c && v) fams.add(c.group);
  }
  if (!fams.size) return undefined;
  return [...fams].map((g, i) => `linear-gradient(${hue(g)}, ${hue(g)}) 0 calc(100% - ${i * 4}px) / 100% 2px no-repeat`).join(', ');
}

function selectionIn(el: HTMLElement): { start: number; end: number; rect: DOMRect } | null {
  const sel = window.getSelection();
  if (!sel || !sel.rangeCount) return null;
  const range = sel.getRangeAt(0);
  if (!el.contains(range.startContainer) || !el.contains(range.endContainer)) return null;
  const offset = (node: Node, off: number) => {
    const seg = (node.nodeType === Node.TEXT_NODE ? node.parentElement : (node as HTMLElement))?.closest<HTMLElement>('[data-start]');
    if (!seg) return node === el ? (off ? Number.MAX_SAFE_INTEGER : 0) : 0;
    // A tag chip is atomic: its label isn't the markup it stands for.
    if (seg.dataset.end) return off ? Number(seg.dataset.end) : Number(seg.dataset.start);
    return Number(seg.dataset.start) + (node.nodeType === Node.TEXT_NODE ? off : 0);
  };
  const a = offset(range.startContainer, range.startOffset);
  const b = offset(range.endContainer, range.endOffset);
  return { start: Math.min(a, b), end: Math.max(a, b), rect: range.getBoundingClientRect() };
}

function LineText({ line }: { line: Line }) {
  const el = useRef<HTMLDivElement>(null);
  const cuts = new Set([0, line.text.length]);
  for (const sp of line.spans) {
    cuts.add(sp.start);
    cuts.add(sp.end);
  }
  // Pronunciation groups render inline with their IPA (phoneme-token checkpoints only).
  const groups = pronTokens.value ? parsePron(line.text, pronTokens.value).groups.filter((g) => !g.empty) : [];
  for (const g of groups) {
    cuts.add(g.start);
    cuts.add(g.end);
  }
  // Nonverbal tags render as chips; span tags mark their words.
  const tagged = tagsOn.value ? parseTags(line.text) : null;
  for (const t of tagged?.tokens ?? []) {
    cuts.add(t.start);
    cuts.add(t.end);
  }
  const replaceGroup = (s: number, e: number, t: string) => setLineText(line.id, line.text.slice(0, s) + t + line.text.slice(e));
  const marks = [...cuts].sort((a, b) => a - b);
  const sel = brush.value?.lineId === line.id ? brush.value : null;
  const open = (start: number, end: number, rect: DOMRect) => {
    brush.value = { lineId: line.id, start, end: Math.min(end, line.text.length), x: rect.left + rect.width / 2, y: rect.top };
  };
  return (
    <div
      ref={el}
      class="line-text"
      onMouseUp={() => {
        const s = selectionIn(el.current!);
        if (s && s.end > s.start) open(s.start, s.end, s.rect);
        else brush.value = null;
      }}
    >
      {marks.slice(0, -1).map((from, i) => {
        const to = marks[i + 1];
        const tag = tagged?.tokens.find((t) => t.start === from);
        if (tag) {
          return tag.kind === 'close' ? (
            <span data-start={from} data-end={to} class="seg tag-close" aria-hidden="true" />
          ) : (
            <span data-start={from} data-end={to} class={`seg tag-inline${tag.kind === 'open' ? ' is-span' : ''}${TAG_INFO[tag.name].weak ? ' is-weak' : ''}`} title={`<${tag.name}${tag.kind === 'point' ? '/' : ''}>: ${TAG_INFO[tag.name].note}`}>
              {tag.kind === 'open' ? `‹${tag.name}›` : tag.name}
            </span>
          );
        }
        const inTag = tagged?.spans.find((s) => from >= s.open.end && from < s.close.start);
        const span = line.spans.find((sp) => from >= sp.start && from < sp.end);
        const picked = sel && from >= sel.start && to <= sel.end;
        const g = groups.find((x) => from >= x.start && from < x.end);
        const check = g ? guessed.value[g.body] : undefined;
        const paintTitle = span ? Object.entries(span.values).map(([k, v]) => `${controlById.value.get(k)?.label ?? k} ${v > 0 ? '+' : ''}${v}`).join(' · ') : '';
        return (
          <span
            data-start={from}
            data-ipa={g && to === g.end ? `${g.ipa}${check ? ' · check' : ''}` : undefined}
            class={`seg${inTag ? ` tag-words is-${inTag.name}` : ''}${span ? ' is-painted' : ''}${picked ? ' is-picked' : ''}${g ? ' is-pron' : ''}${g?.unknown.length ? ' is-pron-bad' : ''}${check ? ' is-pron-check' : ''}`}
            style={span ? { background: underline(span.values) } : undefined}
            onClick={
              g
                ? (e) => {
                    if (window.getSelection()?.isCollapsed) {
                      brush.value = null;
                      openPronEditor(e.currentTarget as HTMLElement, g, replaceGroup);
                    }
                  }
                : span
                  ? (e) => {
                      if (window.getSelection()?.isCollapsed) open(span.start, span.end, (e.currentTarget as HTMLElement).getBoundingClientRect());
                    }
                  : undefined
            }
            title={g ? `(${g.body}) → ${g.ipa}${check ? ` · guessed for “${check}”, check it` : ''}${g.unknown.length ? ` · no token for ${g.unknown.join(' ')}` : ''} · click to edit${paintTitle ? ` · ${paintTitle}` : ''}` : paintTitle || undefined}
          >
            {line.text.slice(from, to)}
          </span>
        );
      })}
    </div>
  );
}

function BrushBar() {
  const b = brush.value;
  const bar = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!b) return;
    const close = (e: Event) => {
      if (e instanceof KeyboardEvent ? e.key === 'Escape' : !bar.current?.contains(e.target as Node) && !(e.target as HTMLElement).closest?.('.line-text')) brush.value = null;
    };
    window.addEventListener('pointerdown', close);
    window.addEventListener('keydown', close);
    return () => {
      window.removeEventListener('pointerdown', close);
      window.removeEventListener('keydown', close);
    };
  }, [b?.lineId, b?.start, b?.end]);
  if (!b) return null;
  const line = script.value.lines.find((l) => l.id === b.lineId);
  if (!line) return null;
  const values = valuesAt(line, b.start);
  const bind: Binding = {
    values,
    begin: () => scripts.begin(),
    set: (id, v) => {
      const c = controlById.value.get(id);
      if (c) paint(line.id, b.start, b.end, id, Math.round(clamp(v, c.range[0], c.range[1]) * 100) / 100);
    },
    end: () => scripts.end(),
  };
  const brushes = spec.value.ui.controls.filter((c) => c.spannable && c.group !== 'mood');
  const hasMood = spec.value.ui.controls.some((c) => c.spannable && c.group === 'mood');
  const w = 560;
  const left = clamp(b.x - w / 2, 12, window.innerWidth - w - 12);
  const top = b.y > 420 ? b.y - 12 : b.y + 40;
  return (
    <div ref={bar} class={`brushbar${b.y > 420 ? ' is-above' : ''}`} style={{ left: `${left}px`, top: `${top}px`, width: `${w}px` }} role="dialog" aria-label="Delivery brushes">
      <header>
        <span class="eyebrow">Paint “{line.text.slice(b.start, b.end)}”</span>
        {phonemizeOn.value && (
          <button
            class="btn sm ghost"
            disabled={phonemizing.value}
            onClick={async () => {
              const next = await phonemizeRange(line.text, b.start, b.end);
              if (next !== null) {
                setLineText(line.id, next);
                brush.value = null;
              }
            }}
            title={`Look up these words in the ${langLabel(lookupLanguage.value)} lexicon and write them as X-SAMPA pronunciations (undo: ⌘Z)`}
          >
            /ə/ Pronounce
          </button>
        )}
        {tagsOn.value && (
          <button
            class="btn sm ghost"
            onClick={(e) => {
              const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
              brush.value = null;
              openTagPaletteAt(r, line.text, b.start, b.end, (next) => setLineText(line.id, next));
            }}
            title="A nonverbal tag: at the start of the selection, or around its words"
          >
            ‹› Tag
          </button>
        )}
        <button class="btn sm ghost" onClick={() => (scripts.begin(), paint(line.id, b.start, b.end, null, 0), scripts.end())}>
          Clear
        </button>
        <button class="icon-btn sm" onClick={() => (brush.value = null)} aria-label="Close brushes">
          <Icon name="x" size={14} />
        </button>
      </header>
      <div class="brushbar-body">
        <div class="brush-sliders">
          {brushes.map((c) => (
            <Slider key={c.id} control={c} value={values[c.id] ?? 0} color={hue(c.group)} onStart={bind.begin} onChange={(v) => bind.set(c.id, v)} onEnd={bind.end} />
          ))}
        </div>
        {hasMood && <MoodControls b={bind} compact />}
      </div>
    </div>
  );
}

function TakeStrip({ line }: { line: Line }) {
  const list = takes.value[line.id] ?? [];
  if (!list.length) return null;
  return (
    <ul class="takes" aria-label="Takes">
      {list.map((t, i) => {
        const id = `take:${t.id}`;
        const playing = nowPlaying.value?.id === id;
        const stale = takeIsStale(line, t);
        return (
          <li class={`take${t.starred ? ' is-starred' : ''}${stale ? ' is-stale' : ''}`}>
            <button class="take-play" onClick={() => player.toggle(id, t.audio.pcm, t.audio.sr)} aria-label={`${playing ? 'Stop' : 'Play'} take ${i + 1}`} title={stale ? 'Rendered with earlier settings' : `seed ${t.seed}`}>
              <span class="take-n">{i + 1}</span>
              <Waveform pcm={t.audio.pcm} color={t.starred ? 'var(--o)' : 'var(--text-2)'} playId={id} height={22} />
            </button>
            {t.audio.spanFollow && <SpanFollowMark follow={t.audio.spanFollow} />}
            {guidanceNote(t.request) && (
              <span class="take-follow is-tags" title={`${guidanceNote(t.request)}: the server takes one guidance per render`} aria-label={guidanceNote(t.request)!}>
                tags
              </span>
            )}
            <button class={`icon-btn sm star${t.starred ? ' is-on' : ''}`} onClick={() => starTake(line.id, t.id)} aria-label={`Star take ${i + 1}`} aria-pressed={t.starred} title="Keeper: the script plays this take">
              <Icon name="star" size={13} filled={t.starred} />
            </button>
            {canEdit.value && (
              <button
                class="icon-btn sm"
                onClick={() => void openClip({ kind: 'take', label: `Line ${script.value.lines.indexOf(line) + 1} · take ${i + 1}`, lineId: line.id, takeId: t.id }, t.audio.pcm, t.audio.sr, t.text)}
                aria-label={`Edit the words of take ${i + 1}`}
                title="Edit the words of this take (Edit screen)"
              >
                <Icon name="edit" size={12} />
              </button>
            )}
            {backend.value?.dub && (
              <button
                class="icon-btn sm"
                onClick={() => void dubTake(t, `Line ${script.value.lines.indexOf(line) + 1} · take ${i + 1}`)}
                aria-label={`Dub take ${i + 1}`}
                title="Dub this take into another language (Dub screen)"
              >
                <Icon name="globe" size={12} />
              </button>
            )}
            <button class="icon-btn sm" onClick={() => deleteTake(line.id, t.id)} aria-label={`Delete take ${i + 1}`}>
              <Icon name="x" size={12} />
            </button>
          </li>
        );
      })}
      {list.length > 1 && <CompareButton lineId={line.id} />}
    </ul>
  );
}

const FOLLOW_TEXT: Record<SpanFollow, [string, string]> = {
  reading: ['read', 'Spans follow the reading: the painted words were steered where the model was reading them'],
  rate: ['est.', 'Spans placed by estimate: the server can’t follow the reading here (no alignment heads for this language, or an older server), so the painting was timed by characters per frame'],
};

/** How the take's painted spans were placed (X-Control-Span-Follow). */
function SpanFollowMark({ follow }: { follow: SpanFollow }) {
  const [text, title] = FOLLOW_TEXT[follow];
  return (
    <span class={`take-follow is-${follow}`} title={title} aria-label={title}>
      {text}
    </span>
  );
}

/** Keeper, then the newest other take, back to back. */
function CompareButton({ lineId }: { lineId: string }) {
  const list = takes.value[lineId] ?? [];
  const keeper = list.find((t) => t.starred) ?? list[0];
  const other = [...list].reverse().find((t) => t !== keeper);
  const id = `cmp:${lineId}`;
  const on = nowPlaying.value?.id === id;
  if (!other) return null;
  return (
    <li>
      <button
        class={`btn sm ghost${on ? ' is-on' : ''}`}
        onClick={() => {
          if (on) return player.stop();
          const { pcm, sr } = sequence([keeper.audio, other.audio], 0.6);
          player.play(id, pcm, sr);
        }}
        title={`Play take ${list.indexOf(keeper) + 1} (keeper) then take ${list.indexOf(other) + 1}`}
      >
        <Icon name={on ? 'stop' : 'play'} size={12} /> Compare
      </button>
    </li>
  );
}

function LineRow({ line, n, playing }: { line: Line; n: number; playing: boolean }) {
  const editEl = useRef<HTMLInputElement>(null);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(line.text);
  const busy = lineBusy.value[line.id];
  const err = lineError.value[line.id];
  const count = (takes.value[line.id] ?? []).length;
  return (
    <li class={`line${playing ? ' is-playing' : ''}`}>
      <span class="line-n">{n}</span>
      <div class="line-main">
        {editing ? (
          <>
          <div class="line-edit-row">
          <input
            ref={editEl}
            class="input line-edit"
            value={draft}
            data-tags
            autoFocus
            aria-label={`Line ${n} text`}
            onInput={(e) => setDraft((e.currentTarget as HTMLInputElement).value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') (e.currentTarget as HTMLInputElement).blur();
              if (e.key === 'Escape') {
                (e.currentTarget as HTMLInputElement).value = line.text;
                setDraft(line.text);
                setEditing(false);
              }
            }}
            onBlur={(e) => {
              // The tag palette borrows focus and gives it back: still editing.
              if ((e.relatedTarget as HTMLElement | null)?.closest?.('.tag-palette')) return;
              // The element's value: an edit right before blurring may not be rendered into `draft` yet.
              const v = (e.currentTarget as HTMLInputElement).value;
              if (v.trim() && v !== line.text) setLineText(line.id, v);
              setEditing(false);
            }}
          />
          <PronounceButton input={() => editEl.current} />
          <TagButton input={() => editEl.current} />
          </div>
          <PronMarks text={draft} />
          <TagMarks text={draft} />
          </>
        ) : (
          <>
            <LineText line={line} />
            <PronMarks text={line.text} issuesOnly />
            <TagMarks text={line.text} replace={(next) => setLineText(line.id, next)} />
          </>
        )}
        <TakeStrip line={line} />
        {err && (
          <div class="error-line" role="alert">
            {isPronError(err) ? 'Not rendered: fix the pronunciation marked above.' : err}
          </div>
        )}
      </div>
      <div class="line-actions">
        <button class={`btn sm${busy ? ' is-busy' : ''}`} onClick={() => void renderLine(line.id)} disabled={busy || !line.text.trim()} title={count ? 'Render another take (new seed)' : 'Render this line'}>
          {busy ? 'Rendering…' : count ? 'New take' : 'Render'}
        </button>
        <div class="line-tools">
          <button class="icon-btn sm" onClick={() => (setDraft(line.text), setEditing(true))} aria-label={`Edit line ${n}`} title="Edit text (paint stays on its words)">
            <Icon name="edit" size={13} />
          </button>
          <button class="icon-btn sm" onClick={() => moveLine(line.id, -1)} aria-label={`Move line ${n} up`}>
            <Icon name="up" size={13} />
          </button>
          <button class="icon-btn sm" onClick={() => moveLine(line.id, 1)} aria-label={`Move line ${n} down`}>
            <Icon name="down" size={13} />
          </button>
          <button class="icon-btn sm" onClick={() => removeLine(line.id)} aria-label={`Delete line ${n}`}>
            <Icon name="trash" size={13} />
          </button>
        </div>
      </div>
    </li>
  );
}

function ScriptText() {
  const [text, setText] = useState(script.value.lines.map((l) => l.text).join('\n'));
  return (
    <div class="script-text">
      <textarea class="input" rows={10} value={text} data-tags onInput={(e) => setText((e.currentTarget as HTMLTextAreaElement).value)} aria-label="Script text" />
      <p class="hint">One line per sentence: the script is split at line breaks and sentence ends. Lines whose text is unchanged keep their paint and takes.</p>
      <button class="btn primary" onClick={() => (replaceScript(text), (editAll.value = false))}>
        Apply script
      </button>
    </div>
  );
}

const editAll = signal(false);

function Sidebar() {
  const s = script.value;
  const d = voiceDesign.value;
  const base = speakers.value.find((x) => x.id === d.baseId);
  const bindBase: Binding = {
    values: s.baseline,
    begin: () => scripts.begin(),
    set: (id, v) => {
      const c = controlById.value.get(id);
      if (c) setBaseline(id, clamp(v, c.range[0], c.range[1]));
    },
    end: () => scripts.end(),
  };
  const ready = keepers.value.filter((k) => k.take).length;
  const total = s.lines.length;
  const rendering = Object.values(lineBusy.value).some(Boolean);
  return (
    <aside class="rail direct-rail" aria-label="Script settings">
      <div class="rail-section">
        <h4>Voice</h4>
        <select class="input" value={s.voice ?? ''} onChange={(e) => setVoice((e.currentTarget as HTMLSelectElement).value || null)} aria-label="Voice for this script">
          <option value="">Bench: {speakers.value.find((x) => x.id === design.value.baseId)?.label ?? '—'} (as designed in Cast)</option>
          {cast.value.map((p) => (
            <option value={p.id}>{p.name}</option>
          ))}
        </select>
        <p class="hint">
          {base?.label ?? 'No voice'} · {d.mode === 'reference' ? 'exact clone' : 'designable'}. Identity is set per character in Cast; only delivery and mood paint onto words.
        </p>
        <LanguageSelect />
        <TagStrength />
      </div>
      <div class="rail-section">
        <h4>Baseline performance</h4>
        {BASELINE.map((id) => controlById.value.get(id))
          .filter((c) => !!c)
          .map((c) => (
            <Slider key={c!.id} control={c!} value={s.baseline[c!.id] ?? 0} color={hue(c!.group)} label={BASELINE_LABEL[c!.id]} onStart={bindBase.begin} onChange={(v) => bindBase.set(c!.id, v)} onEnd={bindBase.end} />
          ))}
      </div>
      <div class="rail-section">
        <h4>Pronunciation</h4>
        <PronHelp />
      </div>
      <div class="rail-section">
        <h4>Brushes</h4>
        <p class="hint">Select words in a line to paint delivery onto them. Underlines show what is painted:</p>
        <div class="legend">
          {spec.value.ui.groups
            .filter((g) => spec.value.ui.controls.some((c) => c.group === g.id && c.spannable))
            .map((g) => (
              <span class="legend-item">
                <i style={{ background: hue(g.id) }} />
                {g.label}
              </span>
            ))}
        </div>
      </div>
      <div class="rail-section">
        <h4>Script</h4>
        <p class="hint">
          {ready}/{total} lines have a take. Play and export use each line’s starred take.
        </p>
        <div class="col gap-s">
          <button
            class="btn"
            disabled={rendering}
            onClick={() => {
              for (const l of s.lines) if (!(takes.value[l.id] ?? []).length) void renderLine(l.id);
            }}
          >
            <Icon name="wand" size={14} /> Render missing lines
          </button>
          <button class="btn primary" onClick={() => (nowPlaying.value?.id === SCRIPT_ID ? player.stop() : playScript())}>
            <Icon name={nowPlaying.value?.id === SCRIPT_ID ? 'stop' : 'play'} size={14} /> {nowPlaying.value?.id === SCRIPT_ID ? 'Stop' : 'Play script'}
          </button>
          <button class="btn" onClick={exportScript}>
            <Icon name="download" size={14} /> Export WAV
          </button>
        </div>
      </div>
    </aside>
  );
}

export function DirectScreen() {
  useEffect(() => {
    activeHistory.value = scripts as History<unknown>;
    primaryAction.value = playScript;
    return () => {
      activeHistory.value = null;
      primaryAction.value = null;
      brush.value = null;
    };
  }, []);
  const lines = script.value.lines;
  const now = playingLine();
  return (
    <div class="direct">
      <Sidebar />
      <section class="script" aria-label="Script">
        <header class="screen-head">
          <div>
            <span class="eyebrow">Direct</span>
            <h2>Script</h2>
          </div>
          <button class="btn ghost" onClick={() => (editAll.value = !editAll.value)}>
            <Icon name="edit" size={14} /> {editAll.value ? 'Back to lines' : 'Edit as text'}
          </button>
        </header>
        {editAll.value ? (
          <ScriptText />
        ) : lines.length ? (
          <ol class="lines">
            {lines.map((l, i) => (
              <LineRow key={l.id} line={l} n={i + 1} playing={now === l.id} />
            ))}
          </ol>
        ) : (
          <div class="empty">
            <Icon name="edit" size={26} />
            <h3>No lines yet</h3>
            <p>Paste a script, or add lines one by one; each line is rendered and directed on its own.</p>
            <button class="btn" onClick={() => (editAll.value = true)}>
              Paste a script
            </button>
          </div>
        )}
        {!editAll.value && (
          <button class="btn ghost add-line" onClick={() => addLine(null, 'A new line.')}>
            <Icon name="plus" size={14} /> Add line
          </button>
        )}
      </section>
      <BrushBar />
    </div>
  );
}
