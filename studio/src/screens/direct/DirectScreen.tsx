import { signal } from '@preact/signals';
import { useEffect, useRef, useState } from 'preact/hooks';
import { exportWav, sequence } from '../../audio/mix';
import * as player from '../../audio/player';
import { nowPlaying, playhead } from '../../audio/player';
import { Icon } from '../../components/Icon';
import { Rail, RailBar, RailToggle } from '../../components/Rail';
import { phone } from '../../lib/media';
import { langLabel } from '../../lib/languages';
import { guessed, isPronError, lookupLanguage, openPronEditor, phonemizeOn, phonemizeRange, phonemizing, PronHelp, PronMarks, PronounceButton } from '../../components/Pron';
import { parsePron } from '../../lib/pron';
import { Slider } from '../../components/Slider';
import { Waveform } from '../../components/Waveform';
import { clamp, uid } from '../../lib/math';
import { activeHistory, controlById, hue, notify, primaryAction, pronTokens, spec } from '../../state/app';
import { design } from '../../state/design';
import type { History } from '../../state/history';
import { cast, savePersona, speakers, type Persona } from '../../state/library';
import {
  addLine,
  deleteTake,
  keepers,
  lineBusy,
  lineError,
  lineVoiceId,
  matchLines,
  mixdown,
  moveLine,
  paint,
  regenerateStale,
  removeLine,
  renderLine,
  replaceScript,
  restoreTake,
  script,
  scripts,
  setBaseline,
  setGap,
  setLineText,
  setLineVoice,
  setVoice,
  splitScript,
  staleLines,
  starTake,
  takeIsStale,
  takes,
  timeline,
  valuesAt,
  voiceDesign,
  type Line,
} from '../../state/script';
import { confirm } from '../../components/ConfirmDialog';
import { MoodControls, type Binding } from '../cast/QuickControls';
import { canEdit, openClip } from '../../state/edit';
import { LanguageSelect } from '../../components/LanguageSelect';
import { guidanceNote, openTagPaletteAt, TagButton, TagMarks, tagsOn, TagStrength } from '../../components/Tags';
import { parseTags, TAG_INFO } from '../../lib/tags';
import type { SpanFollow } from '../../api';

const SCRIPT_ID = 'script';

/** Baseline performance controls (whole script). */
const BASELINE = ['chatty', 'arousal', 'pace'];
const BASELINE_LABEL: Record<string, string> = { chatty: 'Read ↔ chatty', arousal: 'Energy', pace: 'Pace' };

/** Cast voices tell apart by the data hues (theme-aware tokens); the bench voice is neutral. */
const VOICE_COLORS = ['var(--c1)', 'var(--c2)', 'var(--c6)', 'var(--c3)', 'var(--c5)', 'var(--c4)', 'var(--o-hi)', 'var(--o)'];
function voiceColor(id: string | null): string {
  const i = id ? cast.value.findIndex((p) => p.id === id) : -1;
  return i < 0 ? 'var(--text-3)' : VOICE_COLORS[i % VOICE_COLORS.length];
}
/** A voice's display name: the cast persona, or the voice on the Cast bench (null). */
const voiceName = (id: string | null) =>
  id ? (cast.value.find((p) => p.id === id)?.name ?? '(removed voice)') : `${speakers.value.find((s) => s.id === design.value.baseId)?.label ?? 'Voice'} (bench)`;

interface Brush {
  lineId: string;
  start: number;
  end: number;
  x: number;
  y: number;
}
const brush = signal<Brush | null>(null);

function playScript() {
  const mix = mixdown();
  if (!mix) return notify('Render some lines first: each line’s starred take plays in order.');
  player.play(SCRIPT_ID, mix.pcm, mix.sr);
}

function exportScript() {
  const mix = mixdown();
  if (!mix) return notify('Nothing rendered yet.');
  exportWav(mix.pcm, mix.sr, 'script');
  const missing = keepers.value.filter((k) => !k.take).length;
  if (missing) notify(`Exported ${keepers.value.length - missing} lines; ${missing} without a take were skipped.`);
}

/** Where the playing script is: the line under the playhead and its time on the timeline (lines without a take aren't played). */
function playPosition(): { lineId: string; t: number } | null {
  if (nowPlaying.value?.id !== SCRIPT_ID) return null;
  const at = playhead.value;
  const gap = script.value.gap;
  let m = 0;
  for (const c of timeline.value.clips) {
    if (!c.take) continue;
    if (at < m + c.duration + gap) return { lineId: c.line.id, t: c.start + at - m };
    m += c.duration + gap;
  }
  return null;
}

function addBench() {
  const p: Persona = { id: uid(), name: voiceName(null), design: structuredClone(design.value), signature: '', createdAt: Date.now() };
  savePersona(p);
  notify(`Added ${p.name} to the cast`, 'ok');
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
  const pick = (touch?: boolean) => {
    const s = selectionIn(el.current!);
    if (s && s.end > s.start) open(s.start, s.end, s.rect);
    else if (!touch) brush.value = null;
  };
  return (
    // Touch: a long-press selection fires no mouseup; read it once the finger lifts and the selection has settled
    // (only to open: a tap on painted words opens the brushes through its click).
    <div ref={el} class="line-text" onMouseUp={() => pick()} onTouchEnd={() => setTimeout(() => pick(true), 50)}>
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
      if (e instanceof KeyboardEvent) {
        if (e.key !== 'Escape' || e.defaultPrevented) return;
        e.preventDefault();
        brush.value = null;
      } else if (!bar.current?.contains(e.target as Node) && !(e.target as HTMLElement).closest?.('.line-text')) brush.value = null;
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
    <div ref={bar} class={`brushbar${b.y > 420 ? ' is-above' : ''}`} style={phone.value ? undefined : { left: `${left}px`, top: `${top}px`, width: `${w}px` }} role="dialog" aria-label="Delivery brushes">
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
            <button
              class="icon-btn sm"
              onClick={() => {
                const r = deleteTake(line.id, t.id);
                if (r) notify(`Take ${i + 1} deleted`, 'info', { label: 'Undo', run: () => restoreTake(line.id, r.take, r.index, r.wasStarred) });
              }}
              aria-label={`Delete take ${i + 1}`}
              title="Delete this take"
            >
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
  rate: ['est.', 'Spans placed by estimate: the server can’t follow the reading here (no alignment heads for this language), so the painting was timed by characters per frame'],
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
  const [editing, setEditing] = useState(() => editNew.peek() === line.id);
  const [draft, setDraft] = useState(line.text);
  const busy = lineBusy.value[line.id];
  const err = lineError.value[line.id];
  const list = takes.value[line.id] ?? [];
  const keeper = list.find((t) => t.starred) ?? list[list.length - 1];
  const stale = !!keeper && takeIsStale(line, keeper);
  const voice = lineVoiceId(line);
  return (
    <li class={`line${playing ? ' is-playing' : ''}`} style={{ '--voice': voiceColor(voice) }}>
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
            placeholder="What do they say?"
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
              if (editNew.peek() === line.id) editNew.value = null;
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
        <select
          class="input line-voice"
          value={line.voice ?? ''}
          onChange={(e) => setLineVoice(line.id, (e.currentTarget as HTMLSelectElement).value || null)}
          aria-label={`Voice for line ${n}`}
          title={`Speaks as ${voiceName(voice)}`}
        >
          <option value="">Script voice ({voiceName(script.value.voice)})</option>
          {line.voice && !cast.value.some((p) => p.id === line.voice) && <option value={line.voice}>(voice unavailable)</option>}
          {cast.value.map((p) => (
            <option value={p.id}>{p.name}</option>
          ))}
        </select>
        <button
          class={`btn sm${busy ? ' is-busy' : ''}${stale ? ' is-stale' : ''}`}
          onClick={() => void renderLine(line.id)}
          disabled={busy || !line.text.trim()}
          title={stale ? 'Settings or voice changed since this take' : list.length ? 'Render another take (new seed)' : 'Render this line'}
        >
          {busy ? 'Rendering…' : stale ? (
            <>
              <Icon name="refresh" size={13} /> Regenerate
            </>
          ) : list.length ? 'New take' : 'Render'}
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

/** The textarea's text differs from the lines (Cancel asks before discarding it). */
const scriptTextDirty = signal(false);
const linesAsText = () => script.value.lines.map((l) => l.text).join('\n');

async function closeScriptText() {
  if (scriptTextDirty.value && !(await confirm('Discard your text changes?', { ok: 'Discard', danger: true }))) return;
  scriptTextDirty.value = false;
  editAll.value = false;
}

function ScriptText() {
  const pasting = script.value.lines.length === 0;
  const [text, setText] = useState(linesAsText);
  useEffect(() => () => void (scriptTextDirty.value = false), []);
  const apply = async () => {
    const rows = splitScript(text, { sentences: pasting });
    const { dropped } = matchLines(script.value.lines, rows);
    const takesLost = dropped.reduce((n, l) => n + (takes.value[l.id]?.length ?? 0), 0);
    if (dropped.length && takesLost) {
      const ok = await confirm(`${dropped.length} line${dropped.length === 1 ? '' : 's'} will be removed with ${takesLost} take${takesLost === 1 ? '' : 's'}. Apply anyway?`, { ok: 'Apply', danger: true });
      if (!ok) return;
    }
    replaceScript(rows);
    scriptTextDirty.value = false;
    editAll.value = false;
  };
  return (
    <div class="script-text">
      <textarea
        class="input"
        rows={10}
        value={text}
        data-tags
        onInput={(e) => {
          const v = (e.currentTarget as HTMLTextAreaElement).value;
          setText(v);
          scriptTextDirty.value = v !== linesAsText();
        }}
        aria-label="Script text"
        placeholder={pasting ? 'Paste or type the script' : undefined}
      />
      <p class="hint">{pasting ? 'The script is split at line breaks and sentence ends.' : 'One line per row. Rows you leave unchanged keep their paint, voice and takes.'}</p>
      <button class="btn primary" onClick={() => void apply()}>
        Apply script
      </button>
    </div>
  );
}

const editAll = signal(false);
/** A line added from a "says…" button opens with its text field. */
const editNew = signal<string | null>(null);

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
  const people = cast.value;
  return (
    <aside class="rail direct-rail" aria-label="Voices and performance">
      <div class="rail-section">
        <h4>Default voice</h4>
        <select class="input" value={s.voice ?? ''} onChange={(e) => setVoice((e.currentTarget as HTMLSelectElement).value || null)} aria-label="Default voice for this script">
          <option value="">Bench: {speakers.value.find((x) => x.id === design.value.baseId)?.label ?? '—'} (as designed in Cast)</option>
          {people.map((p) => (
            <option value={p.id}>{p.name}</option>
          ))}
        </select>
        <p class="hint">
          {base?.label ?? 'No voice'} · {d.mode === 'reference' ? 'exact clone' : 'designable'}. Lines speak in this voice unless they pick their own; identity is set per voice in Cast, only delivery and mood paint onto words.
        </p>
        {people.length ? (
          <div class="voice-adds">
            {people.map((p) => (
              <button class="btn ghost sm" style={{ '--voice': voiceColor(p.id) }} onClick={() => (editNew.value = addLine(s.lines[s.lines.length - 1]?.id ?? null, '', p.id))} title={`Add a line spoken by ${p.name}`}>
                <span class="voice-dot" /> {p.name} says…
              </button>
            ))}
          </div>
        ) : (
          <button class="btn sm voice-bench" onClick={addBench} disabled={!design.value.baseId} title="Save the Cast bench voice as a cast member, so lines can pick it">
            <Icon name="plus" size={13} /> Add the bench voice to the cast
          </button>
        )}
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
    </aside>
  );
}

/** The right sidebar: the script as a whole (renders, playback, export, the gap between lines). */
function ScriptSide() {
  const s = script.value;
  const ready = keepers.value.filter((k) => k.take).length;
  const stale = staleLines.value.length;
  const rendering = Object.values(lineBusy.value).some(Boolean);
  const playing = nowPlaying.value?.id === SCRIPT_ID;
  return (
    <aside class="rail direct-side" aria-label="Script">
      <div class="rail-section">
        <h4>Script</h4>
        <p class="hint">
          {ready}/{s.lines.length} lines have a take. Play and export use each line’s starred take.
        </p>
        <div class="col gap-s">
          {stale > 0 && (
            <button class="btn primary" disabled={rendering} onClick={() => void regenerateStale()} title="Settings or voice changed since these lines’ starred takes: render a new take of each">
              <Icon name="refresh" size={14} /> Regenerate {stale} changed line{stale === 1 ? '' : 's'}
            </button>
          )}
          <button
            class="btn"
            disabled={rendering}
            onClick={() => {
              for (const l of s.lines) if (!(takes.value[l.id] ?? []).length) void renderLine(l.id);
            }}
          >
            <Icon name="wand" size={14} /> Render missing lines
          </button>
          <button class={`btn${stale ? '' : ' primary'}`} onClick={() => (playing ? player.stop() : playScript())}>
            <Icon name={playing ? 'stop' : 'play'} size={14} /> {playing ? 'Stop' : 'Play script'}
          </button>
          <button class="btn" onClick={exportScript}>
            <Icon name="download" size={14} /> Export WAV
          </button>
        </div>
      </div>
      <div class="rail-section">
        <h4>Timing</h4>
        <label class="gap">
          Gap between lines
          <input type="range" min={0} max={1.5} step={0.05} value={s.gap} onChange={(e) => setGap(Number((e.currentTarget as HTMLInputElement).value))} aria-label="Gap between lines (s)" />
          <b>{s.gap.toFixed(2)} s</b>
        </label>
      </div>
    </aside>
  );
}

/** One lane per voice; each line's keeper take placed where the script plays it. */
function Timeline({ at }: { at: number | null }) {
  const { clips, total } = timeline.value;
  if (!clips.length) return null;
  const lanes = [...new Set(clips.map((c) => lineVoiceId(c.line)))];
  const pct = (t: number) => `${(t / (total || 1)) * 100}%`;
  return (
    <section class="timeline" aria-label="Timeline">
      <div class="timeline-head">
        <span class="eyebrow">Timeline</span>
        <span class="muted small">
          {total.toFixed(1)} s{clips.some((c) => !c.take) && ' (estimated until rendered)'}
        </span>
      </div>
      <div class="lanes">
        {lanes.map((vid) => {
          const color = voiceColor(vid);
          return (
            <div key={vid ?? ''} class="lane" style={{ '--voice': color }}>
              <span class="lane-name" title={voiceName(vid)}>
                {voiceName(vid)}
              </span>
              <div class="lane-track">
                {clips
                  .filter((c) => lineVoiceId(c.line) === vid)
                  .map((c) => (
                    <button
                      key={c.line.id}
                      class={`clip${c.take ? '' : ' is-pending'}`}
                      style={{ left: pct(c.start), width: pct(c.duration) }}
                      title={c.line.text}
                      onClick={() => c.take && player.toggle(`take:${c.take.id}`, c.take.audio.pcm, c.take.audio.sr)}
                    >
                      {c.take ? <Waveform pcm={c.take.audio.pcm} color={color} height={28} /> : <span>{c.line.text}</span>}
                    </button>
                  ))}
              </div>
            </div>
          );
        })}
        {at !== null && <div class="timeline-head-line" style={{ left: `calc(var(--lane-name) + (100% - var(--lane-name)) * ${at / (total || 1)})` }} />}
      </div>
    </section>
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
  const now = playPosition();
  return (
    <div class="direct">
      <Rail id="direct-voices" side="left" title="Voices">
        <Sidebar />
      </Rail>
      <section class="script" aria-label="Lines">
        <RailBar>
          <RailToggle id="direct-voices" title="Voices" icon="mic" />
          <RailToggle id="direct-script" title="Script" icon="play" />
        </RailBar>
        <header class="screen-head">
          <div>
            <span class="eyebrow">Direct</span>
            <h2>Script</h2>
          </div>
          <button class="btn ghost" onClick={() => (editAll.value ? void closeScriptText() : (editAll.value = true))}>
            <Icon name={editAll.value ? 'x' : 'edit'} size={14} /> {editAll.value ? (scriptTextDirty.value ? 'Cancel' : 'Back to lines') : 'Edit as text'}
          </button>
        </header>
        {editAll.value ? (
          <ScriptText />
        ) : lines.length ? (
          <ol class="lines">
            {lines.map((l, i) => (
              <LineRow key={l.id} line={l} n={i + 1} playing={now?.lineId === l.id} />
            ))}
          </ol>
        ) : (
          <div class="empty">
            <Icon name="edit" size={26} />
            <h3>No lines yet</h3>
            <p>Paste a script, or add lines one by one; each line is rendered and directed on its own, in its own voice.</p>
            <button class="btn" onClick={() => (editAll.value = true)}>
              Paste a script
            </button>
          </div>
        )}
        {!editAll.value && (
          <>
            <button class="btn ghost add-line" onClick={() => addLine(null, 'A new line.')}>
              <Icon name="plus" size={14} /> Add line
            </button>
            <Timeline at={now?.t ?? null} />
          </>
        )}
      </section>
      <Rail id="direct-script" side="right" title="Script">
        <ScriptSide />
      </Rail>
      <BrushBar />
    </div>
  );
}
