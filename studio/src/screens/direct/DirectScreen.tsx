import { signal } from '@preact/signals';
import { useEffect, useRef, useState } from 'preact/hooks';
import { exportWav, sequence } from '../../audio/mix';
import * as player from '../../audio/player';
import { nowPlaying, playhead } from '../../audio/player';
import { Icon } from '../../components/Icon';
import { dialogKeys, focusFirst } from '../../components/focusTrap';
import { Rail, RailBar, RailToggle } from '../../components/Rail';
import { slug } from '../../lib/files';
import { phone } from '../../lib/media';
import { langLabel } from '../../lib/languages';
import { guessed, isPronError, lookupLanguage, openPronEditor, phonemizeOn, phonemizeRange, phonemizing, PronHelp, PronMarks, PronounceButton } from '../../components/Pron';
import { parsePron } from '../../lib/pron';
import { Slider } from '../../components/Slider';
import { Waveform } from '../../components/Waveform';
import { clamp, uid } from '../../lib/math';
import { activeHistory, backend, controlById, hue, modalOpen, notify, primaryAction, projectSettings, pronTokens, settingsOpen, spec } from '../../state/app';
import { design } from '../../state/design';
import type { History } from '../../state/history';
import { cast, savePersona, speakers, type Persona } from '../../state/library';
import { current } from '../../state/projects';
import {
  addLine,
  cancelLine,
  cancelRenders,
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
  renderLines,
  renderQueue,
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
  takeStaleReason,
  takes,
  timeline,
  valuesAt,
  voiceDesign,
  type Line,
  type Take,
} from '../../state/script';
import { confirm } from '../../components/ConfirmDialog';
import { MoodControls, type Binding } from '../cast/QuickControls';
import { canEdit, openClip } from '../../state/edit';
import { guidanceNote, openTagPaletteAt, TagButton, TagMarks, tagsOn } from '../../components/Tags';
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
  const missing = keepers.value.filter((k) => !k.take).length;
  if (missing) notify(`${missing} line${missing === 1 ? '' : 's'} without a take ${missing === 1 ? 'is' : 'are'} skipped`);
  player.play(SCRIPT_ID, mix.pcm, mix.sr);
}

function exportScript() {
  const mix = mixdown();
  if (!mix) return notify('Nothing rendered yet.');
  exportWav(mix.pcm, mix.sr, slug(current.value?.name ?? 'script'));
  const missing = keepers.value.filter((k) => !k.take).length;
  if (missing) notify(`Exported ${keepers.value.length - missing} lines; ${missing} without a take were skipped.`);
}

/** Where a rendered clip starts in the script mix (lines without a take take no time there). */
function mixOffset(lineId: string): number | null {
  const gap = script.value.gap;
  let m = 0;
  for (const c of timeline.value.clips) {
    if (!c.take) continue;
    if (c.line.id === lineId) return m;
    m += c.duration + gap;
  }
  return null;
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

/** Widen [start, end) of `text` to whole words (painting half a word is never meant). */
function snapToWords(text: string, start: number, end: number): [number, number] {
  const isWord = (ch: string) => /[\p{L}\p{N}'’]/u.test(ch);
  let s = start;
  let e = end;
  while (s > 0 && isWord(text[s - 1]) && isWord(text[s] ?? '')) s--;
  while (e < text.length && isWord(text[e]) && isWord(text[e - 1] ?? '')) e++;
  return [s, e];
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
    const [s, e] = snapToWords(line.text, start, Math.min(end, line.text.length));
    brush.value = { lineId: line.id, start: s, end: e, x: rect.left + rect.width / 2, y: rect.top };
  };
  const pick = (touch?: boolean) => {
    const s = selectionIn(el.current!);
    if (s && s.end > s.start) open(s.start, s.end, s.rect);
    else if (!touch) brush.value = null;
  };
  // While the bar is open, a changed selection in this line moves the painted range with it.
  useEffect(() => {
    if (!sel) return;
    const onSel = () => {
      const s = selectionIn(el.current!);
      if (s && s.end > s.start && (s.start !== sel.start || s.end !== sel.end)) {
        const [a, b] = snapToWords(line.text, s.start, Math.min(s.end, line.text.length));
        if (a !== sel.start || b !== sel.end) brush.value = { ...sel, start: a, end: b };
      }
    };
    document.addEventListener('selectionchange', onSel);
    return () => document.removeEventListener('selectionchange', onSel);
  }, [sel?.lineId, sel?.start, sel?.end, line.text]);
  return (
    // Touch: a long-press selection fires no mouseup; read it once the finger lifts and the selection has settled
    // (only to open: a tap on painted words opens the brushes through its click).
    <div ref={el} class="line-text" data-line={line.id} tabIndex={-1} onMouseUp={() => pick()} onTouchEnd={() => setTimeout(() => pick(true), 50)}>
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

/** Open the brush bar over a whole line (the keyboard path: the brush button in the line tools). */
function paintWholeLine(line: Line, from: HTMLElement) {
  const el = from.closest('.line')?.querySelector<HTMLElement>('.line-text');
  const rect = (el ?? from).getBoundingClientRect();
  brush.value = { lineId: line.id, start: 0, end: line.text.length, x: rect.left + rect.width / 2, y: rect.top };
}

function BrushBar() {
  const b = brush.value;
  const bar = useRef<HTMLDivElement>(null);
  const close = () => (brush.value = null);
  useEffect(() => {
    if (!b) return;
    const outside = (e: PointerEvent) => {
      if (!bar.current?.contains(e.target as Node) && !(e.target as HTMLElement).closest?.('.line-text')) brush.value = null;
    };
    window.addEventListener('pointerdown', outside);
    return () => window.removeEventListener('pointerdown', outside);
  }, [!!b]);
  // Open: modal for the shell's keys, first control focused; closed: focus back on the line's text.
  useEffect(() => {
    if (!b) return;
    const lineId = b.lineId;
    modalOpen.value = true;
    focusFirst(bar.current!, '.brushbar-body');
    if (phone.value) document.querySelector<HTMLElement>(`.line-text[data-line="${lineId}"]`)?.scrollIntoView({ block: 'start', behavior: 'smooth' });
    return () => {
      modalOpen.value = false;
      document.querySelector<HTMLElement>(`.line-text[data-line="${lineId}"]`)?.focus({ preventScroll: true });
    };
  }, [b?.lineId]);
  if (!b) return null;
  const line = script.value.lines.find((l) => l.id === b.lineId);
  if (!line) return null;
  const values = valuesAt(line, b.start);
  // A control is "mixed" when the selection covers characters with different values of it.
  const mixed = new Set<string>();
  for (let i = b.start + 1; i < b.end; i++) {
    const v = valuesAt(line, i);
    for (const id of new Set([...Object.keys(values), ...Object.keys(v)])) if ((values[id] ?? 0) !== (v[id] ?? 0)) mixed.add(id);
  }
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
  const families = spec.value.ui.groups.filter((g) => spec.value.ui.controls.some((c) => c.group === g.id && c.spannable));
  const w = 560;
  const left = clamp(b.x - w / 2, 12, window.innerWidth - w - 12);
  const top = b.y > 420 ? b.y - 12 : b.y + 40;
  return (
    <div
      ref={bar}
      class={`brushbar${b.y > 420 ? ' is-above' : ''}`}
      style={phone.value ? undefined : { left: `${left}px`, top: `${top}px`, width: `${w}px` }}
      role="dialog"
      aria-modal="true"
      aria-label="Delivery brushes"
      onKeyDown={dialogKeys(() => bar.current, close)}
    >
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
        <button class="btn sm ghost" onClick={() => (scripts.begin(), paint(line.id, b.start, b.end, null, 0), scripts.end())} title="Remove every painted value from these words">
          Clear paint
        </button>
        <button class="icon-btn sm" onClick={close} aria-label="Close brushes" title="Close (Esc)">
          <Icon name="x" size={14} />
        </button>
      </header>
      <div class="legend brushbar-legend" aria-hidden="true">
        {families.map((g) => (
          <span class="legend-item">
            <i style={{ background: hue(g.id) }} />
            {g.label}
          </span>
        ))}
      </div>
      <div class="brushbar-body">
        <div class="brush-sliders">
          {brushes.map((c) => (
            <div key={c.id} class={mixed.has(c.id) ? 'slider-wrap is-mixed' : undefined} title={mixed.has(c.id) ? 'Mixed: these words carry different values; moving the slider sets them all' : undefined}>
              <Slider control={c} value={values[c.id] ?? 0} color={hue(c.group)} label={mixed.has(c.id) ? `${c.label} · mixed` : undefined} onStart={bind.begin} onChange={(v) => bind.set(c.id, v)} onEnd={bind.end} />
            </div>
          ))}
        </div>
        {hasMood && <MoodControls b={bind} compact />}
      </div>
    </div>
  );
}

/** Takes seen by this session's strips: a take not yet seen is new (flashes; the first one toasts how to keep it). */
const seenTakes = new Set<string>();
let keeperHintShown = false;

function TakeStrip({ line }: { line: Line }) {
  const list = takes.value[line.id] ?? [];
  const fresh = list.filter((t) => !seenTakes.has(t.id) && Date.now() - t.createdAt < 3000).map((t) => t.id);
  useEffect(() => {
    for (const t of list) seenTakes.add(t.id);
    if (fresh.length && !keeperHintShown && list.length > 1 && !list.find((t) => fresh.includes(t.id))?.starred) {
      keeperHintShown = true;
      notify(`Take ${list.length} added — ★ to make it the keeper`);
    }
  }, [list.length]);
  const cmp = nowPlaying.value?.id === `cmp:${line.id}` ? comparePlaying(line.id) : null;
  if (!list.length) return null;
  return (
    <ul class="takes" aria-label="Takes">
      {list.map((t, i) => {
        const id = `take:${t.id}`;
        const playing = nowPlaying.value?.id === id || cmp === t.id;
        const why = takeStaleReason(line, t);
        return (
          <li class={`take${t.starred ? ' is-starred' : ''}${why ? ' is-stale' : ''}${fresh.includes(t.id) ? ' is-flash' : ''}${playing ? ' is-playing' : ''}`}>
            <button class="take-play" onClick={() => player.toggle(id, t.audio.pcm, t.audio.sr)} aria-label={`${playing ? 'Stop' : 'Play'} take ${i + 1}`} title={why ? `Rendered before the ${why.replace(' changed', '')} changed` : `seed ${t.seed}`}>
              <span class="take-n">{i + 1}</span>
              <Waveform pcm={t.audio.pcm} color={t.starred ? 'var(--o)' : 'var(--text-2)'} playId={id} height={22} />
            </button>
            {why && (
              <span class="chip take-stale" title="The line changed since this take was rendered; Regenerate makes a fresh one">
                {why}
              </span>
            )}
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

const COMPARE_GAP = 0.6;

/** The keeper and the newest other take of a line, in the order Compare plays them. */
function comparePair(lineId: string): [Take, Take] | null {
  const list = takes.value[lineId] ?? [];
  const keeper = list.find((t) => t.starred) ?? list[0];
  const other = [...list].reverse().find((t) => t !== keeper);
  return keeper && other ? [keeper, other] : null;
}

/** Which take of a running Compare is sounding now. */
function comparePlaying(lineId: string): string | null {
  const pair = comparePair(lineId);
  if (!pair) return null;
  const [keeper, other] = pair;
  return playhead.value < keeper.audio.pcm.length / keeper.audio.sr + COMPARE_GAP ? keeper.id : other.id;
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
  const pair = comparePair(lineId);
  const id = `cmp:${lineId}`;
  const on = nowPlaying.value?.id === id;
  if (!pair) return null;
  const [keeper, other] = pair;
  return (
    <li>
      <button
        class={`btn sm ghost${on ? ' is-on' : ''}`}
        onClick={() => {
          if (on) return player.stop();
          const { pcm, sr } = sequence([keeper.audio, other.audio], COMPARE_GAP);
          player.play(id, pcm, sr);
        }}
        title={`Play take ${list.indexOf(keeper) + 1} (keeper) then take ${list.indexOf(other) + 1}`}
      >
        <Icon name={on ? 'stop' : 'play'} size={12} /> Keeper vs take {list.indexOf(other) + 1}
      </button>
    </li>
  );
}

/** Grow a one-row textarea to its content. */
function autoGrow(el: HTMLTextAreaElement) {
  el.style.height = 'auto';
  el.style.height = `${el.scrollHeight}px`;
}

function LineRow({ line, n, playing }: { line: Line; n: number; playing: boolean }) {
  const editEl = useRef<HTMLTextAreaElement>(null);
  const isNew = editNew.peek() === line.id;
  const [editing, setEditing] = useState(isNew);
  const [draft, setDraft] = useState(line.text);
  const [emptyWarn, setEmptyWarn] = useState(false);
  const busy = lineBusy.value[line.id];
  const err = lineError.value[line.id];
  const list = takes.value[line.id] ?? [];
  const keeper = list.find((t) => t.starred) ?? list[list.length - 1];
  const staleWhy = keeper ? takeStaleReason(line, keeper) : null;
  const stale = !!staleWhy;
  const voice = lineVoiceId(line);
  useEffect(() => {
    if (editing && editEl.current) {
      autoGrow(editEl.current);
      editEl.current.focus();
      editEl.current.setSelectionRange(editEl.current.value.length, editEl.current.value.length);
    }
  }, [editing]);
  const startEdit = () => {
    setDraft(line.text);
    setEmptyWarn(false);
    setEditing(true);
  };
  /** Leave editing: commit `v` (a cleared existing line stays in editing with a warning; a cleared new line is removed). */
  const finish = (v: string, cancel = false) => {
    const wasNew = editNew.peek() === line.id;
    if (wasNew) editNew.value = null;
    if (cancel) {
      if (wasNew && !line.text.trim()) removeLine(line.id);
      setEditing(false);
      return;
    }
    if (!v.trim()) {
      if (wasNew || !line.text.trim()) return removeLine(line.id);
      setEmptyWarn(true);
      return;
    }
    if (v !== line.text) setLineText(line.id, v);
    setEditing(false);
  };
  return (
    <li class={`line${playing ? ' is-playing' : ''}`} style={{ '--voice': voiceColor(voice) }}>
      <span class="line-n">{n}</span>
      <div class="line-main">
        {editing ? (
          <>
            <div class="line-edit-row">
              <textarea
                ref={editEl}
                class="input line-edit"
                rows={1}
                value={draft}
                data-tags
                placeholder="What do they say?"
                aria-label={`Line ${n} text`}
                aria-invalid={emptyWarn || undefined}
                onInput={(e) => {
                  const el = e.currentTarget as HTMLTextAreaElement;
                  setDraft(el.value);
                  if (el.value.trim()) setEmptyWarn(false);
                  autoGrow(el);
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    finish((e.currentTarget as HTMLTextAreaElement).value);
                  } else if (e.key === 'Escape') {
                    e.preventDefault();
                    e.stopPropagation();
                    finish('', true);
                  }
                }}
                onBlur={(e) => {
                  // The tag palette borrows focus and gives it back: still editing.
                  if ((e.relatedTarget as HTMLElement | null)?.closest?.('.tag-palette, .pron-editor')) return;
                  // The element's value: an edit right before blurring may not be rendered into `draft` yet.
                  const v = (e.currentTarget as HTMLTextAreaElement).value;
                  if (!v.trim() && line.text.trim() && editNew.peek() !== line.id) return finish('', true);
                  finish(v);
                }}
              />
              <PronounceButton input={() => editEl.current} />
              <TagButton input={() => editEl.current} />
            </div>
            <p class={`hint line-edit-hint${emptyWarn ? ' is-bad' : ''}`}>{emptyWarn ? 'Text can’t be empty — Esc to cancel or delete the line' : 'Enter saves · Esc cancels'}</p>
            <PronMarks text={draft} />
            <TagMarks text={draft} />
          </>
        ) : (
          <div onDblClick={startEdit}>
            <LineText line={line} />
            <PronMarks text={line.text} issuesOnly />
            <TagMarks text={line.text} replace={(next) => setLineText(line.id, next)} />
          </div>
        )}
        <TakeStrip line={line} />
        {err && (
          <div class="error-line" role="alert">
            {isPronError(err) ? (
              'Not rendered: fix the pronunciation marked above.'
            ) : (
              <>
                Render failed: {err}{' '}
                <button class="link" onClick={() => void renderLine(line.id)}>
                  Retry
                </button>
              </>
            )}
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
          onClick={() => (busy ? cancelLine(line.id) : void renderLine(line.id))}
          disabled={!busy && !line.text.trim()}
          title={busy ? 'Stop rendering this line' : stale ? `The ${staleWhy!.replace(' changed', '')} changed since this take: render a fresh one` : list.length ? 'Render another take (new seed)' : 'Render this line'}
        >
          {busy ? 'Cancel' : stale ? (
            <>
              <Icon name="refresh" size={13} /> Regenerate
            </>
          ) : list.length ? 'New take' : 'Render'}
        </button>
        <div class="line-tools">
          <button class="icon-btn sm" onClick={startEdit} aria-label={`Edit line ${n}`} title="Edit text (paint stays on its words) · double-click the line">
            <Icon name="edit" size={13} />
          </button>
          <button class="icon-btn sm" disabled={!line.text.trim()} onClick={(e) => paintWholeLine(line, e.currentTarget as HTMLElement)} aria-label={`Paint delivery onto line ${n}`} title="Paint delivery onto this line (or select words in it)">
            <Icon name="brush" size={13} />
          </button>
          <button class="icon-btn sm" onClick={() => (editNew.value = addLine(line.id, '', line.voice))} aria-label={`Add a line below line ${n}`} title="Add a line below">
            <Icon name="plus" size={13} />
          </button>
          <button class="icon-btn sm" onClick={() => moveLine(line.id, -1)} aria-label={`Move line ${n} up`} title="Move up">
            <Icon name="up" size={13} />
          </button>
          <button class="icon-btn sm" onClick={() => moveLine(line.id, 1)} aria-label={`Move line ${n} down`} title="Move down">
            <Icon name="down" size={13} />
          </button>
          <button class="icon-btn sm" onClick={() => removeLine(line.id)} aria-label={`Delete line ${n}`} title="Delete line">
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
        <div class="voice-adds">
          {people.map((p) => (
            <button class="btn ghost sm" style={{ '--voice': voiceColor(p.id) }} onClick={() => (editNew.value = addLine(s.lines[s.lines.length - 1]?.id ?? null, '', p.id))} title={`Add a line spoken by ${p.name}`}>
              <span class="voice-dot" /> {p.name} says…
            </button>
          ))}
          {!people.length && (
            <>
              <button class="btn ghost sm" style={{ '--voice': voiceColor(null) }} disabled={!design.value.baseId} onClick={() => (editNew.value = addLine(s.lines[s.lines.length - 1]?.id ?? null, ''))} title="Add a line spoken by the Cast bench voice">
                <span class="voice-dot" /> Bench says…
              </button>
              <button class="btn sm voice-bench" onClick={addBench} disabled={!design.value.baseId} title="Save the Cast bench voice as a cast member, so lines can pick it">
                <Icon name="plus" size={13} /> Add the bench voice to the cast
              </button>
            </>
          )}
        </div>
        <p class="hint">
          Language {langLabel(projectSettings.value.language)} · tag strength {backend.value?.tags ? (projectSettings.value.tagStrength > 1 ? projectSettings.value.tagStrength.toFixed(1) : 'off') : 'n/a'} —{' '}
          <button class="link" onClick={() => (settingsOpen.value = true)}>
            Project settings
          </button>
        </p>
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
        <p class="hint">Select words in a line to paint delivery onto them (or use the brush in the line’s tools for the whole line). Click painted words to change or clear them. Underlines show what is painted:</p>
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
  const missing = s.lines.filter((l) => l.text.trim() && !(takes.value[l.id] ?? []).length);
  const stale = staleLines.value.length;
  const q = renderQueue.value;
  const rendering = !!q || Object.values(lineBusy.value).some(Boolean);
  const playing = nowPlaying.value?.id === SCRIPT_ID;
  return (
    <aside class="rail direct-side" aria-label="Script">
      <div class="rail-section">
        <h4>Script</h4>
        <p class="hint">
          {ready}/{s.lines.length} lines have a take. Play and export use each line’s starred take.
        </p>
        <div class="col gap-s">
          {q ? (
            <div class="render-progress" aria-live="polite">
              <span>
                Rendering {q.done}/{q.total}
                {q.failed ? ` · ${q.failed} failed` : ''}
              </span>
              <button class="btn sm" onClick={cancelRenders}>
                Cancel
              </button>
            </div>
          ) : (
            <>
              {stale > 0 && (
                <button class="btn primary" disabled={rendering} onClick={() => void regenerateStale()} title="Settings or voice changed since these lines’ starred takes: render a new take of each">
                  <Icon name="refresh" size={14} /> Regenerate {stale} changed line{stale === 1 ? '' : 's'}
                </button>
              )}
              <button class="btn" disabled={rendering || !missing.length} onClick={() => void renderLines(missing.map((l) => l.id))} title={missing.length ? 'Render every line that has no take yet' : 'Every line has a take'}>
                <Icon name="wand" size={14} /> Render {missing.length} missing
              </button>
            </>
          )}
          <button class={`btn${stale ? '' : ' primary'}`} disabled={!ready} onClick={() => (playing ? player.stop() : playScript())} title={ready ? undefined : 'Render a line first'}>
            <Icon name={playing ? 'stop' : 'play'} size={14} /> {playing ? 'Stop' : 'Play script'}
          </button>
          <button class="btn" disabled={!ready} onClick={exportScript} title={ready ? `Save the script as ${slug(current.value?.name ?? 'script')}.wav` : 'Render a line first'}>
            <Icon name="download" size={14} /> Export WAV
          </button>
        </div>
      </div>
      <div class="rail-section">
        <h4>Timing</h4>
        <label class="gap">
          Gap between lines
          <input type="range" min={0} max={1.5} step={0.05} value={s.gap} aria-valuetext={`${s.gap.toFixed(2)} seconds`} onInput={(e) => setGap(Number((e.currentTarget as HTMLInputElement).value))} aria-label="Gap between lines (s)" />
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
                      class={`clip${c.take ? '' : ' is-pending'}${!c.take && at !== null ? ' is-skipped' : ''}`}
                      style={{ left: pct(c.start), width: pct(c.duration) }}
                      title={c.take ? c.line.text : `${c.line.text} (not rendered: estimated length)`}
                      aria-label={`Line ${script.value.lines.indexOf(c.line) + 1}: ${c.line.text}${c.take ? '' : ' (not rendered)'}`}
                      onClick={() => {
                        if (!c.take) return;
                        if (nowPlaying.value?.id === SCRIPT_ID) {
                          const mix = mixdown();
                          const off = mixOffset(c.line.id);
                          if (mix && off !== null) player.play(SCRIPT_ID, mix.pcm, mix.sr, off);
                        } else player.toggle(`take:${c.take.id}`, c.take.audio.pcm, c.take.audio.sr);
                      }}
                    >
                      {c.take ? (
                        <Waveform pcm={c.take.audio.pcm} color={color} height={28} />
                      ) : (
                        <span>
                          <i class="clip-est">est.</i> {c.line.text}
                        </span>
                      )}
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
            <button class="btn ghost add-line" onClick={() => (editNew.value = addLine(null, ''))}>
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
