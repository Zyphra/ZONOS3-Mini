import { effect, signal } from '@preact/signals';
import { useEffect, useRef, useState } from 'preact/hooks';
import { exportWav, sequence } from '../../audio/mix';
import * as player from '../../audio/player';
import { nowPlaying, playhead } from '../../audio/player';
import { autoplay } from '../../audio/preview';
import { Icon } from '../../components/Icon';
import { dialogKeys, focusFirst } from '../../components/focusTrap';
import { Rail, RailBar, RailToggle } from '../../components/Rail';
import { slug } from '../../lib/files';
import { phone } from '../../lib/media';
import { isTyping } from '../../lib/dom';
import { decoderName } from '../../lib/decoders';
import { LACI_LABEL } from '../../lib/laci';
import { langLabel } from '../../lib/languages';
import { guessed, isPronError, lookupLanguage, openPronEditor, phonemizeOn, phonemizeRange, phonemizing, PronHelp, PronMarks, PronounceButton } from '../../components/Pron';
import { parsePron } from '../../lib/pron';
import { Slider } from '../../components/Slider';
import { Waveform } from '../../components/Waveform';
import { clamp, uid } from '../../lib/math';
import { activeHistory, backend, controlById, hue, modalOpen, notify, primaryAction, projectSettings, pronTokens, requestContext, screenKeys, settingsOpen, spec } from '../../state/app';
import { design } from '../../state/design';
import type { History } from '../../state/history';
import { cast, savePersona, speakers, type Persona } from '../../state/library';
import { current } from '../../state/projects';
import { coversLine } from '../../state/requests';
import { failedLines, GAP_RANGE, playingLineId, restoreLine, scriptDraft, scriptTextDirty, setVoiceColor } from '../../state/script';
import {
  addLine,
  cancelLine,
  cancelRenders,
  deleteTake,
  keepers,
  lineBusy,
  lineError,
  lineLanguage,
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
  setLineText,
  setLineStart,
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

/** Cast voices tell apart by the data hues (theme-aware tokens), assigned on first appearance and kept in the script; the bench voice is neutral. */
const VOICE_COLORS = ['var(--c1)', 'var(--c2)', 'var(--c6)', 'var(--c3)', 'var(--c5)', 'var(--c4)'];
function voiceColor(id: string | null): string {
  if (!id) return 'var(--text-3)';
  const have = script.value.voiceColors?.[id];
  if (have) return have;
  // Unassigned (an older project, or a new cast member): the next hue not yet taken.
  const taken = new Set(Object.values(script.value.voiceColors ?? {}));
  return VOICE_COLORS.find((c) => !taken.has(c)) ?? VOICE_COLORS[Object.keys(script.value.voiceColors ?? {}).length % VOICE_COLORS.length];
}
/** Every cast voice the script mentions gets its colour pinned (so it never shifts when the cast is reordered). */
effect(() => {
  const s = script.value;
  const ids = new Set<string>();
  if (s.voice) ids.add(s.voice);
  for (const l of s.lines) if (l.voice) ids.add(l.voice);
  for (const id of ids) if (!s.voiceColors?.[id]) setVoiceColor(id, voiceColor(id));
});
/** A voice's display name: the cast persona, or the voice on the Cast bench (null). */
const voiceName = (id: string | null) => (id ? (cast.value.find((p) => p.id === id)?.name ?? '(removed voice)') : (speakers.value.find((s) => s.id === design.value.baseId)?.label ?? 'Voice'));

interface Brush {
  lineId: string;
  start: number;
  end: number;
  /** Picked by tapping words (touch): taps extend it, and its words are highlighted (no native selection shows them). */
  touch?: boolean;
}
const brush = signal<Brush | null>(null);
/** The kind of the last pointer that pressed on a line's text: touch taps pick words, mouse drags select them. */
let lastPointer = 'mouse';

function playScript() {
  const mix = mixdown();
  if (!mix) return notify('Render some lines first: each line’s keeper plays in order.');
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
/** The playing line as a signal: rows subscribe to this one boolean instead of the playhead. */
effect(() => {
  const id = playPosition()?.lineId ?? null;
  if (playingLineId.peek() !== id) playingLineId.value = id;
});

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

/** [start, end) of every word of `text`, in order (the keyboard selection steps through these). */
function wordRanges(text: string): [number, number][] {
  const out: [number, number][] = [];
  for (const m of text.matchAll(/[\p{L}\p{N}'’]+/gu)) out.push([m.index, m.index + m[0].length]);
  return out;
}

/** The offset into the line text under the point (x, y) in `root`'s segments; null off its words (tag chips included). */
function offsetAt(root: HTMLElement, x: number, y: number): number | null {
  let node: Node | null = null;
  let off = 0;
  if (document.caretPositionFromPoint) {
    const p = document.caretPositionFromPoint(x, y);
    if (p) [node, off] = [p.offsetNode, p.offset];
  } else {
    const r = document.caretRangeFromPoint?.(x, y);
    if (r) [node, off] = [r.startContainer, r.startOffset];
  }
  if (!node || !root.contains(node)) return null;
  const seg = (node.nodeType === Node.TEXT_NODE ? node.parentElement : (node as HTMLElement))?.closest<HTMLElement>('[data-start]');
  if (!seg || seg.dataset.end) return null;
  return Number(seg.dataset.start) + (node.nodeType === Node.TEXT_NODE ? off : 0);
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
  const sel = brush.value?.lineId === line.id ? brush.value : null;
  // A tapped range shows on its own segments (no native selection draws it); a selected one keeps the text nodes whole
  // so re-rendering doesn't drop the selection.
  if (sel?.touch) {
    cuts.add(sel.start);
    cuts.add(sel.end);
  }
  const marks = [...cuts].sort((a, b) => a - b);
  const open = (start: number, end: number, touch?: boolean) => {
    const [s, e] = snapToWords(line.text, start, Math.min(end, line.text.length));
    brush.value = { lineId: line.id, start: s, end: e, touch };
  };
  const pick = (touch?: boolean) => {
    const s = selectionIn(el.current!);
    if (s && s.end > s.start) open(s.start, s.end);
    else if (!touch) brush.value = null;
  };
  // Touch (no drag-select there): a tap on a word opens the brushes on it (on its whole span when painted). With a
  // tapped range open in this line, a tap outside it stretches it to that word, a tap inside starts over from that word,
  // and a tap on its only word puts it down. A long press still selects text natively. Returns whether it took the tap.
  const tap = (e: MouseEvent) => {
    if (lastPointer !== 'touch' || !window.getSelection()?.isCollapsed) return false;
    const pos = offsetAt(el.current!, e.clientX, e.clientY);
    // The word holding the tapped offset, else the one ending there (a tap just past a word's last letter).
    const words = pos === null ? [] : wordRanges(line.text);
    const w = words.find(([s, end]) => pos! >= s && pos! < end) ?? words.find(([, end]) => pos === end);
    if (!w) return false;
    const cur = brush.peek();
    if (cur?.lineId === line.id && cur.touch) {
      if (w[0] < cur.start || w[1] > cur.end) open(Math.min(cur.start, w[0]), Math.max(cur.end, w[1]), true);
      else if (w[0] === cur.start && w[1] === cur.end) brush.value = null;
      else open(w[0], w[1], true);
      return true;
    }
    const span = line.spans.find((sp) => w[0] >= sp.start && w[0] < sp.end);
    open(span ? span.start : w[0], span ? span.end : w[1], true);
    return true;
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
  // Keyboard: ⇧← / ⇧→ grow a word selection (mirrored into the document selection), Enter paints it, Esc drops it.
  const [kbd, setKbd] = useState<[number, number] | null>(null);
  useEffect(() => setKbd(null), [line.text]);
  const words = wordRanges(line.text);
  const mirror = (range: [number, number] | null) => {
    setKbd(range);
    const root = el.current;
    const s = window.getSelection();
    if (!root || !s) return;
    if (!range) return s.removeAllRanges();
    const segs = [...root.querySelectorAll<HTMLElement>('.seg[data-start]')].filter((x) => !x.dataset.end);
    const at = (pos: number) => {
      // The text node holding `pos`, and the offset inside it.
      for (const seg of segs) {
        const start = Number(seg.dataset.start);
        const node = seg.firstChild;
        if (!node || node.nodeType !== Node.TEXT_NODE) continue;
        const len = node.textContent?.length ?? 0;
        if (pos >= start && pos <= start + len) return { node, offset: pos - start };
      }
      return null;
    };
    const a = at(range[0]);
    const b = at(range[1]);
    if (!a || !b) return;
    const r = document.createRange();
    r.setStart(a.node, a.offset);
    r.setEnd(b.node, b.offset);
    s.removeAllRanges();
    s.addRange(r);
  };
  const onKey = (e: KeyboardEvent) => {
    if (e.target !== el.current || !words.length) return;
    if (e.key === 'Escape' && kbd) {
      e.preventDefault();
      mirror(null);
      return;
    }
    if (e.key === 'Enter' && kbd) {
      e.preventDefault();
      open(kbd[0], kbd[1]);
      return;
    }
    if (!e.shiftKey || (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight')) return;
    e.preventDefault();
    const first = words.findIndex((w) => kbd && w[0] >= kbd[0]);
    const last = kbd ? words.findLastIndex((w) => w[1] <= kbd[1]) : -1;
    let lo = kbd ? first : e.key === 'ArrowRight' ? 0 : words.length - 1;
    let hi = kbd ? last : lo;
    if (kbd) {
      if (e.key === 'ArrowRight') hi = Math.min(words.length - 1, hi + 1);
      else lo = Math.max(0, lo - 1);
    }
    mirror([words[lo][0], words[hi][1]]);
  };
  return (
    // Touch: a long-press selection fires no mouseup; read it once the finger lifts and the selection has settled
    // (only to open). A tap (no selection) picks words in the capture phase, before a painted word's own click.
    <div
      ref={el}
      class={`line-text${sel?.touch ? ' is-tapping' : ''}`}
      data-line={line.id}
      tabIndex={0}
      aria-label={`Line text: ${line.text}. Shift with the arrow keys selects words; Enter paints them.`}
      onPointerDown={(e) => (lastPointer = e.pointerType)}
      onClickCapture={(e) => {
        if (!(e.target as Element).closest('.is-pron') && tap(e)) e.stopPropagation();
      }}
      onMouseUp={() => lastPointer !== 'touch' && pick()}
      onTouchEnd={() => setTimeout(() => pick(true), 50)}
      onKeyDown={onKey}
      onBlur={() => kbd && setKbd(null)}
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
                  ? () => {
                      if (window.getSelection()?.isCollapsed) open(span.start, span.end);
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
function paintWholeLine(line: Line) {
  brush.value = { lineId: line.id, start: 0, end: line.text.length };
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
  const whole = coversLine(line.text, b.start, b.end);
  const brushes = spec.value.ui.controls.filter((c) => c.spannable && c.group !== 'mood' && !(whole && c.span_only));
  const hasMood = spec.value.ui.controls.some((c) => c.spannable && c.group === 'mood');
  const families = spec.value.ui.groups.filter((g) => spec.value.ui.controls.some((c) => c.group === g.id && c.spannable));
  // Desktop: the bar sits inside its line (`.line-main`, position: relative), under the text; phone: a bottom sheet.
  return (
    <div
      ref={bar}
      class={`brushbar${phone.value ? '' : ' is-anchored'}`}
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
      {b.touch && <p class="hint brushbar-tap">Tap more words to stretch the selection; tap a word inside it to start over from there.</p>}
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

/** Takes seen by this session's strips: a take not yet seen is new (flashes; with auto-play on, a single-line render plays it). */
const seenTakes = new Set<string>();

function TakeStrip({ line }: { line: Line }) {
  const list = takes.value[line.id] ?? [];
  const fresh = list.filter((t) => !seenTakes.has(t.id) && Date.now() - t.createdAt < 3000).map((t) => t.id);
  useEffect(() => {
    const newest = list[list.length - 1];
    const isNew = newest && !seenTakes.has(newest.id) && fresh.includes(newest.id);
    for (const t of list) seenTakes.add(t.id);
    // One line rendered by hand (no batch running) and auto-play on: hear it right away.
    if (isNew && autoplay.value && !renderQueue.peek() && !nowPlaying.peek()) player.play(`take:${newest.id}`, newest.audio.pcm, newest.audio.sr);
  }, [list.length]);
  const cmp = nowPlaying.value?.id === `cmp:${line.id}` ? comparePlaying(line.id) : null;
  if (!list.length) return null;
  const decoders = backend.value?.decoders ?? [];
  return (
    <ul class="takes" aria-label="Takes">
      {list.map((t, i) => {
        const id = `take:${t.id}`;
        const playing = nowPlaying.value?.id === id || cmp === t.id;
        const why = takeStaleReason(line, t);
        // The decoder that made it (X-Decoder, else what the request asked for), where there is a choice or it isn't this server's default.
        const made = t.audio.decoder ?? t.request.decoder;
        const decoderNote = made && (decoders.length > 1 || made !== decoders[0]?.id) ? `Decoder ${decoderName(made)}` : null;
        return (
          <li class={`take${t.starred ? ' is-starred' : ''}${why ? ' is-stale' : ''}${fresh.includes(t.id) ? ' is-flash' : ''}${playing ? ' is-playing' : ''}`}>
            <button
              class="take-play"
              onClick={() => player.toggle(id, t.audio.pcm, t.audio.sr)}
              aria-label={`${playing ? 'Stop' : 'Play'} take ${i + 1}`}
              title={[why ? `Rendered before the ${why.replace(' changed', '')} changed` : `Seed ${t.seed}`, decoderNote, t.audio.spanFollow ? FOLLOW_TEXT[t.audio.spanFollow] : null].filter(Boolean).join(' · ')}
            >
              <span class="take-n">{i + 1}</span>
              <Waveform pcm={t.audio.pcm} color={t.starred ? 'var(--o)' : 'var(--text-2)'} playId={id} height={22} />
            </button>
            {why && (
              <span class="chip take-stale" title="The line changed since this take was rendered; Regenerate makes a fresh one">
                {why}
              </span>
            )}
            {guidanceNote(t.request) && (
              <span class="take-follow is-tags" title={`${guidanceNote(t.request)}: the server takes one guidance per render`} aria-label={guidanceNote(t.request)!}>
                tags
              </span>
            )}
            <button class={`icon-btn sm star${t.starred ? ' is-on' : ''}`} onClick={() => starTake(line.id, t.id)} aria-label={`Make take ${i + 1} the keeper`} aria-pressed={t.starred} title="Script plays this one">
              <Icon name="star" size={13} filled={t.starred} />
            </button>
            {canEdit.value && (
              <button
                class="icon-btn sm"
                onClick={() => void openClip({ kind: 'take', label: `Line ${script.value.lines.indexOf(line) + 1} · take ${i + 1}`, lineId: line.id, takeId: t.id }, t.audio.pcm, t.audio.sr, t.text, t.request.language ?? 'en_us')}
                aria-label={`Edit the words of take ${i + 1}`}
                title="Edit the words of this take (Edit screen)"
              >
                <Icon name="edit-speech" size={12} />
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

/** How the take's painted spans were placed (X-Control-Span-Follow), for the take's hover text. */
const FOLLOW_TEXT: Record<SpanFollow, string> = {
  reading: 'Delivery painted by reading position',
  rate: 'Delivery painted by estimate (no alignment heads for this language)',
};

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

function LineRow({ line, n, count }: { line: Line; n: number; count: number }) {
  const playing = playingLineId.value === line.id;
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
                  if (e.key === 'Enter' && (!e.shiftKey || e.metaKey || e.ctrlKey)) {
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
          <div onDblClick={() => lastPointer !== 'touch' && startEdit()}>
            <LineText line={line} />
            <PronMarks text={line.text} issuesOnly />
            <TagMarks text={line.text} replace={(next) => setLineText(line.id, next)} />
          </div>
        )}
        <TakeStrip line={line} />
        {brush.value?.lineId === line.id && !phone.value && <BrushBar />}
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
          title={`Speaks as ${voiceName(voice)}, in ${langLabel(lineLanguage(line))}`}
        >
          <option value="">Default · {voiceName(script.value.voice)}</option>
          {line.voice && !cast.value.some((p) => p.id === line.voice) && <option value={line.voice}>(voice unavailable)</option>}
          {cast.value.map((p) => (
            <option value={p.id}>{p.name}</option>
          ))}
        </select>
        {backend.value?.startToken && (
          <button
            class={`icon-btn sm line-start${line.startImmediately ? ' is-on' : ''}`}
            aria-pressed={!!line.startImmediately}
            onClick={() => setLineStart(line.id, !line.startImmediately)}
            aria-label={`Start line ${n} immediately`}
            title={line.startImmediately ? 'Starts at once, no lead-in silence (click to allow it)' : 'Start immediately: speech from the first frame, no lead-in silence'}
          >
            <Icon name="start" size={13} />
          </button>
        )}
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
          <button class="icon-btn sm" disabled={!line.text.trim()} onClick={() => paintWholeLine(line)} aria-label={`Paint delivery onto line ${n}`} title="Paint delivery onto this line (or select words in it)">
            <Icon name="brush" size={13} />
          </button>
          <button class="icon-btn sm" onClick={() => (editNew.value = addLine(line.id, '', line.voice))} aria-label={`Add a line below line ${n}`} title="Add a line below">
            <Icon name="plus" size={13} />
          </button>
          <button class="icon-btn sm" disabled={n === 1} onClick={() => moveLine(line.id, -1)} aria-label={`Move line ${n} up`} title="Move up">
            <Icon name="up" size={13} />
          </button>
          <button class="icon-btn sm" disabled={n === count} onClick={() => moveLine(line.id, 1)} aria-label={`Move line ${n} down`} title="Move down">
            <Icon name="down" size={13} />
          </button>
          <button
            class="icon-btn sm"
            onClick={() => {
              const r = removeLine(line.id);
              if (r) notify(`Line ${n} deleted`, 'info', { label: 'Undo', run: () => restoreLine(r.line, r.index, r.takes) });
            }}
            aria-label={`Delete line ${n}`}
            title="Delete line"
          >
            <Icon name="trash" size={13} />
          </button>
        </div>
      </div>
    </li>
  );
}

const linesAsText = () => script.value.lines.map((l) => l.text).join('\n');

async function closeScriptText() {
  if (scriptTextDirty.value && !(await confirm('Discard your script text changes?', { ok: 'Discard', danger: true }))) return;
  scriptDraft.value = null;
  editAll.value = false;
}

/** The whole script as text. The draft is module state (`scriptDraft`): leaving Direct and coming back finds it again. */
function ScriptText() {
  const pasting = script.value.lines.length === 0;
  const text = scriptDraft.value ?? linesAsText();
  const apply = async () => {
    const rows = splitScript(text, { sentences: pasting });
    const { dropped } = matchLines(script.value.lines, rows);
    const takesLost = dropped.reduce((n, l) => n + (takes.value[l.id]?.length ?? 0), 0);
    if (dropped.length && takesLost) {
      const ok = await confirm(`${dropped.length} line${dropped.length === 1 ? '' : 's'} will be removed with ${takesLost} take${takesLost === 1 ? '' : 's'}. Apply anyway?`, { ok: 'Apply', danger: true });
      if (!ok) return;
    }
    replaceScript(rows);
    scriptDraft.value = null;
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
          scriptDraft.value = v === linesAsText() ? null : v;
        }}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
            e.preventDefault();
            void apply();
          }
        }}
        aria-label="Script text"
        placeholder={pasting ? 'Paste or type the script' : undefined}
      />
      <p class="hint">{pasting ? 'The script is split at line breaks and sentence ends.' : 'One line per row. Rows you leave unchanged keep their paint, voice and takes.'} ⌘↵ applies.</p>
      <button class="btn primary" onClick={() => void apply()}>
        Apply script
      </button>
    </div>
  );
}

/** Edit-as-text open; a kept draft (`scriptDraft`) opens it again on return. */
const editAll = signal(scriptDraft.peek() !== null);
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
          <option value="">{speakers.value.find((x) => x.id === design.value.baseId)?.label ?? '—'} (from Cast)</option>
          {people.map((p) => (
            <option value={p.id}>{p.name}</option>
          ))}
        </select>
        <p class="hint">
          {base?.label ?? 'No voice'} · {d.mode === 'reference' ? 'kept as is' : 'shaped'}. Lines use the default voice unless they pick their own. Change a voice’s identity in Cast; here only delivery and mood paint onto words.
        </p>
        <div class="voice-adds">
          {people.map((p) => (
            <button class="btn ghost sm" style={{ '--voice': voiceColor(p.id) }} onClick={() => (editNew.value = addLine(s.lines[s.lines.length - 1]?.id ?? null, '', p.id))} title={`Add a line spoken by ${p.name}`}>
              <span class="voice-dot" /> {p.name} says…
            </button>
          ))}
          {!people.length && (
            <>
              <button class="btn ghost sm" style={{ '--voice': voiceColor(null) }} disabled={!design.value.baseId} onClick={() => (editNew.value = addLine(s.lines[s.lines.length - 1]?.id ?? null, ''))} title={design.value.baseId ? 'Add a line spoken by the default voice' : 'Pick a voice in Cast first'}>
                <span class="voice-dot" /> {voiceName(null)} says…
              </button>
              <button class="btn sm voice-bench" onClick={addBench} disabled={!design.value.baseId} title={design.value.baseId ? 'Save the Cast voice as a cast member, so lines can pick it' : 'Pick a voice in Cast first'}>
                <Icon name="plus" size={13} /> Add this voice to the cast
              </button>
              {!design.value.baseId && <p class="hint">Pick a voice in Cast first.</p>}
            </>
          )}
        </div>
        <p class="hint">
          Language {langLabel(projectSettings.value.language)} (library voices speak their own) · tag strength {backend.value?.tags ? (projectSettings.value.tagStrength > 1 ? projectSettings.value.tagStrength.toFixed(1) : 'off') : 'n/a'}
          {(backend.value?.decoders.length ?? 0) > 1 && ` · decoder ${decoderName(requestContext.value.decoder ?? backend.value!.decoders[0].id)}`}
          {(backend.value?.laciLanguages.length ?? 0) > 0 && ` · LACI ${LACI_LABEL[projectSettings.value.laci].toLowerCase()}`} —{' '}
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
        <p class="hint">Select words in a line to paint delivery onto them (on a phone, tap a word, then tap more to stretch the selection; or use the brush in the line’s tools for the whole line). Click painted words to change or clear them. Underlines show what is painted:</p>
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

/** Scroll the first failed line into view and flash it. */
function showFailed() {
  const id = failedLines.value[0];
  const el = id && document.querySelector<HTMLElement>(`.line-text[data-line="${id}"]`)?.closest<HTMLElement>('.line');
  if (!el) return;
  el.scrollIntoView({ block: 'center', behavior: 'smooth' });
  el.classList.add('is-flash');
  setTimeout(() => el.classList.remove('is-flash'), 1200);
}

/** Render / Regenerate / Play / progress: the rail on desktop, the sticky bottom bar on phones. */
function Transport({ bar }: { bar?: boolean }) {
  const s = script.value;
  const ready = keepers.value.filter((k) => k.take).length;
  const missing = s.lines.filter((l) => l.text.trim() && !(takes.value[l.id] ?? []).length);
  const stale = staleLines.value.length;
  const failed = failedLines.value.filter((id) => s.lines.some((l) => l.id === id)).length;
  const q = renderQueue.value;
  const rendering = !!q || Object.values(lineBusy.value).some(Boolean);
  const playing = nowPlaying.value?.id === SCRIPT_ID;
  if (!s.lines.length) return null;
  return (
    <div class={bar ? 'transport' : 'col gap-s'}>
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
            <button class="btn primary" disabled={rendering} onClick={() => void regenerateStale()} title="Settings or a voice changed since these lines’ keepers: render a new take of each">
              <Icon name="refresh" size={14} /> Regenerate {stale} changed
            </button>
          )}
          {missing.length > 0 && (
            <button class="btn" disabled={rendering} onClick={() => void renderLines(missing.map((l) => l.id))} title="Render every line that has no take yet">
              <Icon name="wand" size={14} /> Render {missing.length} missing
            </button>
          )}
        </>
      )}
      <button class={`btn${stale || missing.length ? '' : ' primary'}`} disabled={!ready} onClick={() => (playing ? player.stop() : playScript())} title={ready ? undefined : 'Render a line first'}>
        <Icon name={playing ? 'stop' : 'play'} size={14} /> {playing ? 'Stop' : 'Play script'}
      </button>
      {failed > 0 && !q && (
        <p class="hint is-bad transport-failed">
          {failed} failed ·{' '}
          <button class="link" onClick={showFailed}>
            Show
          </button>
        </p>
      )}
    </div>
  );
}

/** The right sidebar: the script as a whole (renders, playback, export, the gap between lines). On phones the transport sits under the lines instead. */
function ScriptSide() {
  const s = script.value;
  const ready = keepers.value.filter((k) => k.take).length;
  const gapSpec = { id: 'gap', group: 'timing', label: 'Gap between lines', type: 'unipolar' as const, map: { '+': {} }, range: GAP_RANGE, safe: GAP_RANGE, collapse: null, spannable: false, identity: null };
  return (
    <aside class="rail direct-side" aria-label={phone.value ? 'Export' : 'Script'}>
      {!phone.value && (
        <div class="rail-section">
          <h4>Script</h4>
          {s.lines.length > 0 && (
            <p class="hint">
              {ready}/{s.lines.length} lines have a take. Play and export use each line’s keeper.
            </p>
          )}
          <Transport />
        </div>
      )}
      <div class="rail-section">
        <h4>Export</h4>
        <button class="btn" disabled={!ready} onClick={exportScript} title={ready ? `Save the script as ${slug(current.value?.name ?? 'script')}.wav` : 'Render a line first'}>
          <Icon name="download" size={14} /> Export WAV
        </button>
      </div>
      <div class="rail-section">
        <h4>Timing</h4>
        <Slider control={gapSpec} value={s.gap} color="var(--text-2)" compact format={(v) => `${v.toFixed(2)} s`} onStart={() => scripts.begin()} onChange={(v) => scripts.set({ ...script.value, gap: Math.round(v * 100) / 100 })} onEnd={() => scripts.end()} />
        <p class="hint">Silence between lines when the script plays or exports.</p>
      </div>
    </aside>
  );
}

/** One lane per voice; each line's keeper take placed where the script plays it. Reads the playhead itself (the rows don't). */
function Timeline() {
  const { clips, total } = timeline.value;
  if (!clips.length) return null;
  const at = playPosition()?.t ?? null;
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
                      title={c.take ? c.line.text : `Not rendered · ${c.line.text} (click to render)`}
                      aria-label={`Line ${script.value.lines.indexOf(c.line) + 1}: ${c.line.text}${c.take ? '' : ' (not rendered: click to render)'}`}
                      onClick={() => {
                        if (!c.take) return void renderLine(c.line.id);
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
                          <i class="clip-est">Not rendered</i> · {c.line.text}
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

/** Direct's own keys (the shell asks first): R renders the focused line, S stars the take that is playing. */
function directKeys(e: KeyboardEvent): boolean {
  if (e.metaKey || e.ctrlKey || e.altKey || isTyping(e.target) || modalOpen.value) return false;
  if (e.key === 'r' || e.key === 'R') {
    const id = (e.target as HTMLElement | null)?.closest?.<HTMLElement>('.line')?.querySelector<HTMLElement>('.line-text')?.dataset.line;
    if (!id) return false;
    e.preventDefault();
    void renderLine(id);
    return true;
  }
  if (e.key === 's' || e.key === 'S') {
    const np = nowPlaying.value;
    const takeId = np?.id.startsWith('take:') ? np.id.slice(5) : null;
    if (!takeId) return false;
    const lineId = Object.keys(takes.value).find((lid) => takes.value[lid].some((t) => t.id === takeId));
    if (!lineId) return false;
    e.preventDefault();
    starTake(lineId, takeId);
    notify('Keeper: the script plays this take');
    return true;
  }
  return false;
}

export function DirectScreen() {
  useEffect(() => {
    activeHistory.value = scripts as History<unknown>;
    primaryAction.value = playScript;
    screenKeys.value = directKeys;
    // A kept edit-as-text draft reopens where it was left.
    if (scriptDraft.peek() !== null) editAll.value = true;
    return () => {
      activeHistory.value = null;
      primaryAction.value = null;
      screenKeys.value = null;
      brush.value = null;
    };
  }, []);
  const lines = script.value.lines;
  const onPhone = phone.value;
  return (
    <div class="direct">
      <Rail id="direct-voices" side="left" title="Voices">
        <Sidebar />
      </Rail>
      <section class="script" aria-label="Lines">
        <RailBar>
          <RailToggle id="direct-voices" title="Voices" icon="mic" />
          <RailToggle id="direct-script" title="Export" icon="download" />
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
              <LineRow key={l.id} line={l} n={i + 1} count={lines.length} />
            ))}
          </ol>
        ) : (
          <div class="empty">
            <Icon name="edit" size={26} />
            <h3>No lines yet</h3>
            <p>Paste a script, or add lines one by one; each line is rendered and directed on its own, in its own voice.</p>
            <div class="row gap-s">
              <button class="btn primary" onClick={() => (editAll.value = true)}>
                <Icon name="edit" size={14} /> Paste a script
              </button>
              <button class="btn" onClick={() => (editNew.value = addLine(null, ''))}>
                <Icon name="plus" size={14} /> Add a line
              </button>
            </div>
          </div>
        )}
        {!editAll.value && lines.length > 0 && (
          <>
            <button class="btn ghost add-line" onClick={() => (editNew.value = addLine(null, ''))}>
              <Icon name="plus" size={14} /> Add line
            </button>
            <Timeline />
            {onPhone && <Transport bar />}
          </>
        )}
      </section>
      <Rail id="direct-script" side="right" title={onPhone ? 'Export' : 'Script'}>
        <ScriptSide />
      </Rail>
      {onPhone && <BrushBar />}
    </div>
  );
}
