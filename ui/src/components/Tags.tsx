/**
 * Nonverbal tags in text fields: the palette (point tags at the caret, span tags around the selected words; `<` in
 * a field marked `data-tags`, or the `<>` button beside it), chips that remove or convert a tag, an overlay that
 * shows a resting field's markup as chips, and the tag-strength control. Everything hides when the server has no
 * tag support (`capabilities.tag_cfg`); markup then reads as plain text, and the chips say so.
 */
import { computed, signal } from '@preact/signals';
import type { ComponentChildren } from 'preact';
import { useLayoutEffect, useRef, useState } from 'preact/hooks';
import { clamp } from '../lib/math';
import { insertPoint, parseTags, pointToSpan, removeTag, SPAN_TAG_NAMES, spanToPoint, TAG_INFO, TAG_NAMES, wrapSpan, type Edit, type TagName, type TagSpan, type TagToken } from '../lib/tags';
import { backend, projectSettings } from '../state/app';
import { setSettings } from '../state/projects';
import { Icon } from './Icon';
import { replaceInInput } from './Pron';

export const tagsOn = computed(() => !!backend.value?.tags);

/** An older server doesn't say whether its checkpoint was trained on tags (one that wasn't offers none). */
export const tagsNote = computed(() => (backend.value?.tags?.trained === null ? 'Only the NV checkpoint was trained on these tags; other models read them as text.' : null));

const tagTitle = (name: TagName) => `<${name}/>: ${TAG_INFO[name].note}`;

// ── Palette ───────────────────────────────────────────────────────────────────────────────────────

interface PaletteState {
  x: number;
  y: number;
  text: string;
  start: number;
  end: number;
  apply: (e: Edit) => void;
  /** Opened by typing `<`: Escape types it after all. */
  literal: (() => void) | null;
  returnFocus: HTMLElement | null;
}

const palette = signal<PaletteState | null>(null);

/** The palette for a text field's selection (or caret). */
export function openTagPalette(el: HTMLInputElement | HTMLTextAreaElement, fromKey = false) {
  const start = el.selectionStart ?? el.value.length;
  const end = el.selectionEnd ?? start;
  const r = el.getBoundingClientRect();
  palette.value = {
    x: r.left,
    y: r.bottom + 6,
    text: el.value,
    start,
    end,
    apply: (e) => {
      replaceInInput(el, e.start, e.end, e.insert);
      el.setSelectionRange(e.caret, e.caret);
    },
    literal: fromKey ? () => replaceInInput(el, start, end, '<') : null,
    returnFocus: el,
  };
}

/** The palette for a stretch of text that isn't a field (Direct's painted line): edits come back as whole texts. */
export function openTagPaletteAt(anchor: DOMRect, text: string, start: number, end: number, set: (next: string) => void) {
  palette.value = {
    x: anchor.left,
    y: anchor.bottom + 6,
    text,
    start,
    end,
    apply: (e) => set(text.slice(0, e.start) + e.insert + text.slice(e.end)),
    literal: null,
    returnFocus: null,
  };
}

// `<` in a field marked data-tags opens the palette (capture: before the field's own keys).
addEventListener(
  'keydown',
  (e) => {
    const el = e.target as HTMLElement;
    if (e.key !== '<' || e.metaKey || e.ctrlKey || e.altKey || !tagsOn.value) return;
    if (!(el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement) || !el.hasAttribute('data-tags')) return;
    e.preventDefault();
    openTagPalette(el, true);
  },
  true,
);

export function TagPalette() {
  const st = palette.value;
  return st ? <Palette key={`${st.x}:${st.y}:${st.start}:${st.end}`} st={st} /> : null;
}

function Palette({ st }: { st: PaletteState }) {
  const box = useRef<HTMLDivElement>(null);
  const hasWords = st.end > st.start && /\S/.test(st.text.slice(st.start, st.end));
  // The tags this server reads (all seven on the NV runtime).
  const offered = new Set(backend.value?.tags?.tags ?? TAG_NAMES);
  useLayoutEffect(() => {
    box.current?.querySelector<HTMLElement>('button:not([disabled])')?.focus();
    const outside = (e: PointerEvent) => !box.current?.contains(e.target as Node) && close(false);
    window.addEventListener('pointerdown', outside);
    return () => window.removeEventListener('pointerdown', outside);
  }, []);
  const close = (refocus = true) => {
    palette.value = null;
    if (refocus) st.returnFocus?.focus();
  };
  const point = (name: TagName) => {
    close();
    st.apply(insertPoint(st.text, st.start, name));
  };
  const span = (name: TagName) => {
    const e = wrapSpan(st.text, st.start, st.end, name);
    close();
    if (e) st.apply(e);
  };
  const onKey = (e: KeyboardEvent) => {
    e.stopPropagation();
    const items = [...box.current!.querySelectorAll<HTMLElement>('button:not([disabled])')];
    const i = items.indexOf(document.activeElement as HTMLElement);
    const go = (j: number) => items[(j + items.length) % items.length]?.focus();
    if (e.key === 'ArrowDown' || e.key === 'ArrowRight') go(i + 1);
    else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') go(i - 1);
    else if (e.key === 'Home') go(0);
    else if (e.key === 'End') go(items.length - 1);
    else if (e.key === 'Escape') {
      close();
      st.literal?.();
    } else if (e.key === 'Tab') return close();
    else if (/^[a-z]$/.test(e.key)) {
      // First letter jumps to the tag (again: the next one with that letter).
      const from = i + 1;
      const hit = [...items.slice(from), ...items.slice(0, from)].find((b) => b.dataset.tag?.startsWith(e.key));
      hit?.focus();
    } else return;
    e.preventDefault();
  };
  const left = clamp(st.x, 12, window.innerWidth - 332);
  const top = st.y + 230 > window.innerHeight ? Math.max(12, st.y - 280) : st.y;
  return (
    <div ref={box} class="tag-palette" style={{ left: `${left}px`, top: `${top}px` }} role="dialog" aria-label="Nonverbal tags" onKeyDown={onKey}>
      <span class="eyebrow">At the caret</span>
      <div class="tag-palette-row" role="group" aria-label="Point tags">
        {TAG_NAMES.filter((n) => offered.has(n)).map((n) => (
          <button key={n} class={`tag-pick${TAG_INFO[n].weak ? ' is-weak' : ''}`} data-tag={n} onClick={() => point(n)} title={tagTitle(n)}>
            {n}
            {TAG_INFO[n].weak && <span class="tag-weak">weak</span>}
          </button>
        ))}
      </div>
      <span class="eyebrow">Over the selected words</span>
      <div class="tag-palette-row" role="group" aria-label="Span tags">
        {SPAN_TAG_NAMES.filter((n) => offered.has(n)).map((n) => (
          <button key={n} class="tag-pick is-span is-weak" data-tag={`${n} span`} disabled={!hasWords} onClick={() => span(n)} title={hasWords ? `<${n}>…</${n}>: ${TAG_INFO[n].note}` : 'Select words first'}>
            ‹{n}›
            <span class="tag-weak">weak</span>
          </button>
        ))}
      </div>
      {tagsNote.value && <p class="tag-note">{tagsNote.value}</p>}
      <p class="hint">Type a letter to jump · Esc {st.literal ? 'types “<”' : 'closes'}</p>
    </div>
  );
}

/** The `<>` button beside a field: the palette for its selection. */
export function TagButton({ input }: { input: () => HTMLInputElement | HTMLTextAreaElement | null }) {
  if (!tagsOn.value) return null;
  return (
    <button
      class="icon-btn sm nv-btn"
      onMouseDown={(e) => e.preventDefault()}
      onClick={() => {
        const el = input();
        if (el) openTagPalette(el);
      }}
      aria-label="Insert a nonverbal tag"
      title="Nonverbal tag (type “<” in the text)"
    >
      <span aria-hidden="true" class="nv-btn-glyph">
        ‹›
      </span>
    </button>
  );
}

// ── Chips ─────────────────────────────────────────────────────────────────────────────────────────

/** A tag's actions: remove it, or turn a point into a span over the next word (and back). */
function convert(text: string, t: TagToken, span: TagSpan | undefined): string | null {
  if (span) return spanToPoint(text, span);
  return pointToSpan(text, t);
}

/** The tags of `text` as chips with remove / convert, and markup the server will read as text. */
export function TagMarks({ text, replace }: { text: string; replace?: (next: string) => void }) {
  const parsed = parseTags(text);
  if (!parsed.tokens.length && !parsed.issues.length) return null;
  if (!tagsOn.value) {
    return parsed.tokens.length ? (
      <span class="tag-marks">
        <span class="tag-off" title="The model this server runs wasn’t trained on nonverbal tags (or the server has no tag support): the markup is read as text.">‹tags› read as text</span>
      </span>
    ) : null;
  }
  const heads = parsed.tokens.filter((t) => t.kind !== 'close');
  return (
    <span class="tag-marks" aria-label="Nonverbal tags">
      {heads.map((t) => {
        const span = parsed.spans.find((s) => s.open.start === t.start);
        const words = span ? text.slice(span.open.end, span.close.start).trim() : '';
        const next = replace ? convert(text, t, span) : null;
        return (
          <span key={t.start} class={`tag-chip${span ? ' is-span' : ''}${TAG_INFO[t.name].weak ? ' is-weak' : ''}`} title={`${span ? `<${t.name}> over “${words}”` : `<${t.name}/>`}: ${TAG_INFO[t.name].note}`}>
            <span class="tag-chip-name">{span ? `‹${t.name}› ${words.length > 18 ? `${words.slice(0, 17)}…` : words}` : t.name}</span>
            {replace && next !== null && (
              <button class="tag-chip-act" onClick={() => replace(next)} aria-label={span ? `Make <${t.name}> a point tag` : `Make <${t.name}/> a span over the next word`} title={span ? 'Make it a point tag' : 'Make it a span over the next word'}>
                ⇄
              </button>
            )}
            {replace && (
              <button class="tag-chip-act" onClick={() => replace(removeTag(text, t))} aria-label={`Remove the ${t.name} tag`} title="Remove the tag">
                <Icon name="x" size={10} />
              </button>
            )}
          </span>
        );
      })}
      {parsed.issues.map((x) => (
        <span key={x.at} class="tag-issue" role="note">
          {x.text} (read as text)
        </span>
      ))}
    </span>
  );
}

// ── Resting field overlay ─────────────────────────────────────────────────────────────────────────

/**
 * A text field whose markup shows as chips while it rests (no raw angle brackets); focusing it shows the text
 * as typed. The overlay is visual only (the field takes every click).
 */
export function TagField({ text, children, block }: { text: string; children: ComponentChildren; block?: boolean }) {
  const [focused, setFocused] = useState(false);
  const wrap = useRef<HTMLSpanElement>(null);
  const [box, setBox] = useState<Record<string, string>>({});
  const viewing = tagsOn.value && !focused && parseTags(text).tokens.length > 0;
  useLayoutEffect(() => {
    if (!viewing) return;
    const field = wrap.current?.querySelector('input, textarea');
    if (!field) return;
    const cs = getComputedStyle(field);
    // A single-line field centres its line: the overlay's line is the field's content height.
    const inner = field.clientHeight - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom);
    const lineHeight = field instanceof HTMLInputElement ? `${inner}px` : cs.lineHeight;
    setBox({ padding: cs.padding, font: cs.font, lineHeight, letterSpacing: cs.letterSpacing });
  }, [viewing]);
  return (
    <span ref={wrap} class={`tag-field${block ? ' is-block' : ''}${viewing ? ' is-viewing' : ''}`} onFocusIn={() => setFocused(true)} onFocusOut={() => setFocused(false)}>
      {children}
      {viewing && (
        <span class="tag-view" style={box} aria-hidden="true">
          <TaggedText text={text} />
        </span>
      )}
    </span>
  );
}

/** Text with its tags as inline chips (span tags: a chip before the words, which are marked). */
export function TaggedText({ text }: { text: string }) {
  const { tokens, spans } = parseTags(text);
  const out: ComponentChildren[] = [];
  let pos = 0;
  for (const t of tokens) {
    if (t.start > pos) {
      const inside = spans.find((s) => pos >= s.open.end && pos < s.close.start);
      out.push(<span class={inside ? `tag-words is-${inside.name}` : undefined}>{text.slice(pos, t.start)}</span>);
    }
    if (t.kind !== 'close') out.push(<span class={`tag-inline${t.kind === 'open' ? ' is-span' : ''}`}>{t.kind === 'open' ? `‹${t.name}›` : t.name}</span>);
    pos = t.end;
  }
  if (pos < text.length) out.push(text.slice(pos));
  return <>{out}</>;
}

// ── Tag strength ──────────────────────────────────────────────────────────────────────────────────

/** The project's tag strength: tag guidance for lines with tags (1 = off, then voice guidance as usual). */
export function TagStrength() {
  const t = backend.value?.tags;
  if (!t) return null;
  const v = projectSettings.value.tagStrength;
  const lo = Math.max(1, t.min_scale);
  return (
    <label class="lang-row tag-strength" title="Tag guidance for lines with nonverbal tags: stronger tags (laugh most). It replaces voice guidance on those lines (the server allows one at a time); 1 turns it off">
      tag strength
      <input type="range" min={lo} max={t.max_scale} step={0.1} value={v} onInput={(e) => setSettings({ tagStrength: Number(e.currentTarget.value) })} aria-label="Tag strength" aria-valuetext={v > 1 ? `${v.toFixed(1)}, tag guidance on` : '1, off'} />
      <span class="mono">{v > 1 ? v.toFixed(1) : 'off'}</span>
    </label>
  );
}

/** What a request did about guidance, for takes: tag guidance replaces voice guidance. */
export function guidanceNote(req: { tag_cfg_scale?: number }): string | null {
  return (req.tag_cfg_scale ?? 1) > 1 ? `tag guidance ${req.tag_cfg_scale} on · voice guidance off` : null;
}
