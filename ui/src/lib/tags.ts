/**
 * Nonverbal (NV) markup in a line's text, as the NV checkpoint reads it: point tags `<laugh/>` and span tags over
 * words `<shouting>…</shouting>`. Seven names; anything else in angle brackets, and an open tag without its close
 * (or the reverse), is plain text to the server. The markup travels in the text as typed (it survives text
 * normalization), so painted spans and pronunciation groups keep their offsets on the sent text.
 */

export const TAG_NAMES = ['laugh', 'breath', 'sigh', 'cough', 'panting', 'shouting', 'crying'] as const;
export type TagName = (typeof TAG_NAMES)[number];

/** Tags the palette also offers over words. */
export const SPAN_TAG_NAMES: readonly TagName[] = ['shouting', 'crying'];

/** What the NV model was measured to do with each tag: honest hints for the palette. */
export const TAG_INFO: Record<TagName, { label: string; weak: boolean; note: string }> = {
  laugh: { label: 'laugh', weak: false, note: 'a laugh; clearest with tag strength above 1' },
  breath: { label: 'breath', weak: false, note: 'an audible breath' },
  sigh: { label: 'sigh', weak: false, note: 'a sigh' },
  cough: { label: 'cough', weak: true, note: 'weak: the model barely reacts to it (measured ≈ no effect)' },
  panting: { label: 'panting', weak: false, note: 'panting breaths' },
  shouting: { label: 'shouting', weak: true, note: 'weak: measured no better than without the tag' },
  crying: { label: 'crying', weak: true, note: 'weak: measured no better than without the tag' },
};

export interface TagToken {
  start: number;
  end: number;
  name: TagName;
  kind: 'point' | 'open' | 'close';
}

/** A span tag: the open and close tokens and the words between them. */
export interface TagSpan {
  name: TagName;
  open: TagToken;
  close: TagToken;
}

export interface ParsedTags {
  /** Recognized tokens in text order: points and the open / close of well-formed spans. */
  tokens: TagToken[];
  spans: TagSpan[];
  /** Unmatched opens / closes (read as plain text by the server). */
  issues: { at: number; text: string }[];
}

/** `<laugh />` is read as `<laugh/>` (the server canonicalizes the space away). */
const MARKUP = /<(\/?)([a-z]+)\s*(\/?)>/g;
const NAMES = new Set<string>(TAG_NAMES);

export function parseTags(text: string, names: ReadonlySet<string> = NAMES): ParsedTags {
  const tokens: TagToken[] = [];
  const spans: TagSpan[] = [];
  const issues: ParsedTags['issues'] = [];
  const open: TagToken[] = [];
  for (const m of text.matchAll(MARKUP)) {
    const [all, slash, name, selfClose] = m;
    if (!names.has(name) || (slash && selfClose)) continue;
    const t: TagToken = { start: m.index!, end: m.index! + all.length, name: name as TagName, kind: selfClose ? 'point' : slash ? 'close' : 'open' };
    if (t.kind === 'point') tokens.push(t);
    else if (t.kind === 'open') open.push(t);
    else {
      const k = open.findLastIndex((o) => o.name === t.name);
      if (k < 0) {
        issues.push({ at: t.start, text: `</${t.name}> closes no <${t.name}>` });
        continue;
      }
      const [o] = open.splice(k, 1);
      tokens.push(o, t);
      spans.push({ name: t.name, open: o, close: t });
    }
  }
  for (const o of open) issues.push({ at: o.start, text: `<${o.name}> is never closed` });
  tokens.sort((a, b) => a.start - b.start);
  return { tokens, spans, issues };
}

/** Whether the text holds any markup the NV model reads. */
export const hasTags = (text: string) => parseTags(text).tokens.length > 0;

/** The text without its recognized markup (spaces around a removed tag collapse): what is spoken. */
export function stripTags(text: string): string {
  const { tokens } = parseTags(text);
  if (!tokens.length) return text;
  let out = '';
  let pos = 0;
  for (const t of tokens) {
    out += text.slice(pos, t.start);
    pos = t.end;
  }
  return (out + text.slice(pos)).replace(/ {2,}/g, ' ').replace(/ ([,.!?;:])/g, '$1').trim();
}

/** One message about markup the server would read as plain text, or null. */
export function tagProblem(text: string): string | null {
  const { issues } = parseTags(text);
  return issues.length ? `Tags: ${issues.map((x) => x.text).join('; ')} (read as plain text).` : null;
}

export interface Edit {
  /** Replace text[start, end) with `insert`. */
  start: number;
  end: number;
  insert: string;
  /** Where the caret goes after the edit (offset in the new text). */
  caret: number;
}

const isWordChar = (c: string | undefined) => !!c && /[\p{L}\p{N}'’)]/u.test(c);

/** A point tag at `at`, padded with a space where it would touch a word. */
export function insertPoint(text: string, at: number, name: TagName): Edit {
  const tag = `<${name}/>`;
  const before = isWordChar(text[at - 1]) || /[,.!?;:]/.test(text[at - 1] ?? '') ? ' ' : '';
  const after = isWordChar(text[at]) ? ' ' : '';
  const insert = before + tag + after;
  return { start: at, end: at, insert, caret: at + insert.length };
}

/** Grow [start, end) to whole words (no surrounding spaces). */
export function wordRange(text: string, start: number, end: number): [number, number] {
  let s = Math.max(0, Math.min(start, end));
  let e = Math.min(text.length, Math.max(start, end));
  while (s < e && /\s/.test(text[s])) s++;
  while (e > s && /\s/.test(text[e - 1])) e--;
  while (s > 0 && isWordChar(text[s - 1])) s--;
  while (e < text.length && isWordChar(text[e])) e++;
  return [s, e];
}

/** A span tag around the words of [start, end) (grown to whole words); null when there are no words there. */
export function wrapSpan(text: string, start: number, end: number, name: TagName): Edit | null {
  const [s, e] = wordRange(text, start, end);
  if (e <= s) return null;
  const insert = `<${name}>${text.slice(s, e)}</${name}>`;
  return { start: s, end: e, insert, caret: s + insert.length };
}

const tidy = (text: string) => text.replace(/ {2,}/g, ' ').replace(/^ | $/g, '');

/** The text without one tag (a span tag loses both ends; its words stay). */
export function removeTag(text: string, t: TagToken): string {
  const { spans } = parseTags(text);
  const span = t.kind === 'point' ? null : spans.find((s) => s.open.start === t.start || s.close.start === t.start);
  if (!span) return tidy(text.slice(0, t.start) + text.slice(t.end));
  return tidy(text.slice(0, span.open.start) + text.slice(span.open.end, span.close.start) + text.slice(span.close.end));
}

/** A span tag as a point tag where its words start (the words stay, untagged). */
export function spanToPoint(text: string, span: TagSpan): string {
  const inner = text.slice(span.open.end, span.close.start);
  return text.slice(0, span.open.start) + `<${span.name}/> ` + inner.replace(/^ /, '') + text.slice(span.close.end);
}

/** A point tag as a span tag over the word after it (or before it, at the end of the text); null without one. */
export function pointToSpan(text: string, t: TagToken): string | null {
  const rest = text.slice(0, t.start) + text.slice(t.end);
  const at = t.start;
  let s = at;
  while (s < rest.length && /\s/.test(rest[s])) s++;
  let [ws, we] = wordRange(rest, s, s + 1);
  if (we <= ws || !isWordChar(rest[ws])) {
    [ws, we] = wordRange(rest, Math.max(0, at - 1), at);
    if (we <= ws || !isWordChar(rest[ws])) return null;
  }
  return tidy(rest.slice(0, ws) + `<${t.name}>${rest.slice(ws, we)}</${t.name}>` + rest.slice(we));
}
