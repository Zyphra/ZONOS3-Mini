/**
 * The lite Script's grammar: one tolerant, range-aware parse that the compiler, the editor's coloring and its
 * completion all read. Offsets are UTF-16 indices into the whole script.
 */

export const DIRECTIVES: readonly string[] = ['language', 'mode', 'seed', 'laci', 'decoder', 'tags', 'gap', 'baseline', 'voice'];
export const NUMBER = /^[+-]?(\d+(\.\d*)?|\.\d+)$/;
export const VOICE = /^([^:{}\[\]=@#,\\]+?)\s*(?:=\s*([^{}]+?))?\s*(?:\{([^{}]*)\})?$/;
/** After a voice name at the start of a line: optional `{delivery}`, then the colon. */
const PREFIX = /^\s*(?:\{([^{}]*)\})?\s*:/;
const SPAN = /\[([^\[\]]*)\]\{([^{}]*)\}/g;
export const VOICE_NAME_MAX = 40;
/** The characters VOICE's name group excludes. */
export const NAME_FORBIDDEN = /[:{}\[\]=@#,\\]/;

export interface Loc {
  from: number;
  to: number;
}
export interface Word extends Loc {
  text: string;
}
/** One comma-separated piece of an item list, trimmed: `head arg…`. */
export interface Item extends Loc {
  text: string;
  head: Word;
  args: Word[];
}
/** The inside of `{…}` (from/to: the body between the braces), or a @baseline argument (no braces). */
export interface ItemList extends Loc {
  open: number | null;
  close: number | null;
  items: Item[];
}
/** A @voice argument, scanned leniently; `valid`: VOICE accepts it (the compiler's rule). */
export interface VoiceDecl {
  valid: boolean;
  name: Word;
  eq: number | null;
  ref: Word | null;
  items: ItemList | null;
}
/** `Name {…}:` / `Name:` / `{…}:` at the start of a speech line. */
export interface Prefix {
  name: Word | null;
  items: ItemList | null;
  colon: number;
}
/** An unclosed `{` at a speech line's start (optionally after a voice name), running to the line end: completion and coloring only. */
export interface Draft {
  name: Word | null;
  items: ItemList;
}
/** `[words]{items}`; for a draft span, items.close is null and `to` is the line end. */
export interface SpanMark extends Loc {
  words: Word;
  items: ItemList;
}
interface LineBase {
  /** 1-based. */
  line: number;
  /** The whole line. */
  raw: Loc;
  /** The line trimmed. */
  loc: Loc;
}
export type ScriptLine =
  | (LineBase & { kind: 'blank' })
  | (LineBase & { kind: 'comment' })
  | (LineBase & { kind: 'directive'; at: Word; name: string | null; arg: Word; voice: VoiceDecl | null; items: ItemList | null })
  | (LineBase & {
      kind: 'speech';
      escape: number | null;
      prefix: Prefix | null;
      draft: Draft | null;
      stray: Word | null;
      text: Word;
      spans: SpanMark[];
      draftSpan: SpanMark | null;
    });

const isSpace = (c: string) => /\s/.test(c);

/** The trimmed word inside [from, to); empty: at `to`. */
function word(text: string, from: number, to: number): Word {
  let a = from;
  let b = to;
  while (a < b && isSpace(text[a])) a++;
  while (b > a && isSpace(text[b - 1])) b--;
  return a < b ? { from: a, to: b, text: text.slice(a, b) } : { from: to, to, text: '' };
}

function words(text: string, from: number, to: number): Word[] {
  return [...text.slice(from, to).matchAll(/\S+/g)].map((m) => ({ from: from + m.index, to: from + m.index + m[0].length, text: m[0] }));
}

function itemList(text: string, from: number, to: number, open: number | null, close: number | null): ItemList {
  const items: Item[] = [];
  let start = from;
  for (let i = from; i <= to; i++) {
    if (i < to && text[i] !== ',') continue;
    const ws = words(text, start, i);
    if (ws.length) {
      const a = ws[0].from;
      const b = ws[ws.length - 1].to;
      items.push({ from: a, to: b, text: text.slice(a, b), head: ws[0], args: ws.slice(1) });
    }
    start = i + 1;
  }
  return { from, to, open, close, items };
}

function scanVoice(text: string, arg: Word): VoiceDecl {
  const s = arg.text;
  const lb = s.indexOf('{');
  const head = lb < 0 ? s : s.slice(0, lb);
  const eqAt = head.indexOf('=');
  const eq = eqAt < 0 ? null : arg.from + eqAt;
  const headEnd = lb < 0 ? arg.to : arg.from + lb;
  let items: ItemList | null = null;
  if (lb >= 0) {
    const open = arg.from + lb;
    const rb = s.indexOf('}', lb + 1);
    const close = rb < 0 ? null : arg.from + rb;
    items = itemList(text, open + 1, close ?? arg.to, open, close);
  }
  return {
    valid: VOICE.test(s),
    name: word(text, arg.from, eq ?? headEnd),
    eq,
    ref: eq === null ? null : word(text, eq + 1, headEnd),
    items,
  };
}

/** `{…}` of a prefix match `m` (PREFIX, or `{…}:`) found at `base`. */
function braces(text: string, base: number, m: RegExpExecArray): ItemList | null {
  if (m[1] === undefined) return null;
  const open = base + m[0].indexOf('{');
  const close = open + 1 + m[1].length;
  return itemList(text, open + 1, close, open, close);
}

function speechLine(text: string, base: LineBase, candidates: readonly string[]): ScriptLine {
  const { loc } = base;
  const escape = text[loc.from] === '\\' ? loc.from : null;
  const s0 = escape === null ? loc.from : loc.from + 1;
  const rest = text.slice(s0, loc.to);
  /** The first candidate `rest` starts with (ignoring case) that `re` matches after: its Word, where `re` ran, the match. */
  const afterName = (re: RegExp): { name: Word; at: number; m: RegExpExecArray } | null => {
    for (const c of candidates) {
      if (rest.slice(0, c.length).toLowerCase() !== c.toLowerCase()) continue;
      const m = re.exec(rest.slice(c.length));
      if (m) return { name: { from: s0, to: s0 + c.length, text: text.slice(s0, s0 + c.length) }, at: s0 + c.length, m };
    }
    return null;
  };
  let prefix: Prefix | null = null;
  let draft: Draft | null = null;
  let stray: Word | null = null;
  if (escape === null) {
    const hit = afterName(PREFIX);
    if (hit) prefix = { name: hit.name, items: braces(text, hit.at, hit.m), colon: hit.at + hit.m[0].length - 1 };
    const m = prefix ? null : /^\{([^{}]*)\}\s*:/.exec(rest);
    if (m) prefix = { name: null, items: braces(text, s0, m), colon: s0 + m[0].length - 1 };
    if (!prefix) {
      if (/^\{[^{}]*$/.test(rest)) draft = { name: null, items: itemList(text, s0 + 1, loc.to, s0, null) };
      else {
        const d = afterName(/^\s*\{[^{}]*$/);
        if (d) {
          const open = d.at + d.m[0].indexOf('{');
          draft = { name: d.name, items: itemList(text, open + 1, loc.to, open, null) };
        }
      }
      const w = /^([\p{L}\p{N}_'’-]+)\s*:\s/u.exec(rest);
      if (w) stray = { from: s0, to: s0 + w[1].length, text: w[1] };
    }
  }
  const said = word(text, prefix ? prefix.colon + 1 : s0, loc.to);
  const spans: SpanMark[] = [];
  let tail = 0;
  for (const m of said.text.matchAll(SPAN)) {
    const a = said.from + m.index;
    const w = m[1];
    const end = a + m[0].length;
    spans.push({ from: a, to: end, words: { from: a + 1, to: a + 1 + w.length, text: w }, items: itemList(text, a + w.length + 3, end - 1, a + w.length + 2, end - 1) });
    tail = m.index + m[0].length;
  }
  let draftSpan: SpanMark | null = null;
  const d = /\[([^\[\]]*)\]\{([^{}]*)$/.exec(said.text.slice(tail));
  if (d) {
    const a = said.from + tail + d.index;
    const w = d[1];
    draftSpan = { from: a, to: loc.to, words: { from: a + 1, to: a + 1 + w.length, text: w }, items: itemList(text, a + w.length + 3, loc.to, a + w.length + 2, null) };
  }
  return { ...base, kind: 'speech', escape, prefix, draft, stray, text: said, spans, draftSpan };
}

/** Each line of `text`; `speakers`: the library's names a line may start with (`speakerNames`). */
export function parseScript(text: string, speakers: readonly string[]): ScriptLine[] {
  const out: ScriptLine[] = [];
  const speech: { i: number; base: LineBase }[] = [];
  const voices: string[] = [];
  let start = 0;
  text.split('\n').forEach((line, i) => {
    const raw = { from: start, to: start + line.length };
    start = raw.to + 1;
    const w = word(text, raw.from, raw.to);
    const base: LineBase = { line: i + 1, raw, loc: { from: w.from, to: w.to } };
    const t = w.text;
    if (!t) return void out.push({ ...base, kind: 'blank' });
    if (t[0] === '#') return void out.push({ ...base, kind: 'comment' });
    if (t[0] !== '@') {
      // Parsed once every @voice name is known (a voice may be defined after its first line); a placeholder till then.
      speech.push({ i: out.length, base });
      return void out.push({ ...base, kind: 'blank' });
    }
    const m = /^@([a-z]+)\b/i.exec(t);
    const atText = m ? m[0] : '@' + t.slice(1).split(/\s/)[0];
    const at: Word = { from: w.from, to: w.from + atText.length, text: atText };
    let s = at.to;
    while (s < raw.to && isSpace(text[s])) s++;
    const arg = s > w.to ? { from: s, to: s, text: '' } : word(text, s, w.to);
    const name = m ? m[1].toLowerCase() : null;
    const voice = name === 'voice' ? scanVoice(text, arg) : null;
    const items = name === 'baseline' ? itemList(text, arg.from, arg.to, null, null) : null;
    if (voice?.valid && voice.name.text.length <= VOICE_NAME_MAX) voices.push(voice.name.text);
    out.push({ ...base, kind: 'directive', at, name, arg, voice, items });
  });
  // The names the compiler can resolve, longest first (stable: voices, then labels, then ids).
  const candidates = [...voices, ...speakers].sort((x, y) => y.length - x.length);
  for (const { i, base } of speech) out[i] = speechLine(text, base, candidates);
  return out;
}

/** The library's names a speech line may start with: labels, then ids. */
export function speakerNames(speakers: readonly { label: string; id: string }[]): string[] {
  return [...speakers.map((s) => s.label), ...speakers.map((s) => s.id)];
}

/** The comma piece of `list` up to `caret`: its words, and whether it ends in whitespace. */
export function pieceAt(text: string, list: ItemList, caret: number): { words: Word[]; trailing: boolean } {
  const from = Math.max(list.from, text.lastIndexOf(',', caret - 1) + 1);
  return { words: words(text, from, caret), trailing: /\s$/.test(text.slice(from, caret)) };
}
