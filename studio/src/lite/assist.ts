/**
 * What the Script editor suggests at the caret and how it colors the script, read off the parse in `./syntax`.
 */
import type { AccentSpec, ControlSpec } from '../api';
import { decoderName } from '../lib/decoders';
import { LACI_LABEL, LACI_MODES } from '../lib/laci';
import { langLabel } from '../lib/languages';
import { parsePron } from '../lib/pron';
import { DEFAULT_GAP, GAP_RANGE } from '../state/lines';
import { NO_STEERING, type ScriptEnv } from './script';
import { DIRECTIVES, NAME_FORBIDDEN, NUMBER, parseScript, pieceAt, speakerNames, VOICE_NAME_MAX, type ItemList } from './syntax';

export type AssistEnv = Pick<ScriptEnv, 'spec' | 'steering' | 'speakers' | 'languages' | 'caps'>;
export type ItemKind = 'directive' | 'value' | 'control' | 'keyword' | 'preset' | 'accent' | 'voice';

export interface CompletionItem {
  label: string;
  /** Replaces the completion's [from, to); never holds a newline. */
  insert: string;
  detail: string;
  /** More about the item, shown while it is the active one. */
  info?: string;
  kind: ItemKind;
  /** A theme hue (`c1`…): the control's group. */
  hue?: string;
  /** Where the caret lands in `insert`; default: its end. */
  caret?: number;
}

export interface Completion {
  from: number;
  to: number;
  items: CompletionItem[];
  /** What the caret's place takes, when there is nothing to pick there. */
  hint: string | null;
}

export type TokenClass = 'comment' | 'escape' | 'directive' | 'voice' | 'ref' | 'value' | 'punct' | 'key' | 'num' | 'span' | 'pron';
export const TOKEN_CLASSES: readonly TokenClass[] = ['comment', 'escape', 'directive', 'voice', 'ref', 'value', 'punct', 'key', 'num', 'span', 'pron'];

export interface Token {
  from: number;
  to: number;
  cls: TokenClass;
}

/** An item and the names typed text matches it by. */
type Named = [CompletionItem, readonly string[]];
/** Where an item list sits: on a @voice, over a whole line (or @baseline), over words. */
type Context = 'voice' | 'line' | 'span';

const num = (n: number) => String(n).replace('-', '−');

const DIRECTIVE_DETAIL: Record<string, string> = {
  language: 'language of every line · default: Language above',
  mode: 'embedding | reference for library voices',
  seed: 'one seed for every line · default: Advanced → Seed',
  laci: 'off | stream | whole · default: Advanced → LACI',
  decoder: 'an audio decoder of this server',
  gap: `silence between lines, ${GAP_RANGE[0]}–${GAP_RANGE[1]} s`,
  baseline: 'delivery for every line',
  voice: 'a designed voice: @voice Name = library voice {…}',
};

const MODES: Named[] = [
  [{ label: 'embedding', insert: 'embedding', detail: 'the voice embedding: every control applies', kind: 'value' }, ['embedding']],
  [{ label: 'reference', insert: 'reference', detail: 'the recording: only delivery and mood apply', kind: 'value' }, ['reference']],
];

/**
 * The items whose names match `q`: 0 for a name starting with it (or an empty `q`), 1 for a name holding it (unless
 * `prefixOnly`), in that order; others drop.
 */
function rank(items: readonly Named[], q: string, prefixOnly = false): CompletionItem[] {
  const lq = q.toLowerCase();
  const kept: [CompletionItem, number][] = [];
  for (const [item, names] of items) {
    const ns = names.map((n) => n.toLowerCase());
    if (!lq || ns.some((n) => n.startsWith(lq))) kept.push([item, 0]);
    else if (!prefixOnly && ns.some((n) => n.includes(lq))) kept.push([item, 1]);
  }
  return kept.sort((a, b) => a[1] - b[1]).map(([item]) => item);
}

function controlItems(env: AssistEnv, ctx: Context): Named[] {
  if (!env.steering) return [];
  const { ui } = env.spec;
  return ui.controls
    .filter((c) => (ctx === 'voice' ? !c.span_only : ctx === 'line' ? c.spannable && !c.span_only : c.spannable))
    .map((c): Named => {
      const info = [c.poles && `${c.poles[0]} ↔ ${c.poles[1]}`, `safe ${num(c.safe[0])}…${num(c.safe[1])}`, c.span_only && 'words only', c.effect];
      const item: CompletionItem = {
        label: c.id,
        insert: `${c.id} `,
        detail: `${c.label} ${num(c.range[0])}…${num(c.range[1])}`,
        info: info.filter(Boolean).join(' · '),
        kind: 'control',
        hue: ui.groups.find((g) => g.id === c.group)?.hue,
      };
      return [item, [c.id, c.label]];
    });
}

/** The library's voices; `after`: what follows the name when inserted. */
function libraryItems(env: AssistEnv, after: string): Named[] {
  return env.speakers.map((s) => [
    { label: s.label, insert: s.label + after, detail: `${s.scope === 'default' ? 'library voice' : 'this tab'} · ${s.speaker_modes.join(', ')}`, kind: 'voice' },
    [s.label, s.id],
  ]);
}

export function completeAt(text: string, caret: number, env: AssistEnv): Completion | null {
  const lines = parseScript(text, speakerNames(env.speakers));
  const l = lines.find((x) => x.raw.from <= caret && caret <= x.raw.to);
  if (!l || l.kind === 'comment') return null;
  const { spec, steering } = env;
  const served = env.caps.decoders ?? [];
  /** The caret plus what `re` (anchored) matches right after it. */
  const wordEnd = (re: RegExp) => caret + (re.exec(text.slice(caret))?.[0].length ?? 0);
  const done = (from: number, to: number, items: CompletionItem[], hint: string | null = null): Completion | null => {
    const typed = text.slice(from, to);
    const kept = items.filter((i) => i.insert !== typed);
    return kept.length || hint ? { from, to, items: kept, hint } : null;
  };
  const hint = (h: string): Completion => ({ from: caret, to: caret, items: [], hint: h });
  /** `list` when the caret is inside it (an unclosed list runs to the line's end), else null. */
  const listAt = (list: ItemList | null | undefined) => (list && caret >= list.from && (list.close === null || caret <= list.close) ? list : null);

  /** The caret in an item list: an item's head, or what the head takes. */
  const itemList = (list: ItemList, ctx: Context): Completion | null => {
    const { words, trailing } = pieceAt(text, list, caret);
    const w0 = words[0];
    if (!words.length || (words.length === 1 && !trailing)) {
      const items = controlItems(env, ctx);
      if (ctx === 'voice') {
        const keywords: [string, string, boolean][] = [
          ['preset', 'a Studio preset (sets mode embedding)', steering && Object.keys(spec.ui.presets).length > 0],
          ['accent', 'accent id strength', steering && spec.ui.accents.length > 0],
          ['mode', 'embedding | reference', true],
        ];
        for (const [kw, detail, offered] of keywords) if (offered) items.push([{ label: kw, insert: `${kw} `, detail, kind: 'keyword' }, [kw]]);
      } else if (!steering) return hint(NO_STEERING);
      return done(w0?.from ?? caret, wordEnd(/^[^\s,{}]*/), rank(items, w0?.text ?? ''));
    }
    const head = w0.text.toLowerCase();
    const args = words.slice(1);
    const typingArg = !args.length || (args.length === 1 && !trailing);
    const argFrom = args[0]?.from ?? caret;
    if (ctx === 'voice' && head === 'preset') {
      const presets = Object.entries(spec.ui.presets).map(([key, entries]): Named => {
        const detail = Object.entries(entries)
          .map(([k, v]) => (k.startsWith('accent:') ? `accent ${k.slice(7)} ${v}` : `${k} ${v}`))
          .join(', ');
        return [{ label: key, insert: key, detail, kind: 'preset' }, [key]];
      });
      const to = caret + (/^[^,}]*/.exec(text.slice(caret))?.[0].trimEnd().length ?? 0);
      return done(argFrom, to, rank(presets, text.slice(argFrom, caret)));
    }
    if (ctx === 'voice' && head === 'accent') {
      if (!typingArg) {
        const a: AccentSpec | undefined = spec.ui.accents.find((x) => x.id === args[0].text.toLowerCase());
        return a ? hint(`${a.label} strength ${num(a.range[0])}…${num(a.range[1])} · safe ${num(a.safe[0])}…${num(a.safe[1])}`) : null;
      }
      const accents = spec.ui.accents.map((a): Named => {
        const info = `safe ${num(a.safe[0])}…${num(a.safe[1])}` + (a.effect ? ` · ${a.effect}` : '');
        return [{ label: a.id, insert: `${a.id} `, detail: `${a.label} ${num(a.range[0])}…${num(a.range[1])}`, info, kind: 'accent' }, [a.id, a.label]];
      });
      return done(argFrom, wordEnd(/^[^\s,{}]*/), rank(accents, args[0]?.text ?? ''));
    }
    if (ctx === 'voice' && head === 'mode') return typingArg ? done(argFrom, wordEnd(/^[^\s,{}]*/), rank(MODES, args[0]?.text ?? '')) : null;
    const c: ControlSpec | undefined = steering ? spec.ui.controls.find((x) => x.id === head) : undefined;
    if (!c) return null;
    const poles = c.poles ? ` · ${c.poles[0]} ↔ ${c.poles[1]}` : '';
    return hint(`${c.label} ${num(c.range[0])}…${num(c.range[1])} · safe ${num(c.safe[0])}…${num(c.safe[1])}${poles}${c.span_only ? ' · words only' : ''}`);
  };

  /** A voice name starting a speech line: this script's @voice names, then the library's. */
  const voiceNames = (from: number): Completion | null => {
    const q = text.slice(from, caret);
    if (NAME_FORBIDDEN.test(q)) return null;
    const own: Named[] = [];
    const seen = new Set<string>();
    for (const o of lines) {
      const v = o.kind === 'directive' ? o.voice : null;
      if (!v?.valid || v.name.text.length > VOICE_NAME_MAX || seen.has(v.name.text.toLowerCase())) continue;
      seen.add(v.name.text.toLowerCase());
      own.push([{ label: v.name.text, insert: `${v.name.text}: `, detail: `@voice, line ${o.line}`, kind: 'voice' }, [v.name.text]]);
    }
    return done(from, caret, rank([...own, ...libraryItems(env, ': ')], q, true));
  };

  if (l.kind === 'blank') return voiceNames(caret);
  if (l.kind === 'directive') {
    if (l.at.from < caret && caret <= l.at.to) {
      const offered: Record<string, boolean> = { laci: env.caps.laci_languages.length > 0, decoder: served.length > 0, baseline: steering };
      const items = DIRECTIVES.filter((d) => offered[d] !== false && (d === 'voice' || !lines.some((o) => o !== l && o.kind === 'directive' && o.name === d))).map(
        (d): Named => [{ label: `@${d}`, insert: `@${d} `, detail: DIRECTIVE_DETAIL[d], kind: 'directive' }, [d]],
      );
      return done(l.at.from, l.at.to, rank(items, text.slice(l.at.from + 1, caret), true));
    }
    if (caret < l.arg.from) return null;
    const q = text.slice(l.arg.from, caret);
    const to = Math.max(l.arg.to, caret);
    switch (l.name) {
      case 'language':
        return done(l.arg.from, to, rank(env.languages.map((code): Named => [{ label: code, insert: code, detail: langLabel(code), kind: 'value' }, [code, langLabel(code)]]), q));
      case 'mode':
        return done(l.arg.from, to, rank(MODES, q));
      case 'laci':
        if (!env.caps.laci_languages.length) return hint('This server has no LACI alignment heads.');
        return done(l.arg.from, to, rank(LACI_MODES.map((m): Named => [{ label: m, insert: m, detail: LACI_LABEL[m], kind: 'value' }, [m, LACI_LABEL[m]]]), q));
      case 'decoder':
        return done(
          l.arg.from,
          to,
          rank(
            served.map((d, i): Named => [{ label: d.id, insert: d.id, detail: decoderName(d.id) + (i === 0 ? ' · default' : ''), kind: 'value' }, [d.id, decoderName(d.id)]]),
            q,
          ),
        );
      case 'seed':
        return hint('a whole number from 0 to 4294967295');
      case 'gap':
        return hint(`seconds from ${GAP_RANGE[0]} to ${GAP_RANGE[1]} (default ${DEFAULT_GAP})`);
      case 'baseline': {
        const list = listAt(l.items);
        return list && itemList(list, 'line');
      }
      case 'voice': {
        const v = l.voice!;
        const list = listAt(v.items);
        if (list) return itemList(list, 'voice');
        if (v.eq === null || caret <= v.eq || (v.items && caret > v.items.open!)) return null;
        let start = v.eq + 1;
        while (start < caret && /\s/.test(text[start])) start++;
        return done(start, Math.max(v.ref?.to ?? caret, caret), rank(libraryItems(env, ''), text.slice(start, caret)));
      }
      default:
        return null;
    }
  }

  const lists: [ItemList | null | undefined, Context][] = [[l.prefix?.items, 'line'], [l.draft?.items, 'line'], ...l.spans.map((sp): [ItemList, Context] => [sp.items, 'span']), [l.draftSpan?.items, 'span']];
  for (const [list, ctx] of lists) {
    const at = listAt(list);
    if (at) return itemList(at, ctx);
  }

  if (l.escape === null && !l.prefix && !l.draft && caret === l.loc.to) return voiceNames(l.loc.from);
  return null;
}

export function highlight(text: string, env: AssistEnv): Token[] {
  const out: Token[] = [];
  const put = (from: number, to: number, cls: TokenClass) => out.push({ from, to, cls });
  const items = (list: ItemList | null) => {
    if (!list) return;
    if (list.open !== null) put(list.open, list.open + 1, 'punct');
    for (const item of list.items) {
      put(item.head.from, item.head.to, 'key');
      for (const a of item.args) put(a.from, a.to, NUMBER.test(a.text) ? 'num' : 'value');
    }
    if (list.close !== null) put(list.close, list.close + 1, 'punct');
  };
  for (const l of parseScript(text, speakerNames(env.speakers))) {
    if (l.kind === 'comment') put(l.loc.from, l.loc.to, 'comment');
    else if (l.kind === 'directive') {
      put(l.at.from, l.at.to, 'directive');
      if (l.voice) {
        const v = l.voice;
        put(v.name.from, v.name.to, 'voice');
        if (v.eq !== null) put(v.eq, v.eq + 1, 'punct');
        if (v.ref) put(v.ref.from, v.ref.to, 'ref');
        items(v.items);
      } else if (l.items) items(l.items);
      else if (l.name && DIRECTIVES.includes(l.name)) put(l.arg.from, l.arg.to, NUMBER.test(l.arg.text) ? 'num' : 'value');
    } else if (l.kind === 'speech') {
      if (l.escape !== null) put(l.escape, l.escape + 1, 'escape');
      for (const p of [l.prefix, l.draft]) {
        if (p?.name) put(p.name.from, p.name.to, 'voice');
        items(p?.items ?? null);
      }
      if (l.prefix) put(l.prefix.colon, l.prefix.colon + 1, 'punct');
      for (const sp of l.draftSpan ? [...l.spans, l.draftSpan] : l.spans) {
        put(sp.from, sp.from + 1, 'span');
        put(sp.words.to, sp.words.to + 1, 'span');
        items(sp.items);
      }
      const at = l.text.from;
      if (env.caps.phoneme_tokens) for (const g of parsePron(l.text.text, null).groups) put(at + g.start, at + g.end, 'pron');
    }
  }
  out.sort((a, b) => a.from - b.from);
  const kept: Token[] = [];
  for (const t of out) if (t.to > t.from && !(kept.length && t.from < kept[kept.length - 1].to)) kept.push(t);
  return kept;
}
