/**
 * Pronunciation input: X-SAMPA in parentheses, "I met (S@vQn)". Ported from the phoneme branch's tts_ui.html
 * (same table and rules: "_" ties two symbols, symbols without an X-SAMPA spelling are typed as-is, stress /
 * syllable dots / linking / spaces are dropped as in training). Phoneme-token checkpoints read each group as
 * U+E000 IPA U+E001; every offset the studio sends refers to that SENT text.
 */

// [X-SAMPA, IPA]; the first pair for an IPA symbol is its canonical spelling.
const XSAMPA_PAIRS: [string, string][] = [
  ['a', 'a'], ['b', 'b'], ['b_<', 'ɓ'], ['c', 'c'], ['d', 'd'], ['d`', 'ɖ'], ['d_<', 'ɗ'],
  ['e', 'e'], ['f', 'f'], ['g', 'ɡ'], ['g_<', 'ɠ'], ['h', 'h'], ['h\\', 'ɦ'], ['i', 'i'],
  ['j', 'j'], ['j\\', 'ʝ'], ['k', 'k'], ['l', 'l'], ['l`', 'ɭ'], ['l\\', 'ɺ'], ['m', 'm'],
  ['n', 'n'], ['n`', 'ɳ'], ['o', 'o'], ['p', 'p'], ['p\\', 'ɸ'], ['q', 'q'], ['r', 'r'],
  ['r`', 'ɽ'], ['r\\', 'ɹ'], ['r\\`', 'ɻ'], ['s', 's'], ['s`', 'ʂ'], ['s\\', 'ɕ'], ['t', 't'],
  ['t`', 'ʈ'], ['u', 'u'], ['v', 'v'], ['v\\', 'ʋ'], ['P', 'ʋ'], ['w', 'w'], ['x', 'x'],
  ['x\\', 'ɧ'], ['y', 'y'], ['z', 'z'], ['z`', 'ʐ'], ['z\\', 'ʑ'],
  ['A', 'ɑ'], ['B', 'β'], ['B\\', 'ʙ'], ['C', 'ç'], ['D', 'ð'], ['E', 'ɛ'], ['F', 'ɱ'],
  ['G', 'ɣ'], ['G\\', 'ɢ'], ['G\\_<', 'ʛ'], ['H', 'ɥ'], ['H\\', 'ʜ'], ['I', 'ɪ'], ['I\\', 'ᵻ'],
  ['J', 'ɲ'], ['J\\', 'ɟ'], ['J\\_<', 'ʄ'], ['K', 'ɬ'], ['K\\', 'ɮ'], ['L', 'ʎ'], ['L\\', 'ʟ'],
  ['M', 'ɯ'], ['M\\', 'ɰ'], ['N', 'ŋ'], ['N\\', 'ɴ'], ['O', 'ɔ'], ['O\\', 'ʘ'], ['Q', 'ɒ'],
  ['R', 'ʁ'], ['R\\', 'ʀ'], ['S', 'ʃ'], ['T', 'θ'], ['U', 'ʊ'], ['U\\', 'ᵿ'], ['V', 'ʌ'],
  ['W', 'ʍ'], ['X', 'χ'], ['X\\', 'ħ'], ['Y', 'ʏ'], ['Z', 'ʒ'],
  ['@', 'ə'], ['@\\', 'ɘ'], ['@`', 'ɚ'], ['{', 'æ'], ['}', 'ʉ'], ['1', 'ɨ'], ['2', 'ø'],
  ['3', 'ɜ'], ['3\\', 'ɞ'], ['3`', 'ɝ'], ['4', 'ɾ'], ['5', 'ɫ'], ['6', 'ɐ'], ['7', 'ɤ'],
  ['8', 'ɵ'], ['9', 'œ'], ['&', 'ɶ'], ['?', 'ʔ'], ['?\\', 'ʕ'], ['<\\', 'ʢ'], ['>\\', 'ʡ'],
  ['!\\', 'ǃ'], ['|\\', 'ǀ'], ['|\\|\\', 'ǁ'], ['=\\', 'ǂ'],
  [':', 'ː'], [':\\', 'ˑ'], ['_>', 'ʼ'], ['_?\\', 'ˤ'], ['_G', 'ˠ'], ['_h', 'ʰ'], ['_j', 'ʲ'],
  ["'", 'ʲ'], ['_l', 'ˡ'], ['_n', 'ⁿ'], ['_w', 'ʷ'], ['`', '˞'],
  ['_T', '˥'], ['_H', '˦'], ['_M', '˧'], ['_L', '˨'], ['_B', '˩'],
  ['_"', '\u0308'], ['_+', '\u031F'], ['_-', '\u0320'], ['_/', '\u030C'], ['_R', '\u030C'],
  ['_0', '\u0325'], ['=', '\u0329'], ['_=', '\u0329'], ['_\\', '\u0302'], ['_F', '\u0302'],
  ['_^', '\u032F'], ['_}', '\u031A'], ['~', '\u0303'], ['_~', '\u0303'], ['_A', '\u0318'],
  ['_a', '\u033A'], ['_c', '\u031C'], ['_d', '\u032A'], ['_e', '\u0334'], ['_k', '\u0330'],
  ['_m', '\u033B'], ['_N', '\u033C'], ['_O', '\u0339'], ['_o', '\u031E'], ['_q', '\u0319'],
  ['_r', '\u031D'], ['_t', '\u0324'], ['_v', '\u032C'], ['_X', '\u0306'], ['_x', '\u033D'],
  // A lone "_" ties two symbols (t_S = t͡ʃ). Stress, syllable breaks, linking and spaces are dropped.
  ['_', '\u0361'], ['"', ''], ['%', ''], ['.', ''], ['-\\', ''], [' ', ''],
];
const TO_IPA = new Map(XSAMPA_PAIRS);
const TO_XSAMPA = new Map<string, string>();
for (const [x, ipa] of XSAMPA_PAIRS) if (ipa && !TO_XSAMPA.has(ipa)) TO_XSAMPA.set(ipa, x);
const MAX_KEY = Math.max(...XSAMPA_PAIRS.map(([x]) => x.length));

/** IPA marks the model was trained without (dropped from typed-as-is IPA too). */
const DROPPED = /[ˈˌ.\s‿]/gu;

export function xsampaToIpa(text: string): string {
  let out = '';
  for (let i = 0; i < text.length; ) {
    let n = Math.min(MAX_KEY, text.length - i);
    while (n > 0 && !TO_IPA.has(text.slice(i, i + n))) n--;
    if (n) {
      out += TO_IPA.get(text.slice(i, i + n));
      i += n;
    } else out += text[i++];
  }
  return out.normalize('NFC').replace(DROPPED, '');
}

export function ipaToXsampa(ipa: string): string {
  const tokens: string[] = [];
  for (const ch of ipa.normalize('NFC').replace(DROPPED, '')) {
    const x = TO_XSAMPA.get(ch);
    if (x !== undefined) {
      tokens.push(x);
      continue;
    }
    // Precomposed letters (ẽ, ä) spell as base + X-SAMPA diacritics.
    for (const part of ch.normalize('NFD')) tokens.push(TO_XSAMPA.get(part) ?? part);
  }
  let out = '';
  tokens.forEach((token, i) => {
    // A tie written "_" before a symbol that would read as a diacritic ("_m" is laminal) stays a literal tie bar.
    if (token === '_') {
      const ahead = '_' + tokens.slice(i + 1, i + MAX_KEY).join('');
      for (let n = Math.min(MAX_KEY, ahead.length); n > 1; n--) {
        if (TO_IPA.has(ahead.slice(0, n))) {
          token = '\u0361';
          break;
        }
      }
    }
    out += token;
  });
  return out;
}

export interface PronTokens {
  open_marker: string;
  close_marker: string;
  symbols: string[];
}

export interface Group {
  /** Display offsets of the whole "(…)" group. */
  start: number;
  end: number;
  /** The X-SAMPA between the parentheses. */
  body: string;
  ipa: string;
  /** IPA characters without a token on this checkpoint. */
  unknown: string[];
  empty: boolean;
}

export interface PronIssue {
  at: number;
  text: string;
}

export interface Parsed {
  groups: Group[];
  /** Unclosed "(", stray ")", nested "(", empty groups, symbols without a token. */
  issues: PronIssue[];
}

/** The well-formed "(…)" groups of `text`; a "(" without its ")" (or with another "(" first) is flagged. */
export function parsePron(text: string, tokens: PronTokens | null): Parsed {
  const known = tokens ? new Set(tokens.symbols) : null;
  const groups: Group[] = [];
  const issues: PronIssue[] = [];
  let i = 0;
  while (i < text.length) {
    const ch = text[i];
    if (ch === ')') {
      issues.push({ at: i, text: `“)” at ${i + 1} closes no group` });
      i++;
      continue;
    }
    if (ch !== '(') {
      i++;
      continue;
    }
    const close = text.indexOf(')', i + 1);
    const nested = text.indexOf('(', i + 1);
    if (close < 0 || (nested >= 0 && nested < close)) {
      issues.push({ at: i, text: `“(” at ${i + 1} is never closed` });
      i++;
      continue;
    }
    const body = text.slice(i + 1, close);
    const ipa = xsampaToIpa(body);
    const unknown = known ? [...new Set(ipa)].filter((c) => !known.has(c)) : [];
    const g: Group = { start: i, end: close + 1, body, ipa, unknown, empty: !ipa };
    groups.push(g);
    if (g.empty) issues.push({ at: i, text: `“()” at ${i + 1} is empty` });
    else if (unknown.length) issues.push({ at: i, text: `(${body}): no token on this model for ${unknown.map((c) => `“${c}”`).join(' ')}` });
    i = close + 1;
  }
  return { groups, issues };
}

/** One message about what would make the server refuse the text, or null. */
export function pronProblem(text: string, tokens: PronTokens | null): string | null {
  if (!tokens) return null;
  const { issues } = parsePron(text, tokens);
  return issues.length ? `Pronunciation: ${issues.map((x) => x.text).join('; ')}.` : null;
}

export interface Sent {
  text: string;
  /** Display offset → sent offset; a boundary inside a group moves to the group's start (`side` -1) or end (+1). */
  map: (at: number, side: -1 | 1) => number;
}

/** The text as sent: each well-formed group becomes open + IPA + close; everything else is unchanged. */
export function toSent(text: string, tokens: PronTokens | null): Sent {
  if (!tokens) return { text, map: (at) => at };
  const { groups } = parsePron(text, tokens);
  if (!groups.length) return { text, map: (at) => at };
  let out = '';
  let pos = 0;
  const pieces: { d0: number; d1: number; s0: number; s1: number; group: boolean }[] = [];
  for (const g of groups) {
    if (g.start > pos) {
      pieces.push({ d0: pos, d1: g.start, s0: out.length, s1: out.length + g.start - pos, group: false });
      out += text.slice(pos, g.start);
    }
    const sent = g.empty ? text.slice(g.start, g.end) : tokens.open_marker + g.ipa + tokens.close_marker;
    pieces.push({ d0: g.start, d1: g.end, s0: out.length, s1: out.length + sent.length, group: !g.empty });
    out += sent;
    pos = g.end;
  }
  pieces.push({ d0: pos, d1: text.length, s0: out.length, s1: out.length + text.length - pos, group: false });
  out += text.slice(pos);
  return {
    text: out,
    map: (at, side) => {
      for (const p of pieces) {
        if (at < p.d0 || at > p.d1) continue;
        if (!p.group) return p.s0 + (at - p.d0);
        if (at === p.d0) return p.s0;
        if (at === p.d1) return p.s1;
        return side < 0 ? p.s0 : p.s1;
      }
      return out.length;
    },
  };
}

// Rough English spelling of IPA ("ʃəvɔːn" → "shivon"): what a recognizer would likely write for a group.
const GLOSS: Record<string, string> = {
  ʃ: 'sh', ʒ: 'zh', θ: 'th', ð: 'th', ŋ: 'ng', ʧ: 'ch', ʤ: 'j', ə: 'i', ɪ: 'i', i: 'ee', ɛ: 'e', æ: 'a', ɑ: 'a', ɒ: 'o', ɔ: 'o',
  ʊ: 'u', u: 'oo', ʌ: 'u', ɜ: 'er', ɝ: 'er', ɚ: 'er', ɹ: 'r', ɾ: 't', j: 'y', ɡ: 'g', ɫ: 'l', ʔ: '', ɐ: 'a', ø: 'eu', œ: 'eu', y: 'u',
};

/** A plain-letters guess at how a pronunciation is spelled (used where a recognized word is missing). */
export function ipaGloss(ipa: string): string {
  let out = '';
  for (const ch of ipa.normalize('NFD').replace(/\p{M}/gu, '').replace(/t͡?ʃ/g, 'ʧ').replace(/d͡?ʒ/g, 'ʤ')) {
    const g = GLOSS[ch];
    if (g !== undefined) out += g;
    else if (/[a-z]/.test(ch)) out += ch;
  }
  return out || 'uh';
}

/**
 * `text` as it was spoken, in plain words: each group replaced by what the recognizer heard there (`heard[k]` for
 * the k-th group), else a spelling guessed from its IPA. For texts that describe recorded audio (transcripts).
 */
export function spokenText(text: string, heard: (string | null | undefined)[] = []): string {
  const { groups } = parsePron(text, null);
  let out = '';
  let pos = 0;
  groups.forEach((g, k) => {
    if (g.empty) return;
    out += text.slice(pos, g.start) + (heard[k]?.trim() || ipaGloss(g.ipa));
    pos = g.end;
  });
  return out + text.slice(pos);
}
