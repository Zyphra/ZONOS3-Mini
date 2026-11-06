// The texts of a speech edit: edit tokens (zonos3.tts.textmatch.tokens), forced-alignment tokens
// (zonos3.tts.align_text.align_tokens), nonverbal markup and pronunciation groups in a target
// (zonos3.tts.edit_align canonical_markup / target_tokens / token_chars). Offsets and lengths are code points.
import languages from '../../../../model/src/zonos3/registries/languages.json';
import { NONVERBAL_MARKUP_TAGS, pyIsSpace } from '../request/markup';
import { pyLen, pySplit } from './py';

/** Python's str.isspace() code points as a regex class body (markup.ts pyIsSpace). */
const SPACE = '\\t-\\r\\x1c-\\x20\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000';
const MARKER = `</?(?:${NONVERBAL_MARKUP_TAGS.join('|')})[${SPACE}]*/?>`;
/** zonos3.tts.markup _NONVERBAL_MARKUP. */
const MARKER_G = new RegExp(MARKER, 'gu');
const MARKER_ANY = new RegExp(MARKER, 'u');
const MARKER_FULL = new RegExp(`^${MARKER}$`, 'u');
export const GROUP_OPEN = '\ue000', GROUP_CLOSE = '\ue001';
/** A whitespace word's pieces: pronunciation groups, nonverbal markers and the text around them (_PIECE_RE). */
const PIECE = new RegExp(`(\ue000[^\ue000\ue001]*\ue001|${MARKER})`, 'u');
/** Letter-equivalents per IPA symbol of a pronunciation group (edit_align GROUP_LETTERS_PER_SYMBOL). */
export const GROUP_LETTERS_PER_SYMBOL = 1.4;

// textmatch: curly quotes fold to ', '_' separates, Python's \w (letters and numbers of any script) and ' are kept.
const QUOTES = /[\u2018\u2019\u201b\u02bc]/g;
const NON_TOKEN = /[^\p{L}\p{N}']+/gu;

/** textmatch.normalize: upper-cased, everything but word characters and ' one space, stripped. */
export function normalize(text: string): string {
  return text.replace(QUOTES, "'").replaceAll('_', ' ').toUpperCase().replace(NON_TOKEN, ' ').trim();
}

/** textmatch.tokens: the edit tokens of a text (the words of its normalize). */
export function tokens(text: string): string[] {
  return pySplit(normalize(text));
}

/** The Whisper-style codes written without spaces between words (zonos3.languages SPOKEN unspaced: ja, zh, yue). */
const UNSPACED = new Set(languages.spoken.filter(l => 'unspaced' in l && l.unspaced).map(l => l.code));
const CJK_BLOCKS: [number, number][] = [
  [0x3040, 0x30ff], [0x31f0, 0x31ff], [0x3400, 0x4dbf], [0x4e00, 0x9fff], [0xf900, 0xfaff], [0xff66, 0xff9f],
];
const WORDY = /^[\p{L}\p{N}']$/u;

/** align_text.text_units' unit texts: one per CJK character or run of other letters / digits, punctuation attached to
 *  the unit before it (the first unit takes the punctuation ahead of it). */
function unitTexts(text: string): string[] {
  const cps = Array.from(text);
  const spans: [number, number][] = [];
  let lead: number | null = null;
  let runEnd = -1;
  cps.forEach((ch, i) => {
    if (pyIsSpace(ch.codePointAt(0)!)) return;
    if (WORDY.test(ch)) {
      const o = ch.codePointAt(0)!;
      if (CJK_BLOCKS.some(([lo, hi]) => lo <= o && o <= hi)) {
        spans.push([i, i + 1]);
        runEnd = -1;
      } else if (runEnd === i) {
        spans[spans.length - 1][1] = i + 1;
        runEnd = i + 1;
      } else {
        spans.push([i, i + 1]);
        runEnd = i + 1;
      }
    } else if (spans.length) {
      spans[spans.length - 1][1] = i + 1;
    } else {
      lead ??= i;
    }
  });
  if (spans.length && lead !== null) spans[0][0] = lead;
  return spans.map(([a, b]) => cps.slice(a, b).join(''));
}

/** align_text.align_tokens: the words of `text` FuseAlign places one interval each; `lang` is a Whisper-style code.
 *  Unspaced languages take text_units' unit texts (whitespace inside a unit removed), every other the whitespace
 *  words. */
export function alignTokens(text: string, lang: string): string[] {
  if (!UNSPACED.has(lang)) return pySplit(text);
  return unitTexts(text).map(u => pySplit(u).join('')).filter(Boolean);
}

/** `text` with the whitespace inside its nonverbal markers removed (`<laugh />` -> `<laugh/>`). */
export function canonicalMarkup(text: string): string {
  return text.replace(MARKER_G, m => pySplit(m).join(''));
}

export function isGroupToken(token: string): boolean {
  return token.startsWith(GROUP_OPEN) && token.endsWith(GROUP_CLOSE);
}

/** Text length of a target token: characters, or a group's IPA symbols in letter-equivalents (a Python float). */
export function tokenChars(token: string): { value: number; int: boolean } {
  return isGroupToken(token)
    ? { value: GROUP_LETTERS_PER_SYMBOL * (pyLen(token) - 2), int: false }
    : { value: pyLen(token), int: true };
}

/** `[target tokens, tokens of target word 0, of word 1, ...]` over the whitespace words of a canonical target: plain
 *  pieces normalized, each pronunciation group one token, each nonverbal marker none. */
export function targetTokens(targetText: string): string[][] {
  const words = pySplit(targetText);
  if (!targetText.includes(GROUP_OPEN) && !MARKER_ANY.test(targetText))
    return [tokens(targetText), ...words.map(tokens)];
  const wordTokens = words.map(word => word.split(PIECE).filter(Boolean).flatMap(piece =>
    isGroupToken(piece) ? [piece] : MARKER_FULL.test(piece) ? [] : tokens(piece)));
  return [wordTokens.flat(), ...wordTokens];
}
