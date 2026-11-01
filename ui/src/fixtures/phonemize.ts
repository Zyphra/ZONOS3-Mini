/**
 * Client-side stand-in for `POST /tts/phonemize` and the server's 400s for pronunciation groups. A small gold
 * lexicon (US English), a few ambiguous / unsupported entries, and a letter-to-sound "guess" for the rest in the
 * guess languages, so every status the studio shows can be exercised offline.
 */
import { ApiError, type PhonemeStatus, type PhonemizeResponse, type PronCaps } from '../api';
import type { PronTokens } from '../lib/pron';

const LEXICON: Record<string, string> = {
  siobhan: 'ʃəvɔːn',
  worcestershire: 'wʊstɚʃɚ',
  hello: 'həloʊ',
  there: 'ðɛɹ',
  how: 'haʊ',
  are: 'ɑɹ',
  you: 'ju',
  today: 'tədeɪ',
  welcome: 'wɛlkəm',
  back: 'bæk',
  to: 'tu',
  the: 'ðə',
  late: 'leɪt',
  show: 'ʃoʊ',
  tonight: 'tənaɪt',
  story: 'stɔɹi',
  starts: 'stɑɹts',
  somewhere: 'sʌmwɛɹ',
  quiet: 'kwaɪət',
  little: 'lɪtəl',
  after: 'æftɚ',
  midnight: 'mɪdnaɪt',
  good: 'ɡʊd',
  evening: 'ivnɪŋ',
  and: 'ænd',
  lovely: 'lʌvli',
  finally: 'faɪnəli',
  meet: 'mit',
  been: 'bɪn',
  caramel: 'kɛɹəmɛl',
  nike: 'naɪki',
  gnocchi: 'njɑki',
  quinoa: 'kinwɑ',
  // espeak would only guess these; the lexicon has an entry this model has no token for.
  tsk: 'ǀ',
  brr: 'ʙ',
};

/** Several readings in the lexicon (the guess picks one when the language allows guessing). */
const AMBIGUOUS: Record<string, string> = { read: 'ɹid', live: 'lɪv', lead: 'lid', wind: 'wɪnd', tear: 'tɛɹ', bass: 'beɪs', gif: 'ɡɪf', route: 'ɹut' };

const RULES: [RegExp, string][] = [
  [/^tch/, 'tʃ'], [/^ch/, 'tʃ'], [/^sh/, 'ʃ'], [/^th/, 'θ'], [/^ng/, 'ŋ'], [/^ph/, 'f'], [/^ck/, 'k'], [/^qu/, 'kw'],
  [/^ee/, 'i'], [/^ea/, 'i'], [/^oo/, 'u'], [/^ou/, 'aʊ'], [/^ow/, 'aʊ'], [/^ai/, 'eɪ'], [/^ay/, 'eɪ'], [/^oa/, 'oʊ'], [/^igh/, 'aɪ'],
  [/^a/, 'æ'], [/^e/, 'ɛ'], [/^i/, 'ɪ'], [/^o/, 'ɑ'], [/^u/, 'ʌ'], [/^y$/, 'i'], [/^y/, 'j'],
  [/^c(?=[eiy])/, 's'], [/^c/, 'k'], [/^g(?=[eiy])/, 'dʒ'], [/^g/, 'ɡ'], [/^j/, 'dʒ'], [/^q/, 'k'], [/^r/, 'ɹ'], [/^x/, 'ks'],
  [/^([bdfhklmnpstvwz])\1?/, '$1'],
];

function guess(word: string): string {
  let w = word.toLowerCase().replace(/[^a-z]/g, '');
  let out = '';
  while (w) {
    const rule = RULES.find(([re]) => re.test(w));
    if (!rule) {
      w = w.slice(1);
      continue;
    }
    const m = w.match(rule[0])!;
    out += rule[1].replace('$1', m[1] ?? '');
    w = w.slice(m[0].length);
  }
  return out;
}

function wait(ms: number) {
  const { promise, resolve } = Promise.withResolvers<void>();
  setTimeout(resolve, ms);
  return promise;
}

export async function phonemize(caps: PronCaps, text: string, language: string, allowGuess: boolean): Promise<PhonemizeResponse> {
  if (!caps.phonemize_enabled) throw new ApiError(404, 'This server has no pronunciation lexicon.');
  if (!caps.phonemize_languages.includes(language)) throw new ApiError(400, `No pronunciation lexicon for ${language}.`);
  await wait(120);
  const canGuess = allowGuess && caps.phonemize_guess_languages.includes(language);
  const known = caps.phoneme_tokens ? new Set(caps.phoneme_tokens.symbols) : null;
  const segments: PhonemizeResponse['segments'] = [];
  for (const m of text.matchAll(/([\p{L}']+)|([^\p{L}']+)/gu)) {
    if (m[2]) {
      segments.push({ text: m[2], status: null });
      continue;
    }
    const word = m[1];
    const key = word.toLowerCase();
    let status: PhonemeStatus;
    let ipa: string | null = null;
    if (LEXICON[key]) [status, ipa] = ['ok', LEXICON[key]];
    else if (AMBIGUOUS[key]) [status, ipa] = canGuess ? ['guessed', AMBIGUOUS[key]] : ['ambiguous', null];
    else if (word.length === 1 && !/[aiAI]/.test(word)) status = 'skipped';
    else if (canGuess) [status, ipa] = ['guessed', guess(word)];
    else status = 'unknown';
    if (ipa && known && ![...ipa].every((c) => known.has(c))) [status, ipa] = ['unsupported', null];
    segments.push({ text: word, status, ...(ipa ? { ipa } : {}) });
  }
  return { language, segments };
}

/** The server's checks on U+E000 … U+E001 groups in a request text. */
export function validateGroups(text: string, tokens: PronTokens | null) {
  const open = tokens?.open_marker ?? '\ue000';
  const close = tokens?.close_marker ?? '\ue001';
  if (!text.includes(open) && !text.includes(close)) return;
  if (!tokens) throw new ApiError(400, 'This checkpoint has no phoneme tokens: remove the U+E000 / U+E001 pronunciation groups');
  const known = new Set(tokens.symbols);
  let pos = 0;
  while (pos < text.length) {
    const start = text.indexOf(open, pos);
    const stretch = start < 0 ? text.slice(pos) : text.slice(pos, start);
    if (stretch.includes(close)) throw new ApiError(400, 'Pronunciation groups: Unbalanced phoneme group markers');
    if (start < 0) break;
    const end = text.indexOf(close, start + 1);
    const group = end >= 0 ? text.slice(start + 1, end) : '';
    if (end < 0 || !group || group.includes(open)) throw new ApiError(400, 'Pronunciation groups: Unbalanced or empty phoneme group');
    const unknown = [...new Set(group)].filter((c) => !known.has(c));
    if (unknown.length) throw new ApiError(400, `Pronunciation groups: Phones outside the inventory: ${JSON.stringify(unknown.sort())}`);
    pos = end + 1;
  }
}
