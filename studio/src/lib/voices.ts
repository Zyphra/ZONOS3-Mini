/** Default voices by language. Their labels name it second: "Imogen en British F 1" (the file Imogen_en-British_F_1.wav). */
import { SERVER_LANGUAGES, SPOKEN_TTS } from './languages';

const names = new Intl.DisplayNames(['en'], { type: 'language', fallback: 'none' });

/** The language a voice's label names ("Imogen en British F 1": English); null when it names none. */
export function voiceLanguage(label: string): string | null {
  const code = label.split(' ')[1];
  return (code && /^[a-z]{2,3}$/.test(code) && names.of(code)) || null;
}

/**
 * The server language (/v2/generate `language`) a voice speaks, `fallback` (the project's) when it names none: a
 * default voice's label names it ("Liesel de F 1": de, "Meilin zh energetic F 1": cmn), and `fallback` stays when it
 * speaks the same language (an English voice in a UK English project). Other voices (their labels are the user's)
 * and languages no server language speaks (Hindi, Telugu) take `fallback`.
 */
export function voiceServerLanguage(s: { label: string; is_default: boolean } | null | undefined, fallback: string): string {
  const code = s?.is_default ? s.label.split(' ')[1] : undefined;
  const own = code && SPOKEN_TTS.get(code);
  if (!own || SERVER_LANGUAGES.find((l) => l.code === fallback)?.speech === code) return fallback;
  return own;
}

export interface VoiceGroup<T> {
  /** The language; "Other" for the voices that name none; null when no voice names one (show them as one list). */
  label: string | null;
  voices: T[];
}

/** English first, then the other languages by name, then the voices that name no language; each keeps its order. */
export function byLanguage<T extends { label: string }>(voices: readonly T[]): VoiceGroup<T>[] {
  const groups = new Map<string | null, T[]>();
  for (const v of voices) {
    const lang = voiceLanguage(v.label);
    const list = groups.get(lang);
    if (list) list.push(v);
    else groups.set(lang, [v]);
  }
  const rank = (l: string | null) => (l === 'English' ? 0 : l === null ? 2 : 1);
  const sorted = [...groups].sort(([a], [b]) => rank(a) - rank(b) || (a ?? '').localeCompare(b ?? ''));
  if (sorted.length === 1 && sorted[0][0] === null) return [{ label: null, voices: sorted[0][1] }];
  return sorted.map(([lang, list]) => ({ label: lang ?? 'Other', voices: list }));
}
