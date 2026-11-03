/** The language registry every server and the studio share (zonos3.languages; its `description` explains the columns). */
import registry from '../../../model/src/zonos3/registries/languages.json';

/** A server language (/v2/generate `language`), in display order. */
export interface ServerLanguage {
  code: string;
  label: string;
  nemo: string;
  lexicon: string;
  speech: string;
}

/** A language speech recognition, alignment and dub references understand (Whisper-style code). */
export interface SpokenLanguage {
  code: string;
  name: string;
  /** The server language a dub into it speaks; none when it is a reference language only. */
  tts?: string;
  /** Written without spaces between words. */
  unspaced?: boolean;
}

export const SERVER_LANGUAGES: readonly ServerLanguage[] = registry.server;
export const SPOKEN_LANGUAGES: readonly SpokenLanguage[] = registry.spoken;

/** Server language codes (as /v2/generate `language`) and their names. */
export const LANG_LABEL: Record<string, string> = Object.fromEntries(SERVER_LANGUAGES.map((l) => [l.code, l.label]));

export const langLabel = (l: string) => LANG_LABEL[l] ?? l;

/** Languages speech editing transcribes (Qwen3-ASR) and aligns (FuseAlign): the server's EDIT_LANGUAGES. */
export const EDIT_LANGUAGES = SERVER_LANGUAGES.map((l) => l.code);

/** A dub target language (`/v2/dub/capabilities` code) as the server's language code. */
export const DUB_TO_SERVER: Record<string, string> = Object.fromEntries(SPOKEN_LANGUAGES.flatMap((l) => (l.tts ? [[l.code, l.tts]] : [])));
