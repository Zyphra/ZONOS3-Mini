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

export const SERVER_LANGUAGES: readonly ServerLanguage[] = registry.server;

/** Server language codes (as /v2/generate `language`) and their names. */
export const LANG_LABEL: Record<string, string> = Object.fromEntries(SERVER_LANGUAGES.map((l) => [l.code, l.label]));

export const langLabel = (l: string) => LANG_LABEL[l] ?? l;

/** The server language that speaks a spoken language (Whisper-style code, as default voice labels name it): the registry's `spoken[].tts` ("en" → en_us, "zh" → cmn). */
export const SPOKEN_TTS: ReadonlyMap<string, string> = new Map(registry.spoken.flatMap((l) => (l.tts ? [[l.code, l.tts] as const] : [])));

/** Languages speech editing transcribes (Qwen3-ASR) and aligns (FuseAlign): the server's EDIT_LANGUAGES. */
export const EDIT_LANGUAGES = SERVER_LANGUAGES.map((l) => l.code);
