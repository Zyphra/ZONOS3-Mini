/** Server language codes (as /v2/generate `language`) and their names. */
export const LANG_LABEL: Record<string, string> = { en_us: 'US English', en_gb: 'UK English', fr_fr: 'French', de: 'German', es: 'Spanish', it: 'Italian', pt_br: 'Portuguese', ja: 'Japanese', cmn: 'Mandarin', ko: 'Korean' };

export const langLabel = (l: string) => LANG_LABEL[l] ?? l;

/** Languages speech editing aligns and transcribes (an MMS-1B-all adapter each). */
export const EDIT_LANGUAGES = ['en_us', 'fr_fr', 'de', 'es', 'it', 'pt_br', 'ja', 'cmn', 'ko'];

/** A dub target language (`/v2/dub/capabilities` code) as the server's language code. */
export const DUB_TO_SERVER: Record<string, string> = { en: 'en_us', fr: 'fr_fr', de: 'de', es: 'es', it: 'it', pt: 'pt_br', ja: 'ja', zh: 'cmn', ko: 'ko' };
