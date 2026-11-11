/**
 * LACI (the server's guard against skipped words, babble and cut-off endings, `/v2/capabilities.laci_languages`):
 * the project's mode and the request fields it sends. Signal-free: the MCP server (`studio/mcp`) shares it.
 */

/** Off; on with the progressive decode (playback starts about a phrase late); on with the whole take decoded at the end. */
export type LaciMode = 'off' | 'stream' | 'whole';
export const LACI_MODES: readonly LaciMode[] = ['off', 'stream', 'whole'];
export const LACI_LABEL: Record<LaciMode, string> = { off: 'Off', stream: 'On, streamed', whole: 'On, whole take' };

/**
 * The fields a request in `language` sends for `mode`: none when off or when the server has no alignment heads for
 * the language. Streamed sends no `stream_decode`: the request defaults already ask for the progressive decode.
 */
export function laciFields(mode: LaciMode | undefined, served: readonly string[], language: string): { laci?: true; stream_decode?: 'whole' } {
  if (!mode || mode === 'off' || !served.includes(language)) return {};
  return mode === 'whole' ? { laci: true, stream_decode: 'whole' } : { laci: true };
}
