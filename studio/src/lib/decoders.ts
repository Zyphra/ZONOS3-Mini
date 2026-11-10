/**
 * Audio decoders (`/v2/capabilities.decoders`, the default first): their names in the UI and which one a request
 * picks. Signal-free: the MCP server (`studio/mcp`) shares it.
 */

/** UI names of the known decoder heads; another id shows as itself. */
const NAMES: Record<string, string> = { keep2_plain: 'Fast (keep2)', full_mr: 'Full-size' };

export const decoderName = (id: string): string => NAMES[id] ?? id;

/**
 * The `decoder` a request sends for the project's choice `want`: null (the server default) when it is the default
 * (`served[0]`) or this server does not serve it, so a project opened on another server never asks for one it lacks.
 */
export function requestDecoder(served: readonly { id: string }[], want: string | null | undefined): string | null {
  return want && want !== served[0]?.id && served.some((d) => d.id === want) ? want : null;
}

/** The server's 400 for a decoder it does not serve (the fixture backend answers the same). */
export function unknownDecoder(id: string, served: readonly { id: string }[]): string {
  return `Unknown decoder '${id}'; this server serves: ${served.map((d) => d.id).join(', ')}.`;
}
