/** Shared tool plumbing: input schemas, results, errors. */
import type { CallToolResult } from '@modelcontextprotocol/server';
import { z } from 'zod';
import { EDIT_LANGUAGES, LANG_LABEL } from '../../../src/lib/languages';
import type { WidgetPayload } from '../payload';

export const PLAYER_URI = 'ui://zonos/player.html';

/** Standard MCP Apps metadata plus the Codex/ChatGPT compatibility alias. */
export const PLAYER_META = {
  ui: { resourceUri: PLAYER_URI },
  'openai/outputTemplate': PLAYER_URI,
};

const langs = Object.keys(LANG_LABEL) as [string, ...string[]];
export const language = z.enum(langs).describe(`Server language code: ${langs.map((l) => `${l} (${LANG_LABEL[l]})`).join(', ')}`);
export const editLanguage = z.enum(EDIT_LANGUAGES as [string, ...string[]]).describe('Language of the clip (picks the alignment model)');
export const values = z.record(z.string(), z.number());
export const project = z.string().optional().describe('Project id; becomes the active project. Default: the active project (a new one is created when none is open).');
export const seed = z.number().int().min(0).max(4294967295).optional();
export const lineRef = z.union([z.number().int().min(1), z.string()]).describe('1-based line number or line id');

export function text(body: string, structured?: Record<string, unknown>): CallToolResult {
  return { content: [{ type: 'text', text: body }], ...(structured ? { structuredContent: structured } : {}) };
}

export function widget(body: string, payload: WidgetPayload): CallToolResult {
  // Codex can resolve the template from the result when descriptor metadata is lost
  // across a tool adapter. Keep this on player results only, never audio fetches.
  return { ...text(body, { ...payload }), _meta: PLAYER_META };
}

/** A tool body whose errors come back as `isError` results with the message. */
export function guard<A>(fn: (args: A) => Promise<CallToolResult>): (args: A) => Promise<CallToolResult> {
  return async (args) => {
    try {
      return await fn(args);
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      console.error('[zonos-mcp] tool error:', message);
      return { isError: true, content: [{ type: 'text', text: message }] };
    }
  };
}

/** Warnings as a text block (empty when none). */
export function warningsText(ws: string[]): string {
  return ws.length ? `\nWarnings:\n${ws.map((w) => `- ${w}`).join('\n')}` : '';
}
