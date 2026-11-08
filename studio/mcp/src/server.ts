/** The MCP server: tools, the player widget resource, instructions. */
import { RESOURCE_MIME_TYPE, registerAppResource, registerAppTool } from '@modelcontextprotocol/ext-apps/server';
import { McpServer } from '@modelcontextprotocol/server';
import { readFile } from 'node:fs/promises';
import { z } from 'zod';
import { VERSION } from './config';
import { b64 } from './project';
import { state } from './state';
import { guard, PLAYER_URI } from './tools/common';
import { registerEditTools } from './tools/edit';
import { registerInfoTools } from './tools/info';
import { registerSpeakTool } from './tools/speak';
import { registerVoiceTools } from './tools/voice';
import * as zonos from './zonos';

const INSTRUCTIONS = `Zonos Voice Studio: design voices, speak and edit speech. Everything you make is saved into a Studio project the user can open.
- Call zonos_controls once to learn the steering vocabulary, and zonos_voices to pick base voices (the "who" hints give each voice's measured gender/age).
- Design a voice with zonos_design_voice and iterate on it: the design persists by name as a Cast persona.
- Speak lines with zonos_speak. Use delivery for how the whole line is said, and spans for specific words.
- Change words of recorded speech with zonos_edit_speech, then keep a take with zonos_keep_edit.
- Values are UI control values within each control's range. Stay inside the safe range unless the user asks for something extreme. Rough scale: "slightly" ≈ 1, "clearly" ≈ half of the safe max, "very" ≈ the safe max. Named moods have a recommended dose.
- Compatible MCP Apps hosts show results in an audio player widget. If no widget appears, provide the Studio project link from the result; do not claim a player is visible. Summarise the parameters briefly instead of repeating them.`;

/** dist/player.html next to dist/server.js. */
const PLAYER_HTML = new URL('./player.html', import.meta.url);

export function createServer(): McpServer {
  const server = new McpServer({ name: 'zonos-voice-studio', version: VERSION }, { instructions: INSTRUCTIONS });

  registerAppResource(server, 'Zonos player', PLAYER_URI, { description: 'Plays takes with their waveform and every generation parameter.', mimeType: RESOURCE_MIME_TYPE }, async () => ({
    contents: [{ uri: PLAYER_URI, mimeType: RESOURCE_MIME_TYPE, text: await readFile(PLAYER_HTML, 'utf8'), _meta: { ui: { prefersBorder: true } } }],
  }));

  registerInfoTools(server);
  registerVoiceTools(server);
  registerSpeakTool(server);
  registerEditTools(server);

  registerAppTool(
    server,
    'zonos_take_audio',
    {
      title: 'Take audio',
      description: 'The WAV of a take, for the player widget.',
      inputSchema: { take_id: z.string(), media: z.string().optional() },
      _meta: { ui: { resourceUri: PLAYER_URI, visibility: ['app'] } },
    },
    guard(async ({ take_id, media }) => {
      let wav = state.takes.get(take_id)?.wav;
      if (!wav && media) {
        wav = await zonos.getMedia(media);
        state.takes.set(take_id, { wav, media });
      }
      if (!wav) throw new Error('This take’s audio is gone (the MCP server restarted and it was never saved)');
      return { content: [{ type: 'text', text: 'audio' }], structuredContent: { take_id, mime: 'audio/wav', base64: b64(wav) } };
    }),
  );
  return server;
}
