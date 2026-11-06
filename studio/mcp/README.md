# Zonos Voice Studio MCP server

An MCP server that lets LLM clients (Claude Desktop / claude.ai / Claude Code, ChatGPT, Cursor, …) design voices
and generate and edit speech through the ZONOS3 v2 API. Everything it makes goes into a Studio project on the server
(`/v2/projects`, `/v2/media`): voice designs become Cast personas, lines become Direct lines with takes, speech edits
become Edit versions. Open the project in the Studio and it looks exactly like one the Studio wrote. Tool results
render an MCP Apps widget (`ui://zonos/player.html`): each take with its waveform and playback, plus every parameter
the render used (voice, design values with their safety zones, accents requested vs applied, painted delivery,
steering doses, sampling, seed, timings, the exact request body).

## Build and run

```bash
pnpm --dir studio install          # the bundle includes studio sources and their deps (world-atlas, topojson-client)
pnpm --dir studio/mcp install
pnpm --dir studio/mcp build        # tsc, vite (dist/player.html), esbuild (dist/server.js)
pnpm --dir studio/mcp test

ZONOS3_STUDIO_PROJECTS=/path/to/projects servers/mlx/scripts/serve.sh   # or any ZONOS3 server with projects
node studio/mcp/dist/server.js            # Streamable HTTP on http://127.0.0.1:3001/mcp
node studio/mcp/dist/server.js --stdio    # stdio
```

Claude Desktop / Claude Code (stdio):

```json
{ "command": "node", "args": ["<abs>/studio/mcp/dist/server.js", "--stdio"], "env": { "ZONOS_SERVER": "http://127.0.0.1:1919" } }
```

| Variable | Default | Meaning |
|---|---|---|
| `ZONOS_SERVER` | `http://127.0.0.1:1919` | The ZONOS3 server (the same variable the studio's dev proxy uses) |
| `ZONOS3_ACCESS_TOKEN` | — | Sent as `Authorization: Bearer <token>` |
| `ZONOS_MCP_HOST` | `127.0.0.1` | HTTP bind address. Loopback binds check the Host / Origin headers; bind `0.0.0.0` behind a tunnel |
| `ZONOS_MCP_PORT` | `3001` | HTTP port |
| `ZONOS_MCP_PROJECT` | — | Initial active project id |

A server without projects (`/v2/capabilities.projects.enabled` false) still works: nothing is saved and every result
says so. Without a steering bundle (`/v2/controls` 404) only plain voices work: steering input is refused.

## Tools

| Tool | Widget | What it does |
|---|---|---|
| `zonos_status` | | Server, steering, speech editing, projects, languages, the active project |
| `zonos_controls` | | The steering vocabulary: controls (range, safe zone, delivery or voice), accents, presets, moods |
| `zonos_voices` | | Library voices (measured gender / age hints) and the project's personas and created voices |
| `zonos_project` | | `list`, `open` (by id or name: personas, lines, takes, Edit clip), `create` |
| `zonos_design_voice` | ✓ | Create / refine a persona by name (base voice, mode, preset, values, accents); renders a preview |
| `zonos_speak` | ✓ | A new Direct line or more takes of one (`line`); `delivery` for the whole line, `spans` for words |
| `zonos_edit_speech` | ✓ | Change the words of a Direct take, a WAV file / URL or the current Edit clip; returns takes |
| `zonos_keep_edit` | | Keep an edit take as the next Edit version; `send_back` replaces the Direct take |
| `zonos_clone_voice` | | Register a voice from an audio file / URL; adds a persona |
| `zonos_save_voice` | | Bake a persona's design into a new embedding voice |
| `zonos_take_audio` | app only | A take's WAV for the widget (audio never enters the model's context) |

## How it maps onto the Studio

- Requests come from the studio's own `buildRequest` (`src/state/requests.ts`) with the same key order, so Studio
  does not mark the takes "Rendered with earlier settings". The document types, `summaryOf` and `remapIds` come from
  `src/state/projectSchema.ts`. Only signal-free studio modules are imported at runtime (`engine`, `atlas`, `geo`,
  `requests`, `projectSchema`, `media`, `lib/*`); `@preact/signals`, `location` or `localStorage` would crash Node.
- Each Direct line has its own voice: speaking with `voice` sets that line's voice; other lines follow the script
  voice (or the Cast bench).
- Accents: the Studio places one accent pin, so the strongest requested accent wins and the result reports the
  strength it really gets.
- Saves go against the revision loaded; on a conflict the tool reloads and applies its change again (3 attempts).
  Voices the project created are registered again when the server lost them (as the Studio does on open), and ids
  are mapped back before saving.
- Edit sources are WAV only (convert with ffmpeg); pronunciation groups in the target are not supported; regions
  come from the server's word diff. Edit draft takes live in this process only, as in the Studio.
- State is per process: one active project, shared by every HTTP client of the process.

## Layout

- `src/main.ts`: transports. `src/server.ts`: tools, the widget resource and the instructions.
- `src/zonos.ts`: the only server access (openapi-fetch client typed by the studio's `paths`, keep-alive edits).
- `src/backend.ts`: capabilities / controls cache, request context. `src/state.ts`: process state (take cache).
- `src/design.ts`: values with zones, the accent pin, painted spans, voice lookup. `src/project.ts`: load, reconcile,
  patch, save. `src/params.ts`: widget parameters and the text the model reads.
- `src/payload.ts`: the widget payload contract (server and widget). `widget/`: the Preact player.
