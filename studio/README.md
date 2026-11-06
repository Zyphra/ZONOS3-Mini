# Zonos Voice Studio (UI)

Web UI for the ZONOS3 Mini TTS server: **Cast** (find or design a voice), **Direct** (perform a script with one or
more voices: delivery painted onto words, takes per line, a timeline with one lane per voice), **Edit** (change the
words of recorded speech; only the changed stretch is regenerated). Vite + Preact + TypeScript + @preact/signals,
plus world-atlas + topojson-client for the
accent map and fflate for project files; hand-written CSS, SVG and canvas.

Cast opens on **Speak**: type a line, pick a voice in the left rail, press Speak (or Enter; Shift+Enter is a newline). The line is
also what every bench move re-renders, so the same bar is the A/B monitor while shaping the voice below. Nothing
renders on load (boot or opening a project); **auto-generate** (on by default, kept in the browser) re-renders
after each change, off leaves the take marked out of date until Speak / A / Space. The small cards open large from
their header; the Texture card opens from anywhere on it and the Mood card from its pad too (its mood chips and the
Room / Delivery sliders work in place). Drop an audio file anywhere on Cast to clone it.

**Direct** is one script whose every line has its own voice. The left rail sets the **default voice** (a cast
persona, or the voice on the Cast bench) that lines follow, with a “{name} says…” button per cast persona that adds a
line in that voice (an empty cast offers to add the bench voice), plus the language, tag strength, baseline
performance, pronunciation help and the brushes legend. Each line has a voice select (“Script voice”, or any cast
persona; one undo step) and a colored edge in its voice's color; select words to paint delivery onto them; each line
keeps its takes and a starred keeper. Under the lines the **timeline** lays the keepers end to end with one lane per
voice (lines without a take get an estimated slot; click a clip to hear it; a playhead follows the script). The
right **Script** sidebar counts the lines with a take, renders the missing ones, plays and exports the script (the
keepers in order, the gap between lines set by its slider) and, when a line's keeper no longer matches its text,
painting, baseline or voice (changing the default voice or a line's voice does this), offers **Regenerate N changed
lines**; such a line's own render button reads **Regenerate**. A regenerated take becomes the keeper.

## Design system

Zyphra's design language, all through the tokens in `src/styles/base.css` `:root` (dark, the default) and
`:root[data-theme='light']`: surfaces `--bg --bg-2 --surface*`, hairlines `--line --line-2`, text `--text*`,
the inverted primary `--inv --inv-text`, Zyphra orange `--o` (focus, the active act, selection and the one primary
highlight only), status `--ok --warn --bad`, data hues `--c1…--c6`, radii `--r` / `--r-s` (5 px nav and chips),
motion `--ease --t`. Inter and JetBrains Mono are bundled (`@fontsource`). Buttons are pills with hairlines;
`.btn.primary` is inverted. `src/styles/base.css` holds the tokens, the top bar, the primitives (`.btn`, `.input`,
`.pill`, `.panel`, `.mode` …) shared with the lite page; `src/styles.css` imports it and adds the screens. The brand
marks and icon paths live in `src/lib/marks.ts` for the same reason. Never hard-code a colour: canvases read tokens
through `token()` in `src/state/theme.ts`, which also holds the theme toggle (browser-wide,
`localStorage['zvs.theme']`, applied before first paint by every page's HTML). The pages without Preact (the
chooser and the lite page) mount the same top bar, icons and toggle through `src/lib/chrome.ts`. One screen header
(`.screen-head`) and one empty state (`.empty`) for every screen.

## Phone layout

`/studio` also works on a phone (portrait, ≈360–430 px; the chooser at `/` offers it on every device). Everything
is in `src/styles/mobile.css` (`@media (max-width: 760px)`, loaded after `styles.css`) plus the `phone` signal in
`src/lib/media.ts` (the same breakpoint) for the few places where the markup differs:

- Top bar: the Zyphra mark, the project, undo / redo and a ⋯ menu (backend, theme, shortcuts); Cast · Direct · Edit
  as a tab row under it.
- Side columns are `Rail`s (`src/components/Rail.tsx`): on phones they are drawers opened from the row of toggles at
  the top of each screen (`RailBar` / `RailToggle`: Cast *Voices* / *More*, Direct *Voices* / *Script*, Edit
  *Sources*), closed by the backdrop, ✕ or Esc. On desktop each rail has a fold toggle on its inner edge
  (folded rails are remembered in `localStorage['zvs.rails']`; the screens size their rail columns with
  `--rail-l` / `--rail-r`).
- Cast: Speak first, then the six controls as tiles (read-only summaries, `src/screens/cast/Tiles.tsx`); a tap opens
  the full control in the card dialog, full screen (`who` and `accent` have `large` variants for it).
- Direct: line actions wrap under the text; selecting words (long-press) opens the brushes as a bottom sheet. Edit:
  the header and tools wrap, Takes go under the transcript.
- Touch targets are at least 40 px; text fields use 16 px type (no iOS focus zoom).

## Develop and build

```bash
pnpm --dir studio install
echo "ZONOS_SERVER=https://<server>" > studio/.env.local   # or export ZONOS_SERVER; /v1 and /v2 are proxied there
pnpm --dir studio dev         # http://localhost:5173/ (the chooser), /studio, /lite
pnpm --dir studio typecheck   # gen:api, then tsc
pnpm --dir studio build       # gen:api, tsc, vite → studio/dist/ (commit it)
```

Three pages, three Vite entries: `index.html` (`src/home/`) is the chooser at `/` that opens either app,
`studio.html` (`src/main.tsx`) is the Voice Studio at `/studio`, and `lite.html` (`src/lite/`) is the lite page at
`/lite`. The chooser passes its query on (`/?api=web` opens `/studio?api=web`) and shows both apps on every device,
phones included. Every server maps `/<page>` to
`<page>.html` (the studio serving rule of `../api/openapi.yaml`; Vite's dev and preview servers and Cloudflare's
static assets do the same).

`studio/dist/` is committed so a checkout serves the studio without Node: every server serves it (pages uncached,
hashed assets immutable; `--studio-dir` points elsewhere). The API contract is `../api/openapi.json`:
`pnpm gen:api` (run by `build` and `typecheck`) regenerates `src/api/schema.ts` from it (committed with the spec), and
`src/api.ts` aliases its schemas and sends every JSON call through an `openapi-fetch` client typed by its `paths`;
the streamed `/v2/generate` and the raw-body / binary routes (`/v2/speakers/{id}/preview`,
`/v2/media/{sha256}`) stay on `fetch` / XHR. `openapi-typescript`
drives the TypeScript 5 compiler API, which the native TypeScript 7 the app type-checks with lacks: `.pnpmfile.cjs`
gives it its own TypeScript 5.

Backends are detected at startup: **live** (server has `GET /v2/controls`), **fixture** (server answers but
`/v2/controls` is 404: bundled controls, renders sent without steering fields), **web** (no server answered, but the
browser has WebGPU and the site serves the in-browser engine at `/web-engine/engine.js`: ZONOS3 runs in the tab,
`../servers/web`) or **offline** (no server and no engine: fixture voices and a client-side babbling synthesizer that
applies every control). Force one with `?api=live|fixture|offline|web` (`?engine=URL` points at another engine
build); point a build at another server with `?server=https://…` (CORS is open). Every request goes through one
transport (`fetch`, or the engine's `localFetch` on the web backend; `src/lib/web-engine.ts`, shared with the lite
page). On a Hugging Face Space framed by huggingface.co (the ZeroGPU Space of `../space/`), that transport adds the
visitor's ZeroGPU headers to `/v2/generate` (`src/lib/zerogpu.ts`: the `supports-zerogpu-headers` /
`zerogpu-headers` handshake Gradio's client does), so renders count against the visitor's GPU quota.
`pnpm dev` / `pnpm preview` serve the engine
(`../servers/web/dist`, `pnpm --dir ../servers/web build`) at `/web-engine` and its models
(`../servers/web/public/models`, `pnpm --dir ../servers/web export`) at `/models`; `pnpm build:web` (after `build`)
copies the engine into `dist/web-engine/` for a deployment (`pnpm build` empties `dist/` except `web-engine/`).
Speech editing follows `/v2/capabilities.speech_editing` (and `edit_transcribe`: the server has speech recognition)
on a live server and runs on the client-side fixture offline; `?edit=fixture` forces the fixture, `?edit=0` a
checkpoint that cannot edit, `?edit_transcribe=0` a server without speech recognition (works with the fixture too).

## Lite UI

`lite.html` + `src/lite/` are a Vite entry, built by the same `pnpm build` into `dist/lite.html`: a
one-job page for voice cloning from a speaker embedding, served by every server at `/lite`. It is plain
TypeScript and DOM, no Preact, in the studio's design language: the same tokens, top bar, primitives and fonts
(`src/styles/base.css`, `src/lib/marks.ts`, `@fontsource`) plus its own one-column layout (`src/lite/lite.css`), and
the studio's theme toggle (`localStorage['zvs.theme']`, so the two pages switch together). Beyond those it shares
only the generated `src/api/schema.ts` types, `openapi-fetch`,
`src/lib/files.ts` and `src/lib/web-engine.ts`. It runs on an HTTP server or on the in-browser server exactly as the
studio does (`src/lib/web-engine.ts` is the studio's own engine loader and transport): `?api=web` forces the
in-browser server, `?api=live` the HTTP server, and `auto` (the default) takes the in-browser server when no ZONOS
server answers `GET /v2/capabilities` and the browser has WebGPU and the site serves the engine (`?engine=URL`,
default `/web-engine/engine.js`, loaded at run time and never bundled; `?models=URL` is the engine's own). Every
request, the raw preview and generate ones included, goes through that one transport. While the engine loads its
models the page shows a download bar (`components/DownloadBar.tsx` in the studio, the same `.dl` markup in
`lite.html`; `lib/web-engine.ts` turns the engine's readouts into it): "Downloading backbone + zac" (or "Loading …"
from this browser's cache), the percent, and `420 MB of 1.9 GB · 18 MB/s · about 1 min 20 s left`. The rate counts
network bytes only, over the last 8 s, so cached shards don't skew the estimate. The form stays inert meanwhile;
the top bar's pill then says **Live** or **In-browser**. The encoder models a clip voice's first use downloads
(generate or embedding download) get the same bar (in the studio, in the corner).
On load it reads `GET /v2/capabilities` and, when `speaker_modes` lacks `embedding`, says the server cannot clone
from an embedding and disables the form.
A voice comes from a default voice of `GET /v2/speakers` (those whose `speaker_modes` include `embedding`), an
uploaded or recorded clip (a recording is sent as a 16-bit WAV) or a saved `.npy`/`.npz` embedding (both
`POST /v2/speakers`); the chosen voice offers its preview (`has_preview`) and **Download embedding**
(`GET /v2/speakers/{id}/embedding` saved as a float32 `.npy`, which loads back as an embedding file). Generate posts
`/v2/generate` with `speaker_mode: "embedding"`, `stream: true`, `format: "pcm"`, plays the float32 PCM through
WebAudio as it streams and draws it live as the studio's preview waveform (`src/lib/wave.ts`, shared with
`components/Waveform.tsx`) with a playhead; click the waveform to play from there. Then it offers the WAV (built in
the browser) and shows `X-Seed`; errors show the server's
`detail` (the 422 list as `field: message`). `?server=https://…` points it at another server; `X-TTS-Session-ID` is
random per tab (sessionStorage), so the voices a tab adds are its own.

## Projects

Everything made in the studio belongs to a project. **Where projects live**: on the server, shared by every browser
with the studio link and kept across restarts and tunnel URLs, when `/v2/capabilities.projects.enabled` (store:
`/v2/projects` + `/v2/media`, `src/state/projectStore.ts` `ServerStore`); otherwise in this browser (IndexedDB
`zvs-projects`, `BrowserStore`; offline and old servers). `?projects=fixture` runs the server contract on a
client-side fixture store (`src/fixtures/projects.ts`, its own IndexedDB, shared between tabs like a server);
`?projects=browser` forces the browser store. A project holds the Cast bench and cast, the voices created on the server
(uploads, recordings, saved designs), the Direct script (lines with their voices, the gap) and its takes, the
Edit clip with its versions, plus per-project settings (default language for new Edit clips and pronunciation
lookups, default
voice mode: the Cast bench takes it when it leaves an embedding-only voice). Undo history, drafts (the Edit text and
regions, the last generation's takes) and in-flight work stay per session.

- **Header switcher**: the open project's name and save state (dot: saved / saving / error / changed elsewhere /
  not stored), a menu with recent projects, New, Rename, Duplicate, Export, Import and All projects. **Launcher**
  (All projects, refreshed on open): search, open, rename, duplicate, export, delete (confirmed; on the server it is
  gone for everyone), new, import, the open project's settings and voice report, and storage use.
- **Saving**: autosave 0.8 s after a change (also when the tab is hidden); an unchanged snapshot is not written.
  Media are addressed by SHA-256 (`src/lib/hash.ts`, plain JS: the http cluster URL has no `crypto.subtle`):
  16-bit WAV, or the exact WAV an Edit version sent to the server; voice embeddings as raw float32;
  the document lists them in `media` (`{<sha256>: {type, size}}`) and its `summary` feeds project lists. On the
  server a save uploads the media it lacks (`HEAD /v2/media/<sha256>`, then `PUT`) and then `PUT /v2/projects/<id>` with
  the `base_revision` it was opened at (the browser store keeps revisions too). A failed save says so once, keeps
  everything open and blocks switching away until a save succeeds.
- **Changed elsewhere**: a save refused with 409 (another tab or browser saved first, or deleted it) stops
  autosave and shows a bar under the header: **Reload** (drop this tab's changes) or **Keep mine as a copy** (a new
  project with this tab's state; the other stays as saved).
- **Moving to the server**: when the server holds projects and this browser still has some, the studio says so on
  boot and the launcher offers “Move N to the server”: each is re-addressed to SHA-256 (older ones used MurmurHash3),
  uploaded and then removed from the browser.
- **First visit**: what earlier versions kept in localStorage (`zvs.design`, `cast`, `voiceOrigins`,
  `previewText`, `script`, `scene`, `edit.languages`, `pron.language`) becomes the project “My project”, then those
  keys are dropped (an old Scene's lines are appended to the Direct script, each in its character's voice).
  Browser-wide preferences (theme, autoplay, edit options) stay in localStorage. Projects saved with a Scene open the
  same way: its lines join the script, each with its last render as the starred take.
- **Voices across server restarts**: session voices die with the server and belong to one browser's session. A
  project keeps each voice it created with its clip or embedding; opening it lists `/v2/speakers`, registers the
  missing ones again (`POST /v2/speakers`) and maps the saved ids to the new ones in this tab (designs, cast, voice
  origins, and the requests / render keys derived from them, so takes and renders still match their
  lines). Saves map them back, so re-registering is not a change: two browsers opening one project don't overwrite
  each other. Session voices a project uses but did not create are captured while the server still has them. What
  can't be restored is listed in the launcher and announced on open.
- **Export / import**: one zip (`<name>.voice-studio.zip`): `project.json` (the document exactly as stored, schema
  1, `src/state/projectSchema.ts`) and `media/<hash>.<wav|f32|…>`. Import checks every file against its hash (SHA-256,
  or MurmurHash3 for older zips, re-addressed), keeps the project id unless it exists already, stores it (on the
  server: uploads the media, then creates the project) and opens it.
- Without a store (blocked browser storage, or the server's projects not answering) the studio works for the
  session and says that nothing is saved.
- **Opening a given project**: `/studio?project=<id>` opens that project at boot (the MCP server's links); an unknown
  id falls back to the last one.
- **MCP server**: `mcp/` (`zonos-studio-mcp`) lets LLM clients design voices, speak and edit speech into these
  projects through the same request and schema code; see `mcp/README.md`.

## Moods

Named moods are the unipolar controls of the `mood` group; `ui.moods` (`{id, label, control, valence, arousal,
dose, effect}`) gives each one's MEASURED place on the valence / arousal pad (−1…1) at its default dose. The Mood
pad (Cast card and dialog, Direct's paint bar) draws every measured mood as a landmark: click one to add it at its
default dose (one undo step; click again to remove), and the active ones glow, with a handle on the ray from neutral
through the landmark. Drag the landmark or the handle along that ray to set strength; the ray turns dashed past the
control's tested-safe dose. Moods combine (their doses add up, each with its slider in the dialog), and dragging
the free pad still moves the valence / arousal axes. Keyboard: landmarks are focusable, arrow keys go to the
nearest landmark in that direction, Enter toggles, + / − change the dose, Delete removes. Every mood also has a
chip (Cast, Direct, Edit regions), most aroused first; hovering a chip or landmark shows its measured effect.
Landmarks closer than a label's height are drawn apart, with a short leader line to the measured spot. Labels that don't fit a
small pad appear on hover / focus.

## Painted spans and the line language

Direct lines render in the project's language (the **language** select in the Direct rail,
also in the launcher; `language` on `/v2/generate`, left out for `en_us`): text normalization reads in it, and
pronunciation lookups and new Edit clips start in it (UK English edits as US English). A line with painted spans
asks for `control_span_follow: "reading"`: the server steers each span where the model is reading it (its alignment
heads for that language) instead of by a characters-per-frame estimate. The response header
`X-Control-Span-Follow` says which ran, and each take shows it: **read** (followed the reading) or **est.**
(placed by estimate: no heads for the language). The rate measured from the line's previous
take (`control_span_rate`) is sent only as the fallback: not once a take of that line in that language came back
`reading`. Offline, the fixture answers `reading` for `en_us` / `fr_fr` and the estimate otherwise. Lead / lag
(`control_span_lead` / `control_span_lag`) stay at the server defaults.

## Nonverbal tags

On a server with `/v2/capabilities.tag_cfg.available` (the NV checkpoint), every text that is synthesized (Direct
lines and the script text, the Cast preview line, the Edit text) takes
nonverbal markup: point tags `<laugh/> <breath/> <sigh/> <cough/> <panting/> <shouting/> <crying/>` and span tags
over words `<shouting>…</shouting>`, `<crying>…</crying>` (`src/lib/tags.ts`). **`<`** in such a field (or the `‹›`
button beside it; Direct: “‹› Tag” in the paint bar) opens the palette: a point tag goes in at the caret (padded with
spaces), a span tag around the selected words (grown to whole words); a first letter jumps to a tag, Esc closes (and
types “<” when the palette came from the key). Tags show as chips, not angle brackets: inline in Direct's lines and
over a resting field (the field shows the text as typed while focused), and as a row of chips under the field to
remove a tag or turn it into a point / span (⇄). Tags the NV model barely follows are marked **weak** (cough ≈ no
effect; shouting and crying no better than without). Unclosed span tags and unknown names are read as plain text
and say so. Markup travels in the text as typed, so painted spans and pronunciation groups keep their offsets on the
sent text. The palette appears only when the served checkpoint was trained on tags (`tag_cfg.trained`; the NV
checkpoint); otherwise it is gone and existing markup is flagged “read as text” (an older server without `trained`
gets the palette with a note that only the NV checkpoint reads tags). Markers take no time: the measured
`control_span_rate` is per spoken character.

**Tag strength** (Direct rail; per project, default 1.5, range from the capabilities) is tag guidance:
a line with tags renders with `tag_cfg_scale` = the strength, and since the server allows one guidance per render
its voice guidance is off (`speaker_cfg_scale` 1); such takes show **tags** (“tag guidance on · voice guidance off”).
Strength 1, or a line without tags, renders as before (speaker CFG 1.5). Edit texts go out at scale 1 (the server
refuses tag guidance there): the markers reach the model but stay out of recognition, so a
tag is its own token in the word diff (inserting one between unchanged words plans a region there, which the server
then fills) and is never “not heard”; a region that only adds tags skips the length warnings. Offline (and with
`?nv=fixture`) the synth voices laugh, breath, sigh and panting (stronger with tag strength), barely a cough, and
shapes shouting / crying words mildly; `?nv=0` simulates a server without tags.

## Pronunciation input

On phoneme-token checkpoints (`/v2/capabilities.phoneme_tokens`), every text that is synthesized (Direct
lines, the Cast preview line) takes X-SAMPA pronunciations in parentheses, “I met (S@vQn)” (`_` ties, symbols
without an X-SAMPA spelling are typed as-is, stress / dots / spaces
dropped). Each group shows its IPA inline (Direct) or as a chip; symbols without a token on the checkpoint and
unclosed / empty groups are flagged and block the render. Click a group to edit it (X-SAMPA with live IPA, per
symbol validity). `/ə/` (Pronounce, when `phonemize_enabled`) looks the selected words up via `POST /v2/phonemize`
and writes them as groups; guessed words are marked “check”, ambiguous / unknown / unsupported ones are left as
text with a note (Direct: one undo step; inputs: ⌘Z while focused). Requests carry each group as U+E000 IPA
U+E001, and `control_spans` offsets refer to that sent text (a painted boundary inside a group takes the whole
group). Without phoneme tokens parentheses are plain text and the Direct rail says which checkpoint enables it
(`SERVE_MODEL=<phoneme checkpoint> PHONEME_LEXICON=…`). On an edit checkpoint that also has phoneme
tokens (`speech_editing` + `phoneme_tokens`), the Edit screen's target text takes groups the same way (chips,
group editor, `/ə/`; invalid ones block Generate). A group is one target word for the word diff and region
planning (it never equals a spelled word, so "Siobhan" → "(S@vO:n)" is a one-word replacement); `target_text` is
sent with U+E000 IPA U+E001, `original_text` / `original_words` stay plain. The words-heard check treats a group as
a wildcard word, and each take lists what was recognized at it (`verify.groups`: “(S@vO:n) heard as ‘shivon’”).
Keeping such a take writes the groups as heard into the new version's transcript (re-aligned); sending it back to
Direct keeps the groups in the line. The original transcript (Fix the transcript) is always plain, and a
take whose text has groups is opened by transcription (without speech recognition: typed, starting from the text
with each group spelled as a guess). Without phoneme tokens the target text is plain. Offline, `src/fixtures/phonemes.json` (the real 136-symbol inventory)
and `src/fixtures/phonemize.ts` stand in; `?phon=fixture` forces them, `?phon=0` a checkpoint without tokens. Offline
mode is the edit + phoneme checkpoint (the fixture edit engine sends `verify.groups`); `?edit=fixture&phon=fixture`
forces the same against a live server.

## Edit screen

Every edit call sends the clip's `language` (server code: en_us, en_gb, fr_fr, de, es, it, pt_br, ja, cmn, ko); it picks
the alignment tokenization (and, with speech recognition, the Qwen3-ASR language) for transcription, alignment,
windowing and the take check. Clips use the language last chosen for them (the picker next to the title, default US
English; switching re-aligns the version, or re-transcribes an automatic transcript when the server can, one undo
step). On a server with speech recognition its first transcription or take check loads the recognition model (about
half a minute): the status says so rather than looking stuck. An unsupported language shows the server's message.

Transcript-first editing:
- Open a Direct take, a voice's reference clip (its `transcript` from `/v2/speakers` when present),
  an upload or a recording. Known text → `POST /v2/edit/align`, otherwise `POST /v2/edit/transcribe`; an
  automatic transcript can be corrected ("Fix the transcript") and is re-aligned. Clips up to
  `edit_align_max_seconds`.
- Without speech recognition (`edit_transcribe: false`, e.g. the in-browser engine) a clip with no known text opens
  on "Type what the clip says": type it, Align (`POST /v2/edit/align`), then edit. Voice clips without a transcript
  are tagged "needs text"; takes are not checked (`verify: false`), so listen and keep the best.
- Edit the words in place: the word diff against the aligned transcript plans the regions (frames at
  `edit_frame_rate`, `margin_seconds` 0.08, merged when closer than 5 frames, at most `edit_max_spans`). Region
  edges drag on the zoomable waveform; dragging on empty waveform adds a "regenerate as is" region.
- Word timings are editable in the word lane under the waveform: click a word to select and hear it (optional
  0.3 s pre-roll), drag its edges (a boundary shared with the next word moves both; Alt-drag opens a gap; low
  confidence words are underlined, adjusted ones dotted). With a word selected: ←/→ nudge the active edge 10 ms
  (⇧ 50 ms), `,` / `.` pick the start / end edge, Tab / ⇧Tab next / previous word, ↵ play, Esc done. A drag or a
  burst of nudges is one undo step; "Reset timings" restores the alignment. Corrected timings drive the planning,
  margins, re-say snapping and the model-context estimate, and go to the server as `original_words` (with
  `original_text`) once any word was adjusted.
- Each region can be directed with the spannable controls of `/v2/controls` (`region_controls`), the whole edit
  too (`controls`). `POST /v2/edit` always gets explicit `spans` (frame-exact seconds), `original_text` (the
  transcript the word timings belong to), `num_takes` 1–4 and `verify: edit_transcribe`; "context per side"
  (advanced, 2–8 s) sets `context_seconds`. Long clips are edited on windows around the regions; the response's
  `contexts` show as a faint "model context" band on the waveform.
- Takes show WER vs the target (best preselected), ASR words that differ (both only with speech recognition), fill
  length vs the original region and a warning when a region did not stop. A/B maps the playhead between version and
  take; a region of the selected take can be regenerated on its own. Keeping a take makes a new version (one undo
  step each); export a WAV, or send the version back to the Direct take it came from.

## Architecture

- `src/api.ts` — the only server access: contract types, backend detection, speakers / embeddings / previews /
  uploads, streaming `generate` (float32 PCM). `src/fixtures/` — `controls.json` (a snapshot of the live
  `/v2/controls`), synthetic embeddings and the offline synthesizer.
- `src/state/engine.ts` — contract math: UI value → primitive doses (bipolar/unipolar maps, accents, presets),
  safe/strong/collapse zones, identity estimate, steered embedding `e' = (e+Σ)·|e|/|e+Σ|`, timing (duration
  factors).
- `src/state/` — signals per act: `design.ts` (Cast voice, undo via `history.ts`, roll, neighbours), `who.ts`
  (perceptual Who map from `ui.who`: measured library spots, base spot + gender/age doses), `geo.ts` + `atlas.ts`
  (Natural Earth 110m panels per tab, UK split into England/Scotland/Wales; pin depth → strength, borders blend), `library.ts` (voices, embeddings, saved cast),
  `requests.ts` (design + `RequestContext` → `/v2/generate` body, spans; signal-free, shared with `mcp/`), `script.ts` (Direct lines and
  their voices, non-overlapping painted spans, takes, stale takes, the timeline and mixdown, measured `control_span_rate` as the
  reading-position fallback), `edit.ts` (Edit: versions, draft
  regions and direction, takes, send back). `src/edit/regions.ts` is the pure region planner (difflib port);
  `src/fixtures/edit.ts` emulates the three edit endpoints offline. `projectSchema.ts` (document types, settings,
  summary, id remap; type imports only, shared with `mcp/`), `projectDoc.ts` (snapshot / restore), `media.ts`
  (SHA-256-addressed media), `projectStore.ts` (server and browser stores, revisions, re-addressing) and
  `projects.ts` (autosave, conflicts, voices on open, export / import, moving to the server) keep every act in a
  project; `src/components/Projects.tsx` is the switcher and launcher.
- `src/audio/` — `player.ts` (one AudioContext, clip and streaming playback, playhead signal), `render.ts`
  (request-keyed cache, de-dup, 4 parallel), `preview.ts` (always-on preview: 250 ms debounce, stale-request
  abort, A/B against the unsteered voice, auto-generate toggle, nothing rendered for what a load sets; `speak()`
  plays the current take or renders it now), `mix.ts` (sequences, WAV export).
- `src/state/moods.ts` (named moods and their landmarks) and `src/components/Moods.tsx` (mood chips, pad
  landmarks).
- `src/lib/pron.ts` (X-SAMPA ⇄ IPA, group parsing, sent text + offset map) and `src/components/Pron.tsx` (chips,
  group editor, Pronounce).
- `src/screens/{cast,direct,edit}/` and `src/components/` (zone-aware slider, canvas waveform, panels).

Keys: Space play/stop · A/B design vs original (Cast), version vs take (Edit) · ⌘Z / ⇧⌘Z undo/redo (versions in
Edit) · 1–3 acts (a `#scene` link opens Direct) · Edit: ⌘↵ generate, K keep, [ ] 0 zoom, ← → seek · ? help.
