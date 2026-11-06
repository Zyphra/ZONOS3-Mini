# zonos3-web — the in-browser server

ZONOS3 Mini running in the browser on hand-written WebGPU (WGSL) inside a Web Worker, serving the v2 API of
[`api/openapi.yaml`](../../api/openapi.yaml) through a fetch-compatible in-process transport. The studio uses it as
its `web` backend (`?api=web`, and in `auto` when no server answers). Adapted from web-zonos3.

| Served | Off (capabilities say so; the browser cannot run their Python workers) |
| --- | --- |
| `/v2/generate` (streamed / buffered, pcm / wav), `/v1/audio/speech` | speech recognition: `/v2/edit/transcribe`, edits without `original_text` whose words are read, `verify` (503 "Speech recognition is not available on this server: …"; `edit_transcribe` false) |
| voices: default voices (shipped codes), session uploads (reference encoder + MECT speaker encoder), embedding files, slerp blends | plugins (`plugins` is empty) |
| `/v2/controls` steering-bundle controls (with set calibrations), condition rows | phonemize, text normalization |
| speaker, tag and controls CFG; LACI and span-follow reading | server projects (the studio keeps projects in the browser) |
| speech editing with the clip's transcript: `/v2/edit/align` (FuseAlign, clips ≤ 30 s: `edit_align_max_seconds`), `/v2/edit` (clips ≤ 150 s; `original_text` aligns clips ≤ 30 s, longer clips need `original_words`, or `spans` within `max_context_seconds`) | accent adapters (`accents` is empty; the speech-edit adapter is merged into the `edit` backbone) |
| capabilities, health, models | |

## Layout

| Path | Contents |
| --- | --- |
| `src/client.ts` | Main-thread facade (`dist/engine.js`): `engine.init`, `localFetch`; decodes audio the worker cannot (default voices, `speaker_audio_base64` up to 60 s, the edit routes' `audio_base64` up to 150 s, sent with the upload's whole length); spawns the engine worker and the codec worker and connects them with a `MessageChannel` |
| `src/worker.ts` | The worker: loads models and companions, warms up, serves requests one at a time |
| `src/codec-worker.ts`, `src/spec-sink.ts` | The codec's host half (iSTFT, FIR mixer, resampler) in its own worker: the engine worker's `ZacStream` hands it each chunk's spectra, so prefix-streamed chunks overlap the following backbone steps |
| `src/api/` | The routes (port of `servers/ggml/src/server/`): `router`, `generate` (resolve + one-request scheduler), `edit` (speech editing: `zonos3.tts.edit_runtime` without speech recognition), `speakers`, `capabilities`, `validate` (spec validation, FastAPI 422 shape) |
| `src/edit/` | Pure ports of the speech-edit logic (`zonos3.tts.edit`, `edit_align`, `align_text`, `textmatch`, `edit_runtime.region_doses`: edit tokens, word diff, context windows, fill lengths, edit prompt, fill extraction, splice) and the fill runner (span-end stop, `min_frames`, codes only) with the whole-clip decode |
| `src/aligner/` | FuseAlign on WebGPU (Dia text encoder, joint transformer, refinement heads), one forward per clip ≤ 30 s |
| `src/request/` | Pure ports of the ggml request logic: steering bundle, markup, LACI, span follow |
| `src/generation.ts` | The generation registry ([`model/src/zonos3/registries/generation.json`](../../model/src/zonos3/registries/generation.json)): sampling defaults, the API rate, tag-CFG range, silence prefill (default 3, at most 30 frames), decode context (30), the 900-frame reference prefix (longer reference codes keep their first 900 frames), the 2000-frame budget (shrunk to the slot rows the prompt leaves), 60 s reference clips |
| `src/backbone.ts`, `zac.ts`, `istft.ts`, `speaker.ts`, `encoder.ts`, `ssl.ts`, `rvq.ts` | The networks on WebGPU (`src/gpu/kernels/*.wgsl`) and their host parts. `ZacStream.create` times the codec gemms at load and repacks the matrices of each row class to k-major f16 (`src/gpu/h8.ts`, `gemm_h8*.wgsl`) when an h8 kernel beats the q8 one on the device |
| `scripts/export_web_models.py` | GGUF → sharded web format (`zonos3-web-v1`; the backbone's `output.weight` becomes q8_0), voices, companions, LACI heads |
| `harness/` | Headless-Chrome harness: `server.mjs` (HTTP front for conformance), `parity.mjs` (WebGPU parity checks), `bench.mjs` (desktop / Android benchmark) |
| `worker/`, `wrangler.jsonc` | Cloudflare deployment: the studio with the engine as static assets, models from R2 |

## Build and run

```bash
hf download Zyphra/ZONOS3_Mini_GGUF --local-dir servers/ggml/models
pnpm --dir servers/web export          # public/models/ (~5.4 GB; first load fetches backbone + zac (zac-keep2.gguf), ~1.0 GB; uploads add encoder + teacher, ~2.8 GB, and spk; speech editing adds edit (~1.1 GB) and aligner (~0.5 GB))
pnpm --dir servers/web build           # dist/engine.js + dist/worker.js + dist/codec-worker.js
pnpm --dir studio dev                  # http://localhost:5173/studio?api=web (or /lite?api=web)
```

The studio's dev and preview servers serve `dist/` at `/web-engine` and `public/models/` at `/models`. The worker
caches shards in Cache Storage (`zonos3-web-models-v4`; the name changes whenever an export reuses shard URLs, so a
visitor never mixes old and new shards); `?models=URL` loads them from elsewhere. Models that load
together report as one readout (`engine.init(progress)`: backbone + zac; `engine.onEncoderProgress(cb)`: spk, or
encoder + teacher on an uploaded clip's first use, edit on the first edit, aligner on the first alignment; default
voices ship their codes and speaker embedding beside the clip, so they load neither encoder): `{models, loaded, total, fetched}` in tensor bytes, `total` known from
the first readout (manifests are read first), at most every 100 ms; `fetched` counts the network bytes so a page can
tell a download rate from Cache Storage reads. Without WebGPU (or
with adapter limits too low) `engine.init` rejects and the studio shows the error. The speaker encoder clamps clips to
what one storage binding of the device holds (60 s on Apple Silicon); an edit's speaker row embeds the clip's first
60 s. Speech editing is offered (`speech_editing`) when the export has both `edit/` and `aligner/`.

## Tests

```bash
pnpm --dir servers/web test            # unit tests (Node): tokenizer, sampler, prompt, iSTFT, companions, LACI, API router and edit routes, request rules (model/tests/fixtures/request_errors.json), the speech-edit logic against tests/edit/goldens.json
.venv/bin/python servers/web/tests/edit/make_goldens.py   # regenerate tests/edit/goldens.json from the Python edit logic
pnpm --dir servers/web goldens         # CPU ggml goldens (needs a CPU ggml build at $GGML_BUILD, default /tmp/ggml-build)
pnpm --dir servers/web parity          # WebGPU parity vs the goldens in headless Chrome
pnpm --dir servers/web harness         # HTTP front on :1920, then:
uv run --package zonos3-api pytest api/tests/conformance --base-url http://127.0.0.1:1920
```

The harness and parity runner launch Chrome from `CHROME_PATH`, else `harness/.chrome`
(`pnpm --dir servers/web exec npx @puppeteer/browsers install chrome@stable --path harness/.chrome`);
`HARNESS_HEADFUL=1` opens a visible window.

## Benchmark

```bash
node servers/web/harness/bench.mjs [--mode e2e|micro] [--runs 3]                  # local headless Chrome
ADB=…/platform-tools/adb node servers/web/harness/bench.mjs --android SERIAL [--browser com.android.chrome]
```

`e2e` times engine init, then streamed `/v2/generate` runs (plain and speaker CFG): `ttfa_ms` (first audio byte) and
`rtf` (wall time / audio time). `micro` times one backbone decode step (`bb_step_ms`), the CFG pair in one submission
(`bb_pair_ms`), one 16-frame codec chunk (`zac_chunk_ms`) and the host iSTFT + resampler of 256 columns (`istft_ms`),
and prints the codec gemm plan. `--android` opens the page in the phone's browser over `adb reverse`
(`http://127.0.0.1` is a secure context there); `--browser` names another browser package.

## Deploy (Cloudflare)

```bash
pnpm --dir servers/web upload-models --remote      # public/models -> R2 bucket zonos3-studio-models
pnpm --dir servers/web deploy                      # studio build + engine -> Workers Static Assets
```

`pnpm --dir servers/web upload-models --local && pnpm --dir servers/web exec wrangler dev` runs it locally.
