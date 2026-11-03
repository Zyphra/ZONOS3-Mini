# zonos3-web — the in-browser server

ZONOS3 Mini running in the browser on hand-written WebGPU (WGSL) inside a Web Worker, serving the v2 API of
[`api/openapi.yaml`](../../api/openapi.yaml) through a fetch-compatible in-process transport. The studio uses it as
its `web` backend (`?api=web`, and in `auto` when no server answers). Adapted from web-zonos3.

| Served | Off (capabilities say so; the browser cannot run their Python workers) |
| --- | --- |
| `/v2/generate` (streamed / buffered, pcm / wav), `/v1/audio/speech` | speech editing, dubbing (`/v2/dub/capabilities` answers `enabled: false`) |
| voices: default voices (shipped codes), session uploads (reference encoder + MECT speaker encoder), embedding files, slerp blends | phonemize, text normalization |
| `/v2/controls` steering-bundle controls (with set calibrations), condition rows | server projects (the studio keeps projects in the browser) |
| speaker, tag and controls CFG; LACI and span-follow reading | accent adapters (`accents` is empty) |
| capabilities, health, models | |

## Layout

| Path | Contents |
| --- | --- |
| `src/client.ts` | Main-thread facade (`dist/engine.js`): `engine.init`, `localFetch`; decodes audio the worker cannot (default voices, `speaker_audio_base64`) |
| `src/worker.ts` | The worker: loads models and companions, warms up, serves requests one at a time |
| `src/api/` | The routes (port of `servers/ggml/src/server/`): `router`, `generate` (resolve + one-request scheduler), `speakers`, `capabilities`, `validate` (spec validation, FastAPI 422 shape) |
| `src/request/` | Pure ports of the ggml request logic: steering bundle, markup, LACI, span follow |
| `src/generation.ts` | The generation registry ([`model/src/zonos3/registries/generation.json`](../../model/src/zonos3/registries/generation.json)): sampling defaults, the API rate, tag-CFG range, silence prefill (default 3, at most 30 frames), decode context (30), the 900-frame reference prefix (longer reference codes keep their first 900 frames), the 2000-frame budget (shrunk to the slot rows the prompt leaves), 60 s reference clips |
| `src/backbone.ts`, `zac.ts`, `istft.ts`, `speaker.ts`, `encoder.ts`, `ssl.ts`, `rvq.ts` | The networks on WebGPU (`src/gpu/kernels/*.wgsl`) and their host parts |
| `scripts/export_web_models.py` | GGUF → sharded web format (`zonos3-web-v1`), voices, companions, LACI heads |
| `harness/` | Headless-Chrome harness: `server.mjs` (HTTP front for conformance), `parity.mjs` (WebGPU parity checks) |
| `worker/`, `wrangler.jsonc` | Cloudflare deployment: the studio with the engine as static assets, models from R2 |

## Build and run

```bash
hf download Zyphra/ZONOS3_Mini_GGUF --local-dir servers/ggml/models
pnpm --dir servers/web export          # public/models/ (~4.5 GB; first load fetches backbone + zac, ~1.9 GB; uploads add encoder + teacher, ~2.6 GB, and spk)
pnpm --dir servers/web build           # dist/engine.js + dist/worker.js
pnpm --dir studio dev                  # http://localhost:5173/studio?api=web (or /lite?api=web)
```

The studio's dev and preview servers serve `dist/` at `/web-engine` and `public/models/` at `/models`. The worker
caches shards in Cache Storage (`zonos3-web-models-v2`); `?models=URL` loads them from elsewhere. Models that load
together report as one readout (`engine.init(progress)`: backbone + zac; `engine.onEncoderProgress(cb)`: spk, or
encoder + teacher on a clip voice's first use): `{models, loaded, total, fetched}` in tensor bytes, `total` known from
the first readout (manifests are read first), at most every 100 ms; `fetched` counts the network bytes so a page can
tell a download rate from Cache Storage reads. Without WebGPU (or
with adapter limits too low) `engine.init` rejects and the studio shows the error. The speaker encoder clamps clips to
what one storage binding of the device holds (60 s on Apple Silicon).

## Tests

```bash
pnpm --dir servers/web test            # unit tests (Node): tokenizer, sampler, prompt, iSTFT, companions, LACI, API router, request rules (model/tests/fixtures/request_errors.json)
pnpm --dir servers/web goldens         # CPU ggml goldens (needs a CPU ggml build at $GGML_BUILD, default /tmp/ggml-build)
pnpm --dir servers/web parity          # WebGPU parity vs the goldens in headless Chrome
pnpm --dir servers/web harness         # HTTP front on :1920, then:
uv run --package zonos3-api pytest api/tests/conformance --base-url http://127.0.0.1:1920
```

The harness and parity runner launch Chrome from `CHROME_PATH`, else `harness/.chrome`
(`pnpm --dir servers/web exec npx @puppeteer/browsers install chrome@stable --path harness/.chrome`);
`HARNESS_HEADFUL=1` opens a visible window.

## Deploy (Cloudflare)

```bash
pnpm --dir servers/web upload-models -- --remote   # public/models -> R2 bucket zonos3-studio-models
pnpm --dir servers/web deploy                      # studio build + engine -> Workers Static Assets
```

`pnpm --dir servers/web upload-models -- --local && pnpm --dir servers/web exec wrangler dev` runs it locally.
