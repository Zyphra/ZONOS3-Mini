# ZONOS3

Inference runtime for **ZONOS3 Mini**, Zyphra's multilingual text-to-speech model: three HTTP servers (a CUDA server
built on Mini-SGLang, a portable ggml server and an MLX server for Apple Silicon) that serve the Zonos Voice Studio, a
plain-PyTorch reference model, an offline Python API, and the GGUF conversion tools.

ZONOS3 Mini is an 11-layer language model that mixes gated-DeltaNet layers with global attention (one global layer
in four), uses Full AttnRes residuals and a latent mixture-of-experts. It reads SuperBPE text tokens and predicts
10 × 4096 dual-codec audio codes at 15 Hz. A causal multi-resolution iSTFT decoder turns the codes into 44.1 kHz
audio. Voices are cloned from a raw reference clip (encoded into the same codes) or from a 192-d MECT-B2 speaker
embedding.

## Layout

| Path | Contents |
| --- | --- |
| `model/` | The `zonos3` package: codec, speaker encoders, prompt / conditioning / steering / LACI code, the speech-editing runtime every server serves, the server plugin seam, registries and the Python workers (text normalization, speech recognition, forced alignment, the ggml server's Python host) |
| `servers/sgl/` | The `zonos3-sgl` package (`zonos3_sgl`): the CUDA server built on [Mini-SGLang](https://github.com/sgl-project/mini-sglang), its launch scripts and benchmark |
| `servers/ggml/` | The C++/ggml server and CLI (CPU, CUDA, Metal, Vulkan); ggml is the submodule `servers/ggml/ggml` |
| `servers/mlx/` | The `zonos3-mlx` package (`zonos3_mlx`): the Apple Silicon server running the ggml server's GGUF files in MLX |
| `servers/web/` | The `zonos3-web` package: the in-browser server (WebGPU in a Web Worker, the v2 API behind a fetch-compatible transport) the studio runs with `?api=web` |
| `studio/` | Zonos Voice Studio (Preact, pnpm); `studio/dist/` is the committed production build every server serves at `/` |
| `tools/` | Checkpoint and GGUF conversion tools (`zonos3-tools`) |
| `servers/api/` | The HTTP contract every server implements (`openapi.yaml`, its bundled `openapi.json`) and its conformance tests (`zonos3-api`) |
| `servers/space/` | The ZeroGPU Hugging Face Space (`app.py`) and its deploy script |
| `assets/` | Runtime assets shared by the servers: `default_voices/`, `steering/` (steering bundles), `laci_heads/` (LACI alignment heads) |
| `third_party/` | Third-party license texts (`third_party/licenses/`) and the FuseAlign submodule |

## Requirements

- Linux x86_64 with an NVIDIA GPU (tested on H100) and the CUDA 12.8 toolkit (`CUDA_HOME`, default
  `/usr/local/cuda-12.8`) for the sgl server; an Apple Silicon Mac for the MLX server
  ([servers/mlx/README.md](servers/mlx/README.md)).
- [uv](https://docs.astral.sh/uv/getting-started/installation/); the project pins Python 3.12.
- Access to the Hugging Face repo [`Zyphra/ZONOS3_Mini_release_ckpts`](https://huggingface.co/Zyphra/ZONOS3_Mini_release_ckpts)
  (LM, codec and speaker encoder weights). While it is private, set `HF_TOKEN` or run `huggingface-cli login`.
  Weights are downloaded into the Hugging Face cache on first use (about 15 GB).
- The FuseAlign forced aligner of speech editing: the submodule `third_party/fusealign-multilingual`
  ([Zyphra/fusealign-multilingual](https://github.com/Zyphra/fusealign-multilingual), package `fusealign-hub`,
  Apache-2.0) and its checkpoint from the Hugging Face repo [`Zyphra/fusealign`](https://huggingface.co/Zyphra/fusealign)
  (one 1.34 GB file, downloaded on the first alignment).

## Install

```bash
git clone --recurse-submodules git@github.com:Zyphra/ZONOS3.git
cd ZONOS3
uv sync --locked --all-packages
```

The repo is a uv workspace (`model`, `servers/sgl`, `servers/mlx`, `tools`, `api`); `uv sync` also installs the
FuseAlign package `fusealign-hub` from the submodule `third_party/fusealign-multilingual` (editable). `--group
phonemize` adds espeak-ng guesses for `POST /v2/phonemize` (GPL-3.0, opt-in); the `dev` group (installed by default)
holds the test tools. In an existing checkout, `git submodule update --init third_party/fusealign-multilingual
servers/ggml/ggml` fetches FuseAlign and ggml for the C++ server ([servers/ggml/README.md](servers/ggml/README.md)).
On macOS, `uv sync --locked --package zonos3-mlx` installs the MLX server.

## Serve

```bash
servers/sgl/scripts/serve.sh
```

The launcher sources `servers/sgl/scripts/env.sh` (CUDA toolchain, the TileLang gated-DeltaNet backend and the five
`ZONOS3_USE_TRAINING_*` exact-mode kernels) and serves the release LM
`hf://Zyphra/ZONOS3_Mini_release_ckpts/lm/rate30_20k_r2grpo` on `127.0.0.1:1919` with `--attention-backend fa --cache-type
naive`, two running requests and CUDA graphs. Extra server flags are appended (e.g. `servers/sgl/scripts/serve.sh --port 1920`).
The first start JIT-compiles kernels and builds the text-normalization grammar cache, which takes several minutes.
On Apple Silicon, `servers/mlx/scripts/serve.sh` serves the GGUF files of `Zyphra/ZONOS3_Mini_GGUF` on the same port
with the same variables. Without any server, the studio runs ZONOS3 in the browser on WebGPU (`?api=web`; models from
the GGUF files, see [servers/web/README.md](servers/web/README.md)).
Every server answers `/` with a page that opens either app: Zonos Voice Studio at `/studio`, or a lightweight
voice-cloning page at `/lite` (pick, upload, record or load a voice embedding, then speak with it; see
[studio/README.md](studio/README.md#lite-ui)).

| Variable | Effect |
| --- | --- |
| `SERVE_MODEL` | Checkpoint directory or `hf://` location (`--model-path`); default the release LM |
| `SERVE_VOICES` | Default-voice directory (`--tts-default-voices-dir`); default `assets/default_voices/` |
| `STEER_BUNDLE` | Steering bundle (`--steer-bundle`) behind every steering control (emotion axes, styles, accent directions, word emphasis, …); default `assets/steering/studio_bundle_mect60k_x10.json`, empty serves no controls |
| `PHONEME_LEXICON` | Gold pronunciation lexicon directory (`--tts-phoneme-lexicon`) for `POST /v2/phonemize` |
| `ZONOS3_STUDIO_PROJECTS` | Shared studio project store (`--studio-projects-dir`); empty (default) keeps projects in the browser |
| `LACI_HEADS` | `"lang=path ..."` LACI alignment heads (`--laci-heads`) |
| `LACI_ALL` | `1` passes LACI heads for every server language, from `LACI_HEADS_DIR` (default `assets/laci_heads/`): the directory's `*_heads.json` for every language, overridden by a `*_heads_<speech code>.json` (`*_heads_zh.json` for Chinese); `en_gb` included |
| `PLUGINS` | `"name ..."` server plugins (`--plugins`), e.g. `dub` from [Zyphra/ZONOS3-dub](https://github.com/Zyphra/ZONOS3-dub) |

## Generate

Every request should carry a voice: a reference clip (`speaker_audio_base64`) or the id of a voice listed by
`GET /v2/speakers` (`speaker_embedding_id`). Without one, generation is unconditioned.

```bash
# Request body with a reference clip
python3 - <<'EOF' > body.json
import base64, json
print(json.dumps({"text": "Hello from ZONOS3.", "language": "en_us", "stream": False,
                  "speaker_audio_base64": base64.b64encode(open("assets/default_voices/AmericanFemale.mp3", "rb").read()).decode()}))
EOF

# Whole utterance: raw float32 little-endian PCM, 44.1 kHz mono (the seed used comes back in X-Seed)
curl -X POST http://127.0.0.1:1919/v2/generate -H 'Content-Type: application/json' -d @body.json --output hello.pcm
ffmpeg -f f32le -ar 44100 -ac 1 -i hello.pcm hello.wav

# OpenAI-compatible endpoint: "voice" is a speaker id of GET /v2/speakers; response_format pcm (default) or wav
VOICE=$(curl -s http://127.0.0.1:1919/v2/speakers | python3 -c 'import json, sys; print(json.load(sys.stdin)["speakers"][0]["id"])')
curl -X POST http://127.0.0.1:1919/v1/audio/speech -H 'Content-Type: application/json' \
  -d "{\"model\": \"zonos3\", \"input\": \"Hello from ZONOS3.\", \"voice\": \"$VOICE\", \"response_format\": \"wav\"}" \
  --output hello2.wav
```

`stream: true` returns the audio progressively (an incremental decode of the causal decoder); `"format": "wav"` returns a
buffered 16-bit WAV instead of raw PCM (with `stream: false`). Open `http://127.0.0.1:1919/studio` for the Zonos Voice
Studio (its steering controls come from the default `STEER_BUNDLE`). The API
guide is [docs/api.md](docs/api.md); the schemas are in [servers/api/openapi.yaml](servers/api/openapi.yaml).

## Offline Python API

```bash
source servers/sgl/scripts/env.sh
uv run --locked --package zonos3-sgl python
```

```python
from zonos3_sgl.tts import TTSLLM, TTSSamplingParams

# The server's settings: FlashAttention, per-request (naive) cache, and half the GPU for the KV cache so the codec
# encoder and decoder fit next to it.
tts = TTSLLM(model_path="hf://Zyphra/ZONOS3_Mini_release_ckpts/lm/rate30_20k_r2grpo",
             attention_backend="fa", cache_type="naive", memory_ratio=0.5)

result = tts.generate_one("Hello from the offline Python API.", TTSSamplingParams(seed=42), language="en_us")
tts.save_audio(result["audio"], "offline.wav")

# Clone a voice from a reference clip (raw reference prefix) or a speaker embedding
(cloned,) = tts.generate(["This is spoken in the cloned voice."], TTSSamplingParams(seed=42),
                         speaker_reference_audio="assets/default_voices/AmericanFemale.mp3")
emb = tts.embed_speaker_file("assets/default_voices/AmericanFemale.mp3")
by_embedding = tts.generate_one("Same voice, from its embedding.", TTSSamplingParams(seed=42), speaker_embedding=emb)
```

The offline path applies the server's text normalization and conditioning (speaker, condition rows, accents, LACI).

## Reference model

`model/` holds a plain-PyTorch implementation of the full generation path (`zonos3.reference`), the numerical oracle
for both servers. It runs on CPU or GPU:

```bash
uv run python -m zonos3.reference --model hf://Zyphra/ZONOS3_Mini_release_ckpts/lm/rate30_20k_r2grpo \
  --text "Hello from the reference." -o ref.wav --device cuda
```

`--request req.json` takes a v2 `GenerateRequest` instead of `--text`. GGUF files for the ggml server are built from
the same modules with `uv run --package zonos3-tools python tools/convert_gguf.py <subcommand>` (`--help` lists them).

## Features

| Feature | Where |
| --- | --- |
| HTTP API (`/v2/generate`, `/v1/audio/speech`, `/v2/phonemize`, …) | [docs/api.md](docs/api.md), [servers/api/openapi.yaml](servers/api/openapi.yaml) |
| Raw-reference and embedding voice cloning, speaker CFG, condition rows | [docs/guide.md](docs/guide.md) |
| Steering controls (emotion axes, moods, styles, voice quality, accent directions) and accent adapters | [docs/guide.md](docs/guide.md) |
| Speech editing | [docs/guide.md](docs/guide.md) |
| Pronunciation input (phoneme tokens) and nonverbal tags with tag CFG | [docs/guide.md](docs/guide.md) |
| Zonos Voice Studio, server projects | [docs/guide.md](docs/guide.md), [studio/README.md](studio/README.md) |
| LACI, duration targets and server plugins (Dub: [Zyphra/ZONOS3-dub](https://github.com/Zyphra/ZONOS3-dub)) | [docs/guide.md](docs/guide.md#plugins) |
| Feature overview | [docs/features.md](docs/features.md) |
| Runtime architecture and vocoder | [docs/tts_architecture.md](docs/tts_architecture.md), [docs/structures.md](docs/structures.md) |
| Release checkpoints and registries | [docs/release_checkpoints.md](docs/release_checkpoints.md) |

### Feature prerequisites

- **Speech recognition** (optional for speech editing: transcripts of clips sent without one, and `verify`) is
  Qwen3-ASR (`Qwen/Qwen3-ASR-1.7B`) in its own venv:
  `bash tools/setup_asr_env.sh` once creates `.venv-asr` (qwen-asr pins its own transformers). One worker process per
  server (`model/src/zonos3/workers/asr.py`) starts on first use; `ZONOS3_ASR_PYTHON` overrides its interpreter
  (default `.venv-asr/bin/python`) and `ZONOS3_ASR_DEVICE` its device (default `cuda:0`, else CPU). Without it
  `/v2/capabilities` reports `edit_transcribe: false`, and editing needs the clip's transcript.
- **Speech editing** needs the speech-edit LoRA (sgl: `--accent-adapters <dir with
  lm/rate30_20k_r2grpo/edit_lora/z3m_omni3b_edit_lora.pt>` on the release LM; see [docs/guide.md](docs/guide.md)) or an
  edit checkpoint (`SERVE_MODEL=<edit checkpoint>`) and FuseAlign
  (`fusealign-hub` in the workspace; its checkpoint downloads from `Zyphra/fusealign`) for the word timings.
- **Phoneme input** needs the release phoneme tokens
  (`lm/rate30_20k_r2grpo/phoneme_tokens/phoneme_tokens_s15000.pt`; sgl / reference `--phoneme-rows`, GGUF
  `convert_gguf.py lm --phoneme-rows`) or a checkpoint merged with them (`tools/add_text_extra_rows.py`), and
  `PHONEME_LEXICON=<lexicon dir>` (`lexicons/gold_phonemes_v1` of the release repo) for "Phonemize selection".
  **Nonverbal tags** need a checkpoint with the NV rows.
- **Forced alignment** (FuseAlign, speech editing) runs in one worker process per server
  (`model/src/zonos3/workers/fusealign.py`, the workspace interpreter), started on first use;
  `ZONOS3_FUSEALIGN_DEVICE` picks its device (default `cuda:0`, else CPU). Recordings over 30 s (or transcripts over
  1022 UTF-8 bytes) are aligned long-form, anchored by Qwen3-ASR transcripts of their voice-activity chunks (at most
  25 s each), or without speech recognition by FuseAlign's own voice-activity map; FuseAlign's built-in Whisper
  anchoring is never used.
- **Accent adapters** need `--accent-adapters <dir>`.

Lingua training checkpoints are converted with `tools/convert_lingua_checkpoint.py`.

## Tests

```bash
source servers/sgl/scripts/env.sh
uv run --locked --group phonemize pytest
```

Tests that load weights run when `ZONOS3_TEST_MODEL` names a checkpoint (e.g. the release LM) and a GPU is present.
The API conformance suite runs against a live server of any kind (tests of capabilities the server lacks skip):

```bash
uv run --package zonos3-api pytest servers/api/tests/conformance --base-url http://127.0.0.1:1919
```

The ggml parity tests compare the C++ build with the reference:
`uv run --package zonos3-tools pytest servers/ggml/tests/parity --ggml-build servers/ggml/build`. On Apple Silicon,
`uv run --package zonos3-mlx pytest servers/mlx/tests` runs the MLX server's tests (they skip elsewhere).
`pnpm install` at the root installs the pnpm workspace (`studio`, `studio/mcp`, `servers/web`; one `pnpm-lock.yaml`);
every `typecheck`, `build` and `test` script regenerates the untracked `src/api/schema.ts` from `servers/api/openapi.json`
first (`gen:api`). `pnpm --dir servers/web test` runs the in-browser server's unit tests, `pnpm --dir studio/mcp test`
the MCP server's, and `pnpm --dir servers/web parity` the WebGPU parity checks against CPU ggml goldens in headless
Chrome.

## License

ZONOS3 is released under the [MIT License](LICENSE). Third-party components are listed in [`NOTICE`](NOTICE) and
[`third_party/licenses/`](third_party/licenses/).
