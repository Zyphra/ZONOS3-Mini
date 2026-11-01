# ZONOS3

Inference runtime for **ZONOS3 Mini**, Zyphra's multilingual text-to-speech model: an HTTP server with a web UI and
the Zonos Voice Studio, and an offline Python API.

ZONOS3 Mini is an 11-layer language model that mixes gated-DeltaNet layers with global attention (one global layer
in four), uses Full AttnRes residuals and a latent mixture-of-experts. It reads SuperBPE text tokens and predicts
10 × 4096 dual-codec audio codes at 15 Hz. A causal multi-resolution iSTFT decoder turns the codes into 44.1 kHz
audio. Voices are cloned from a raw reference clip (encoded into the same codes) or from a 192-d MECT-B2 speaker
embedding.

The server is built on [Mini-SGLang](https://github.com/sgl-project/mini-sglang).

## Requirements

- Linux x86_64 with an NVIDIA GPU (tested on H100) and the CUDA 12.8 toolkit (`CUDA_HOME`, default
  `/usr/local/cuda-12.8`).
- [uv](https://docs.astral.sh/uv/getting-started/installation/); the project pins Python 3.12.
- Access to the Hugging Face repo [`Zyphra/ZONOS3_Mini_release_ckpts`](https://huggingface.co/Zyphra/ZONOS3_Mini_release_ckpts)
  (LM, codec and speaker encoder weights). While it is private, set `HF_TOKEN` or run `huggingface-cli login`.
  Weights are downloaded into the Hugging Face cache on first use (about 15 GB).

## Install

```bash
git clone git@github.com:Zyphra/ZONOS3.git
cd ZONOS3
uv sync --locked
```

`--group phonemize` adds espeak-ng guesses for `POST /tts/phonemize` (GPL-3.0, opt-in); `--extra dev` adds the test
tools.

## Serve

```bash
scripts/serve.sh
```

The launcher sources `scripts/env.sh` (CUDA toolchain, the TileLang gated-DeltaNet backend and the five
`ZONOS3_USE_TRAINING_*` exact-mode kernels) and serves the release LM
`hf://Zyphra/ZONOS3_Mini_release_ckpts/lm/mect_60k` on `127.0.0.1:1919` with `--attention-backend fa --cache-type
naive`, two running requests and CUDA graphs. Extra server flags are appended (e.g. `scripts/serve.sh --port 1920`).
The first start JIT-compiles kernels and builds the text-normalization grammar cache, which takes several minutes.

| Variable | Effect |
| --- | --- |
| `SERVE_MODEL` | Checkpoint directory or `hf://` location (`--model-path`); default the release LM |
| `SERVE_VOICES` | Default-voice directory (`--tts-default-voices-dir`); default `default_voices/` |
| `EMOTION_SERVE_DIRECTIONS` | Emotion direction set; default `steering/emotion_directions_mect60k_axes` |
| `ACCENT_SERVE_DIRECTIONS` | Accent direction set (`--accent-directions`), e.g. `steering/accent_directions_mect60k` |
| `STEER_BUNDLE` | Steering bundle (`--steer-bundle`) behind the studio's controls: `steering/studio_bundle_mect60k.json` |
| `PHONEME_LEXICON` | Gold pronunciation lexicon directory (`--tts-phoneme-lexicon`) for `POST /tts/phonemize` |
| `ZONOS3_STUDIO_PROJECTS` | Shared studio project store (`--studio-projects-dir`); empty (default) keeps projects in the browser |
| `LACI_HEADS` | `"lang=path ..."` LACI alignment heads (`--laci-heads`) |
| `DUB` | `1` serves the studio's Dub screen: LACI heads for every dub language from `LACI_HEADS_DIR` (default `laci_heads/`) and the alignment worker of `scripts/setup_dub_align_env.sh` |

## Generate

Every request should carry a voice: a reference clip (`speaker_audio_base64`) or the id of a voice listed by
`GET /tts/speakers` (`speaker_embedding_id`). Without one, generation is unconditioned.

```bash
# Request body with a reference clip
python3 - <<'EOF' > body.json
import base64, json
print(json.dumps({"text": "Hello from ZONOS3.", "language": "en_us", "stream": False,
                  "speaker_audio_base64": base64.b64encode(open("default_voices/AmericanFemale.mp3", "rb").read()).decode()}))
EOF

# Whole utterance: raw float32 little-endian PCM, 44.1 kHz mono
curl -X POST http://127.0.0.1:1919/tts/generate -H 'Content-Type: application/json' -d @body.json --output hello.pcm
ffmpeg -f f32le -ar 44100 -ac 1 -i hello.pcm hello.wav

# OpenAI-compatible endpoint (same raw PCM; "input" instead of "text")
python3 -c 'import json; b = json.load(open("body.json")); print(json.dumps({"model": "zonos3", "input": b["text"], "speaker_audio_base64": b["speaker_audio_base64"]}))' > openai.json
curl -X POST http://127.0.0.1:1919/v1/audio/speech -H 'Content-Type: application/json' -d @openai.json --output hello2.pcm
```

`stream: true` returns the audio progressively (prefix re-decode of the causal decoder). Open
`http://127.0.0.1:1919/` for the web UI and `http://127.0.0.1:1919/studio/` for the Zonos Voice Studio (start the
server with `STEER_BUNDLE=steering/studio_bundle_mect60k.json` for its steering controls). The full request
reference is in [docs/api.md](docs/api.md).

## Offline Python API

```bash
source scripts/env.sh
uv run --locked python
```

```python
from zonos3.message import TTSSamplingParams
from zonos3.tts import TTSLLM

# The server's settings: FlashAttention, per-request (naive) cache, and half the GPU for the KV cache so the codec
# encoder and decoder fit next to it.
tts = TTSLLM(model_path="hf://Zyphra/ZONOS3_Mini_release_ckpts/lm/mect_60k",
             attention_backend="fa", cache_type="naive", memory_ratio=0.5)

result = tts.generate_one("Hello from the offline Python API.", TTSSamplingParams(seed=42), language="en_us")
tts.save_audio(result["audio"], "offline.wav")

# Clone a voice from a reference clip (raw reference prefix) or a speaker embedding
(cloned,) = tts.generate(["This is spoken in the cloned voice."], TTSSamplingParams(seed=42),
                         speaker_reference_audio="default_voices/AmericanFemale.mp3")
emb = tts.embed_speaker_file("default_voices/AmericanFemale.mp3")
by_embedding = tts.generate_one("Same voice, from its embedding.", TTSSamplingParams(seed=42), speaker_embedding=emb)
```

The offline path applies the server's text normalization and conditioning (speaking rate, quality, condition rows,
accents, LACI).

## Features

| Feature | Where |
| --- | --- |
| HTTP API (`/tts/generate`, `/v1/audio/speech`, `/tts/phonemize`, …) | [docs/api.md](docs/api.md) |
| Raw-reference and embedding voice cloning, speaker CFG, condition rows | [docs/guide.md](docs/guide.md) |
| Emotion, mood, style and voice-quality steering; accent directions and adapters | [docs/guide.md](docs/guide.md) |
| Speech editing | [docs/guide.md](docs/guide.md) |
| Pronunciation input (phoneme tokens) and nonverbal tags with tag CFG | [docs/guide.md](docs/guide.md) |
| Zonos Voice Studio, server projects, dubbing | [docs/guide.md](docs/guide.md), [ui/README.md](ui/README.md) |
| LACI duration control and the Dub pipeline | [docs/laci_dub.md](docs/laci_dub.md) |
| Feature overview | [docs/features.md](docs/features.md) |
| Runtime architecture and vocoder | [docs/tts_architecture.md](docs/tts_architecture.md), [docs/structures.md](docs/structures.md) |
| Release checkpoints and registries | [docs/release_checkpoints.md](docs/release_checkpoints.md) |

### Feature prerequisites

- **Speech editing** needs an edit checkpoint (`SERVE_MODEL=<edit checkpoint>`) and `ZSA_ROOT` pointing at a
  [z-speech-alignment](https://github.com/Zyphra/z-speech-alignment) checkout with its uv environment (`uv sync` in
  it); `ZSA_PYTHON` overrides its interpreter.
- **Phoneme input** and **nonverbal tags** need checkpoints merged with the phoneme / NV rows
  (`scripts/add_text_extra_rows.py`) and `PHONEME_LEXICON=<lexicon dir>`.
- **Dubbing** needs `scripts/setup_dub_align_env.sh` once, then `DUB=1 scripts/serve.sh`.
- **Accent adapters** need `--accent-adapters <dir>`.

Lingua training checkpoints are converted with `scripts/convert_lingua_checkpoint.py`.

## Tests

```bash
source scripts/env.sh
uv run --locked --extra dev python -m pytest
```

Tests that load weights run when `ZONOS3_TEST_MODEL` names a checkpoint (e.g. the release LM) and a GPU is present.

## License

ZONOS3 is released under the [MIT License](LICENSE). Third-party components are listed in [`NOTICE`](NOTICE) and
[`licenses/`](licenses/).
