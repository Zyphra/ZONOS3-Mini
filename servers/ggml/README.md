# zonos3.cpp

Native ggml/GGUF inference for **ZONOS3 Mini** ([Zyphra/ZONOS3_Mini_release_ckpts](https://huggingface.co/Zyphra/ZONOS3_Mini_release_ckpts), revision `4d641a9fedd8c5544fca6e6f17ebb5c351c1a510`), LM `lm/mect_60k` at step **60000**. The 11-layer, 1024-wide backbone mixes Gated DeltaNet and global attention with Full AttnRes and top-2/64-expert latent MoE. SuperBPE-8192 text plus a speaker prompt and three condition rows produce ten delayed 4096-entry codebooks at 15 Hz; the multi-resolution iSTFT decoder renders 44.1 kHz mono audio.

Voice cloning has two modes:

- **Speaker embedding** (default): a native MECT-B2 speaker encoder (`spk-encoder.gguf`) turns up to 60 s of reference audio into a 192-d vector that the backbone's `speaker_projection` maps onto one prompt row. Saved embeddings need no encoder and no ffmpeg.
- **Raw reference** (opt-in): the reference is encoded to ordinary codec codes (`dac-encoder.gguf` + `ssl-teacher.gguf`) and prepended to the prompt. Saved codes need no encoder and no ffmpeg.

Every prompt also carries three **condition rows** (loudness, bandwidth `fmax`, speaking rate; each a 10-bucket class, 10 = unknown), an optional **speaker CFG** twin pass, and a **codec-silence prefill** after the text.

CLI/server synthesis and reference encoding run natively without Python. Python is needed for conversion and numerical reference checks; optional NeMo text normalization is a separate Python helper. All project interfaces use `zonos3`/`ZONOS3`; obsolete binaries, flags, environment aliases, and GGUF schemas (including the earlier 375k backbone and the conv-tail decoder) are rejected.

## Build

```bash
git clone --recurse-submodules https://github.com/Zyphra/zonos3.cpp.git
cd zonos3.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# CUDA:
cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j
```

The optional desktop shell uses `-DZONOS3_APP=ON`, CMake >= 3.31, a compatible C++23 compiler, and platform webview dependencies. CPU/CUDA numerical checks are described in [internals](docs/INTERNALS.md#reference-comparison); Metal, Vulkan, and Windows builds depend on the installed toolchain and backend operators.

## Prepare models

The default backbone is the Q4_0-expert/Q8_0-mixer QAT model `zonos3-qat.gguf` (0.66 GB). Download the converted files from the private [Zyphra/ZONOS3_Mini_GGUF](https://huggingface.co/Zyphra/ZONOS3_Mini_GGUF) repository (authenticate with an account that can read it):

```bash
hf auth login
hf download Zyphra/ZONOS3_Mini_GGUF zonos3-qat.gguf dac.gguf spk-encoder.gguf --local-dir models
# Optional raw-reference cloning companions:
hf download Zyphra/ZONOS3_Mini_GGUF dac-encoder.gguf ssl-teacher.gguf --local-dir models
# Optional voice-design steering directions (request `controls`, the studio's design controls, dub tempo buckets):
hf download Zyphra/ZONOS3_Mini_GGUF steer.gguf --local-dir models
# Optional emotion and accent steering directions (`emotion_*`, `accent`):
hf download Zyphra/ZONOS3_Mini_GGUF emotion.gguf accent-directions.gguf --local-dir models
```

The server picks up every companion it finds next to the backbone. Backbones downloaded before 2026-11-01 lack
`zonos3.text_tokenizer_spec` and are refused: re-download `zonos3-qat.gguf` (and `steer.gguf`, which gained the dub's
`rate_extension`). Files downloaded before 2026-10-27 are the previous fullenc_120k release; re-download
`zonos3-qat.gguf` and `spk-encoder.gguf` together (the backbone expects embeddings from the matching speaker encoder).

Or convert the checkpoints locally as below.

### Convert the release

Download the private release (authenticate with an account that can read it) or use a local copy:

```bash
hf auth login
R=release   # or a local copy such as /data/groups/audio/gabriel/hf_release_zonos3mini_20261020
hf download Zyphra/ZONOS3_Mini_release_ckpts \
  --revision 4d641a9fedd8c5544fca6e6f17ebb5c351c1a510 --local-dir "$R"

T="uv run --package zonos3-tools python ../../tools/convert_gguf.py"   # from servers/ggml
mkdir -p models
# F16 reference backbone from the release:
$T lm "$R/lm/mect_60k" --outtype f16 -o models/zonos3-f16.gguf
# Default backbone: QAT directory (same layout as the release LM plus bake.json; step 3462). Q4_0 experts and
# Q8_0 mixers are stored bit-exactly, so there is no outtype to choose.
$T lm QAT/lm/mect_60k_q4q8_qat \
  --silence-frame "$R/registries/silence_frame_opus_dual_3n_600k_sched600k_r4096_step600000.json" \
  -o models/zonos3-qat.gguf
$T codec-decoder "$R/autoencoder/generator.pt" --head "$R/autoencoder/heads/full_mr" -o models/dac.gguf
$T speaker-encoder "$R/speaker_encoders/ownmect_s2/ckpt_0037.pth" -o models/spk-encoder.gguf

# Optional: raw-reference cloning companions.
$T codec-encoder "$R/autoencoder/generator.pt" --outtype f32 -o models/dac-encoder.gguf
$T ssl-teacher "$R/autoencoder/teacher" --outtype f32 -o models/ssl-teacher.gguf

# Optional: steering, emotion and accent companions (repo-root steering/ assets).
$T steer ../../steering/studio_bundle_mect60k.json -o models/steer.gguf
$T directions ../../steering/emotion_directions_mect60k_axes -o models/emotion.gguf
$T directions ../../steering/accent_directions_mect60k -o models/accent-directions.gguf

# Optional: the keep2_plain decoder variant (2 layers per transformer stage).
$T codec-decoder "$R/autoencoder/generator.pt" --head "$R/autoencoder/heads/keep2_plain" -o models/dac-keep2.gguf
```

```text
models/
  zonos3-qat.gguf     default backbone (Q4_0 experts, Q8_0 mixers) + embedded tokenizer, condition edges and silence frame
  zonos3-f16.gguf     optional F16 reference backbone (select with --quant f16)
  dac.gguf            MR-iSTFT waveform decoder (full_mr); needed for WAV/PCM output
  spk-encoder.gguf    speaker encoder; speaker-embedding cloning from audio
  dac-encoder.gguf    optional acoustic/semantic encoder + RVQ
  ssl-teacher.gguf    optional frozen SSL teacher; raw-reference audio cloning needs both
  steer.gguf          optional steering directions (voice design controls); convert_gguf.py steer
  emotion.gguf        optional emotion directions; convert_gguf.py directions
  accent-directions.gguf  optional accent-steering directions; convert_gguf.py directions
  accents.gguf        optional accent adapters; convert_gguf.py accents <adapter dir>
```

`tools/convert_gguf.py` validates the source architecture, tensor shapes and checksums and loads every checkpoint strictly into its PyTorch module. The backbone embeds the pinned tokenizer and its spec, so a separate tokenizer is unnecessary at inference. Older GGUFs (375k backbone, conv-tail `dac.gguf`) fail with a reconvert message. See [conversion details](docs/INTERNALS.md#convert-the-release).

## Launch

```bash
ZONOS3_SERVER_BIN="$PWD/build/zonos3-server" ZONOS3_MODEL_DIR="$PWD/models" \
  bash ./scripts/start-zonos3.sh -y --no-browser
```

Extracted releases place the server beside `start-zonos3.sh` or `start-zonos3.bat`. The launcher defaults to `zonos3-qat.gguf`; `--quant f16` selects `zonos3-f16.gguf` and `--quant q8_0` selects `zonos3-q8_0.gguf`. It checks the required local models, passes `--spk-encoder`, `--dac-encoder` and `--ssl-teacher` when those files exist (printing a note when `spk-encoder.gguf` is missing: speaker-embedding cloning is then disabled), and starts the UI at `http://127.0.0.1:1919/`. Models are never downloaded by the launcher or desktop app. ffmpeg is fetched only when unavailable locally or on `PATH`.

`zonos3-app` offers local file pickers, including an optional speaker-encoder picker. Its one-time configuration migration preserves valid settings but requires selecting the new backbone; incompatible embedding/emotion state is discarded.

## Generate speech

```bash
# Decoder is discovered beside the backbone.
./build/zonos3-cli models/zonos3-qat.gguf --tts 'Hello, world.' out.wav --seed 1

# Clone from audio with the speaker embedding (default) and save the embedding.
./build/zonos3-cli models/zonos3-qat.gguf --tts 'Hello again.' clone.wav \
  --clone default_voices/AmericanFemale.mp3 --save-speaker-embedding voice-emb.npy --seed 1

# Reuse a saved embedding without loading the speaker encoder.
./build/zonos3-cli models/zonos3-qat.gguf --tts 'A cached embedding.' cached.wav \
  --speaker-embedding voice-emb.npy --loudness -23 --condition-buckets 10,3,10 --seed 1

# Raw reference: encode once, save the codes, reuse them.
./build/zonos3-cli models/zonos3-qat.gguf --tts 'Raw reference.' ref.wav \
  --clone default_voices/AmericanFemale.mp3 --speaker-mode reference \
  --save-reference voice-codes.npy --seed 1
./build/zonos3-cli models/zonos3-qat.gguf --tts 'Cached codes.' ref2.wav \
  --reference-codes voice-codes.npy --seed 1

# Speaker CFG (two-slot generation; needs a speaker prompt).
./build/zonos3-cli models/zonos3-qat.gguf --tts 'Guided.' cfg.wav \
  --speaker-embedding voice-emb.npy --cfg-scale 1.5 --seed 1
```

Pass `--gpu` for GPU execution or `--cpu` to force CPU. `--dac`, `--spk-encoder`, `--dac-encoder`, and `--ssl-teacher` override companion discovery; the encoder overrides require `--clone`.

| Flag | Meaning |
|---|---|
| `--clone <audio>` | encode reference audio (`--speaker-mode embedding` by default, `reference` for raw codes) |
| `--speaker-embedding <emb.npy>` | precomputed float32 `[192]` embedding; exclusive with `--clone` and `--reference-codes` |
| `--reference-codes <codes.npy>` | ordinary C-order `[frames,10]` integer codes in `0..4095`, at most 900 frames (60 s); not the delayed matrix the decoder consumes |
| `--save-speaker-embedding` / `--save-reference` | write the computed embedding / codes |
| `--loudness <LUFS>`, `--fmax <Hz>`, `--speaking-rate <utf-8 bytes/s>` | condition values, mapped to buckets via the checkpoint's edges (`bisect_right`) |
| `--condition-buckets a,b,c` | explicit bucket ids `0..10` (10 = unknown) in `[loudness, fmax, speaking_rate]` order; wins per name over values; omitted = unknown |
| `--cfg-scale <s>` | speaker CFG scale, default 1 (off); `guided = uncond + s·(cond − uncond)` on the logits, with an unconditioned twin (no speaker prompt, unknown buckets) |
| `--silence-prefill <n>` | codec-silence frames appended after the text, `0..30`, default 3; their audio is never emitted |
| `--headroom-db <db>` | output attenuation, `0..24`, default 3 (from the decoder); 0 = raw decoder level |
| `--steer <steer.gguf>` | steering directions; found beside the backbone when a control flag is given |
| `--control ID=DOSE` | repeatable; direction ids come from `GET /v2/controls` (`set.name`). Raw-space controls move the speaker embedding (norm kept; embedding mode only), hidden-space controls add to a Full AttnRes bank source on every generated frame |
| `--control-span START:END:ID=DOSE[,ID=DOSE...]` | repeatable; hidden-space controls over the code-point range `[START,END)` of the `--tts` text, ramped over 0.2 s; positions come from `--control-span-rate` (frames per character, default 1) scaled by the timing of the global controls. Needs the KV decode path |
| `--accent NAME` | accent adapter of `accents.gguf` (`--accents PATH`, else found beside the backbone; `convert_gguf.py accents`); `--accent-strength S` scales it (default 1). `--generate`/`--tts` |
| `--probe-heads <heads.json>` | alignment probe of every decode step over the prompt's text (a `laci_heads/` JSON; its heads with weight 1); `--probe-window LAYER:HEAD:LO:HI` (repeatable) masks a head to text indices `[LO,HI]`; `--dump-probe probe.npy` writes float32 `[steps,2]` (argmax, centroid). KV path |
| `--dump-logits <logits.npy>` | float32 `[1+steps,10,4098]` conditional logits: the prompt's last row, then every decode step. KV path |

For file cloning, install ffmpeg or set `ZONOS3_FFMPEG`. `--max` limits generated autoregressive rows, including codebook-delay/EOS draining; prompt plus generation must fit the 32768-position context.

## HTTP API and the studio

`zonos3-server` serves the v2 HTTP API of `api/openapi.yaml` (the same contract as the CUDA server in `servers/sgl`): the native routes under `/v2/`, the OpenAI-compatible `GET /v1/models` and `POST /v1/audio/speech`, and Zonos Voice Studio at `/` (`studio/dist` of the checkout, or `--studio-dir`).

```bash
./build/zonos3-server models/zonos3-qat.gguf \
  --tts-default-voices-dir ../../default_voices --studio-projects-dir ~/.zonos3/projects --port 1919
curl -fsS http://127.0.0.1:1919/v2/capabilities
curl -fsS http://127.0.0.1:1919/v2/generate \
  -d '{"text":"Hello.","stream":false,"format":"wav","seed":1}' -o hello.wav
# Streaming returns mono float32 PCM at 44.1 kHz (X-Seed echoes the seed used).
curl -fsSN http://127.0.0.1:1919/v2/generate \
  -d '{"text":"Streaming hello.","stream":true,"seed":1}' -o hello.pcm
```

Requests are validated against the spec (embedded at build time): a schema violation, an unknown field or a dropped legacy field is a 422 `{"detail": [{"loc", "msg", "type"}]}`; a state-dependent error (unknown voice, a feature this server lacks, a CFG combination, `stream: true` with `format: "wav"`) is a 400 `{"detail"}`.

- **Voices** (`GET|POST /v2/speakers`, `/v2/speakers/<id>/preview`, `/v2/speakers/<id>/embedding`): default voices come from `--tts-default-voices-dir` (audio files, ids `default_<sha1[:16]>` of the relative path; `.npy`/`.npz` speaker embeddings); `POST /v2/speakers` caches a clip (`speaker_audio_base64`) or an embedding (`speaker_embedding_base64`) in the `X-TTS-Session-ID` session. An audio voice with a sibling `<stem>.codes.npy` (ordinary `[T,10]` codes as written by `dac-encoder-cli`) uses those codes in reference mode instead of loading the encoder; regenerate them after changing a voice: `./build/dac-encoder-cli models/dac-encoder.gguf models/ssl-teacher.gguf --audio ../../default_voices/NAME.mp3 ../../default_voices/NAME.codes.npy --gpu`. Audio cloning needs `spk-encoder.gguf` (embedding mode) or `dac-encoder.gguf` + `ssl-teacher.gguf` (reference mode); encoding is lazy per mode and cached, and the reference encoder and SSL teacher are freed after `--reference-idle-unload N` idle seconds (default 60; `0` keeps them loaded).
- **Generation** (`POST /v2/generate`, `POST /v1/audio/speech` with `voice` = a speaker id): `speaker_embedding_id` (or an inline `speaker_audio_base64` / `speaker_embedding_base64` / `speaker_reference_codes`, or a `speaker_blend_embedding_id_a/_b` slerp at `speaker_blend_t`) with `speaker_mode` `"reference"` (the checkpoint default) or `"embedding"`; `condition_buckets` / `condition_values`; `speaker_cfg_scale` (needs a speaker prompt and `--batch >= 2`); `silence_prefill_frames` (server default 3, max 30); `headroom_db` (`0..24`); `decode_context`; `max_tokens`; `seed`. Streaming uses `stream_decode` `"prefix"` (chunks of `ZONOS3_PREFIX_STREAM_CHUNK_FRAMES`=15 frames behind a `ZONOS3_PREFIX_STREAM_HOLD_BACK_FRAMES`=3 hold-back, the tail from the exact whole decode; the default, see `--stream-decode-default`) or `"whole"` (one exact chunk at EOS).
- **Text normalization**: off by default (text is tokenized raw). `--text-normalizer-python PY` runs `model/src/zonos3/workers/textnorm.py` (the `zonos3.tokenizer.textnorm` NeMo normalizer) as one persistent worker under `PY` with `PYTHONPATH` = `--python-src` (default: `model/src` of the checkout holding the executable, else `python/` beside it); `uv run which python` from the repo root is such an interpreter.
- **Speech editing** (`/v2/edit/transcribe`, `/v2/edit/align`, `/v2/edit`): needs an edit checkpoint (a GGUF with `zonos3.span_end_id` and edit marker rows, converted with `tools/convert_gguf.py lm`), `dac-encoder.gguf` + `ssl-teacher.gguf` (the clip's codes) and `ZSA_ROOT` = a [z-speech-alignment](https://github.com/Zyphra/z-speech-alignment) checkout with its uv environment (`ZSA_PYTHON` overrides its `.venv/bin/python`). The MMS aligner runs as `model/src/zonos3/workers/zsa_mms.py` worker processes, one per language (`ZONOS3_MMS_DEVICE`, at most `ZONOS3_MMS_MAX_WORKERS`, default 2); `spk-encoder.gguf` supplies the speaker row of speaker-embedding checkpoints.
- **Dubbing** (`/v2/dub/capabilities`, `/v2/dub/ref`, `/v2/dub/lips/<ref_id>`, `/v2/dub/transcribe`, `/v2/dub/analyze`, `/v2/dub/generate`, `/v2/dub/regenerate`, `/v2/dub/mux/<dub_id>`): needs `--laci-heads` for the target languages (as `servers/sgl/scripts/serve.sh` `DUB=1`: `cmn=laci_heads/zonos3_mini_mect60k_heads_zh.json` and `en_us fr_fr de es it pt_br ja ko` = `laci_heads/zonos3_mini_mect60k_heads.json`), a checkpoint with a speaking-rate condition row, ffmpeg, and the alignment worker of `tools/setup_dub_align_env.sh` (`model/src/zonos3/workers/dub_align.py` in `<repo>/.venv-dub-align`, AV-sync in `<repo>/.avsync`; `ZONOS3_DUB_ALIGN_PYTHON`, `ZONOS3_DUB_AVSYNC_HOME`, `ZONOS3_DUB_ALIGN_DEVICE` override, e.g. `cuda:1` with two visible GPUs). Source voices clone the reference (its prompt clip is encoded with `dac-encoder.gguf` + `ssl-teacher.gguf`); studio voices are speaker ids in embedding mode. Every take is a LACI generation with the segment's duration target; the steering bundle's `rate_extension` (in `steer.gguf`) adds tempo-steered speaking-rate buckets.
- **Projects** (`/v2/projects`, `/v2/media/<sha256>`): `--studio-projects-dir DIR` keeps the studio's projects and content-addressed media on the server, in the same layout as the CUDA server (both can share one directory).
- **Access**: with `ZONOS3_ACCESS_TOKEN` set, every request needs `Authorization: Bearer <token>`, the `zonos3_access` cookie, or one `GET /?access_token=<token>` (which sets the cookie). `X-Keepalive: 1` on the edit and dub routes answers at once and sends a space every 10 s until the JSON body.

### Steering and the voice studio

A `steer.gguf` next to the backbone (or `--steer PATH`) enables voice-design controls. Build it from a steering bundle (the JSON plus the per-set direction directories; semantics follow the minisgl `steering.py` reference):

```bash
uv run --package zonos3-tools python ../../tools/convert_gguf.py steer ../../steering/studio_bundle_mect60k.json -o models/steer.gguf
```

Requests accept `controls` (`{"delivery.tempo": 2, "identity.gender": -1}`), `control_spans` (`[{"start": 0, "end": 12, "controls": {"delivery.whisper": 3}}]`, code-point offsets into the request text, answered with `X-Control-Span-Follow: rate`) and `control_span_rate`. With `--laci-heads LANG=PATH` for the request's language, `control_span_follow: "reading"` places the spans by the alignment heads' reading position instead (`control_span_lead` / `control_span_lag`; answered with `X-Control-Span-Follow: reading`), and `laci: true` runs LACI: skips, stalls and early EOS are rolled back and regenerated under an ACI attention window (no CFG). `GET /v2/controls` lists the sets, directions (raw vectors included), duration timing tables and the UI metadata. Raw controls need `speaker_mode` `embedding`; hidden controls work in both modes. The directions were fitted on the `mect_60k` backbone; other checkpoints may respond differently.

## Standalone tools

```bash
./build/zonos3-cli models/zonos3-qat.gguf --build-prompt 'Hello.' ids.npy   # + ids.cond.json
./build/dac-cli models/dac.gguf codes.npy decoded.wav
./build/spk-encoder-cli models/spk-encoder.gguf --audio voice.mp3 voice-emb.npy --cpu
./build/dac-encoder-cli models/dac-encoder.gguf models/ssl-teacher.gguf \
  --audio voice.mp3 voice-codes.npy --cpu
./build/quantize-cli models/zonos3-f16.gguf models/zonos3-q8_0.gguf q8_0
./build/quantize-cli models/zonos3-f16.gguf models/zonos3-q6_k-exp.gguf \
  q6_k --experts-only --spine-f16
```

Prompt dumps carry a `.cond.json` sidecar (speaker vector, condition buckets, row layout) next to the ids. `dac-cli` honours `<codes>.eos.npy` (`[eos]` or `[eos, prefill]`), `--skip-frames N`, and `--headroom-db X`.

The default `zonos3-qat.gguf` matches the QAT checkpoint's baked weights exactly and was checked against the F16 backbone (golden logits cos 0.99995, eight-sentence Whisper-large-v3 WER 3.5% for both). Post-training quantization profiles (`quantize-cli`), pruning and calibration artifacts have not been qualified for Mini, and earlier (375k) measurements do not transfer. See [quantization](docs/INTERNALS.md#quantization) and [reference comparison](docs/INTERNALS.md#reference-comparison).

## Verify

```bash
ctest --test-dir build --output-on-failure
bash tests/readme-commands.sh build/zonos3-cli build/zonos3-server
bash tests/launcher-smoke.sh build/zonos3-server
```

For the real frozen teacher cache/replay regression, build `ssl-reference-state` and run it with `models/ssl-teacher.gguf` (add `--gpu` for an allocated GPU). Set `-DZONOS3_TEST_SSL_TEACHER=/absolute/path/to/ssl-teacher.gguf` at configure time to include its CPU check in CTest.

The tiny fixtures exercise interfaces, prompt layouts (speaker row, reference prefix, condition rows, silence prefill), conditioning-row replacement across chunk boundaries, the speaker-CFG twin, the iSTFT decoder seam, pruning, and evaluation tools—not speech quality. Actual-model Python/native comparisons are required for numerical parity; see [reference comparison](docs/INTERNALS.md#reference-comparison).
