# zonos3.cpp

Native ggml/GGUF inference for **ZONOS3 Mini** ([Zyphra/ZONOS3_Mini_release_ckpts](https://huggingface.co/Zyphra/ZONOS3_Mini_release_ckpts), revision `4d1d9efca3fb58ca880e028586d9f98bbc002288`), LM `lm/rate30_20k_r2grpo` (zonos3-rate30-20k-r2grpo, step **587**; the previous `lm/mect_60k`, step 60000, still loads). The 11-layer, 1024-wide backbone mixes Gated DeltaNet and global attention with Full AttnRes and top-2/64-expert latent MoE. SuperBPE-8192 text plus a speaker prompt and three condition rows produce ten delayed 4096-entry codebooks at 15 Hz; the multi-resolution iSTFT decoder renders 44.1 kHz mono audio.

Voice cloning has two modes:

- **Speaker embedding** (default): a native MECT-B2 speaker encoder (`spk-encoder.gguf`) turns up to 60 s of reference audio into a 192-d vector that the backbone's `speaker_projection` maps onto one prompt row. Saved embeddings need no encoder and no ffmpeg.
- **Raw reference** (opt-in): the reference is encoded to ordinary codec codes (`zac-encoder.gguf` + `ssl-teacher.gguf`) and prepended to the prompt. Saved codes need no encoder and no ffmpeg.

Every prompt also carries three **condition rows** (loudness, bandwidth `fmax`, speaking rate; each a bucket class with the checkpoint's per-table count n, ids `0..n-1`, `n` = unknown), an optional **speaker CFG** twin pass, and a **codec-silence prefill** after the text.

CLI/server synthesis and reference encoding run natively without Python. Python is needed for conversion and numerical reference checks; optional NeMo text normalization is a separate Python helper. All project interfaces use `zonos3`/`ZONOS3`; obsolete binaries, flags, environment aliases, and GGUF schemas (including the earlier 375k backbone and the conv-tail decoder) are rejected.

## Build

```bash
# In a ZONOS3 checkout (from its root, the ggml submodule only: git submodule update --init servers/ggml/ggml)
cd servers/ggml
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# CUDA:
cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j
```

The optional desktop shell uses `-DZONOS3_APP=ON`, CMake >= 3.31, a compatible C++23 compiler, and platform webview dependencies. CPU/CUDA numerical checks are described in [internals](docs/INTERNALS.md#reference-comparison); Metal, Vulkan, and Windows builds depend on the installed toolchain and backend operators.

## Prepare models

The default backbone is the Q4_0-expert/Q8_0-mixer post-training-quantized model `zonos3-ptq.gguf` (0.66 GB, `lm/rate30_20k_r2grpo`); `zonos3-qat.gguf` (0.66 GB, the previous `lm/mect_60k` QAT backbone) remains available with `--quant qat`. Download the converted files from the private [Zyphra/ZONOS3_Mini_GGUF](https://huggingface.co/Zyphra/ZONOS3_Mini_GGUF) repository (authenticate with an account that can read it):

```bash
hf auth login
hf download Zyphra/ZONOS3_Mini_GGUF zonos3-ptq.gguf zac-keep2.gguf spk-encoder.gguf --local-dir models
# Optional raw-reference cloning companions (the repo keeps the encoder under its previous name dac-encoder.gguf):
hf download Zyphra/ZONOS3_Mini_GGUF dac-encoder.gguf ssl-teacher.gguf --local-dir models
mv models/dac-encoder.gguf models/zac-encoder.gguf
# Optional voice-direction steering bundle (request `controls`: voice design, emotion, accent directions; the
# studio's design controls; tempo-steered speaking-rate buckets; hidden-state directions fitted on lm/mect_60k):
hf download Zyphra/ZONOS3_Mini_GGUF steer.gguf --local-dir models
# Optional speech editing (/v2/edit; also needs spk-encoder.gguf, the encoder companions above and --worker-python):
# the release LM with the speech-edit LoRA's vocabulary, and the LoRA as an edit adapter bank. Keep edit/accents.gguf
# where it lands: as models/accents.gguf it would be picked up by zonos3-ptq.gguf, which refuses it.
hf download Zyphra/ZONOS3_Mini_GGUF zonos3-ptq-edit.gguf edit/accents.gguf --local-dir models
./build/zonos3-server models/zonos3-ptq-edit.gguf --accent-adapters models/edit/accents.gguf --worker-python <PY>
```

The server picks up every companion it finds next to the backbone. `zonos3-ptq.gguf` carries per-table condition bucket
counts (`zonos3.condition.bucket_counts`) and needs a build that reads them. Backbones downloaded before 2026-11-01 lack
`zonos3.text_tokenizer_spec` and are refused: re-download `zonos3-qat.gguf`. A `steer.gguf` without
`zonos3_steer.set.calibrations` (converted before 2026-11-03) is refused as well: re-download or reconvert it. Files
downloaded before 2026-10-27 are the previous fullenc_120k release; re-download
`zonos3-qat.gguf` and `spk-encoder.gguf` together (the backbone expects embeddings from the matching speaker encoder).

Or convert the checkpoints locally as below.

### Convert the release

Download the private release (authenticate with an account that can read it) or use a local copy:

```bash
hf auth login
R=release   # or a local copy such as /data/groups/audio/gabriel/hf_release_zonos3mini_20261020
hf download Zyphra/ZONOS3_Mini_release_ckpts \
  --revision 4d1d9efca3fb58ca880e028586d9f98bbc002288 --local-dir "$R"

T="uv run --package zonos3-tools python ../../tools/convert_gguf.py"   # from servers/ggml
mkdir -p models
# F16 reference backbone from the release:
$T lm "$R/lm/rate30_20k_r2grpo" --outtype f16 -o models/zonos3-f16.gguf
# Default backbone: the same LM quantized post-training (round to nearest): Q4_0 experts, Q8_0 mixers.
$T lm "$R/lm/rate30_20k_r2grpo" --outtype q4_0-exp-q8_0-mix -o models/zonos3-ptq.gguf
# Previous backbone (--quant qat): mect_60k QAT directory (same layout as the release LM plus bake.json; step 3462).
# Q4_0 experts and Q8_0 mixers are stored bit-exactly, so there is no outtype to choose.
$T lm QAT/lm/mect_60k_q4q8_qat \
  --silence-frame "$R/registries/silence_frame_opus_dual_3n_600k_sched600k_r4096_step600000.json" \
  -o models/zonos3-qat.gguf
$T codec-decoder "$R/autoencoder/generator.pt" --head "$R/autoencoder/heads/keep2_plain" -o models/zac-keep2.gguf
$T speaker-encoder "$R/speaker_encoders/ownmect_s2/ckpt_0037.pth" -o models/spk-encoder.gguf

# Optional: raw-reference cloning companions.
$T codec-encoder "$R/autoencoder/generator.pt" --outtype f32 -o models/zac-encoder.gguf
$T ssl-teacher "$R/autoencoder/teacher" --outtype f32 -o models/ssl-teacher.gguf

# Optional: the steering bundle (repo-root steering/ assets).
$T steer ../../steering/studio_bundle_mect60k.json -o models/steer.gguf

# Optional: the full-size full_mr decoder (32/12/12/12/12 layers per transformer stage; select with --zac models/zac.gguf).
$T codec-decoder "$R/autoencoder/generator.pt" --head "$R/autoencoder/heads/full_mr" -o models/zac.gguf
```

```text
models/
  zonos3-ptq.gguf     default backbone (Q4_0 experts, Q8_0 mixers) + embedded tokenizer, condition edges and silence frame
  zonos3-qat.gguf     optional previous mect_60k QAT backbone (select with --quant qat)
  zonos3-f16.gguf     optional F16 reference backbone (select with --quant f16)
  zac-keep2.gguf      default MR-iSTFT waveform decoder (keep2_plain: 2 layers per stage); needed for WAV/PCM output
  zac.gguf            optional full-size decoder (full_mr; select with --zac, launcher --zac zac)
  spk-encoder.gguf    speaker encoder; speaker-embedding cloning from audio
  zac-encoder.gguf    optional acoustic/semantic encoder + RVQ
  ssl-teacher.gguf    optional frozen SSL teacher; raw-reference audio cloning needs both
  steer.gguf          optional steering bundle (voice design, emotion and accent directions); convert_gguf.py steer
  accents.gguf        optional accent adapters; convert_gguf.py accents <adapter dir>
```

`tools/convert_gguf.py` validates the source architecture, tensor shapes and checksums and loads every checkpoint strictly into its PyTorch module. The backbone embeds the pinned tokenizer and its spec, so a separate tokenizer is unnecessary at inference. Older GGUFs (375k backbone, conv-tail `zac.gguf`) fail with a reconvert message. See [conversion details](docs/INTERNALS.md#convert-the-release).

## Launch

```bash
ZONOS3_SERVER_BIN="$PWD/build/zonos3-server" ZONOS3_MODEL_DIR="$PWD/models" \
  bash ./scripts/start-zonos3.sh -y --no-browser
```

Extracted releases place the server beside `start-zonos3.sh` or `start-zonos3.bat`. The launcher defaults to `zonos3-ptq.gguf`; `--quant qat` selects the previous mect_60k backbone `zonos3-qat.gguf`, `--quant f16` selects `zonos3-f16.gguf` and `--quant q8_0` selects `zonos3-q8_0.gguf`. It checks the required local models, passes `--spk-encoder`, `--zac-encoder` and `--ssl-teacher` when those files exist (printing a note when `spk-encoder.gguf` is missing: speaker-embedding cloning is then disabled), and starts the UI at `http://127.0.0.1:1919/`. Models are never downloaded by the launcher or desktop app. ffmpeg is fetched only when unavailable locally or on `PATH`.

`zonos3-app` offers local file pickers, including an optional speaker-encoder picker. Its one-time configuration migration preserves valid settings but requires selecting the new backbone; incompatible embedding/emotion state is discarded.

## Generate speech

```bash
# Decoder is discovered beside the backbone.
./build/zonos3-cli models/zonos3-ptq.gguf --tts 'Hello, world.' out.wav --seed 1

# Clone from audio with the speaker embedding (default) and save the embedding.
./build/zonos3-cli models/zonos3-ptq.gguf --tts 'Hello again.' clone.wav \
  --clone default_voices/AmericanFemale.mp3 --save-speaker-embedding voice-emb.npy --seed 1

# Reuse a saved embedding without loading the speaker encoder.
./build/zonos3-cli models/zonos3-ptq.gguf --tts 'A cached embedding.' cached.wav \
  --speaker-embedding voice-emb.npy --condition-buckets 4,3,10 --seed 1

# Raw reference: encode once, save the codes, reuse them.
./build/zonos3-cli models/zonos3-ptq.gguf --tts 'Raw reference.' ref.wav \
  --clone default_voices/AmericanFemale.mp3 --speaker-mode reference \
  --save-reference voice-codes.npy --seed 1
./build/zonos3-cli models/zonos3-ptq.gguf --tts 'Cached codes.' ref2.wav \
  --reference-codes voice-codes.npy --seed 1

# Speaker CFG (two-slot generation; needs a speaker prompt).
./build/zonos3-cli models/zonos3-ptq.gguf --tts 'Guided.' cfg.wav \
  --speaker-embedding voice-emb.npy --cfg-scale 1.5 --seed 1
```

Pass `--gpu` for GPU execution or `--cpu` to force CPU. `--zac`, `--spk-encoder`, `--zac-encoder`, and `--ssl-teacher` override companion discovery; the encoder overrides require `--clone`.

| Flag | Meaning |
|---|---|
| `--clone <audio>` | encode reference audio (`--speaker-mode embedding` by default, `reference` for raw codes) |
| `--speaker-embedding <emb.npy>` | precomputed float32 `[192]` embedding; exclusive with `--clone` and `--reference-codes` |
| `--reference-codes <codes.npy>` | ordinary C-order `[frames,10]` integer codes in `0..4095`; frames past 900 (60 s, the registry's `reference_max_frames`) are dropped; not the delayed matrix the decoder consumes |
| `--save-speaker-embedding` / `--save-reference` | write the computed embedding / codes |
| `--loudness <LUFS>`, `--fmax <Hz>`, `--speaking-rate <utf-8 bytes/s>` | condition values, mapped to buckets via the checkpoint's edges (`bisect_right`) |
| `--condition-buckets a,b,c` | explicit bucket ids `0..n` per table (n = the checkpoint's count for that condition = unknown) in `[loudness, fmax, speaking_rate]` order; wins per name over values; omitted = unknown |
| `--cfg-scale <s>` | speaker CFG scale, default 1 (off); `guided = uncond + s·(cond − uncond)` on the logits, with an unconditioned twin (no speaker prompt, unknown buckets) |
| `--silence-prefill <n>` | codec-silence frames appended after the text, `0..30`, default 3 (the registry's `silence_prefill_frames`); their audio is never emitted |
| `--headroom-db <db>` | output attenuation, `0..24`, default 3 (from the decoder); 0 = raw decoder level |
| `--steer <steer.gguf>` | steering directions; found beside the backbone when a control flag is given |
| `--control ID=DOSE` | repeatable; direction ids come from `GET /v2/controls` (`set.name`). Raw-space controls move the speaker embedding (norm kept; embedding mode only), hidden-space controls add to a Full AttnRes bank source on every generated frame |
| `--control-span START:END:ID=DOSE[,ID=DOSE...]` | repeatable; hidden-space controls over the code-point range `[START,END)` of the `--tts` text, ramped over 0.2 s; positions come from `--control-span-rate` (frames per character, default 1) scaled by the timing of the global controls. Needs the KV decode path |
| `--accent NAME` | accent adapter of `accents.gguf` (`--accents PATH`, else found beside the backbone; `convert_gguf.py accents`); `--accent-strength S` scales it (default 1). `--generate`/`--tts` |
| `--probe-heads <heads.json>` | alignment probe of every decode step over the prompt's text (a `laci_heads/` JSON; its heads with weight 1); `--probe-window LAYER:HEAD:LO:HI` (repeatable) masks a head to text indices `[LO,HI]`; `--dump-probe probe.npy` writes float32 `[steps,2]` (argmax, centroid). KV path |
| `--dump-logits <logits.npy>` | float32 `[1+steps,10,4098]` conditional logits: the prompt's last row, then every decode step. KV path |

For file cloning, install ffmpeg or set `ZONOS3_FFMPEG`. `--max` limits generated autoregressive rows, including codebook-delay/EOS draining; prompt plus generation must fit the 32768-position context.

## HTTP API and the studio

`zonos3-server` serves the v2 HTTP API of `api/openapi.yaml` (the same contract as the CUDA server in `servers/sgl`): the native routes under `/v2/`, the OpenAI-compatible `GET /v1/models` and `POST /v1/audio/speech`, and the studio build (`studio/dist` of the checkout, or `--studio-dir`): a chooser at `/`, Zonos Voice Studio at `/studio` and the lite voice-cloning page at `/lite`.

```bash
./build/zonos3-server models/zonos3-ptq.gguf \
  --tts-default-voices-dir ../../default_voices --studio-projects-dir ~/.zonos3/projects --port 1919
curl -fsS http://127.0.0.1:1919/v2/capabilities
curl -fsS http://127.0.0.1:1919/v2/generate \
  -d '{"text":"Hello.","stream":false,"format":"wav","seed":1}' -o hello.wav
# Streaming returns mono float32 PCM at 44.1 kHz (X-Seed echoes the seed used).
curl -fsSN http://127.0.0.1:1919/v2/generate \
  -d '{"text":"Streaming hello.","stream":true,"seed":1}' -o hello.pcm
```

Requests are validated against the spec (embedded at build time): a schema violation, an unknown field or a dropped legacy field is a 422 `{"detail": [{"loc", "msg", "type"}]}`; a state-dependent error (unknown voice, a feature this server lacks, a CFG combination, `stream: true` with `format: "wav"`) is a 400 `{"detail"}`.

- **Voices** (`GET|POST /v2/speakers`, `/v2/speakers/<id>/preview`, `/v2/speakers/<id>/embedding`): default voices come from `--tts-default-voices-dir` (audio files, ids `default_<sha1[:16]>` of the relative path; `.npy`/`.npz` speaker embeddings); `POST /v2/speakers` caches a clip (`speaker_audio_base64`) or an embedding (`speaker_embedding_base64`) in the `X-TTS-Session-ID` session. An audio voice with a sibling `<stem>.codes.npy` (ordinary `[T,10]` codes as written by `zac-encoder-cli`) uses those codes in reference mode instead of loading the encoder, and one with a sibling `<stem>.embedding.npy` (float32 `[192]`, as written by `spk-encoder-cli`) uses that embedding in embedding mode instead of loading the speaker encoder; regenerate both after changing a voice: `./build/zac-encoder-cli models/zac-encoder.gguf models/ssl-teacher.gguf --audio ../../default_voices/NAME.mp3 ../../default_voices/NAME.codes.npy --gpu` and `./build/spk-encoder-cli models/spk-encoder.gguf --audio ../../default_voices/NAME.mp3 ../../default_voices/NAME.embedding.npy --cpu`. Audio cloning needs `spk-encoder.gguf` (embedding mode) or `zac-encoder.gguf` + `ssl-teacher.gguf` (reference mode); encoding is lazy per mode and cached, and the reference encoder and SSL teacher are freed after `--reference-idle-unload N` idle seconds (default 60; `0` keeps them loaded).
- **Generation** (`POST /v2/generate`, `POST /v1/audio/speech` with `voice` = a speaker id): `speaker_embedding_id` (or an inline `speaker_audio_base64` / `speaker_embedding_base64` / `speaker_reference_codes`, or a `speaker_blend_embedding_id_a/_b` slerp at `speaker_blend_t`) with `speaker_mode` `"reference"` (the checkpoint default) or `"embedding"`; `condition_buckets` / `condition_values`; `speaker_cfg_scale` (needs a speaker prompt and `--batch >= 2`); `silence_prefill_frames` (server default 3, `0..30`, larger or negative is a 400); `headroom_db` (`0..24`); `decode_context`; `max_tokens`; `seed`. Reference codes (inline, a voice's `.codes.npy` or an encoded clip) keep their first 900 frames. `max_tokens` (at most `--max`, default 2000) shrinks to the slot rows the prompt leaves; a prompt that fills the slot is a 400 `TTS input len {rows} exceeds {capacity}`. Sampling defaults and these limits come from the generation registry `model/src/zonos3/registries/generation.json`, embedded at build time. Streaming uses `stream_decode` `"prefix"` (chunks of `ZONOS3_PREFIX_STREAM_CHUNK_FRAMES`=15 frames, the first one `ZONOS3_PREFIX_STREAM_FIRST_CHUNK_FRAMES`=4, behind a `ZONOS3_PREFIX_STREAM_HOLD_BACK_FRAMES`=3 hold-back, the tail from the exact whole decode; the default, see `--stream-decode-default`) or `"whole"` (one exact chunk at EOS).
- **Python workers** (`--worker-python PY`): text normalization and the Python host run as persistent workers of `model/src/zonos3/workers` under `PY` with `PYTHONPATH` = `--python-src` (default: `model/src` of the checkout holding the executable, else `python/` beside it); the Python host keeps the caller's `PYTHONPATH` after it, and `--plugins` checks at startup that each `zonos3_<name>` imports under that same environment. The Python host (`pyhost.py`) runs speech editing (`zonos3.tts.edit_runtime`, the Python implementation the other servers run) and the `--plugins` routers; it starts on first use and again after it exited, and calls back into the server for everything that needs the served model (upload decode, speaker embedding and codes of a clip, each region's infill generation, decodes, LACI takes with a duration target on the scheduler). Its word timings come from FuseAlign, so `PY` needs the workspace's packages, including `fusealign-hub` (the `third_party/fusealign-multilingual` submodule: `git submodule update --init third_party/fusealign-multilingual`, then `uv sync`; the checkpoint downloads from the [`Zyphra/fusealign`](https://huggingface.co/Zyphra/fusealign) Hugging Face repository on first use, `ZONOS3_FUSEALIGN_DEVICE` picks its device); `uv run which python` from the repo root is such an interpreter. Release packages ship `fusealign_hub` (Apache-2.0) as `python/fusealign_hub` next to `python/zonos3`; `PY` there needs its dependencies (`torch`, `torchaudio`, `transformers`, `safetensors`, `huggingface_hub`, `silero-vad`), or `pip install git+https://github.com/Zyphra/fusealign-multilingual.git`.
- **Speech recognition** (speech editing and plugins): the Python host runs Qwen3-ASR (`Qwen/Qwen3-ASR-1.7B`, `model/src/zonos3/workers/asr.py`) in its own interpreter, `ZONOS3_ASR_PYTHON` (default `<repo>/.venv-asr/bin/python`, set up by `tools/setup_asr_env.sh`; `ZONOS3_ASR_DEVICE` picks its device, e.g. `cuda:1`).
- **Text normalization**: off without `--worker-python` (text is tokenized raw); with it, `model/src/zonos3/workers/textnorm.py` (the `zonos3.tokenizer.textnorm` NeMo normalizer) normalizes request text.
- **Speech editing** (`/v2/edit/transcribe`, `/v2/edit/align`, `/v2/edit`): needs an edit backbone (a GGUF with `zonos3.span_end_id` and edit marker rows: the release LM converted with `tools/convert_gguf.py lm --edit-adapter <z3m_omni3b_edit_lora.pt>`, with that `zonos-accent-lora/3` adapter in `accents.gguf` (`convert_gguf.py accents`; edit fills run on it at strength 1 and it is not an accent name), or an edit checkpoint converted with `lm`), `zac-encoder.gguf` + `ssl-teacher.gguf` (the clip's codes), `--worker-python` with `fusealign-hub` (FuseAlign places every word) and the ASR interpreter (Qwen3-ASR transcribes the original when no `original_text` is given and every take of `verify`). `spk-encoder.gguf` supplies the speaker row of speaker-embedding checkpoints.
- **Plugins** (`--plugins NAME[,NAME...]`, repeatable; `GET /v2/capabilities` lists them as `plugins`): plugin `NAME` is the `zonos3_<NAME>` package (`zonos3.server.plugins`; see "Plugins" in `docs/guide.md`), which must be installed in `--worker-python` (`uv pip install --python PY --no-deps -e <plugin checkout>`; checked at startup; `--plugins` needs `--worker-python`). Every request of `/v2/<NAME>`, `/v2/<NAME>/*` and `GET|HEAD /<NAME>`, `/<NAME>/*` is forwarded to the Python host, which runs it through the plugin's FastAPI router (bodies cross as temporary files; `X-Keepalive: 1` applies to the POST paths the plugin lists). Plugins reach the served model through the host protocol: LACI takes with a duration target on this server's scheduler (`--laci-heads` for their languages, e.g. `LACI_ALL=1` of `servers/sgl/scripts/serve.sh`; the steering bundle's `rate_extension` in `steer.gguf` adds tempo-steered speaking-rate buckets) and reference encodes.
- **Projects** (`/v2/projects`, `/v2/media/<sha256>`): `--studio-projects-dir DIR` keeps the studio's projects and content-addressed media on the server, in the same layout as the CUDA server (both can share one directory).
- **Access**: with `ZONOS3_ACCESS_TOKEN` set, every request needs `Authorization: Bearer <token>`, the `zonos3_access` cookie, or one `GET /?access_token=<token>` (which sets the cookie). `X-Keepalive: 1` on the edit routes and the POST paths a plugin lists answers at once and sends a space every 10 s until the JSON body.

### Steering and the voice studio

A `steer.gguf` next to the backbone (or `--steer PATH`) enables voice-design controls. Build it from a steering bundle (the JSON plus the per-set direction directories; semantics follow the minisgl `steering.py` reference):

```bash
uv run --package zonos3-tools python ../../tools/convert_gguf.py steer ../../steering/studio_bundle_mect60k.json -o models/steer.gguf
```

Requests accept `controls` (`{"delivery.tempo": 2, "identity.gender": -1}`), `control_spans` (`[{"start": 0, "end": 12, "controls": {"delivery.whisper": 3}}]`, code-point offsets into the request text, answered with `X-Control-Span-Follow: rate`) and `control_span_rate`. Emotion (`mood.*`) and accent directions (`accents_en.*`, `accent_hidden.hin`, ...) are bundle sets like any other. `controls_cfg_scale` (default 1; needs `speaker_mode` `embedding` and `--batch >= 2`) guides a request with a hidden-space delta against a twin of the same prompt without it, `uncond + scale * (cond - uncond)`. With `--laci-heads LANG=PATH` for the request's language, `control_span_follow: "reading"` places the spans by the alignment heads' reading position instead (`control_span_lead` / `control_span_lag`; answered with `X-Control-Span-Follow: reading`), and `laci: true` runs LACI: skips, stalls and early EOS are rolled back and regenerated under an ACI attention window (no CFG). `GET /v2/controls` lists the sets (with the set's `calibration`: the doses of its `calibration.json`, when it has one), directions (raw vectors included), duration timing tables and the UI metadata; the server never applies a calibration (controls are raw doses), and every speaker object lists the `disabled_controls` its calibration doses to 0. Raw controls need `speaker_mode` `embedding`; hidden controls work in both modes. `accent` / `accent_strength` select an accent adapter of `accents.gguf` (`--accent-adapters`). The directions were fitted on the `mect_60k` backbone; other checkpoints may respond differently.

## Standalone tools

```bash
./build/zonos3-cli models/zonos3-ptq.gguf --build-prompt 'Hello.' ids.npy   # + ids.cond.json
./build/zac-cli models/zac-keep2.gguf codes.npy decoded.wav
./build/spk-encoder-cli models/spk-encoder.gguf --audio voice.mp3 voice-emb.npy --cpu
./build/zac-encoder-cli models/zac-encoder.gguf models/ssl-teacher.gguf \
  --audio voice.mp3 voice-codes.npy --cpu
./build/quantize-cli models/zonos3-f16.gguf models/zonos3-q8_0.gguf q8_0
./build/quantize-cli models/zonos3-f16.gguf models/zonos3-q6_k-exp.gguf \
  q6_k --experts-only --spine-f16
```

Prompt dumps carry a `.cond.json` sidecar (speaker vector, condition buckets, row layout) next to the ids. `zac-cli` honours `<codes>.eos.npy` (`[eos]` or `[eos, prefill]`), `--skip-frames N`, and `--headroom-db X`.

The default `zonos3-ptq.gguf` quantizes the BF16 `lm/rate30_20k_r2grpo` weights post-training (round to nearest) into the QAT file's layout (30 Q4_0 and 28 Q8_0 tensors); no QAT run exists for this LM. On six speaker-less CLI prompts × three seeds its Qwen3-ASR WER is 11.1% against 22.2% for its F16 backbone (0.0% vs 6.7% without the one prompt every backbone, the previous QAT included, misreads on this path; [details](docs/INTERNALS.md#quantization)). The previous `zonos3-qat.gguf` matches the mect_60k QAT checkpoint's baked weights exactly and was checked against its F16 backbone (golden logits cos 0.99995, eight-sentence Whisper-large-v3 WER 3.5% for both). Quantized backbones run their F16 logits head as q8_0, quantized at load (single-slot decode 14% faster on Metal and 17% on CPU on an M3; the F16 backbone keeps its F16 head; [details](docs/INTERNALS.md#quantization)); the WER figures above predate it. Other post-training quantization profiles (`quantize-cli`), pruning and calibration artifacts have not been qualified for Mini, and earlier (375k) measurements do not transfer. See [quantization](docs/INTERNALS.md#quantization) and [reference comparison](docs/INTERNALS.md#reference-comparison).

## Verify

```bash
ctest --test-dir build --output-on-failure
bash tests/readme-commands.sh build/zonos3-cli build/zonos3-server
bash tests/launcher-smoke.sh build/zonos3-server
```

For the real frozen teacher cache/replay regression, build `ssl-reference-state` and run it with `models/ssl-teacher.gguf` (add `--gpu` for an allocated GPU). Set `-DZONOS3_TEST_SSL_TEACHER=/absolute/path/to/ssl-teacher.gguf` at configure time to include its CPU check in CTest.

The tiny fixtures exercise interfaces, prompt layouts (speaker row, reference prefix, condition rows, silence prefill), conditioning-row replacement across chunk boundaries, the speaker-CFG twin, the iSTFT decoder seam, pruning, and evaluation tools—not speech quality. Actual-model Python/native comparisons are required for numerical parity; see [reference comparison](docs/INTERNALS.md#reference-comparison).
