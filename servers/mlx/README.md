# zonos3-mlx

The ZONOS3 Mini HTTP server on Apple Silicon: the v2 API of [servers/api/openapi.yaml](../api/openapi.yaml) and the Zonos
Voice Studio, running every network in [MLX](https://github.com/ml-explore/mlx) from the ggml server's GGUF files.

| Network | GGUF | Notes |
| --- | --- | --- |
| LM backbone | `zonos3-ptq.gguf` (default, `lm/rate30_20k_r2grpo`), `zonos3-qat.gguf` (previous `lm/mect_60k`, `SERVE_MODEL`) or `zonos3-f16.gguf` | GDN + global attention, Full AttnRes, latent MoE; Q4_0 / Q8_0 blocks become MLX affine-quantized weights, bit-exact; per-table condition bucket counts |
| Codec decoders | `zac-keep2.gguf` (default, `keep2_plain` head) and `zac.gguf` (full-size `full_mr` head) | causal multi-resolution iSTFT heads, 44.1 kHz; a request picks one with `decoder` |
| Speaker encoder | `spk-encoder.gguf` | MECT-B2 + ASTP, 192-d embeddings (`speaker_mode=embedding`) |
| Reference encoder | `zac-encoder.gguf` + `ssl-teacher.gguf` | raw-reference cloning and speech editing; the BF16 autocast recipe of the release; loaded on first use, freed after `--reference-idle-unload` idle seconds |
| Companions | `steer.gguf`, `accents.gguf` | the steering bundle (request `controls` / `control_spans` / `controls_cfg_scale`; `/v2/controls` with each set's calibration), accent adapters (request `accent`) |

Every file is auto-detected next to the backbone. Every decoder found (`zac-keep2.gguf`, then `zac.gguf`) or named
with a repeated `--zac` (the first is the default) is loaded; its id is the GGUF's `mossdual_istft.head.variant`
(two files with one id are refused). `/v2/capabilities` lists them in `decoders`, a request's `decoder` picks one
(an id the server does not serve is a 400; null = the default), including the decode of an edit, and audio responses
name it in `X-Decoder`. Generation runs on one worker thread that owns the LM; up to
`--batch` requests (default 4; a CFG request takes two slots) advance one row per tick, a CFG request's two forwards in
one MLX evaluation. Sampling runs on the GPU on the reference sampler's random stream (the host draws the
`torch.multinomial` exponentials, so a seed gives the reference rows up to float rounding), and without a probe reader
(LACI, span follow) the next row's forwards are queued on the lazy sample before the host reads it back. Streamed audio
decodes on a second (codec) thread while the backbone steps on: progressive prefix
chunks by default, the one exact decode at EOS with `ZONOS3_STREAM_DECODE_DEFAULT=whole` (as on the sgl server; a
request's `stream_decode` wins). The first prefix chunk decodes before its frame is complete (early partial-frame
decode, the request's `stream_codebooks` / `stream_codebooks_scope`, registry default 1 on the first chunk): the
residual codebooks not yet generated are left out of that frame, and later chunks wait for complete frames. It needs
the per-codebook residual biases (`res.{i}.bias`) in the decoder GGUF; with an older GGUF (only the summed
`res.bias`) every request streams complete frames.

## Install and run

```bash
uv sync --locked --package zonos3-mlx
hf download Zyphra/ZONOS3-Mini --include 'gguf/*' --local-dir servers/ggml   # servers/ggml/gguf/
servers/mlx/scripts/serve.sh            # 127.0.0.1:1919: / chooser, /studio, /lite
```

`serve.sh` takes the same variables as the sgl launcher (`SERVE_MODEL`, `SERVE_VOICES`, `PHONEME_LEXICON`,
`ZONOS3_STUDIO_PROJECTS`, `LACI_HEADS`, `LACI_ALL=1`, `PLUGINS`) plus `MODELS_DIR` and `WORKER_PYTHON`
(`--worker-python`); extra flags are appended. `python -m zonos3_mlx --help` lists the flags.

- **Helper workers** (`--worker-python PY`, as on the ggml server): the FuseAlign forced aligner
  (`model/src/zonos3/workers/fusealign.py`) runs as a persistent worker process under `PY`; without the flag it runs
  under the server's own interpreter.
- **Text normalization** is built in (the Rust `zyphra_textnorm` extension of the `third_party/zyphra-textnorm` submodule; `ZONOS3_TTS_NORM=0` turns it
  off).
- **Forced alignment** (every word timing of speech editing and plugins) is FuseAlign (`fusealign-hub`, the
  `third_party/fusealign-multilingual` submodule: `git submodule update --init third_party/fusealign-multilingual`
  before `uv sync`; the checkpoint downloads from [`Zyphra/fusealign`](https://huggingface.co/Zyphra/fusealign) on
  first use): one worker per server, started on the first alignment. `ZONOS3_FUSEALIGN_DEVICE` picks its torch
  device; without it the worker takes `cuda:0` when available, else the CPU, so it runs on the CPU on macOS.
  Recordings over 30 s are aligned long-form on Qwen3-ASR transcripts of their speech chunks (never Whisper), or on
  FuseAlign's own speech map without ASR.
- **Speech recognition** (speech editing and plugins) is Qwen3-ASR in `.venv-asr` (`bash tools/setup_asr_env.sh`
  once): one worker per server (`ZONOS3_ASR_PYTHON` overrides the interpreter; `ZONOS3_ASR_DEVICE`, default `mps` on
  macOS). **Speech editing** needs an edit backbone and FuseAlign: the release LM converted with
  `tools/convert_gguf.py lm --edit-adapter <z3m_omni3b_edit_lora.pt>` plus that adapter in `accents.gguf` (edit fills
  run on it at strength 1; it is not a request `accent`), or an edit checkpoint. `.venv-asr` is optional: it lets the
  server transcribe clips sent without their transcript and run `verify` (`edit_transcribe` in `/v2/capabilities`).
- **Plugins** (`--plugins NAME...`, `PLUGINS="..."` in `serve.sh`) import `zonos3_<NAME>` from the server's
  interpreter and mount its routes under `/v2/<NAME>/` and `/<NAME>`; they share the server's ASR and FuseAlign
  workers. Dubbing is the `dub` plugin of [Zyphra/ZONOS3-dub](https://github.com/Zyphra/ZONOS3-dub) (its README
  covers install, `LACI_ALL=1` and its workers).

## Tools

`python -m zonos3_mlx.tools` runs single networks (`decode`, `embed`, `ssl`, `encode`) and one in-process synthesis
(`generate`); `python -m zonos3_mlx.validate` dumps one LM forward's captures for `tools/compare_golden.py`.

## Verify

```bash
uv run --package zonos3-mlx pytest servers/mlx/tests                       # tiny-model parity, decoder, server
uv run --package zonos3-api pytest servers/api/tests/conformance --base-url http://127.0.0.1:1919
```

Against the release goldens of `tools/dump_golden.py`: the F16 backbone (`zonos3-f16.gguf`) matches the reference
logits at cosine 1.000000 with 100 % argmax agreement. The quantized backbones (`zonos3-ptq.gguf`, `zonos3-qat.gguf`)
run their F16 logits head as Q8_0 (quantized at load, the quants the web server ships; about 43 % faster decode steps
on an M-series GPU): against the same model with its F16 head, the logits on the web goldens' `full_ids.npy` agree at
cosine ≥ 0.99996 (prefill) / 0.999999 (decode) with 98.4 % / 100 % argmax agreement. On the CPU device (`--cpu`) the
SSL teacher matches at cosine 1.000000 (a few dozen of ~800k elements in three late layers sit just outside atol 1e-4 /
rtol 1e-3) and the speaker encoder at cosine 1.000000. Reference codes of the release voices agree with the committed
`<stem>.codes.npy` on 97–98 % of the first codebook and 67–96 % of the residual ones (RVQ residuals amplify the BF16
rounding differences); the decoded clips transcribe identically.
