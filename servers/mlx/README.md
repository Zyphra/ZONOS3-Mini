# zonos3-mlx

The ZONOS3 Mini HTTP server on Apple Silicon: the v2 API of [api/openapi.yaml](../../api/openapi.yaml) and the Zonos
Voice Studio, running every network in [MLX](https://github.com/ml-explore/mlx) from the ggml server's GGUF files.

| Network | GGUF | Notes |
| --- | --- | --- |
| LM backbone | `zonos3-qat.gguf` (default) or `zonos3-f16.gguf` | GDN + global attention, Full AttnRes, latent MoE; Q4_0 / Q8_0 blocks become MLX affine-quantized weights, bit-exact |
| Codec decoder | `dac.gguf` | causal multi-resolution iSTFT head, 44.1 kHz |
| Speaker encoder | `spk-encoder.gguf` | MECT-B2 + ASTP, 192-d embeddings (`speaker_mode=embedding`) |
| Reference encoder | `dac-encoder.gguf` + `ssl-teacher.gguf` | raw-reference cloning and speech editing; the BF16 autocast recipe of the release; loaded on first use, freed after `--reference-idle-unload` idle seconds |
| Companions | `steer.gguf`, `emotion.gguf`, `accents.gguf`, `accent-directions.gguf` | request `controls`, emotion, accents |

Every file is auto-detected next to the backbone. Generation runs on one worker thread that owns every MLX call; up to
`--batch` requests (default 4; a CFG request takes two slots) advance one row per tick.

## Install and run

```bash
uv sync --locked --package zonos3-mlx
hf download Zyphra/ZONOS3_Mini_GGUF --local-dir servers/ggml/models
servers/mlx/scripts/serve.sh            # 127.0.0.1:1919, the studio at /
```

`serve.sh` takes the same variables as the sgl launcher (`SERVE_MODEL`, `SERVE_VOICES`, `PHONEME_LEXICON`,
`ZONOS3_STUDIO_PROJECTS`, `LACI_HEADS`, `DUB=1`) plus `MODELS_DIR` and `WORKER_PYTHON` (`--worker-python`); extra
flags are appended. `python -m zonos3_mlx --help` lists the flags.

- **Helper workers** (`--worker-python PY`, as on the ggml server): text normalization
  (`model/src/zonos3/workers/textnorm.py`) and the FuseAlign forced aligner (`model/src/zonos3/workers/fusealign.py`)
  run as persistent worker processes under `PY`. Without the flag, text normalization is off and FuseAlign runs under
  the server's own interpreter.
- **Text normalization** (NeMo grammars) needs pynini, which has no macOS wheel on PyPI: point `--worker-python` at
  an interpreter that has it (e.g. a conda env with `pynini==2.1.6`; FuseAlign then runs there too, so it also needs
  the workspace's `fusealign-hub`). Without it requests are served unnormalized.
- **Forced alignment** (every word timing of speech editing and dubbing) is FuseAlign (`fusealign-hub`, the
  `third_party/fusealign-multilingual` submodule: `git submodule update --init third_party/fusealign-multilingual`
  before `uv sync`; the checkpoint downloads from [`Zyphra/fusealign`](https://huggingface.co/Zyphra/fusealign) on
  first use): one worker per server, started on the first alignment. `ZONOS3_FUSEALIGN_DEVICE` picks its torch
  device; without it the worker takes `cuda:0` when available, else the CPU, so it runs on the CPU on macOS.
  Recordings over 30 s are aligned long-form on Qwen3-ASR transcripts of their speech chunks (never Whisper).
- **Speech recognition** (speech editing and dubbing) is Qwen3-ASR in `.venv-asr` (`bash tools/setup_asr_env.sh`
  once): one worker per server (`ZONOS3_ASR_PYTHON` overrides the interpreter; `ZONOS3_ASR_DEVICE`, default `mps` on
  macOS). **Speech editing** needs an edit checkpoint, FuseAlign and `.venv-asr`; **dubbing** FuseAlign and `.venv-asr`
  (its SimAlign word aligner and AV-sync lip-sync run in a second worker on the same device).

## Tools

`python -m zonos3_mlx.tools` runs single networks (`decode`, `embed`, `ssl`, `encode`) and one in-process synthesis
(`generate`); `python -m zonos3_mlx.validate` dumps one LM forward's captures for `tools/compare_golden.py`.

## Verify

```bash
uv run --package zonos3-mlx pytest servers/mlx/tests                       # tiny-model parity, decoder, server
uv run --package zonos3-api pytest api/tests/conformance --base-url http://127.0.0.1:1919
```

Against the release goldens of `tools/dump_golden.py`: the F16 backbone matches the reference logits at cosine 1.000000
with 100 % argmax agreement; on the CPU device (`--cpu`) the SSL teacher matches at cosine 1.000000 (a few dozen of
~800k elements in three late layers sit just outside atol 1e-4 / rtol 1e-3) and the speaker encoder at cosine
1.000000. Reference codes of the release voices agree with the committed `<stem>.codes.npy` on 97–98 % of the first
codebook and 67–96 % of the residual ones (RVQ residuals amplify the BF16 rounding differences); the decoded clips
transcribe identically.
