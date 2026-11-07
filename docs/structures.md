# Structure

Zonos3 is a TTS-only serving stack for ZONOS3 checkpoints (an `hf://` location, a Hugging Face repo id or a local path).
This page describes the CUDA server (`servers/sgl`, package `zonos3_sgl`), the MLX server (`servers/mlx`, package
`zonos3_mlx`) and the model package they build on (`model`, package `zonos3`); the ggml server is described in
[servers/ggml/README.md](../servers/ggml/README.md).

## Processes

- **API server**: FastAPI frontend for `/v2/generate`, `/v1/audio/speech`, capabilities, speaker cache, edit and studio project endpoints, plus the routes of `--plugins`; it also serves the studio build at `/`.
- **Tokenizer worker**: Builds SuperBPE / byte-token TTS prompt frames, encodes reference audio with the matched dual codec, and decodes audio frames to PCM with the codec's iSTFT decoder.
- **Scheduler worker**: Runs one scheduler per tensor-parallel rank, manages request queues, KV cache slots, and decode scheduling.
- **Engine**: Owns the model, attention backend, KV cache, CUDA graphs, and checkpoint weights for one rank.

The processes communicate through ZeroMQ. Tensor-parallel ranks use `torch.distributed` and NCCL/PyNCCL for GPU communication.

## Request Flow

1. The API server receives text plus optional speaker, condition-row and steering inputs.
2. The tokenizer worker turns text into prompt frames and inserts the optional speaker row, condition rows and raw reference prefix.
3. Rank 0 scheduler receives the request and broadcasts it to other TP ranks when needed.
4. The engine runs Zonos3 prefill/decode and samples multi-codebook audio frames.
5. The scheduler sends audio code frames to the tokenizer worker.
6. The tokenizer worker decodes frames through the matched codec (`model/src/zonos3/registries/codec_registry_keep2_plain.json` by default: the keep2 decoder head) and sends PCM bytes back to the API server.

## Code Organization

`zonos3` (`model/src/zonos3`, plain PyTorch, never imports `zonos3_sgl`):

- `zonos3.codec`: The vendored dual codec (encoder, quantizer, causal multi-resolution iSTFT decoder).
- `zonos3.models`: Model config conversion, the MECT speaker encoder, speaker cloning and the accent adapter file format.
- `zonos3.tokenizer`: Text normalization (`textnorm.py`), the matched-codec registry (`matched_codec.py`) and the reference encoder (`dual_reference.py`).
- `zonos3.tts`: Prompt building, markup, conditioning, direction sets (`directions.py`) and steering bundles (`steering.py`), LACI and span-follow, speech-edit prompts and alignment, the speech-edit runtime (`edit_runtime.py`: `/v2/edit*` of every server over the server's `EditEngine`), sampling helpers, the delay pattern (`delay.py`) and the streaming decode of the generated rows (`stream.py`: `StreamDecoder`, `StreamSettings` and `prefix_window`, the whole / prefix stream of the reference and the MLX server and the window math of the sgl vocoder manager). `plan.py` holds the v2 GenerateRequest rules and prompt plan of every Python server (the reference, the MLX server's `plan_request`, the sgl frontend's `_tts_generate_msg` and the row helpers of its tokenizer worker): request head (tag CFG, speaker mode, condition rows), the rules after the speaker inputs (language, max_tokens, silence prefill, reference-code cut, decode context, accent, controls, span follow, CFG, pronunciation groups, LACI, seed) and the prompt rows with the CFG twin, over a server-supplied context; the ggml and in-browser servers port the same rules (`model/tests/fixtures/request_errors.json`).
- `zonos3.server`: Pieces the Python servers share: request schemas, the edit routes (`edit.py`), the plugin seam (`plugins.py`), the HTTP error mapping (`errors.py`), studio projects and the keepalive middleware.
- `zonos3.reference`: The plain-PyTorch reference LM and synthesis (`python -m zonos3.reference`), the numerical oracle for every server; `synth.plan_prompt` resolves a request's speaker inputs and runs `zonos3.tts.plan`; `synth.synthesize` decodes at EOS or, with `on_audio`, prefix-streams the take (`zonos3.tts.stream.StreamDecoder`, the servers' prefix mode: `PrefixDecoder` on a stateful `ISTFTStream`, chunks ramping 1, 2, 4, 8, 15 frames).
- `zonos3.generation`: The generation registry's loader: sampling defaults, the tag-CFG range, the silence-prefill default and maximum, the decode-context default, the GGUF servers' reference limit and frame budget, the API sample rate and the audio length limits (reference clip, upload, edit).
- `zonos3.registries`: Codec, speaker-encoder, language and generation registries. `languages.json` (loaded by `zonos3.languages`; embedded by the ggml server, imported by the studio and the in-browser server) holds every language table: the request codes with their labels, NeMo packages and lexicon codes, the speech-recognition languages and the espeak-ng voices. `generation.json` (loaded by `zonos3.generation`; embedded by the ggml server, imported by the in-browser server) holds the request defaults and limits every server reads.
- `zonos3.workers`: Standalone worker scripts the servers spawn (text normalization, Qwen3-ASR speech recognition (optional: speech editing aligns clips sent with their transcript without it), FuseAlign forced alignment and speech segmentation, and the ggml server's Python host: `pyhost.py` runs speech editing and the `--plugins` routes and calls back into the C++ engine).

`zonos3_sgl` (`servers/sgl/src/zonos3_sgl`):

- `zonos3_sgl.core`: TTS request, batch, context, and sampling dataclasses.
- `zonos3_sgl.message`: ZeroMQ message dataclasses, with TTS messages in `message/tts.py`.
- `zonos3_sgl.tts`: Sequence helpers, the alignment probe and the offline `TTSLLM` wrapper.
- `zonos3_sgl.tokenizer`: TTS tokenizer/detokenizer worker and the matched-codec vocoder manager (`vocoder.py`).
- `zonos3_sgl.scheduler`: TTS scheduler, decode manager, table manager, and cache wiring.
- `zonos3_sgl.engine`: Model execution, checkpoint loading, KV cache setup, and CUDA graph replay.
- `zonos3_sgl.models`: Zonos3 model (GDN / global attention, Full AttnRes, latent MoE), local checkpoint weight loading, and accent adapters.
- `zonos3_sgl.attention`: FlashAttention and FlashInfer backends.
- `zonos3_sgl.kvcache`: KV cache pools and cache managers.
- `zonos3_sgl.server`: CLI parsing, process launch and the HTTP API.
- `zonos3_sgl.kernel`: Custom CUDA kernels and bindings.

`zonos3_mlx` (`servers/mlx/src/zonos3_mlx`, Apple Silicon; one process, the GGUF files of the ggml server):

- `zonos3_mlx.gguf_io`, `zonos3_mlx.audio_io`: GGUF reading (Q4_0 / Q8_0 blocks repacked as MLX affine-quantized weights) and ffmpeg decoding.
- `zonos3_mlx.models`: The LM backbone (with the gated-delta Metal kernel of mlx-lm), codec decoder, speaker encoder, SSL teacher and reference encoder.
- `zonos3_mlx.companions`: The steering bundle GGUF (`steer.gguf`) as the `zonos3.tts` dataclasses; the accent adapter bank.
- `zonos3_mlx.engine`: The loaded networks and companions, and `plan_request` (a v2 request's prompt rows, conditioning and CFG twin).
- `zonos3_mlx.scheduler`: The worker thread that owns every MLX call: batched slots, CFG, LACI and span-follow, work items.
- `zonos3_mlx.server`: The FastAPI app (speakers, generate, capabilities, phonemize, edit, projects, plugins, studio); `python -m zonos3_mlx` launches it.
- `zonos3_mlx.tools`, `zonos3_mlx.validate`: Single-network CLIs and LM captures for `tools/compare_golden.py`.
