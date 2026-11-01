# Structure

Zonos3 is a TTS-only serving stack for ZONOS3 checkpoints (an `hf://` location, a Hugging Face repo id or a local path).
This page describes the CUDA server (`servers/sgl`, package `zonos3_sgl`) and the model package it builds on
(`model`, package `zonos3`); the ggml server is described in [servers/ggml/README.md](../servers/ggml/README.md).

## Processes

- **API server**: FastAPI frontend for `/v2/generate`, `/v1/audio/speech`, capabilities, speaker cache, edit, dub and studio project endpoints; it also serves the studio build at `/`.
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
6. The tokenizer worker decodes frames through the matched codec (`model/src/zonos3/registries/codec_registry.json`) and sends PCM bytes back to the API server.

## Code Organization

`zonos3` (`model/src/zonos3`, plain PyTorch, never imports `zonos3_sgl`):

- `zonos3.codec`: The vendored dual codec (encoder, quantizer, causal multi-resolution iSTFT decoder).
- `zonos3.models`: Model config conversion, the MECT speaker encoder, speaker cloning and the accent adapter file format.
- `zonos3.tokenizer`: Text normalization (`textnorm.py`), the matched-codec registry (`matched_codec.py`) and the reference encoder (`dual_reference.py`).
- `zonos3.tts`: Prompt building, markup, conditioning, emotion, steering, LACI and span-follow, speech-edit prompts and alignment, sampling helpers.
- `zonos3.dub`: The dub planner (`plan.py`).
- `zonos3.reference`: The plain-PyTorch reference LM and synthesis (`python -m zonos3.reference`), the numerical oracle for both servers.
- `zonos3.registries`: Codec and speaker-encoder registries.
- `zonos3.workers`: Standalone worker scripts the servers spawn (text normalization, MMS alignment, dub alignment).

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
- `zonos3_sgl.dub`: The dub runtime behind `/v2/dub/*`.
- `zonos3_sgl.server`: CLI parsing, process launch, request schemas, studio projects, and the HTTP API.
- `zonos3_sgl.kernel`: Custom CUDA kernels and bindings.
