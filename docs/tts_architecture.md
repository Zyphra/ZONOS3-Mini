# TTS System Architecture

## Overview

mini-sglang serves a multi-codebook TTS transformer (Zonos3) through a multi-process pipeline with streaming audio output. Text is encoded as SuperBPE (or UTF-8 byte) token frames, fed through an autoregressive transformer that predicts 10 parallel audio codebook streams of the matched dual codec, and decoded to 44.1 kHz PCM audio by that codec's causal iSTFT decoder - with streaming responses to the HTTP client.

Paths below are relative to the two packages: `zonos3/` (`model/src/zonos3`, model code shared with the ggml server and the reference) and `zonos3_sgl/` (`servers/sgl/src/zonos3_sgl`, the CUDA server).

## Process Architecture

Four process types communicate over ZMQ:

```
                  ZMQ                ZMQ                ZMQ
  Frontend  ──────────►  Tokenizer  ──────────►  Scheduler  ──────────►  Detokenizer
  (FastAPI)  ◄──────────             ◄──────────  (Engine)   ◄──────────  (Vocoder)
             TTSAudioReply          TTSUserMsg             TTSDetokenizeMsg
```

| Process | Count | Role |
|---------|-------|------|
| Frontend | 1 | FastAPI server. Accepts `POST /v2/generate` and `POST /v1/audio/speech`, streams PCM chunks back. Logs TTFB, E2E latency, RTF. |
| Tokenizer | N (default 0 = shared with detokenizer) | Runs `TTSPromptBuilder`: text → SuperBPE / byte tokens → 2D prompt tensor (speaker row, condition rows, reference prefix, text). Sends `TTSUserMsg` to scheduler. |
| Scheduler | 1 per TP rank | Runs `TTSScheduler` wrapping `Engine`. Autoregressive decode loop: prefill → decode → sample → emit frames. |
| Detokenizer | 1 | Runs `TTSVocoderManager`: audio codes → matched codec decoder → PCM bytes. Sends `TTSAudioReply` to frontend. |

Spawned by `launch_server()` in `zonos3_sgl/server/launch.py`. Each scheduler process holds one GPU; tensor parallelism across ranks is coordinated via NCCL.

### Message types

Defined in `zonos3_sgl/message/tts.py`:

| Message | Direction | Key fields |
|---------|-----------|------------|
| `TTSTokenizeMsg` | Frontend → Tokenizer | `uid`, `text`, `sampling_params`, optional speaker/condition conditioning |
| `TTSUserMsg` | Tokenizer → Scheduler | `uid`, `input_ids: Tensor (seq_len, frame_width)`, `sampling_params` |
| `TTSDetokenizeMsg` | Scheduler → Detokenizer | `uid`, `audio_codes: List[int]` (one frame), `finished`, `eos_frame` |
| `TTSAudioReply` | Detokenizer → Frontend | `uid`, `audio_data: bytes` (float32 PCM), `finished`, `sample_rate` |

## Token Format

Each time step is a **frame** of width `n_codebooks + 1`:

```
[cb0, cb1, cb2, cb3, cb4, cb5, cb6, cb7, cb8, cb9, text_token]
 -------------- 10 audio codebooks ---------------   text id
```

- **Audio codebook tokens**: integers in `[0, 4095]` (codebook_size = 4096).
- **Special tokens**: `eoa_id = 4096` (end-of-audio), `audio_pad_id = 4097` (padding); speech-editing checkpoints add `span_end_id = 4098`.
- **Text column**: SuperBPE (or byte) token ID during prompt; `text_vocab` placeholder during generation. Extra rows after the text padding row hold the edit markers and phoneme-group tokens (`text_extra_tokens`).
- **Conditioning rows**: speaker-embedding checkpoints put a speaker row (`speaker_projection`, 192 → model dim) before the text; checkpoints with condition tokens (ZONOS3 Mini: `loudness`, `fmax`, `speaking_rate`; table i has `condition_bucket_counts[i]` buckets plus an unknown row, 10 / 10 / 30 on the default `lm/rate30_20k_r2grpo`, 10 each on `lm/mect_60k`) add one condition row per condition (`condition_embeddings`). Raw-reference cloning places the delayed reference codes (the reference prefix) before the text.

During the prompt phase, audio columns are filled with `audio_pad_id` and the text column carries the token sequence. An optional codec-matched silence prefill (`silence_prefill_frames`; default 3 frames = 0.2 s, `ZONOS3_SILENCE_PREFILL_FRAMES` / `ZONOS3_SILENCE_PREFILL_DEFAULT`) appends the delayed rows of the registry's `silence_frame` after the text; the vocoder decodes them as left context and never emits them.

### Shear pattern

The model is trained with an inter-codebook delay: codebook `j` is shifted by `j` frames. `shear()` (`zonos3/tts/prompt.py`) applies this delay to the prompt; `shear_up()` (`zonos3/tts/delay.py`) inverts it before vocoder decoding.

## Request Lifecycle

```
1. HTTP POST /v2/generate (or /v1/audio/speech)
   ├─ FrontendManager sends TTSNormalizeMsg(text, language) first: the tokenizer normalizes the text while the
   │  frontend resolves the request's rules and speaker (NeMo costs ~0.2 ms per English character)
   └─ FrontendManager assigns uid, creates TTSTokenizeMsg

2. Tokenizer process
   ├─ text normalization                   # TTSTextNormalizer, memoized per (text, language): a hit after step 1
   ├─ TTSPromptBuilder.build(text)         # SuperBPE / byte tokenization (zonos3/tts/prompt.py)
   │   └─ per text token: [audio_pad_id]*10 + [text_token_id]
   ├─ speaker row / condition rows / reference prefix before the text
   ├─ append sheared silence prefill rows  # optional, codec-matched silence frame (default 3 frames)
   └─ emit TTSUserMsg(input_ids: 2D tensor)

3. Scheduler
   ├─ Prefill: engine.forward_batch_tts()  # full prompt in one pass (GDN checkpoints: a CUDA graph at padded shapes)
   │   └─ extract last-token logits per sequence
   └─ Decode loop (frame by frame):
       ├─ engine.forward_batch_tts()       # single frame input
       ├─ sample_tts(logits)               # → [cb0..cb9, text_placeholder] on the GPU, without a host sync
       │                                   #   (the host schedules the next step while the GPU runs this one; the
       │                                   #   scheduler stream waits on the engine stream before touching the token pool)
       ├─ TTSSequence.append_token()       # EOS detection
       ├─ emit TTSDetokenizeMsg per frame
       └─ repeat until is_finished

4. Detokenizer (TTSVocoderManager)
   ├─ accumulate frames per uid
   ├─ shear_up() to remove delay pattern
   ├─ matched codec decode (iSTFT head) → float32 PCM @ 44.1 kHz
   │   (whole utterance at EOS, or progressive prefix chunks with stream_decode="prefix")
   └─ emit TTSAudioReply

5. Frontend streams audio_data bytes to HTTP client
```

## Model Architecture (Zonos3)

`Zonos3ForCausalLM` in `zonos3_sgl/models/zonos3.py` (the plain-PyTorch twin with the same parameter names is `Zonos3LM` in `zonos3/reference/lm.py`).

```
Input (seq_len, frame_width)
  │
  ▼
MultiEmbedding              # sum of per-column embeddings → (seq_len, hidden_size)
  │
  ▼
emb_norm                    # RMSNorm (no learnable params)
  │
  ▼
TransformerBlock × N        # residual stream with fused RMSNorm
  │  ├─ attention_norm (RMSNormFused)
  │  ├─ Attention
  │  │   ├─ wq, wkv, wo projections
  │  │   ├─ QK norm + learnable temperature (per-head)
  │  │   ├─ RoPE (interleaved format)
  │  │   ├─ FlashAttention / FlashInfer backend
  │  │   └─ headwise gating (sigmoid)
  │  ├─ ffn_norm (RMSNormFused)
  │  ├─ FeedForward (dense) or MoEFeedForward
  │
  ▼
out_norm (RMSNormFused)
  │
  ▼
MultiOutputHead             # linear: hidden → (n_codebooks × audio_vocab)
  │                         #   reshape → (*, 10, 4098) for ZONOS3 Mini
  ▼
softcap(logits, loss_softcap)  # tanh-based logit capping (15.0 for ZONOS3 Mini)
```

### MultiEmbedding

Maintains separate `VocabParallelEmbedding` tables for each of the `n_codebooks` audio codebooks (10 for ZONOS3 Mini) plus the text column. Forward pass sums all column embeddings element-wise.

### Attention

- **QK normalization**: `F.rms_norm` on Q and K, then Q is scaled by `|temp|` (learnable per-head parameter).
- **Gating**: After attention output, applies `sigmoid(gater(x))` either per-head (`headwise`) or per-element (`elementwise`), then multiplies with the attention result.
- **RoPE**: Interleaved format (`is_neox=False`).

### MoE

Activated for middle layers (`moe_start_from_layer` to `num_layers - moe_end_from_layer`).

- **Router**: Projects hidden → `router_dim` via `down_proj`, then a 3-layer MLP with GELUs produces expert logits.
- **Expert Dropout Augmentation (EDA)**: Non-first MoE layers blend the previous layer's router hidden states (`router_states * scale + current`) before routing. This threads router context across consecutive MoE layers.
- **Experts**: `FusedGroupedExperts` using fused gate+up projection (`w1 || w3` → `gate_up_proj`) with SiLU activation, and a down projection (`w2` → `down_proj`). Uses the `FusedMoE` kernel for efficient dispatch.

## Sampling

`sample_tts()` in `zonos3/tts/sampler.py`. Input: `(B, n_codebooks, vocab_size)` logits.

1. **Repetition penalty**: For each request and selected codebook, gather the last `repetition_window` generated tokens from that codebook only and apply `repetition_penalty` to matching logits. `repetition_codebooks=N` applies this to CB0 through CB(N-1), while a negative value means all codebooks. `repetition_window=0` or `repetition_penalty=1.0` disables this path.
2. **Temperature**: Per-sequence scaling, broadcast over codebooks.
3. **Top-k**: Single `torch.topk` with `min(top_ks)` across the batch.
4. **Top-p / Min-p**: Applied to the softmax probabilities using the most restrictive value across the batch; `top_p=0` and `min_p=0` disable them.
5. **Multinomial sampling**: Per-codebook independently. Supports per-request `torch.Generator` for seeded determinism.
6. **Output**: `[cb0, ..., cb9, text_vocab]` — audio tokens plus text placeholder.

Default params (`TTSSamplingParams`): temperature=0.75, topk=80, top_p=0.69, min_p=0.0, max_tokens=1024, repetition_window=96, repetition_penalty=2.4, repetition_codebooks=6.

## EOS Detection

Managed by `TTSReq.check_eos()` in `zonos3_sgl/core.py`. Each request has one stop id (`TTSReq.stop_id`): EOA for utterances
(`end_token="utterance"`), the span-end token for edit fills (`end_token="span"`, edit checkpoints only).

1. Each frame, check whether any delayed audio codebook sampled the request's stop id.
2. When detected: align `eos_frame` by subtracting the highest stop-token codebook index, then start countdown = `n_codebooks + 1` (11 steps for 10 codebooks). `--eoa-policy codebook0` lets only a stop token in codebook 0 start it (default `any`).
3. Sequence marked finished when countdown reaches 0 — this buffer allows frame alignment across codebooks due to the shear pattern.
4. Also terminates if `num_completion_tokens >= max_tokens`.

The sampler (`sample_tts` in `zonos3/tts/sampler.py`) masks padding and every special id except the request's stop id, so an
utterance never samples span-end and a fill never samples EOA. Utterances draw over codes, EOA and padding only: with
one random number per column in `torch.multinomial`, an edit checkpoint then produces the same audio for the same
seed as its weights without the span-end row.

## Vocoder (matched dual codec)

`TTSVocoderManager` in `zonos3_sgl/tokenizer/vocoder.py` decodes with the codec the LM's training data was tokenized with (the
checkpoint's `audio_codec`). Only matched codecs of the codec registry are accepted. Each request decodes through the
decoder it names (`decoder`, a served decoder id; null = the default `keep2_plain`), carried on its sampling params to
every `TTSDetokenizeMsg` and on `TTSDecodeCodesMsg` (edits).

### Codec registry

`zonos3/registries/codec_registry.json` holds one entry per codec. The ZONOS3 Mini entry
(`opus_dual_3n_600k_sched600k_r4096_step600000`, family `dual`) declares:

- `codebooks` 10, `codebook_sizes` 10 × 4096, `hop` 2940 samples at 44.1 kHz (`frame_rate` 15), `codec_step`;
- `checkpoint` + `checkpoint_sha256`: the codec generator (`hf://…/autoencoder/generator.pt`), whose sha256 is checked
  before the first decode;
- `teacher`: the semantic SSL teacher the encoder needs (`hf://…/autoencoder/teacher`);
- `silence_frame`: the steady-state codes of digital silence, used for the silence prefill;
- `decoders`: the decoders a request may pick by id, and `decoder_default` (`keep2_plain`). Each is `kind:
  "istft_mr"` with its `campaign` folder, `config` and `checkpoint`, `expect_step`, and `headroom_db` (3.0: the default
  output gain of −3 dB, overridable per request with `headroom_db`; the iSTFT head is unbounded):
  - `keep2_plain` (the default): the KEEP-2 pruned head (`autoencoder/heads/keep2_plain`,
    `config_full_unfreezeall_mr_drop30_4x10.json`: two layers per transformer stage);
  - `full_mr`: the full-size head (`autoencoder/heads/full_mr`, `config_full_unfreezeall_mr.json`, decoder.0-8
    unfrozen).

  Both share the codec, its frame rate, codebooks and the stream / flush semantics; they differ only in layer counts.
  The GGUF servers serve the same two heads as `zac-keep2.gguf` and `zac.gguf` (the id is the GGUF's
  `mossdual_istft.head.variant`), the web engine as its exported model directories.

### Decoder

`zonos3.tokenizer.matched_codec.load_model` builds `zonos3.codec.istft_head_a_20261002.model.ISTFTCodec` from the
decoder config, loads the generator's pretrained weights (`generator.pt`: quantizer and the initial decoder stages),
then the MR head's trainable state (decoder stages + causal multi-resolution iSTFT head) from the decoder checkpoint;
the generator's conv-tail decoder is never used. The model is cached per decoder id and device
(`ZONOS3_AUDIO_CODEC_DEVICE`, default `cuda:0`); `decode`, `stream`, `capture_stream_graphs`, `decoder_info` and
`decoder_headroom_db` take the decoder id (None = the default). `decode_model` runs `[B, frames, 10]` codes in fp32 with
TF32 off and checks the output length (`frames × hop`) and finiteness.

The sgl server loads every decoder of the registry entry at startup (`--decoders keep2_plain,full_mr` restricts the
set; the registry default stays the default when listed, else the first listed is); `GET /v2/capabilities` lists them
in `decoders` (default first, also `decoder`), a request picks one with `decoder` (unknown id: 400) and audio
responses name it in `X-Decoder`. Everything below runs per decoder exactly as for the default.

### Streaming decode

- Frames accumulate in a per-request buffer (after any silence prefill rows, which are decoded but never emitted).
- With `N` accumulated rows, only `N - (n_codebooks - 1)` output frames are complete (the rest need later rows for
  `shear_up`); the EOS frame (from the scheduler's `eos_frame` or an EOA seen locally in a row) bounds them further.
- **Whole decode** (default for non-streaming responses, and `stream_decode="whole"`): at EOS, `_decode_final`
  undelays the rows with `shear_up` and decodes every complete pre-EOS frame in one codec call. The last
  `overlap_frames` (default 4) already-counted rows (the silence prefill) are re-decoded as left context.
- **Prefix decode** (`stream_decode="prefix"`, the server default for streaming responses;
  `ZONOS3_STREAM_DECODE_DEFAULT=whole` turns it off): the frames are decoded incrementally on a stateful codec stream
  (conv histories, per-layer KV caches, overlap-add iSTFT, FIR mixer and resampler state;
  `zonos3.tts.stream.PrefixDecoder` over `ISTFTStream`), nothing decoded twice. Chunks ramp: the first waits
  for `ZONOS3_PREFIX_STREAM_FIRST_CHUNK_FRAMES` (1) complete frames past the silence prefill, each later one for
  `first + frames emitted so far`, capped at `ZONOS3_PREFIX_STREAM_CHUNK_FRAMES` (15, 1 s): 1, 2, 4, 8, 15, 15, ...
  `ZONOS3_PREFIX_STREAM_HOLD_BACK_FRAMES` (0) complete frames stay undecoded until EOS, so first audio needs
  9 (shear) + 1 generated frames. At EOS the stream continues to the end and is flushed. The stream emits only final
  samples, so the concatenated chunks equal the whole decode in fp32 (CPU); on CUDA the trunk runs in bf16 and any
  change of chunk boundaries moves the PCM by ~1e-3 RMS. `ZONOS3_PREFIX_STREAM_CONTEXT_FRAMES` is still validated
  (>= chunk) but unused.
- **Primed stream** (prefix decode with the silence prefill rows and no decode context): the vocoder decodes the
  prefill frames once as pure silence and every such request continues a fork of that stream
  (`TTSVocoderManager._prefix_stream`, one primed stream per decoder; `ISTFTStream.fork`: the trunk state tensors are
  shared, never written in place; the head's overlap-add buffers are copied), so the first chunk is the first
  generated frame alone. The
  prefill frames' codebooks that fall in generated rows (frame f, codebook j with f + j >= prefill) are the LM's and
  are not decoded: the PCM moves ~1.7e-3 RMS (signal ~0.16), the same size as moving a chunk boundary
  (~1.4e-3); the codes are unchanged.
- **Codec CUDA graphs**: at tokenizer start `warm_prefix_stream` primes, for every served decoder, the default prefill's stream and captures the
  chunk shapes of the primed ramp (3, then 1, 2, 4, 8, 15 ... until every KV cache is full at 150 frames, then the
  steady 15-frame chunk; `zonos3.tts.stream.prefix_chunks`, `matched_codec.capture_stream_graphs`). A chunk whose
  shape (frames, stream position up to `ISTFTStream.full_frames`) was captured replays `ChunkGraphs` (state copied
  in and out, position as a graph input); other shapes (decode context, other prefills, EOS tails) run eagerly.
  Replays are bit-identical to the eager decode; they cut the GPU half of the first chunk from ~11.8 ms of launches
  to ~2.9 ms.
- **Device iSTFT head**: the overlap-add iSTFT, FIR mixer and resampler of a stream run on the codec's device
  (`zonos3.codec.stream_device`, float64 like the NumPy ports in `zonos3.codec.stream_host` that the MLX server keeps,
  and bit-identical to them), so a chunk's only host copy is its final samples. On one H100 the first chunk's
  `decode_frames` falls from ~6.8 to ~4.3 ms (iSTFT + mixer 2.25 -> 0.60 ms, resampler 0.30 -> 0.19 ms, primed-stream
  fork 0.60 -> 0.06 ms), leaving the trunk's graph replay (~3.1 ms) as the cost.
- **Early partial-frame decode** (on by default in every server: `stream_codebooks` = 1 on the `first` chunk, the
  generation registry's `stream_codebooks`; k = n_codebooks waits for complete frames): a frame is decodable once its
  first k codebooks arrived (row f + k - 1 instead of f + 9; `zonos3.tts.stream.PrefixDecoder.ready`, capped at EOS
  and before any frame whose semantic code is padding/EOA). The missing residual codes reach the codec as -1 and the
  codec leaves those codebooks out of the RVQ sum, bias included (the codec trained with quantizer dropout 0.5; the
  GGUF codecs need the per-codebook `res.{i}.bias` tensors, without them early decode is off for that model); a frame
  is decoded once, never refined. With scope `first` (default) only the first chunk is early and later chunks wait
  for complete frames; `all` decodes every chunk early. The codes are unchanged; the complete-code decode stays
  bit-identical. Measured on sgl / one H100 (`early_sweep.py`): TTFA ~52.5 ms at k = 10 falls ~3.1 ms per codebook to
  ~24.4 ms at k = 1; against the k = 10 take of the same seed the waveform SNR is 23.9 dB with scope `first` (6.5 dB
  with `all`).
- With a raw-reference prompt, `decode_context_frames` (default 30) of the reference's tail codes are prepended as
  decoder context whenever a decode starts at the stream start, and their samples dropped.

### Decode pipeline

```
delayed rows (seq_len, 10)
  → shear_up(): remove inter-codebook delay, fill with audio_pad_id
  → slice to the complete pre-EOS frames
  → matched_codec.decode(codes, codec, decoder): sha256 check, ISTFTCodec of that decoder (fp32, TF32 off)
  → waveform (float32, 44.1 kHz, frames × 2940 samples)
```

### Reference encoder

Raw-reference cloning and the silence-frame fallback encode audio with the same codec:
`zonos3/tokenizer/dual_reference.py` takes the registry's single `dual`-family entry and builds
`zonos3.codec.codec_adapter.MossAutoencoder` from its generator checkpoint, SSL teacher, step and codebook sizes
(SSL teacher → semantic Q1, acoustic residual RVQ). Reference audio is mixed to mono, resampled to 44.1 kHz, peak
normalized only above 1.0 and cut to whole codec frames within the reference limit; the result is int32 codes
`[frames, 10]`.

## GDN / Full-AttnRes family (2026-10)

Checkpoints with `attention_pattern != "global"` use `zonos3_sgl/models/gdn.py` (`OriginalGDN`): the canonical 64-frame chunked
gated-delta-rule recompute (FLA's `chunk_gated_delta_rule` + `causal_conv1d` forward functions, called directly with
`cu_seqlens` and chunk indices built on the host and copied without blocking; FLA would derive the indices with a host
sync per call) with per-slot float32 recurrent and
convolution states (`--cache-type naive`). The unfinished chunk of every slot is kept in a fixed 64-row window
(`_window`, `_win_len`), which the eager prefill path reads and writes and which makes decode a static-shape
computation: `_forward_decode_static` processes every padded request as one 64-row sequence (real rows first, zero
rows after; the kernels are causal, so the real rows are bit-identical to the ragged call), gathers the new row's
output and commits the states only when a window fills. `Engine` captures that step in CUDA graphs per batch size
(`GraphRunner` hooks `before_capture` / `after_capture` / `before_replay` fill the static slot buffer and switch the
layers' static mode; `note_graph_forward` keeps the host mirrors current after a replay). Two details keep replays
valid: all layers share one set of static `cu_seqlens` / chunk-index tensors (FLA memoizes results derived from them by
tensor identity in 4-entry caches), and the engine pins the cache results those tensors produced (`pin_fla_cache`) because
the captured kernels hold raw pointers to them and eager prefills would otherwise evict and free them. The fused
Full-AttnRes bank is one persistent context per graphed shape (`_attnres_state`).

Prefills of fresh requests run as CUDA graphs too (`zonos3_sgl/engine/prefill_graph.py`, `PrefillGraphs`): eagerly a
prefill launches ~900 kernels (~1,400 with a reference prefix) for 3-6 ms of GPU work in 40-60 ms of host time. Each
request is padded to its complete 64-row chunks plus one 64-row window (`PrefillLayout`), so the layer stack's shapes
depend only on the requests' complete-chunk counts; one graph per tuple of counts, captured on first use (single
requests of up to 5 complete chunks at startup) and kept in an LRU (`ZONOS3_PREFILL_GRAPHS`, 16) for prefills of up to
`ZONOS3_PREFILL_GRAPH_MAX_ROWS` padded rows (1024); `ZONOS3_PREFILL_CUDA_GRAPH=0` turns them off. The embedding
prologue (speaker and condition rows) runs eagerly on the real rows (`Zonos3ForCausalLM.embed`), which are scattered
into the graph's padded input; the graph runs `forward_layers`, the head and the last-row gather. No real row reads a
padded one: attention is causal over each request's own page-table row (padded positions write their KV to the dummy
page), `_forward_prefill_static` runs the eager path's committed pass over the complete chunks and its uncommitted tail
pass as a zero-padded 64-row window (the decode window's layout), and every other kernel is batch-invariant (below), so
graph and eager prefills leave bit-identical logits, GDN states and KV rows. Continuations (a cached prefix: LACI
rollbacks) and accent adapters stay eager. `residual_type == "full_attnres"` routes
every sub-layer output through `full_attnres_route()` in `zonos3_sgl/models/zonos3.py` (reference schedule
`_forward_full_attnres_reference`, or the trainer's Triton kernels in `zonos3_sgl/models/training_attnres*.py` when
`ZONOS3_USE_TRAINING_ATTNRES=1`). The `ZONOS3_USE_TRAINING_{RMSNORM,ROUTER_WEIGHTS,SONIC_EXPERTS,ATTNRES,ROPE}` switches
make the forward bit-exact with the packed training forward ("exact mode"); the scheduler samples these models from
float32 logits (`sample_logits_in_fp32` in `zonos3/tts/sampler.py`). Exact mode is also batch-invariant and exact at long
context: `MultiOutputHead` pads the head GEMM to `head_min_rows()` (2 for this family) because cuBLAS uses a different,
one-ulp-off kernel for a single row, and `FlashAttentionBackend` passes `fa_num_splits()` (1 for this family, the
training kernel's setting) because FA3's automatic KV split rounds differently for caches beyond ~512 rows.
`Engine._check_head_batch_invariance()` verifies the head property at load. Every other kernel in the exact path (all
other linears, the FLA chunk kernel, SonicMoE, the training RMSNorm and AttnRes kernels) is batch-invariant as is.

Two exact-mode kernels are Hopper-only: SonicMoE's quack grouped GEMM ("Gemm Sm80 is not implemented yet") and FLA's
TileLang chunk kernels. On a GPU below sm_90 (A100) `servers/sgl/scripts/env.sh` leaves
`ZONOS3_USE_TRAINING_SONIC_EXPERTS` and `FLA_TILELANG` off, so the experts run the Triton `FusedMoE` and the GDN chunks
FLA's Triton kernels; the rest of exact mode (RMSNorm, router weights, AttnRes, RoPE) is portable, and the `fa` backend's
sgl-kernel FA3 has an sm_80 build (head_dim 32 runs on its 64-wide kernels, as on Hopper). The forward is no longer
bit-exact with training. Forced on an H100 (`FLA_TILELANG=0 ZONOS3_USE_TRAINING_SONIC_EXPERTS=0`), the profile measured:
the expert layer 20 vs 42 us at 1 token and 97 vs 204 us at 1024 (SonicMoE is a training kernel), TTFA p50 13.7 vs 14.7 ms,
1700 CV3 / MiniMax samples at 256 running requests in 110 vs 115 s, and the same WER / SIM within seed noise (pooled word
errors -1.6 %).

Text uses the SuperBPE convention in `zonos3/tts/superbpe.py` (`[2] + (ids + 4) + [3]`, artifact pinned by SHA-256 through
`params.json`). Audio uses 10 codebooks of 4096 (`eoa_id` 4096, `audio_pad_id` 4097) from the dual codec; the matched
codec registry (`zonos3/tokenizer/matched_codec.py`) provides both the encoder for reference audio
(`zonos3/tokenizer/dual_reference.py`) and the vocoder. The raw-audio reference prefix (`speaker_reference_prefix_tokens` in
`zonos3/tts/prompt.py`, or caller-supplied `speaker_reference_codes`) precedes the text rows; speaker CFG removes it for the
unconditional twin. Speech-editing checkpoints add a span-end row per codebook (`span_end_id` 4098) and
`text_extra_tokens` marker rows after the text vocabulary; `zonos3/tts/edit.py` builds the fill prompts
([guide: Speech editing](guide.md#speech-editing)).

## Tensor Parallelism

Weights are sharded across TP ranks:

| Component | Sharding |
|-----------|----------|
| `wq`, `gater` | Column-parallel (split output dim) |
| `wkv` | Split KV dim (dim 1 of 3D weight) |
| `wo` | Row-parallel with all-reduce |
| `w_in` (FFN) | Split intermediate dim (dim 1) |
| `w_out` (FFN) | Row-parallel with all-reduce |
| MoE `gate_up_proj` | Split intermediate (dim 1) |
| MoE `down_proj` | Split intermediate (dim 2) |
| Embeddings | Vocabulary-parallel |
| `multi_output` | Output vocabulary sharded |
| `attention.temp` | Split heads (dim 1) |

Sharding logic in `zonos3_sgl/models/weight.py`. Communication via NCCL all-reduce in output projections.

## Configuration

Key `ServerArgs` fields (in `zonos3_sgl/server/args.py`):

| Field | Default | Description |
|-------|---------|-------------|
| `tts_n_codebooks` | `9` | Number of audio codebooks (replaced by the checkpoint's `n_codebooks`, 10 for ZONOS3 Mini, while left at the default) |
| `tts_audio_pad_id` | `1025` | Audio padding token (replaced by the checkpoint's `audio_pad_id`, 4097, while left at the default) |
| `tts_text_vocab` | `None` | Text vocabulary size (auto-detected from config) |
| `tts_sample_rate` | `44100` | Output audio sample rate |
| `tts_audio_codec` | `opus_dual_3n_600k_sched600k_r4096_step600000` | Matched codec, taken from the checkpoint's `audio_codec` |
| `tts_default_voices_dir` | `None` | Optional folder of audio or `.npy`/`.npz` embedding files listed as default voices in `GET /v2/speakers` and the studio's speaker picker (`servers/sgl/scripts/serve.sh`: `default_voices/`) |
| `studio_dir` | `<repo>/studio/dist` | Studio build served at `/` |
| `steer_bundle` | `None` | Steering bundle JSON behind `controls` / `control_spans` and `GET /v2/controls` (`servers/sgl/scripts/serve.sh`: `steering/studio_bundle_mect60k_x10.json`) |

Model-level config is loaded from checkpoint via `cached_load_checkpoint_config()` (`zonos3/utils/hf.py`) and `ModelConfig.from_checkpoint_config()` in `zonos3/models/config.py`. Key fields: `n_codebooks`, `codebook_size`, `text_vocab`, `eoa_id`, `audio_pad_id`, `loss_softcap`, `moe_n_experts`, `moe_start_from_layer`, `moe_router_dim`.

## Key Files

| Path | Role |
|------|------|
| `zonos3_sgl/server/api_server.py` | HTTP endpoints, `FrontendManager` |
| `zonos3/server/schemas.py` | Request models (the v2 `GenerateRequest`, `SpeechRequest`, …), shared by the Python servers |
| `zonos3/tts/plan.py` | v2 request rules and prompt plan of every Python server (the frontend's checks, the tokenizer worker's row helpers) |
| `zonos3/generation.py` | Generation registry (`registries/generation.json`): sampling defaults and request limits |
| `zonos3_sgl/server/launch.py` | Process spawning |
| `zonos3_sgl/server/args.py` | `ServerArgs` configuration |
| `zonos3_sgl/tokenizer/server.py` | Tokenizer/detokenizer worker loop |
| `zonos3/tts/prompt.py` | Prompt rows (speaker, condition, reference prefix, text), shear, silence prefill rows |
| `zonos3/tts/superbpe.py` | SuperBPE text convention, phoneme groups |
| `zonos3_sgl/tokenizer/vocoder.py` | `TTSVocoderManager`: undelay, whole / prefix decode |
| `zonos3/tokenizer/matched_codec.py` | Codec registry, `ISTFTCodec` decoder load and decode, per decoder id |
| `zonos3/tokenizer/dual_reference.py` | Reference-audio encoder (`MossAutoencoder`) |
| `zonos3/codec/` | Vendored dual codec: encoder adapter, quantizer, causal MR iSTFT decoder |
| `zonos3_sgl/engine/engine.py` | `Engine.forward_batch_tts()` |
| `zonos3_sgl/models/zonos3.py` | `Zonos3ForCausalLM` |
| `zonos3/models/config.py` | `ModelConfig` |
| `zonos3_sgl/models/weight.py` | Weight loading and TP sharding |
| `zonos3_sgl/tts/sequence.py` | `TTSSequence`, EOS detection |
| `zonos3/tts/sampler.py` | `sample_tts()` |
| `zonos3_sgl/tts/llm.py` | `TTSLLM` offline generation interface |
| `zonos3_sgl/message/tts.py` | Message dataclasses, `TTSSamplingParams` |
| `zonos3_sgl/attention/` | FlashInfer / FlashAttention backends |
| `zonos3_sgl/layers/` | Linear, norm, RoPE, MoE layers |
| `zonos3/reference/` | Plain-PyTorch reference LM and synthesis (`python -m zonos3.reference`) |
