# zonos3.cpp internals

The supported backbone is **ZONOS3 Mini** (`Zyphra/ZONOS3_Mini_release_ckpts` revision `4d641a9fedd8c5544fca6e6f17ebb5c351c1a510`), LM `lm/mect_60k`, step 60000. Its codec identity is `opus_dual_3n_600k_sched600k_r4096_step600000`. Backbone files use `general.architecture=zonos3` and `zonos3.*` metadata (including speaker, condition and silence-frame keys); the 375k backbone schema, the conv-tail decoder and embedding-free conditioning are rejected, with no compatibility path.

## Model pipeline

```text
UTF-8 text -> embedded SuperBPE-8192 -> BOS + ids+4 + EOS
reference audio -> native MECT-B2 speaker encoder -> 192-d embedding        (default)
reference audio -> native acoustic encoder + frozen SSL teacher -> [frames,10] codes (raw reference, opt-in)
    -> [speaker row | delayed reference prefix] + 3 condition rows + text prompt + silence rows
    -> 11-layer, 1024-wide mixed GDN/global Full AttnRes backbone (speaker/condition rows replace their embeddings)
    -> 10 delayed audio codebooks at 15 Hz (optional speaker-CFG twin)
    -> semantic RVQ/ConvNeXt head + nine residual RVQ codebooks
    -> five windowed-causal MOSS decoder stages (15 -> 240 Hz) + ConvNeXt head
    -> multi-resolution iSTFT (n_fft 736/2208, hop 184) + FIR mixer at 44160 Hz
    -> exact 44160-to-44100 resampling, output gain -> 44.1 kHz mono waveform
```

### Tokenizer and prompt

`src/zonos3-tokenizer.cpp` reproduces the HF tokenizer's added-token splitting, Split pre-tokenizer, GPT-2 byte mapping and rank-ordered BPE merges. Unicode categories come from generated `src/unicode-data.h`. The pinned tokenizer SHA-256 is `90322f910600027dee7fe63d400d1d9dc1035c167d560aadf0871e3661d08bf3`; the target artifact is identical to the previously used SuperBPE artifact.

Text IDs are BOS `2`, payload IDs plus `4`, EOS `3`, and text padding `8196`. Audio ordinary codes are `0..4095`, EOA is `4096`, and audio padding is `4097`. Each input row has ten audio columns and one text column.

Prompt rows, in order: `[speaker row (1) | reference prefix (frames+9)] [loudness][fmax][speaking_rate] BOS text… EOS [silence rows]`. The speaker row and the reference prefix are mutually exclusive. Speaker and condition rows are ordinary all-pad rows (ten `4097` audio columns, text column `8196`); unconditioned prompts still carry the three condition rows.

- **Row replacement.** After the 11-way embedding sum and before the weightless `emb_norm`, the summed embedding of the speaker row is replaced by `speaker_projection(vec)` (raw vector, no normalization) and condition row *i* by `condition_embeddings[i][bucket_i]` (`src/zonos3-graph.cpp`, inputs `ovr_keep/ovr_sel/ovr_spk/ovr_bkt`, functional ops only). The graph maps absolute prompt rows to chunk-local rows, so chunked prefill equals single-shot prefill; `zonos3_check_conditioning` rejects uncovered all-pad rows.
- **Buckets.** `bisect_right(edges, value)` over the nine per-condition edges stored in the GGUF gives `0..9`; `10` is unknown. Explicit bucket ids win over values; omitted = unknown.
- **Reference prefix.** For N ordinary frames, N+9 rows: audio column c at prefix row t holds reference frame t−c when in range, otherwise audio padding; text columns are text padding. No reference EOA, transcript or language marker is inserted. References are at most 60 seconds; context accounting includes all prompt rows, generation rows and EOS draining.
- **Silence prefill.** N (default 3, `0` disables, max 30) copies of the checkpoint's silence frame, sheared (row t, column c holds frame t−c if ≥ 0 else padding; only the first N rows, no flush tail), appended after EOS with text padding.
- **Sidecar.** `--build-prompt`/`--dump-ids` write `<ids>.cond.json` (speaker row/vector, condition row/buckets, `speaker_rows`, `prefill_frames`), which `--validate`, `--generate` and `zonos3-perplexity` load.
- **Tokenizer spec and extra text rows.** The loader requires `zonos3.text_tokenizer_spec` (the checkpoint's `params.text_tokenizer_spec`; checked like `superbpe.validate_spec`) and reads the optional `zonos3.span_end_id` (edit checkpoints: `audio_pad_id + 1`, audio vocabulary 4099). Text rows beyond the padding row come from the `text_embd` shape (`hp.text_extra_tokens`): speech-edit marker rows (`text_vocab + 1 + j`, `zonos3_edit_marker_rows`/`zonos3_edit_marker_id`) then, on a phoneme-token checkpoint, the phoneme block, which must end them. `GET /v2/capabilities` reports the spec as stored.
- **Pronunciation groups.** On a phoneme-token checkpoint `U+E000 IPA U+E001` encodes as `open_id`, one id per IPA code point (`open_id + 2 + index`), `close_id`, the text between groups as SuperBPE on its own (`superbpe.payload_ids`, `zonos3_text_ids`); malformed groups and groups on other checkpoints are the sgl server's 400s (`zonos3_check_phoneme_groups`). Nonverbal markup is ordinary text.
- **CFG twins.** Speaker CFG drops the speaker row / reference prefix and sets every condition bucket unknown (`zonos3_build_cfg_twin`); tag CFG (`tag_cfg_scale > 1`) builds the same prompt from `zonos3_strip_nonverbal_markup(normalized text)`. Both run as a second slot guided by `uncond + scale · (cond − uncond)`.
- **Request prompts.** `zonos3-cli <model.gguf> --request req.json --build-prompt ids.npy [--text-normalizer-python PY --python-src DIR] [--steer F]` runs the server's request resolver (`plan_generate`, `src/server/generate.cpp`) and writes the prompt (+ `ids.twin.npy` for a CFG twin) with sidecars, the files of `python -m zonos3.reference --request req.json --dump-prompt ids.npy`; `tests/parity/test_ggml_prompt.py` compares the two over `model/tests/fixtures/prompt_requests.json`.

### Pronunciation lexicon

`src/zonos3-phonemes.cpp` ports `zonos3/tts/phonemes.py` (`/v2/phonemize`, `zonos3-cli --phonemize <lexicon dir> --language L [--no-guess] [--symbols J] --text T`). It reads the training lexicon LMDB through vendored LMDB 0.9.36 (`vendor/lmdb`, read-only, no lock file): a `meta` JSON record (`version` 1, `languages: {code: {mode}}`) and one named DB per language keyed by `normalize_headword` (NFC, `’`→`'`, `str.casefold`; tables in `src/unicode-data.h`, Unicode 15.0 like Python 3.12). Segmentation (letter runs with marks and inner apostrophes; Korean eojeol; Japanese/Chinese forward max-match of 2..8 characters, okurigana skipping), statuses and the espeak-ng harmonization rules are the Python ones. espeak-ng is optional and only `dlopen`ed from `ZONOS3_ESPEAK_NG_LIBRARY` (data dir `ZONOS3_ESPEAK_NG_DATA`, e.g. the paths of `espeakng_loader`); without it guessing is off. Parity: `servers/ggml/tests/parity/test_ggml_phonemize.py`.

### Mixers and MoE

Layers 4 and 8 (one-based) use global GQA: 16 query heads, four KV heads, width 32, QK-RMSNorm, absolute interleaved RoPE, query temperature and per-head sigmoid output gates. Other layers use the original Gated DeltaNet mixer:

- Four 128-wide heads, inner width 512.
- Bias-free `in_proj` with 2056 outputs: Q/K/V, z gate, and four decay/update logits each.
- Depthwise causal width-four convolution followed by SiLU on Q/K/V.
- Q/K normalization divides by `sqrt(sum(x*x)+1e-6)`. This differs from `ggml_l2_norm`'s epsilon semantics; the native graph expresses it through scaled RMSNorm.
- Per-head log decay is `-exp(A_log)*softplus(a+dt_bias)` and update weight is `sigmoid(b)`.
- FP32 recurrent state is decayed, corrected using the delta rule, and read with Q scaled by `1/sqrt(128)`.
- Per-head RMSNorm (`eps=1e-5`), affine `out_weight`, SiLU(z) gating, then the output projection.

Layer 0 has a dense SwiGLU FFN with hidden width 2048. Layers 1–10 have top-2/64-expert latent MoE: latent width 256, expert hidden width 1536, and a bias-free 1024-to-64 router after weighted RMSNorm. Selection uses `topk(raw_logits-balancing_biases)`; weights are softmax over the selected raw logits. The config's `moe_router_dim=128` does not apply to this linear router.

**Checkpoint packing:** the release stores experts as `experts.gate_up_proj` (64×3072×256) built as `cat(w13[:, 0::2], w13[:, 1::2])`: rows 0..1535 are the gate and 1536..3071 the up projection, so conversion splits it into two contiguous halves for `ffn_gate_exps.weight`/`ffn_up_exps.weight` (the earlier alternating-row layout no longer applies). `experts.down_proj` becomes the down stack; the release has no `router.ent_denom`.

### Full AttnRes

The source bank contains the normalized embedding and every individual mixer/FFN output. Before each sublayer, a learned query scores weightless RMS-normalized sources; a softmax across depth mixes the original sources. There are 23 query rows, including the final mix before output normalization.

There is **no additive residual stream** and no block-sum approximation. Full AttnRes is not implemented by reducing a block-size parameter. The bank is local to the current token/chunk forward and adds no cross-token cache. Runtime depth skipping omits both skipped-layer outputs from that bank.

### Inference state

`zonos3_batch_ctx` stores KV bands only for the two global layers. Head width is zero-padded to 64 for flash-attention kernels. CPU and entirely-F32 weight profiles use F32 KV; other GPU profiles use F16 KV. The profile is derived from every loaded tensor, not the filename. F32-cache decode uses FP32 matrix attention because CUDA FlashAttention can narrow F32 KV internally. Each band includes a private idle-write row outside the attended context.

Every GDN layer has an independent FP32 `[4,128,128]` recurrent matrix and `[1536,3]` oldest-to-newest raw convolution history per slot. Initial prefill resets the slot, then processes chunks of up to 256 positions. `zonos3_batch_slot_extend` continues an existing slot at an absolute position. Decode graphs mask recurrence/history updates for inactive columns and commit new state only after all reads of the old state. Slot admission/reuse resets both GDN states and global KV.

The same batch runtime serves CLI synthesis and the continuous server scheduler. `zonos3_batch_step` reports compute failure instead of returning stale logits. The recompute path rebuilds the complete sequence for validation. Generation rejects prompt-plus-output lengths outside the 32768-position context.

### Alignment probe, accents and emotion

Optional per-slot features of the batch runtime (`zonos3.h`), off by default:

- **Alignment probe** (LACI / span-follow; sgl `align_probe._readout/_masked`, `zonos3.tts.probe.readout`). `zonos3_slot::probe` names heads of the global layers, the prompt's BOS..EOS rows and optional per-head ACI windows; the batch must be created with `zonos3_batch_features::probe_max_text` ≥ the span. At each global layer a probed decode step gathers the slot's text keys from the cache (`get_rows`), takes each head's softmax over them (window-restricted where a window is set) and adds the head-weighted sum to a `[probe_max_text, width]` output; the host reduces it to `zonos3_slot::readout` (argmax, centroid). Windowed heads also recompute their attention output over all of the slot's keys with the text keys outside the window blocked (non-text keys never are), for the windowed slots only (gathered rows, scattered back with `set_rows`); other heads and slots keep the FlashAttention / F32 result bit for bit. Window masks are built in-graph (`step` comparisons, 0 / −1e30 addends, which give the same softmax as −inf). Both paths are explicit F32 attention on every KV profile.
- **Accent adapters** (`zonos3-accent.h`, `accents.gguf`). Attached as `zonos3_model::accents`; a prompt (`zonos3_conditioning::accent`) or slot (`zonos3_slot::accent`) picks an adapter and a strength. The bank stacks all adapters' factors per module (`D_all [in, n·K]`, `U_all [n·K, out]`), so every adapted linear adds `U_all·((D_all·x) ⊙ mask)` with a per-row `[n·K, rows]` mask holding the strength on the row's adapter columns (rows without an accent add an exact zero). Expert factors use the same mask through `mul_mat_id` on the gate/up pre-activations and the expert output; router shifts arrive as a per-row `[E, rows]` addend to the balancing biases.
- **Emotion / accent directions** (`zonos3-emotion.h`, `emotion.gguf`, `accent-directions.gguf`). Host ports of `apply_emotion` (raw / lda spaces move the speaker embedding), `emotion_hidden_delta` (proj deltas go to `zonos3_conditioning::speaker_delta`, added to the projected speaker row; attnres_source deltas go into the slot's steering layout via `zonos3_steer_add_delta`, so the batch's steer sources must include the set's `source_index`) and `EmotionCalibration`.
- **Server wiring** (`src/server/generate.cpp`, sgl `_apply_request_emotion` / `_apply_steering_accent` / `_layout_steering_delta`). `load_emotion_accents` loads the three companions and sets `ServerState::steer_sources` = the emotion and accent-direction sources plus the steering bundle's hidden sources (the bundle adopts that layout, and so does the batch). A request's emotion and accent-direction hidden deltas join the controls' constant delta in that layout; an adapter accent sets `GenReq::accent` (prompt cond and slot, both CFG branches). Emotion CFG (an emotion / steering delta with a speaker embedding and `emotion_cfg_scale` ≠ 1) runs a twin on the same prompt without the proj delta, which the scheduler leaves unsteered (`GenReq::twin_steered`); `zonos3-cli --request --build-prompt` writes it as `ids.twin.npy`.

Feature decode graphs are separate ladders (keyed by probe / masked-row count / accent), built on first use; the plain ladder is untouched. `graph-features --selftest` checks the invariants; `tests/parity/test_ggml_graph.py` checks the probe, accent and emotion math against `model/`.

### Sampling and waveform decode

The sampler preserves pad masking, repetition penalty 2.4 over the last 96 generated frames of codebooks 0–5 (prefill rows are not part of the history), temperature 0.75, top-k 80, top-p 0.69, min-p and multinomial sampling. Cold-start forcing uses the stream row `prefill + step`: codebook `c > row` is forced to padding, matching the training layout (with 3 silence rows the first generated row pads codebooks 4..9). The first EOA establishes the aligned end frame, in stream units that include the prefill rows, and starts an eleven-row countdown. Decode consumes `min(eos_frame, rows-9)` ordinary frames. `--max` counts generated rows only.

**Speaker CFG.** With `cfg_scale != 1` a twin prompt (no speaker row/reference prefix, all condition buckets unknown, same text and silence rows) decodes in lockstep in a second slot and always receives the conditional request's sampled frame. `zonos3_cfg_mix` forms `uncond + s·(cond − uncond)` in FP32 on the post-softcap logits before any sampler transform. CFG without a speaker prompt is an error, and `s == 1` builds no twin.

**Waveform decoder.** The `mossdual_istft` decoder uses semantic lookup/causal ConvNeXt reconstruction plus nine residual lookups, five MOSS decoder stages with causal windows, GELU FFNs and per-channel layer scales (15 → 240 Hz, 16 head frames per code frame). The head is eight causal ConvNeXt blocks (768/2048), a LayerNorm and two linear branches that predict log-magnitude and phase for n_fft 736 and 2208 (hop 184). On the host, per branch: `mag = exp(min(logmag, ln 100))`, an inverse real FFT (pocketfft plans built once at load), the periodic Hann window, overlap-add normalized by the window envelope (exactly `torch.istft(center=True)`), then the non-causal 2→1 FIR mixer (127 taps, 63 samples of lookahead), exact 44160→44100 resampling, and the output gain `10^(−headroom_db/20)` (default 3 dB, applied by the CLI/server/`dac-cli`, not by `dac_decode`). Streaming retains MOSS KV windows, convolution histories, the overlap-add/mixer state and resampler lookahead; `dac_stream_flush` emits the held-back tail so each code frame contributes 2940 samples at 44.1 kHz. The silence-prefill rows are decoded as the first rows of the stream and their samples (`prefill_frames · 2940`) are dropped. `full_mr` and `keep2_plain` share the schema and differ only in layer counts.

## Native reference-audio encoding

`dac-encoder.gguf` is separate from the decoder so ordinary generation does not load the roughly billion-parameter acoustic encoder. Raw-reference encoding requires both the acoustic/semantic modules and `ssl-teacher.gguf`; neither the speaker encoder nor folded decoder tables can replace them. The default speaker-embedding path does not use these models.

The matched recipe is in the codec package's `source/hybrid_adapter.py`, `hybrid10_dual_encoding/model.py`, and `s1dac/losses.py`:

1. Decode external mono audio on the 44100-Hz grid.
2. Resample the entire file to 44160 Hz with the frozen sinc kernel before chunking; pad both grids to their 2940/2944-sample hops.
3. Run independent 20-second padded lanes with ten seconds of real left context on continuation lanes. Discard context tokens from the returned reference codes.
4. Run the acoustic S1/Snake/down-patch/MOSS encoder on the internal grid.
5. Independently resample external-grid lane audio to 16 kHz, compute SeamlessM4T fbank/normalization/stride-two features, and run the frozen sixteen-layer Wav2Vec2-BERT teacher. The teacher is bidirectional within the lane; causal decoder streaming is not interchangeable with this recipe.
6. Pool SSL frames with the frozen noncausal box-window averaging, apply the causal semantic ConvNeXt encoder, and select semantic codebook Q1.
7. Reconstruct the semantic latent, subtract it from the acoustic latent, and run nine residual RVQs.

RVQ uses original input/output projections and original eight-dimensional codebook vectors, normalized nearest-code distances, FP32 arithmetic and first-index ties. Acoustic/semantic modules follow production BF16 boundaries. The teacher reproduces CUDA BF16 autocast: LayerNorm, softmax and final layer states are F32; matrix operands and designated outputs narrow to BF16. Linear bias precedes its final BF16 cast, while biased cuDNN Conv1d has both an unbiased-output cast and a post-bias cast. Diagnostic teacher arithmetic can explicitly select F32. Output is ordinary C-order `[ceil(samples/2940),10]` integer codes. Cached codes can be used directly as a prompt prefix without encoder models.

`src/audio-decode.cpp` owns the shared cross-platform ffmpeg process handling. Encoding models are single-threaded allocator owners; server encoding is serialized and cached.

Frontend mean/variance use NumPy's pairwise F32 reduction ordering. Cached teacher relative indices are retained as graph outputs: marking a constant only as an input does not prevent the allocator reclaiming it after its last consumer. This keeps repeated references and graph-size transitions safe without disabling CUDA graphs or refilling unchanged indices.

## Speaker encoder (ownmect_s2 MECT-B2)

`spk-encoder.gguf` (`general.architecture=mect_spk`, all F32) holds the 11.3M-parameter MECT-B2 speaker model (`speaker_encoders/ownmect_s2/ckpt_0037.pth`, checked against the registry's SHA-256) plus its host frontend tables. `src/spk-encoder.cpp` runs: resample the 44.1 kHz decode to 16 kHz (shared polyphase code in `src/resample-16k.h`), crop to 60 s, Kaldi fbank on the host (×32768, 80 mels, 25/10 ms, Hamming, preemphasis 0.97, no dither, pocketfft real FFT of size 512, log floor `1.19e-7`), per-bin cepstral mean subtraction, then a ggml graph: ResNet-style stem, four stages of 2/4/5/3 MECT blocks (grouped 3×3 CNN stack, flatten, rproj, positional depthwise conv of width 113, post-norm 4-head attention, dense-softmax token MoE FFN with NewGELU), softmax-weighted stage fusion, ASTP pooling, batch-norm and a 192-d linear. Graph `MUL_MAT` nodes use F32 precision. The result is the unnormalized 192-d vector the LM's `speaker_projection` consumes. The Mini LM's speaker row and the raw-reference prefix are mutually exclusive.

## Convert the release

```bash
R=/path/to/ZONOS3_Mini_release_ckpts   # revision 4d641a9fedd8c5544fca6e6f17ebb5c351c1a510
T="uv run --package zonos3-tools python ../../tools/convert_gguf.py"
mkdir -p models
$T lm "$R/lm/mect_60k" --outtype f16 -o models/zonos3-f16.gguf
$T codec-decoder "$R/autoencoder/generator.pt" --head "$R/autoencoder/heads/full_mr" -o models/dac.gguf
$T codec-decoder "$R/autoencoder/generator.pt" --head "$R/autoencoder/heads/keep2_plain" -o models/dac-keep2.gguf
$T speaker-encoder "$R/speaker_encoders/ownmect_s2/ckpt_0037.pth" -o models/spk-encoder.gguf
$T codec-encoder "$R/autoencoder/generator.pt" --outtype f32 -o models/dac-encoder.gguf
$T ssl-teacher "$R/autoencoder/teacher" --outtype f32 -o models/ssl-teacher.gguf
$T steer ../../steering/studio_bundle_mect60k.json -o models/steer.gguf
$T directions ../../steering/emotion_directions_mect60k_axes -o models/emotion.gguf
$T directions ../../steering/accent_directions_mect60k -o models/accent-directions.gguf
$T accents ACCENT_ADAPTER_DIR -o models/accents.gguf          # when accent adapters (<accent>.pt) are available
```

`tools/convert_gguf.py` has one subcommand per GGUF; each loads its weights into the PyTorch module that runs them (`zonos3.reference.lm.Zonos3LM`, `ISTFTCodec`, `MossCodec`, transformers' `Wav2Vec2BertModel`, `MECTFbankSpeakerEmbedding`) with a strict load, so a missing or unknown checkpoint tensor is an error. `--help` on a subcommand documents its GGUF layout.

Backbone conversion (`lm`; `--outtype f32|f16|q8_0`, default f16; `--silence-frame` overrides `<release>/registries/silence_frame_<audio_codec>.json`) reads `params.json`, the flat `model.pth` (177 tensors for the release) and the tokenizer named by `params.text_tokenizer_spec.path`; `conversion.json` (`minisgl-zonos2`, bf16) and `config.yaml` are validated when present. It validates the architecture (`Zonos3LMConfig.from_params`, also against `config.yaml`), the tokenizer SHA-256, condition metadata and the silence frame, and maps parameters through the ordered `LM_NAMES` table (parameter regex → GGUF names/transform), which also fixes the tensor order. Experts arrive as `gate_up_proj` (64×3072×256: rows 0..1535 gate, 1536..3071 up) and `down_proj`; the speaker projection and the three 11-row condition embeddings become `speaker_proj.*` and `cond_embd.{0,1,2}.weight`. The GGUF additionally stores `zonos3.speaker.embedding_length`, `zonos3.condition.{count,bucket_count,names,units,edges.*}`, `zonos3.silence_frame`, `zonos3.text_tokenizer_spec` (the JSON of `params.text_tokenizer_spec`, including phoneme tokens) and, for checkpoints with a span-end token, `zonos3.span_end_id`; derived checkpoints with extra text rows (edit markers, phoneme tokens: `tools/add_text_extra_rows.py`) keep `zonos3.text_vocab` and carry the rows in `text_embd.weight`. BF16-to-F16 export is checked for finite values/overflow; it is not described as lossless.

**QAT backbone (default).** A directory with `bake.json` (`lm/mect_60k_q4q8_qat`, step 3462) is the release LM fine-tuned with Q4_0 routed experts and Q8_0 GDN `in_proj` / attention `wq`,`wkv`,`wo`,`gater`. Its BF16 weights are the values the QAT forward multiplies by, `bf16(q·d)` with an fp16 block scale `d`; `convert_gguf.py lm` (`--outtype q4_0-exp-q8_0-mix`, the only valid and default type for such a directory) maps `bake.json` to GGUF tensors (20 Q4_0 and 26 Q8_0 source tensors → 30 Q4_0 and 28 Q8_0 GGUF tensors; everything else F16, 1-D and `gdn_conv` F32) and recovers each block's exact `(q, d)` by searching the fp16 scales around the BF16-derived one until `bf16(q·d)` reproduces the block bit-for-bit. Re-quantizing the BF16 values naively would shift scales by up to ~0.4%. The conversion fails if any block has no exact reproduction (none does for the shipped checkpoint). `cond_embd.*` and `speaker_proj.weight` stay F16.

`codec-decoder` (`--head` = `full_mr` or `keep2_plain`; the directory must hold one `config_*.json` and `step_0050000.pt`) builds `ISTFTCodec`, loads `quantizer.*`, `semantic_head.*` and `to_external.kernel` from `generator.pt` and the whole transformer trunk and ConvNeXt/iSTFT head from the head checkpoint's `trainable` dict (880 tensors for `full_mr`, 180 for `keep2_plain`); it folds RVQ codebooks into decode-only tables, stores torch's periodic Hann windows for n = 736 and 2208, and records `headroom_db = 3.0`, `head.variant` and `head.step`. `codec-encoder` exports acoustic/semantic modules, original RVQ projections/codebooks and the internal-grid resampling kernel. `ssl-teacher` consumes the actual frozen layer-16 checkpoint, excludes `masked_spec_embed`, and exports per-frame inference tensors plus exact frontend tables. `speaker-encoder` is described under [speaker encoder](#speaker-encoder-ownmect_s2-mect-b2). `steer` packs a steering bundle (`zonos3_steer`, including the bundle's `rate_extension` block as `zonos3_steer.rate_extension`); `directions` packs one emotion/accent direction set (`zonos3_directions`) and `accents` the accent adapters of a directory (`zonos3_accents`); the server finds `emotion.gguf`, `accent-directions.gguf` and `accents.gguf` beside the backbone. The tokenizer remains embedded in the backbone.

## Reference comparison

Run the oracles with `uv run --package zonos3-tools` from `servers/ggml`: `tools/dump_golden.py` imports only the `zonos3` model package (`zonos3.reference.lm.Zonos3LM`, `zonos3.codec`, `zonos3.models.mect_speaker`) and transformers.

```bash
D="uv run --package zonos3-tools python ../../tools/dump_golden.py"
C="uv run --package zonos3-tools python ../../tools/compare_golden.py"
```

Backbone (speaker-embedding prompt with conditions and silence prefill; `dump_golden.py lm` loads the checkpoint through `convert_gguf.load_lm` and captures `Zonos3LM.forward`):

```bash
LM="$R/lm/mect_60k"
$T lm "$LM" --outtype f32 -o out/zonos3-f32.gguf
$D lm "$LM" -o out/golden-emb --text 'The quick brown fox jumps.' \
  --speaker-embedding tests/fixtures/speaker-embedding.npy --condition-buckets 3,5,10 \
  --silence-prefill 3 --mode full
./build/zonos3-cli out/zonos3-f32.gguf --validate out/golden-emb/input_ids.npy out/cpp-emb --cpu
$C out/golden-emb out/cpp-emb
```

`dump_golden.py lm` also accepts `--reference-codes` (raw-reference prompt), omits both for an unconditioned prompt, and writes `input_ids.npy` plus `input_ids.cond.json`; `--mode chunked --chunk-size 2` splits the replaced rows across chunks and `--mode decode` exercises continuation. The oracle captures complete embeddings (`emb_sum` before and `emb_cond` after row replacement), residual mixtures, per-layer inputs/outputs, router logits, GDN normalized vectors/readouts/states/histories and logits. Native `raw-reference-state` tests compare full prefill with continued chunks (splits on and around the conditioning rows) and check slot isolation/reuse. GPU F16 KV effects are measured separately from F32 full-forward parity.

`tools/compare_golden.py` requires every manifest capture, rejects missing/empty/nonfinite tensors, compares all floating elements, and requires exact all-row logits argmax. Integer tensors and codec-code captures require exact values regardless of floating tolerance. It does not silently select only shared files or the final prompt row. `--cos` sets the whole-tensor cosine threshold (default 0.99).

`tests/parity/test_ggml_golden.py` automates the backbone comparison on the release F16 GGUF (`dump_golden.py lm --weight-type f16` vs `--validate`, `--cos 0.9999 --atol 0.05 --rtol 0.01`: the CPU F16 matmul rounds activations to F16) and checks the LACI probe readout of a greedy `--generate` against the reference teacher-forced on the generated rows (identical argmax per step).

Waveform decoder (the oracle loads `zonos3.codec.istft_head_a_20261002.model.ISTFTCodec` like the sgl server, fp32 on CPU; it saves pre-gain audio including the prefill frames):

```bash
$D codec-decoder --generator "$R/autoencoder/generator.pt" \
  --head-dir "$R/autoencoder/heads/full_mr" out/codes.npy -o out/dac-golden
./build/dac-cli models/dac-f32.gguf out/codes.npy out/dac-cpp.npy --skip-frames 0 --headroom-db 0 --cpu
$D codec-decoder --generator "$R/autoencoder/generator.pt" \
  --head-dir "$R/autoencoder/heads/full_mr" out/codes.npy -o out/dac-golden --compare out/dac-cpp.npy
./build/dac-cli --seam models/dac.gguf out/codes.npy --block 3 --cpu
```

Repeat with `heads/keep2_plain` and a `dac-keep2.gguf`.

Speaker encoder (the oracle is `zonos3.models.mect_speaker.MECTFbankSpeakerEmbedding`):

```bash
ffmpeg -nostdin -v error -i ../../default_voices/AmericanFemale.mp3 -ac 1 -ar 44100 -c:a pcm_f32le out/af.wav
$D speaker-encoder "$R/speaker_encoders/ownmect_s2/ckpt_0037.pth" out/af.wav -o out/spk-golden
./build/spk-encoder-cli models/spk-encoder.gguf --audio out/af.wav out/af-emb.npy --dump-dir out/spk-cpp --cpu
$C out/spk-golden out/spk-cpp --cos 0.9999
```

The embedding is not L2-normalized; the AmericanFemale reference has norm 4.21.

Teacher diagnostics accept identical features/masks to separate graph differences from frontend differences:

```bash
$D ssl-teacher --teacher "$R/autoencoder/teacher" \
  --audio ../../default_voices/AmericanFemale.mp3 --dtype f32 -o out/ssl-python-f32
./build/ssl-encoder-cli models/ssl-teacher.gguf \
  --features out/ssl-python-f32/features.npy \
  --mask out/ssl-python-f32/attention_mask.npy \
  --dtype f32 --cpu --out out/ssl-native-f32
```

Production BF16 comparison requires `--dtype bf16 --device cuda` for the Python oracle and an allocated GPU. It runs the actual CUDA autocast context, not a BF16 model without autocast. `--trace-first-layer` exposes operation boundaries; `--diagnose-convolutions` records counterfactuals without replacing actual oracle outputs. `--pcm44100` compares feature extraction from saved external-grid PCM; `--pad-seconds 20` reproduces a padded encoding lane.

The codec oracle runs the **actual frozen adapter and IndependentBatcher** (`zonos3.codec.codec_adapter.MossAutoencoder`, CUDA), not a replacement model; `--teacher` defaults to the generator's sibling `teacher/`:

```bash
$D codec-encoder --generator "$R/autoencoder/generator.pt" --audio ../../default_voices/AmericanFemale.mp3 \
  --samples 2941 --batch-size 1 -o out/encoder-python
./build/dac-encoder-cli models/dac-encoder.gguf models/ssl-teacher.gguf \
  --audio out/encoder-python/audio.npy out/reference.npy \
  --gpu --dump out/encoder-native
$C out/encoder-python out/encoder-native
```

Qualify short inputs, hop boundaries, 20-second lane boundaries, continuation context, and 60-second references. Compare every frame/codebook exactly in `codes.npy` and inspect intermediate numerical differences before attributing code changes to RVQ boundary decisions.

No pass/fail results for the Mini conversion are recorded in this document; earlier 375k observations were removed because they do not describe this checkpoint.

**Fidelity limits:** strict BF16 teacher/codec floating comparisons do not all pass. With identical operands, native PW1 equals PyTorch BF16 Linear and BF16 GEMM/F32-output counterfactuals exactly across 2,045,952 values, but differs from actual cuDNN Conv1d at 511 BF16 rounding decisions. The source MOSS backend is BF16 memory-efficient SDPA; native matrix decomposition does not reproduce every fused-kernel reduction decision. Counterfactuals explain differences, never replace goldens. Short-reference code equality is not a universal or whole padded-lane parity claim. Matching those bits additionally requires the actual CUDA/cuDNN execution arithmetic and platform/version/layout/workspace selection. The speaker encoder, MR-iSTFT decoder and backbone run entirely in F32 graphs and are compared by cosine/absolute tolerance against their PyTorch goldens.

## Quantization

`quantize-cli` dispatches only recognized architectures: `zonos3`, `mossdual_istft`, `mossdual_encoder`, and `w2vbert_ssl`. One-dimensional parameters and convolution/normalization tensors remain F32 where kernels require it. Backbone output/embedding/router/AttnRes tensors are protected, with GDN-sensitive parameters kept high precision; `speaker_proj.weight` stays high precision and `cond_embd.*` (gathered with `ggml_get_rows`) are written F16. Decoder `head.*` matrices are quantizable like the trunk except `head.mixer.weight`, the windows and the resampling kernel. `--experts-only`, `--spine-f16`, `--down-tier-up`, and importance-matrix options apply to the backbone only.

Both latent width 256 and expert hidden width 1536 are K-block-aligned. Importance matrices capture actual expert inputs; they must be collected from Mini traces that include the speaker/condition rows and silence prefill (`--dump-ids` writes the `.cond.json` sidecar that `zonos3-perplexity` reads). `scripts/gen_trace_corpus.sh` encodes references with the native encoder, then reuses the ordinary codes during generation.

Corpus generation replaces `out/gencal` or `out/kldcorp` and its corresponding manifest; preserve any previous run before invoking the script.

```bash
BUILD=build-cuda MODEL=models/zonos3-f16.gguf DAC=models/dac.gguf \
DAC_ENCODER=models/dac-encoder.gguf SSL_TEACHER=models/ssl-teacher.gguf \
  bash scripts/gen_trace_corpus.sh calib
./build/zonos3-perplexity models/zonos3-f16.gguf \
  --manifest out/calib.manifest --imatrix-out out/imatrix.bin --prune-stats out/prune.bin
./build/quantize-cli models/zonos3-f16.gguf models/zonos3-q4_k-exp-imat.gguf \
  q4_k --experts-only --spine-f16 --imatrix out/imatrix.bin
```

KLD bases use magic `Z3KLDIV4` (version 4): after the ids blocks each sequence stores its speaker row, condition row, condition buckets and optional speaker vector, so conditioned rows are replayed exactly. Older bases, calibration corpora, importance matrices and quality tables (including everything measured on the 375k checkpoint) are not transferable. Pruning uses new metadata/magics and clamps top-k to the retained expert count. Runtime masks and rewritten expert stacks are checked for equivalent behavior.

The default QAT GGUF was checked against a Python golden built from the baked BF16 weights (embedding prompt: logits cos 0.99995 on GPU and CPU; argmax 97.4% on GPU and 96.3% on CPU, with every mismatch on a near-tied codebook logit: top-1/top-2 margin of the golden at most 0.32) and against the F16 backbone by Whisper-large-v3 WER on eight sentences (3.5% for both; the four errors are `$42,000` normalization) and by the loudness-condition response (seed 1: −16 LUFS bucket −13.7 vs −15.2 LUFS for F16; −29.5 bucket −32.9 vs −40.8; the −29.5 bucket varies by about 10 LU across seeds 1–4 for F16 itself, −30.2 to −40.8, and −30.6 to −34.9 for QAT, and −16 is louder than −29.5 in every run). Post-training `quantize-cli` profiles are not qualified; treat F16 as the reference and re-qualify any before recommending it. The earlier 375k tables were removed because they do not describe this checkpoint.

## Server, UI and deployment

`zonos3-server` (`src/server/`) serves the v2 API of `api/openapi.yaml`. `main.cpp` parses flags, loads the backbone and waveform decoder (the speaker encoder and the reference-encoding companions load lazily) and registers the routes, the studio serving rule last. `http.cpp` holds the `{"detail"}` errors, CORS (every origin; `X-Audio-*`, `X-Control-Span-Follow` and `X-Seed` exposed), the `ZONOS3_ACCESS_TOKEN` gate (a pre-routing check: bearer, the `zonos3_access` cookie or `GET /?access_token=` with a 303; `Referrer-Policy` and `Cache-Control` on every response), the `X-Keepalive` wrapper (the handler runs on its own thread while a chunked 200 sends a space every 10 s, then the JSON body or `{detail, status_code}`) and `X-TTS-Session-ID`. `schema.cpp` validates request bodies with the vendored `pboettch/json-schema-validator` against `#/components/schemas/<Name>` of `api/openapi.json`, which CMake embeds as `generated/zonos3-openapi.h` (re-configured when the spec changes); violations become the 422 `{"detail": [{"loc": ["body", ...], "msg", "type"}]}`. `capabilities.cpp`, `speakers.cpp`, `generate.cpp` and `projects.cpp` (a line-by-line port of `zonos3/server/projects.py`, same on-disk layout and `flock`) answer their routes; `phonemize.cpp`, `edit.cpp` and `dub.cpp` hold the routes and capability documents of the features served by their later ports. `workers.cpp` runs JSON-lines helper processes (one request per stdin line, one response per stdout line); `textnorm.cpp` keeps one `model/src/zonos3/workers/textnorm.py` process (`PYTHONPATH=--python-src`, `VIRTUAL_ENV` removed), serialized, restarted once when it exits.

Voices are default voices (`--tts-default-voices-dir`: audio files, `.npy`/`.npz` speaker embeddings; ids `default_<sha1[:16]>` of the relative path, as sgl) and session voices (`POST /v2/speakers`, ids `spk_<12 hex>`). Audio voices keep their 44.1 kHz PCM and compute the embedding (spk-encoder) or reference codes (codec encoder + SSL teacher, serialized under one mutex) lazily per mode, caching the result on the voice. A default audio voice with a sibling `<stem>.codes.npy` uses those shipped codes for reference mode instead of encoding (falling back to the encoder if the file fails to load); the shipped codes come from `dac-encoder-cli` with the F32 `dac-encoder.gguf`/`ssl-teacher.gguf` and must be regenerated whenever a voice clip or the encoder changes. The reference encoder and SSL teacher are freed after `--reference-idle-unload N` idle seconds. Speaker resolution, condition rows, steering controls, `max_tokens`, the seed (drawn in `0..2^31-1` when null) and every 400 message follow sgl's `_tts_generate_msg` in its order.

Uploaded audio goes through process-scoped temporary files with exclusive creation, so concurrent server processes cannot overwrite each other's inputs. Parallel evaluation jobs should still set a separate `TMPDIR` per job for artifact isolation.

One worker owns backbone batching (`scheduler.cpp`); each job is pinned to a DAC lane. A job with `speaker_cfg_scale != 1` takes two slots (primary + twin; FIFO admission waits until both are free): per tick the worker mixes the two logit rows with `zonos3_cfg_mix`, samples the primary, copies the sampled frame to the twin and steps both; both are evicted together. The server therefore needs `--batch >= 2` for CFG. `stream_decode=prefix` (the default for streaming responses, `--stream-decode-default`) hands the lane a stream-decode window whenever `ZONOS3_PREFIX_STREAM_CHUNK_FRAMES` (15) new frames are complete beyond `ZONOS3_PREFIX_STREAM_HOLD_BACK_FRAMES` (3) held-back frames; at EOS the exact whole decode (`dac_decode`, the same call as a buffered request) supplies every sample the stream has not emitted. `whole` and buffered requests decode once at EOS. `decode_context` decodes the reference's last `decode_context_frames` ahead of the stream (their delayed rows fill the stream's first, padded rows). Lanes drop the decode-context and silence-prefill samples and apply the headroom gain before the fade. Cancellation releases the slot(s); later admission resets all inference state. Buffered `format: "wav"` is 16-bit PCM; pcm is float32 44.1 kHz.

LACI (`laci: true`, `--laci-heads LANG=PATH`; `zonos3-laci.h`) and reading-position span follow (`control_span_follow: "reading"`; `zonos3-span-follow.h`) share the alignment probe. `make_job` (generate.cpp) calls `attach_probe`, which locates the prompt's text span (at most `probe_max_text` = min(2048, context capacity) rows, else 400), builds the `SpanFollower` from the steering spans' token ranges and the `LaciSession` (prefill rows = the silence-prefill rows, the checkpoint's speaking-rate row as `RateBuckets`). The batch is created with that probe capacity; after every decode step the slot's readout goes to the session (`on_step`, which sets the ACI windows and the stop) or the follower before the next row is sampled. Under LACI the session owns the stream: each sampled row is appended and only `drain()`ed rows reach the job's codes (prefix streaming included); when the attempt ends `finish_attempt` either yields a `Continuation`, and the worker resets the same slot and re-prefills the prompt (without its silence rows) plus the kept rows under the seed `seed + 1000·attempt` (mod 2^32), the continuation's condition rows / steering delta and the kept rows as repetition history, or makes the result final (`drain(true)`, EOS = the session's end row, `record()` kept in `ReqJob::laci_record`). The steering delta of a frame counts the stream's generated rows (`history_rows + step`); with a follower its decisions replace the character-rate schedule and are shared with a speaker / tag CFG twin. Dub takes use `plan_laci_take` / `wait_take` (generate.h): a GenerateRequest with a `DurationTarget` and optionally an extended speaking-rate scale whose steered buckets add the bundle's tempo direction to the take's hidden delta.

Dubbing (`dub.cpp`) ports `zonos3/dub/runtime.py` over `dub-plan.h` (`zonos3/dub/plan.py`). Uploads are decoded with ffmpeg (sound files at their own rate with channels averaged, videos' first audio track at 44.1 kHz mono) and kept with the generated dubs in two LRU stores of 8; lip tracks live in a temporary directory shared by the reference and the dubs that use it. The alignment worker (`dub_align.py`: ASR, forced aligner, SimAlign, AV-sync) is one `JsonLinesWorker` started on first use (a `ping` narrows the reference languages) and again after it died; an `invalid` answer is a 400, any other worker failure a 503. Takes are planned before any is queued and at most `2 × --batch` are in flight across requests; a source voice's prompt clip (`voice_prompt_span`) is encoded to reference codes once per dub. With `steer.gguf`'s `rate_extension` (matched to the checkpoint's row edges) each target language gets its extended `RateScale` (source voices only on the bundle's `reference_sides`). Lip sync writes the mix for `lips_sync`, which warps it with WSOLA; the warp is kept only when it lowers the SyncNet distance. `mux` tries `-c:v copy`, then libx264.

Decoder graph GEMMs explicitly request F32 arithmetic; otherwise buffered and streamed chunk widths can select different F16/TF32 activation paths. The iSTFT overlap-add, the mixer lookahead and the resampler tail are held back and flushed identically in buffered and streamed decoding; `dac-cli --seam` (CTest `dac-seam`) compares them. This is bounded numerical agreement, not bit identity.

Launchers and native desktop packaging use only `zonos3` binary/file names and `ZONOS3_*` environment variables. The launcher and the app's setup page/config (`spk_encoder`) pass `--spk-encoder` when a speaker encoder is present. Desktop config migration preserves valid settings but does not pretend old backbone weights are compatible. Linux desktop compilation additionally needs GTK4, WebKitGTK-6.0, libadwaita and json-glib; macOS/Windows verification requires those hosts/toolchains.

## Smoke tests

```bash
ctest --test-dir build --output-on-failure
bash tests/readme-commands.sh build/zonos3-cli build/zonos3-server
bash tests/launcher-smoke.sh build/zonos3-server
```

The heavyweight teacher regression is optional because its frozen weights are not committed:

```bash
cmake -S . -B build -DZONOS3_TEST_SSL_TEACHER="$PWD/models/ssl-teacher.gguf"
cmake --build build --target ssl-reference-state
./build/ssl-reference-state models/ssl-teacher.gguf
./build/ssl-reference-state models/ssl-teacher.gguf --gpu
```

It checks that new inputs affect per-frame outputs, A/B/A replay is exact, and changing graph size then restoring it does not corrupt cached relative indices.

Fixtures are random-weight interface tests, not speech-quality evidence. `dummy-zonos3.gguf` contains three GDN layers and one global layer, Full AttnRes, the pinned tokenizer and its `text_tokenizer_spec`, the real condition names/edges/units, the real silence frame and random `speaker_proj`/`cond_embd` tensors; `speaker-embedding.npy` is a fixed float32 `[192]` vector. `dummy-dac.gguf` (iSTFT head with the real hop/FFT/FIR geometry, tiny trunk) with `dummy-dac-codes.npy` backs the `dac-seam` CTest. Regenerate with `uv run --package zonos3-tools python scripts/make-dummy-model.py -o tests/fixtures/dummy-zonos3.gguf` (`--edit-markers N`, `--span-end` and `--phonemes SYMBOLS` build edit / phoneme-token variants) and `scripts/make-dummy-dac.py`. Permanent regressions cover prompt layouts and errors, bucket mapping, conditioning-row replacement across chunk boundaries, the CFG twin, pronunciation groups and the tokenizer-spec checks, chunk continuation, isolation/reset, 64-expert pruning policy, KLD rejection, calibration and pruning equivalence, and streamed-versus-buffered decoding. Real checkpoint comparisons and actual CLI/server/browser generation remain required for model qualification.
