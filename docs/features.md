# Features

## TTS Serving

ZONOS3 serves ZONOS3 TTS checkpoints from the Hugging Face Hub (e.g. `hf://Zyphra/ZONOS3-Mini/lm/zonos3_mini`, the default) or a local path. The HTTP API ([api.md](api.md), schemas in [servers/api/openapi.yaml](../servers/api/openapi.yaml)) exposes:

- `POST /v2/generate` for the native TTS API.
- `POST /v1/audio/speech` for an OpenAI-style speech endpoint.
- `GET /v2/capabilities` for model feature flags.
- `GET/POST /v2/speakers` for speaker reference upload, cache, and selection.
- `GET /v2/controls` for the steering bundle's controls.

For command-line arguments, run:

```bash
uv run --package zonos3-sgl python -m zonos3_sgl --help
```

## Text Prompting

Text is encoded with the checkpoint's SuperBPE tokenizer (pinned by SHA-256 in `params.json`), or as UTF-8 byte IDs on byte-text checkpoints. On checkpoints with condition rows the prompt also carries the loudness, fmax and speaking-rate buckets of `condition_buckets` / `condition_values` (per-table counts in `/v2/capabilities` `condition_tokens.bucket_counts`; the default LM has 10 / 10 / 30, the count itself = unknown).

## Speaker Conditioning

Speaker-conditioned checkpoints can use saved `.npy`/`.npz` embeddings or uploaded audio. Audio uploads are embedded by the speaker encoder the checkpoint names in `params.json` (`speaker_embedding_codec`, resolved through `model/src/zonos3/registries/speaker_encoder_registry.json`; ZONOS3 Mini uses `ownmect_s2`, a 192-D MECT-B2 embedding).

GDN / Full-AttnRes checkpoints (2026-10 family) clone from raw reference audio instead: the reference is encoded with the matched dual codec into a 10-codebook prefix (or supplied directly as `speaker_reference_codes`) and placed before the text exactly as in training; `speaker_cfg_scale` mixes the prompt with and without that prefix.

## Steering Controls

Additive direction vectors move a voice along emotion axes (valence, arousal, dominance, loudness, tempo, clarity, articulation), speaking styles, accents, voice qualities, word emphasis and more while preserving speaker identity. Every direction set is served through one steering bundle, `--steer-bundle` (`STEER_BUNDLE` for `servers/sgl/scripts/serve.sh`, default `steering/studio_bundle_mect60k_x10.json`; `steer.gguf` on the ggml, mlx and web servers); per request via `controls` (`"<set>.<direction>": dose`) and `control_spans`, with `controls_cfg_scale` guiding the hidden-space delta in embedding mode. A set's `calibration.json` (the `mood` axes ship one) is served in `GET /v2/controls` and as each speaker's `disabled_controls`; doses stay raw. See [guide.md](guide.md#steering-controls) for every shipped direction set and the studio bundle.

Steer then hold: a constant hidden push integrates over a long line (the model copies its own generated style), so the bundle's `steer_release` runs the request-level hidden `controls` at full dose for an onset, ramps them to a held fraction and holds it; its `exempt` directions, `control_spans` and raw directions stay constant. Requests override the defaults with `steer_hold` / `steer_onset_seconds` / `steer_ramp_seconds` (`steer_hold: 1` is the constant push, bit for bit); `GET /v2/capabilities` and `GET /v2/controls` report them as `steer_release`. See [guide.md](guide.md#steer-then-hold).

## LACI (alignment-guarded decoding)

LACI (Localized Attention-Constrained Inference, arXiv 2609.16989; `model/src/zonos3/tts/laci.py`) guards a generation against skipped text, hallucinated audio and early EOS. Start the server with alignment-head specs per language, e.g. `--laci-heads en_us=laci_heads/zonos3_mini_heads.json fr_fr=laci_heads/zonos3_mini_heads_fr.json` (the heads selected on `lm/zonos3_mini` for every server language ship in `laci_heads/`, fitted by `servers/sgl/scripts/laci_discover_heads.py`; `LACI_HEADS` / `LACI_ALL=1` pass them through `servers/sgl/scripts/serve.sh`), then send `"laci": true` on `/v2/generate` or `/v1/audio/speech`; offline, `TTSLLM(..., laci_heads={...}).generate(..., laci=True)` runs the same code and returns each request's LACI record under `"laci"`.

While the request decodes, its alignment heads' attention over the text (read inside the decode CUDA graphs) tracks the reading position. A skip, a stall or an EOS with two or more text bins unread rolls the request back to the failure onset; it regenerates from there with a new seed (`seed + 1000·retry`) under an ACI hard attention mask, at most 8 times, as a continuation that re-joins the running batch. The text counts as read once its last bin has been read for its share and the reading has reached the last text token; generation stops 1 s (`tail_frames`) later. Six rows after the text is read, the sampler adds 3 (`stop_boost`) to the stop token's logit in every codebook: a take that would otherwise run on past the text with a few extra words (mostly with a speaker embedding, whose stop probability there stays near 0.02) ends at its next likely stop, while a take whose end the model already wants is unaffected. The request's decode picks what leaves the scheduler: with the progressive decode (`stream_decode` prefix on a streamed response) only rows no later rollback can rewrite go out, so the response trails generation by about one text bin and a rare deeper rollback restarts at the first unsent row (`clamped` in the record); with the whole decode (`stream: false`, `stream_decode: "whole"`, offline `generate`) every row is held until the result is final, so every rollback restarts at its own row.

Offline, a LACI prompt can also carry a duration target, `generate(..., laci=True, laci_duration=[DurationTarget(frames, adjust_rate)])` (`zonos3.tts.laci`). Once the reading reaches the last text token, the elapsed rows over the target estimate the take's duration ratio. A take more than 10 % + 3 frames early, or not done 10 % + 3 frames past the target, is retaken from its first row in the speaking-rate bucket that ratio asks for, at most twice and within the retry budget; with `adjust_rate=False` the miss is only recorded. With `adjust_rate`, a truncation (early EOS) also regenerates one speaking-rate bucket slower instead of reseeding in the same bucket. Whatever `adjust_rate` is, a reading position that stops advancing for three text bins at the target pace (at least 15 frames) before the text is read is a stall (`hallucination`), rolled back to its onset as usual. The rate switch re-prefills the prompt with new condition rows, so it needs `cache_type="naive"`. The record adds `pace_retries` and the final `rate_bucket`; each failure record names the `retry_bucket` it switched to.

Constraints: speaker and controls CFG scale 1 (no CFG twin), the `fa` attention backend (chosen automatically with `--laci-heads` when `--attn` is `auto`), tensor parallel size 1. `--align-probe-max-text` (default 2048 with `--laci-heads`) bounds the text the decode graphs read; longer texts decode their probed steps eagerly.

## Short texts

Alone, a text of one or two words often comes out wrong or runs on into words it does not have: the model was trained on sentences. With `short_text: "auto"` (the default) such a text runs as the text, a 0.5 s pause marker (U+E004) and a filler sentence in its language ("Seven birds flew over the quiet river." in English) under LACI, and is cut once the reading position reaches the filler: at the first frame a codes-only silence classifier calls silent, searched from 2 frames before the reading reached the filler, after at least 3 voiced frames. A take that reaches no silent frame within 15 frames (1 s) ran into the filler: it is retaken from its first row with a new seed, at most twice, then cut where the reading reached the filler. The take is held until it is cut, so nothing past the text is decoded or sent; the LACI record ends `short_text` (or `short_text_forced`) and lists retakes as `short_run_on` failures.

A text qualifies when it has at most 2 words and its tokens are at most 2/5 of its filler's, with a checkpoint that has the pause tokens (the release text rows), LACI heads for its language, and no start, pause or phoneme markers, CFG, control spans, speech edit or LACI duration target; `short_text: "off"` generates it alone. The filler sentences, the cut rule and the int8 silence weights live in `model/src/zonos3/registries/short_text.json` (`zonos3.tts.short_text`; ported by the ggml and in-browser servers, pinned by `servers/ggml/tests/fixtures/laci-replay/short_text_*`). Measured on embedding voices with Qwen3-ASR, the cut clip is exactly the word in 97 % of English takes (48 % alone, 56 % alone with LACI) and 88 % across the other first 8 LACI languages (39 % alone); the clips are 0.5–1 s instead of 3–5 s.

## Distributed Serving

Tensor parallelism is available with `--tp-size n`. Rank 0 handles API-facing scheduler messages and broadcasts work to the remaining ranks.

## Attention Backends

Zonos3 supports FlashAttention and FlashInfer backends. Use `--attn` to select a backend, or a comma-separated prefill/decode pair such as `--attn fa,fi`.

## CUDA Graph

Decode CUDA graph capture is enabled by default. Set `--cuda-graph-max-bs 0` to disable it, or use a positive value to control the maximum captured batch size.

On GDN checkpoints, prefills of fresh requests also run as CUDA graphs: each request is padded to its complete 64-row chunks plus one 64-row window, one graph per shape, captured on first use (single requests of up to 5 complete chunks at startup) and kept in an LRU of `ZONOS3_PREFILL_GRAPHS` (16) for prefills of up to `ZONOS3_PREFILL_GRAPH_MAX_ROWS` padded rows (1024). They compute the same bits as the eager prefill; `ZONOS3_PREFILL_CUDA_GRAPH=0` turns them off. See [tts_architecture.md](tts_architecture.md#gdn--full-attnres-family-2026-10).

## Radix Cache

The KV cache manager defaults to radix caching. Use `--cache-type naive` to select the naive cache manager.
