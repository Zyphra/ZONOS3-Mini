# ZONOS3 feature guide

Request fields, server flags and behavior of the ZONOS3 Mini runtime beyond plain TTS. The HTTP endpoints of plain TTS are in [api.md](api.md); dubbing details in [laci_dub.md](laci_dub.md).

## Contents

- [Raw-audio voice cloning](#raw-audio-voice-cloning)
- [Speech editing](#speech-editing)
- [Accent adapters](#accent-adapters)
  - [ZONOS3 Mini release (`lm/mect_60k`)](#zonos3-mini-release-lmmect_60k)
- [Pronunciation input (phoneme tokens)](#pronunciation-input-phoneme-tokens)
- [Nonverbal markup and tag CFG](#nonverbal-markup-and-tag-cfg)
- [Emotion Control](#emotion-control)
  - [Raw-reference checkpoints: hidden-space steering](#raw-reference-checkpoints-hidden-space-steering)
  - [Accent steering](#accent-steering)
- [Zonos Voice Studio](#zonos-voice-studio)
  - [Span steering by reading position](#span-steering-by-reading-position)
  - [Projects on the server](#projects-on-the-server)
  - [Dubbing (Dub screen, `/v2/dub/*`)](#dubbing-dub-screen-v2dub)

## Raw-audio voice cloning

Either upload reference audio (`speaker_audio_base64` / the studio) or pass
already-encoded codes with `speaker_reference_codes` (`[frames, 10]` ints, the
same codes the training loader sees). The prefix is built exactly like
`apps.zonos.reference_packing.encode_reference`: each codebook delayed by its
index, tail flushed with audio padding, text column padded, no separator token.
`speaker_cfg_scale` > 1 runs the prompt without the prefix as the unconditional
twin and mixes `uncond + s * (cond - uncond)` in float32.

Within the runtime, a request alone and the same request inside a batch produce
identical codes.

## Speech editing

Edit checkpoints (speech-editing fine-tunes of `lm/mect_60k`; not released, served
with `SERVE_MODEL=<edit checkpoint>`; an edit checkpoint with phoneme tokens also
reads pronunciation input, see "Pronunciation input") add three text marker rows
(`text_extra_tokens: 3`) and a span-end audio token (`span_end_id` =
`audio_pad_id + 1` = 4098: one more embedding and head row per codebook). The
runtime loads them as they are. Every request has an end token: utterances
(`end_token="utterance"`, all of `/v2/generate`) stop on EOA and never sample
span-end; edit fills (`"span"`) stop on span-end and never sample EOA. An
utterance is drawn over codes, EOA and pad only, so the same seed gives the same
audio as a copy of the checkpoint without the span-end row. Stop boundaries count
host-processed frames, not submitted GPU work.

`TTSLLM.edit(codes, target_text, spans)` generates each masked span
(`(start_frame, length_frames)`, ascending, at most `text_extra_tokens`) as its
own request over `[speaker row][condition rows][text][unmasked context][earlier
fills][marker]` and splices the fills into the untouched codec frames;
`servers/sgl/scripts/edit_speech.py` wraps it (`--span START END` in seconds, or MMS spans
from a word diff; `--max-context-seconds` / `--context-seconds` as below).

Speech editing needs the MMS aligner of z-speech-alignment: clone
`git@github.com:Zyphra/z-speech-alignment.git`, run `uv sync` in it and point
`ZSA_ROOT` at the checkout.

```bash
ZSA_ROOT=<z-speech-alignment checkout> SERVE_MODEL=<edit checkpoint> \
STEER_BUNDLE=steering/studio_bundle_mect60k.json servers/sgl/scripts/serve.sh
```

`/v2/capabilities` then reports `speech_editing: true`, `edit_max_spans` (3),
`edit_frame_rate` (15), `edit_max_seconds` and `edit_align_max_seconds` (150, the
MMS alignment limit, which every windowed edit needs); on other checkpoints
`speech_editing` is false and the edit endpoints answer 400. A `<voice>.txt`
next to a default voice becomes its `transcript` in `GET /v2/speakers`.

- `POST /v2/edit/transcribe` `{audio_base64}`: MMS greedy transcript, `duration`
  and `words` (`word`, `start`, `end`, `confidence`, zsa-normalized `tokens`).
- `POST /v2/edit/align` `{audio_base64, text}`: MMS forced alignment, same fields.
- `POST /v2/edit` `{audio_base64, target_text}` regenerates the edited regions.
  Regions come from `spans` (`[start, end]` seconds in the original, ascending,
  at most `edit_max_spans`) or, without them, from a word diff of `original_text`
  (MMS transcript when omitted) against the target. Optional: `margin_seconds`,
  `max_span_seconds` (30), the `/v2/generate` sampling fields, `seed` (take `k`
  uses `seed + k`), `num_takes` (1-4), `verify` (MMS WER/CER and words per take),
  `speaker_embedding` (default on: the clip's embedding fills the speaker row;
  `false` leaves it out), `condition_buckets` / `condition_values`, `language`,
  `text_normalization`. The response holds `spans`, `original_text`, `words`,
  `contexts`, `expected_seconds` and `takes` (16-bit 44.1 kHz WAV,
  `edited_regions` in the result, `fill_seconds`, `stopped`; `false` = the fill
  hit `max_span_seconds`).
- `min_fill_ratio` (0.7; 0 = off): the span-end token is masked until a fill
  reaches that share of its region's `expected_seconds`, the region's length
  scaled by the target / original text length (zsa token characters) of the
  words in it, read from the MMS alignment of `original_text` (sent → aligned on
  every clip); without `original_text` the region's own length. The
  sampler masks the stop id per codebook while generated frame s < `min_frames`
  + c (`TTSSamplingParams.min_frames`), so the aligned stop frame is at least
  `min_frames`; requests without it build no mask and sample as before.
  `max_span_seconds` stays the upper bound.
- `original_words` (`[{word, start, end, tokens?, confidence?}]`, the align /
  transcribe word objects in order, timings checked or corrected by the caller):
  replace the server's MMS alignment of `original_text` for windows, expected
  fill lengths and the word diff, and come back as `words`. They must spell
  `original_text` (zsa tokens; missing `tokens` come from the MMS worker's
  `normalize` op, no alignment), with finite, ascending, non-overlapping times,
  `start < end`, inside the clip; otherwise 400.
- Clips longer than `max_context_seconds` (16, at most 20) are edited on
  windows: each region plus `context_seconds` (4) of unedited audio per side,
  cut halfway between MMS words of the original (aligned `original_text`, else
  the MMS transcript); an edge inside a changed block of the original → target
  word diff moves out to the block edge, and overlapping windows merge. A
  window's fill prompts hold only its codes and its stretch of the target text;
  every fill splices into the full clip, decoded once, with the full clip's
  speaker row and per-region steering. `contexts` lists the windows (full-clip
  seconds; `null` when the clip is edited whole). Edit training used 2-20 s
  clips, which is why long clips are edited on windows.
- `controls` (`/v2/generate` ids, see `/v2/controls`) apply to every region;
  `region_controls` (one `{id: dose}` or `null` per explicit span) add to them
  per region, doses summed per id. Each region is resolved like `/v2/generate`
  `controls`: raw directions move the clip's speaker-row embedding (400 without a
  speaker row), hidden ones add their delta on the frames the fill generates.
  The prefilled prompt (text, unmasked context, earlier fills) is not steered,
  as `/v2/generate` never steers reference or text rows; the context therefore
  keeps its exact KV and only the new words take the delivery. Unknown ids,
  `region_controls` without `spans` or of the wrong length are 400s.

The frontend runs an edit through the process pipeline: the tokenizer encodes
the audio (`TTSEncodeAudioMsg`) and builds each fill prompt
(`TTSTokenizeMsg.edit_*`), the scheduler generates it with `end_token="span"`,
the detokenizer returns its rows as codes (`TTSCodesReply`), and the tokenizer
decodes the spliced frames (`TTSDecodeCodesMsg`). MMS-1B-all runs in persistent
workers (`model/src/zonos3/workers/zsa_mms.py`), one per language adapter: `language` of `/v2/edit`,
`/v2/edit/transcribe` and `/v2/edit/align` (server codes; en_us, fr_fr, de, es, it,
pt_br, ja, cmn, ko map to zsa's `WHISPER_TO_MMS` adapters, others are a 400) picks it
for every MMS call of the request (transcription, alignment, windowing, verify); at
most `ZONOS3_MMS_MAX_WORKERS` (default 2) stay resident, least recently used first.
They run under the isolated z-speech-alignment interpreter: `ZSA_ROOT` (required, no
default; without it the edit endpoints fail with "Speech editing needs ZSA_ROOT") and
`ZSA_PYTHON` (default `$ZSA_ROOT/.venv/bin/python`) locate it, `HF_HUB_OFFLINE=1` (the
model and its adapters must be cached), `ZONOS3_MMS_DEVICE` picks the device (default
`cuda:0` when available). Explicit spans on clips up to `max_context_seconds`, without
`verify` and without `original_text` (or with `min_fill_ratio` 0), need no MMS.

## Accent adapters

`--accent-adapters DIR` serves accent LoRA adapters on top of the model; each
request picks `accent` and `accent_strength`, and requests with different
accents, strengths or none share prefill batches and decode CUDA graphs. `DIR`
holds `<accent>.pt` files from lingua's
`python -m apps.zonos.accent_adapter --lora NAME=RUN/checkpoints/STEP`
(schema `zonos-accent-lora/1`, or `/2` with per-expert MoE factors, see below);
the file stem is the name requests use and
`/v2/capabilities` lists the loaded names under `accents`. An unknown name
returns 400. The studio shows an accent selector and a strength slider when
adapters are loaded. Tensor parallelism is not supported with adapters.
`--accent-directions DIR` serves accents as hidden-space steering directions
through the same request fields (see [Accent steering](#accent-steering)); both
can be loaded together, a name in both is refused.

```bash
servers/sgl/scripts/serve.sh --accent-adapters <adapter dir>
```

A request runs base + α × adapter, the merge the adapter was scored as offline.
Every loaded adapter owns `rank` columns of one shared space, so an adapted
layer computes `x W^T + ((x D^T) * M) U` with `M` holding each token's strength
on its adapter's columns; the router balancing biases (quantile thresholds) get
each request's scaled shift per token. Per-row slots and strengths are CUDA-graph
inputs. The extra cost is fixed by the total columns loaded, not by how many
accents a batch mixes.

The adapter keys are lingua module names, which the converter keeps, and
`zonos3.models.accent_format.TARGETS` mirrors lingua's
`apps.zonos.lora.DEFAULT_TARGETS`. Every target adds its shift to the flat
`[out, in]` product before the runtime splits it, the product lingua's LoRA
adapts:

| lingua target | served op | layout |
|---|---|---|
| `attention.wq`, `attention.gater` | `Attention.wq` / `.gater` (`SimpleLinear`) | as stored |
| `attention.wkv` | `Attention.wkv` (`ChunkedLinear`) | `[2, kv, in]` as `[2·kv, in]`, K\|V split after |
| `attention.wo` | `Attention.wo` (`SimpleLinearOProj`) or `OriginalGDN.wo` | as stored |
| `attention.in_proj` | `OriginalGDN.in_proj` | fused `[2056, in]`, qkv\|z\|a\|b split after; also applied to the replayed 64-row chunk tail and the graph's fixed windows, with the request's own accent |
| `feed_forward.w_in` / `w_out` | `FeedForward.w_in` (`ChunkedLinear`) / `.w_out` | `[2, inter, in]` as `[2·inter, in]`, h\|gate split after |
| `feed_forward.latent_down` / `latent_up` | `MoEFeedForward.latent_down` / `.latent_up` | as stored, around the experts |
| `router.router_linear` | `Router.router_linear` | logits feed the softmax and the quantile top-k |
| `multi_output` | `MultiOutputHead` | shift on the real rows after the `min_rows` padding |
| `router.balancing_biases` (exact float32 shift) | `Router` balancing bias | per-slot table; legacy ZeroMean `parametrizations.balancing_biases.original` re-centred, `ent_denom` / `normalized_entropy` dropped, as in the converter |

Any other moved tensor is refused at load.

**Expert factors (`zonos-accent-lora/2`).** Adapters trained with
`lora.expert_rank` also shift every MoE expert weight by its own low-rank
product. They export `layers.N.feed_forward.experts.w13` / `.w2` entries with
`up` `[E, out, r]`, `down` `[E, r, in]`, `shape` and `experts: E`, in the
SonicExperts layout: w13 is `[64, 2·1536, 256]` with gate/up rows interleaved,
w2 `[64, 256, 1536]`. The experts sit in the 256-wide latent between
`latent_down` and `latent_up`, top-2, on layers 1-10. The non-expert entries
and the router bias shifts are as in `/1`. Expert factors are served on the
exact SonicMoE path only (`ZONOS3_USE_TRAINING_SONIC_EXPERTS=1`, which
`servers/sgl/scripts/env.sh` sets); elsewhere they are refused, as are expert entries in a
`/1` file, expert weights other than w13 / w2, mismatched shapes or expert
counts, and an adapter whose w13 and w2 ranks differ.

For a routed (token, expert e) pair of a request on adapter a at strength α:

- `α · U13[a,e] (D13[a,e] x)` is added to the gate/up pre-activation, before
  SwiGLU;
- `α · U2[a,e] (D2[a,e] h)` is added to the pair's expert output, before the
  weighted top-2 sum (`h`: the adapted activation).

Training formed `W_e + U_e D_e` in the forward, so this is the same function
with the sum regrouped. The exact path's SonicMoE call fuses gather, GEMM and
SwiGLU, so the pre-activation never reaches memory. The runtime therefore
replays `moe_general_routing_inputs` with the same routing metadata, QuACK
kernels and top-k combine, and hands both grouped GEMMs the shift as their
addend `C` (`D = A·B + C`, added to the float32 accumulator before the
epilogue). A pair without expert factors (no accent, or a `/1` adapter) has a
zero row in `C`, and the kernels with a zero addend return the plain call's
bits.

The shift itself is a Triton kernel (`zonos3_sgl/kernel/triton/expert_lora.py`).
Each program takes one pair and one output tile, reads that pair's
`(adapter, expert)` factors, and computes `α · U (D x)` in float32 in a fixed
order. A pair's bits therefore do not depend on the rest of the batch, and the
kernel has static shapes for the CUDA graphs. Adapters are grouped by expert
rank: each rank keeps one `[adapters·E, in, r]` / `[adapters·E, r, out]` stack
per weight and costs one launch per weight. Pairs off a rank load nothing.
The engine probes these kernels at load like the other side paths, with 1-256
pairs against 1024 on every slot.

Exact mode stays batch-invariant with adapters loaded. A request without an
accent has an all-zero mask row, so its shift is an exact zero. For accented
requests the engine probes every adapted op at load with 1-256 rows against a
1024-row product. Any op whose cuBLAS kernel depends on the row count runs its
side path as fixed 16-row GEMMs instead (on H100, the LM head).

`servers/sgl/scripts/accent_parity.py` (wrapped by `servers/sgl/tests/test_accent_adapters_gpu.py`)
checks this offline on seeded requests and writes a `report.json`:

- (a) requests without an accent, batched with accented requests, against the
  engine without adapters (bit-equal logits and codes);
- (b) every request of a batch mixing accents and strengths against the same
  request run alone (bit-equal);
- (c) an adapter at strength 1 against its merged checkpoint
  (`accent_adapter --reconstruct`, then `tools/convert_lingua_checkpoint.py`),
  teacher-forced on the adapter's codes. These agree closely but not bit for
  bit: the served sum and the merged bf16 weights round differently, and the
  MoE top-k turns some of those ulps into different experts. The base model on
  the same streams gives the scale of the adapter's whole effect.

Requests cycle over no speaker prompt, the raw reference prefix
(`--reference-audio`) and the speaker-embedding row (`--speaker-embedding`),
with unknown condition rows; `--merged-model` is optional. The test wrapper
reads `ZONOS3_TEST_MODEL`, `ZONOS3_TEST_ACCENT_DIR`, `ZONOS3_TEST_MERGED_MODEL`,
`ZONOS3_TEST_MERGED_ACCENT`, and optionally `ZONOS3_TEST_MIXED_ACCENTS`,
`ZONOS3_TEST_REFERENCE_AUDIO`, `ZONOS3_TEST_SPEAKER_EMBEDDING` and
`ZONOS3_TEST_PARITY_OUT`; run it in the exact-mode environment of `servers/sgl/scripts/env.sh`.

`--eoa-policy codebook0` lets only an end-of-audio token sampled in codebook 0
stop a request (lingua `generate --eoa-policy codebook0`); the default `any`
keeps the previous behaviour.

`ZONOS3_ACCESS_TOKEN` keeps the server private: send
`Authorization: Bearer <token>`, or open `/?access_token=<token>` once in a
browser.

### ZONOS3 Mini release (`lm/mect_60k`)

The release LM `hf://Zyphra/ZONOS3_Mini_release_ckpts/lm/mect_60k` serves accent
adapters with `servers/sgl/scripts/serve.sh --accent-adapters <adapter dir>`, with the release
decoder (full-size causal MR iSTFT head) and the speaker encoder from
`model/src/zonos3/registries/`. The server takes the encoder named by the model's `params.json`
(`speaker_embedding_codec: ownmect_s2`), and checks its sha256 at load.

Speaker and condition conditioning are prompt-side. The speaker-embedding row
(`speaker_projection`, Linear 192 → 1024), the three condition rows
(`condition_embeddings`) and the raw reference prefix replace input rows
before the transformer, so the adapters need no path for them. Those tensors
are not LoRA targets: a factor on `speaker_projection` is refused as outside
`TARGETS`, and an exact shift of it or of a condition table is refused as
neither a linear factor nor a router bias (`servers/sgl/tests/test_accent_adapters.py`).
An accent applies on either speaker path. For an uploaded clip or a library
voice the server's default is the raw reference prefix, which is what the
studio sends; `speaker_mode: "embedding"` uses the ownmect_s2 row instead. Speaker
CFG twins carry the request's accent on both paths.

A merged checkpoint (release LM plus an adapter, lingua
`accent_adapter --reconstruct`) converted with `tools/convert_lingua_checkpoint.py`
the way the release LM was converted (its `conversion.json` holds the converter
arguments) is the release model plus the adapter, which is what parity check (c)
compares against.

## Pronunciation input (phoneme tokens)

The ZONOS3 Mini release LM with E2-TTS "X2" pronunciation control: a word can be
written as its IPA. The model is the release LM plus phoneme tokens trained on it
in lingua (`apps/zonos/configs/z3m_mect60k_phoneme_tokens.yaml`, see
`lingua/apps/zonos/PHONEMES.md`): 138 text rows after the padding row, a group
open token, a group close token and one token per IPA symbol of the 136-symbol
inventory of the gold pronunciation lexicon. Only those rows were trained, so the
checkpoint is the release LM bit for bit on text without groups: every steering
direction (`steering/studio_bundle_mect60k.json`, emotion and accent directions) and
the release accent adapters apply to it unchanged. Phoneme checkpoints are not
released; build one from the release LM and an exported row file.

```bash
uv run --package zonos3-tools python tools/add_text_extra_rows.py --base <release LM dir> \
    --rows <phoneme_tokens.pt> --out <phoneme checkpoint>
SERVE_MODEL=<phoneme checkpoint> STEER_BUNDLE=steering/studio_bundle_mect60k.json \
PHONEME_LEXICON=<lexicon dir> servers/sgl/scripts/serve.sh
```

`add_text_extra_rows.py` appends the rows (`zonos-text-extra/1`, exported by
`apps.zonos.text_extra`) to the text table of a converted checkpoint and writes
`text_extra_tokens` and `text_tokenizer_spec.phonemes` (group markers, token ids,
symbols) into `params.json`. `text_extra_tokens` is the same field the
speech-edit checkpoints use for their marker rows. The phoneme block is always the
last extra rows: right after the padding row, or after the edit marker rows (which
stay at text_vocab + 1 + j, where edit training put them); `superbpe.load_spec`
checks that markers plus the phoneme block fill `text_extra_tokens`, and
`edit_max_spans` counts the marker rows. A base with extra rows keeps them and the
new rows' ids shift:

```bash
uv run --package zonos3-tools python tools/add_text_extra_rows.py --base <edit checkpoint> \
    --rows <phoneme_tokens.pt> --out <edit + phoneme checkpoint>
```

The result has `text_extra_tokens` 141 (3 markers, open / close 8200 / 8201, phones
from 8202), `speech_editing` with `edit_max_spans` 3 and the 136 phone symbols. On
text without groups it is the edit checkpoint bit for bit. The rows were trained on
the frozen release LM, not on the edit fine-tune. `PHONEME_LEXICON`
(`servers/sgl/scripts/serve.sh`) passes `--tts-phoneme-lexicon`; the launcher runs with the
`phonemize` dependency group.

- **Request text.** A group is U+E000 IPA U+E001 in `text`
  (`zonos3.tts.superbpe`): the open id, one id per IPA code point, the close id;
  the text between groups is SuperBPE-encoded stretch by stretch, exactly as the
  training loader encodes it. Text normalization leaves groups as typed and
  verbalizes the text between them. Malformed groups and symbols outside the
  inventory are refused with 400 before they reach the tokenizer worker, as are
  groups sent to a checkpoint without phoneme tokens (`/v2/generate`,
  `/v1/audio/speech`, `/v2/edit` `target_text`).
- **Groups in edits** (edit checkpoint with phoneme tokens). `/v2/edit`
  `target_text` takes groups; `original_text` / `original_words` stay plain. MMS
  normalization would mangle a group (IPA upper-cased, the markers and tie bars
  become word breaks, so `t͡ʃiːz` reads `T ƩIːZ`), so a group never goes to MMS: it
  is one target token of its own in the word diff, windows and expected fill length
  (its length = its IPA symbols × 1.4 letter-equivalents,
  `GROUP_LETTERS_PER_SYMBOL`), and window texts keep it as typed, so fill prompts
  carry the group ids as `/v2/generate` encodes them. With `verify`, each group is
  a wildcard word of the WER / CER reference (any one recognized word matches; ties
  go to the alignment with the fewest character edits), the take's word alignment
  uses the target with each group replaced by the word heard, and `verify.groups`
  lists every group of `target_text` in order as `{ipa, heard}`. The 1.4 letters
  per IPA symbol (rather than the lexicon's average of about one) reflects that
  spoken duration follows the word, and respelled names and short words carry more
  letters per phone.
- **Studio.** Pronunciations are typed in X-SAMPA inside parentheses, e.g.
  `I met (S@vQn)`; the studio shows the IPA each group becomes (flagging symbols the
  model has no token for) and sends U+E000 IPA U+E001 groups. On checkpoints
  trained on parenthesized IPA (no `phoneme_tokens` in `/v2/capabilities`) it
  sends `(IPA)` instead. Stress marks, syllable dots and spaces are dropped as in
  training; a lone `_` is a tie (`t_S` = t͡ʃ); symbols without an X-SAMPA spelling
  (superscript tone digits, `ᵝ`, `ˣ`, ...) are typed as-is.
- **Phonemize selection** replaces the selected words with their gold-lexicon
  pronunciation in the Text Norm language (`--tts-phoneme-lexicon`, an LMDB built
  by lingua `apps.zonos.build_phoneme_lexicon`). Only words whose deciding source
  has a single pronunciation come from the lexicon; hiragana, okurigana and single
  kanji/hanzi stay as text, as in training. Ambiguous and unknown words of the
  space-delimited languages are guessed by espeak-ng and mapped to the lexicon's
  notation; the studio lists them as guesses to check. espeak-ng is GPL-3.0: it is the
  opt-in `phonemize` dependency group (`espeakng-loader`; `uv sync --locked --group
  phonemize`, and `servers/sgl/scripts/serve.sh` runs with it); without it unknown words stay as
  text. Chinese, Japanese, Korean and Vietnamese are never guessed.
- `/v2/capabilities` on a phoneme checkpoint lists the phone symbols,
  `phonemize_languages` (the ten Text Norm languages) and the languages espeak-ng
  guesses for (en_us, en_gb, fr_fr, de, es, it, pt_br). `POST /v2/phonemize`
  returns each word's pronunciation and whether it was guessed.

## Nonverbal markup and tag CFG

**Checkpoint.** Nonverbal (NV) markup is read by an NV fine-tune of the speech-edit
model (not released; the edit model's span-end row and 3 edit marker rows, then NV
training steps). Serving one with pronunciation groups appends the phoneme rows after
the edit marker rows, exactly as for the edit checkpoint:

```bash
uv run --package zonos3-tools python tools/add_text_extra_rows.py --base <NV checkpoint> \
    --rows <phoneme_tokens.pt> --out <NV + phoneme checkpoint>
SERVE_MODEL=<NV + phoneme checkpoint> STEER_BUNDLE=steering/studio_bundle_mect60k.json DUB=1 \
PHONEME_LEXICON=<lexicon dir> servers/sgl/scripts/serve.sh
```

The result has `text_extra_tokens` 141 (3 edit markers, group open / close 8200 /
8201, phones from 8202) and `edit_max_spans` 3. On text without groups it is the
converted NV checkpoint bit for bit.

**Markup.** Point tags `<laugh/>`, `<breath/>`, `<sigh/>`, `<cough/>`, `<panting/>`, `<shouting/>`, `<crying/>` and
span tags over words, `<shouting>Get out!</shouting>`, `<crying>…</crying>` (any of the seven names,
`</?(laugh|breath|sigh|cough|panting|shouting|crying)\s*/?>`, lower case). Other tags (`<giggle/>`, `<whispering>`,
`<Laugh/>`) are plain text. Markup at `tag_cfg_scale` 1.0 is ordinary text on the unguided path; a checkpoint never
trained on tags reads it as text.

**Tag CFG.** `tag_cfg_scale` > 1 (`/v2/generate`, `/v1/audio/speech`) runs a plain twin next to the request: the
same normalized text with the markers removed (enclosed words kept, whitespace collapsed) and every other
conditioning identical; both branches advance on the same sampled frames, and every step mixes the raw logits of
all codebooks in float32 before any sampler transform, `plain + s · (tagged − plain)`. Rules: finite `1.0`–`4.0`
(422 otherwise); above `1.0` the text must contain a recognized tag; one twin per request, so `speaker_cfg_scale`
and `emotion_cfg_scale` must be `1.0` and `laci` false (400 otherwise). The offline `TTSLLM` does not build the twin (a request
above 1.0 there raises). The server logs both prompts (`tag CFG uid=…` from the tokenizer, `Tag CFG uid=… plain
twin N rows` from the scheduler).

`GET /v2/capabilities` reports `tag_cfg`: `available` (always true), `trained` (the checkpoint's training config,
`config.yaml` beside `params.json`, has a data source with `nonverbal: true`), `tags`, `min_scale` 1.0, `max_scale`
4.0, `default_scale` 1.0.

**Interactions with the rest of this runtime.**

- *Normalization*: neither markers nor pronunciation groups reach a grammar. With markup, the text between markers is
  normalized piece by piece (each piece with its groups kept as typed) and re-joined with a single space wherever
  the input had whitespace next to a marker (the ja grammar fails on a space next to a marker).
  Text without markup takes the previous path unchanged.
- *Pronunciation groups*: markup and groups mix freely in one text; the plain twin keeps the groups.
- *Steering*: `controls`, `control_spans`, emotion deltas and accent adapters apply to BOTH branches under tag CFG
  (the twin differs in its text rows only), as under speaker CFG: steering is part of the generation, so the
  guidance amplifies only the tag contrast; a steered request with tag CFG is the steered request with stronger tags.
  The reading-position follower's per-frame decisions (from the tagged branch's LACI heads) are shared with the twin.
  Tag CFG and Direct's speaker CFG 1.5 are exclusive (one twin per request): a client picks one per render.
- *Span offsets*: `control_spans` characters index the text as sent, markup included. The character-rate schedule
  gives markers no time (`control_span_rate` is frames per spoken character; a client measuring it from a take
  divides by the length without markers). Reading position: a span maps to the text tokens that spell it; tokens
  inside a marker never open or close a span that also covers words (markers inside stay in), a span over a marker
  alone keeps the marker's tokens.
- */v2/edit*: `tag_cfg_scale` other than 1.0 is refused. Markup in `target_text` (and `original_text`) is allowed:
  markers ride in the model's prompt text but never reach MMS (word diff, alignment, verify): each marker is a
  target piece with no zsa tokens, `<laugh />` is canonicalized to `<laugh/>`, and `original_text` in the response
  is the markup-free text MMS read. A marker alone between unchanged words is therefore no edit (400 "target text
  equals the original transcript") unless `spans` place it; a marker inside an edited region rides with the fill.
- *Dubbing*: target texts may carry markup (LACI runs at tag CFG 1); markers count no bytes for the segment's
  speaking rate and the forced aligner reads the plain text.
- *Sampling*: the sampler draws each row over the narrowest vocabulary prefix (no span-end column for utterances), so
  an edit checkpoint consumes the random numbers of the same checkpoint without that row.

## Emotion Control

You can nudge a voice toward an emotion or along conceptual axes (valence,
arousal, …) **without changing speaker identity**. Emotion is applied as additive
*direction* vectors — no model or checkpoint changes — so the timbre is preserved
while prosody shifts.

The default direction set is `steering/emotion_directions_mect60k_axes/`, and the
server **auto-loads it on startup when that folder is present** — emotion sliders
appear in the studio automatically, no extra flags. Point elsewhere (or disable)
with `--tts-emotion-directions-dir <dir>` (`EMOTION_SERVE_DIRECTIONS` for
`servers/sgl/scripts/serve.sh`; pass an empty string to turn it off).

```bash
curl -X POST http://localhost:1919/v2/generate \
  -H "Content-Type: application/json" \
  -d '{
        "text": "I cannot believe you did that!",
        "emotion_enabled": true,
        "emotion_valence": 0.8,
        "emotion_arousal": 0.8,
        "stream": true
      }' \
  --output excited.pcm
```

`GET /v2/capabilities` reports what the loaded directions expose:
`emotion_enabled`, `emotion_names`, `emotion_axes`, and `emotion_calibrated`
(whether a per-speaker strength calibration is loaded). The shipped set includes a
`calibration.json` so `emotion_strength: 1.0` is already a sensible per-voice,
per-pole default; raise it to exaggerate. `emotion_cfg_scale` > 1 runs an
unsteered twin and amplifies the difference (about 2× compute).

### Raw-reference checkpoints: hidden-space steering

ZONOS3 Mini clones from a raw reference prefix, so for reference-mode requests
there is no speaker embedding for embedding-space directions to act on. The
directions therefore live in the model's hidden space (`manifest.json`:
`"space": "attnres_source"`, `"source_index": k`): on every generated frame the
request's delta is added to source `k` of the Full AttnRes bank (0 = embedding,
`2i+1` = attention/GDN output of layer `i`, `2i+2` = FFN/MoE output of layer `i`)
before later sublayers and the output head route from it. Reference and text rows
are never steered. The add is captured in the decode CUDA graphs with a static row
buffer; with no emotion requested the buffer is zero and the logits are
bit-identical to a server without directions. The request API is unchanged
(`emotion_sliders`, `emotion_valence`/`emotion_arousal`, `emotion_strength`,
`emotion_calibrated`); default voices are selected by `speaker_embedding_id`
and their audio becomes the reference prefix.

Directions are measured, not trained: clips are teacher-forced through the offline
engine, every bank source is mean-pooled over the clip's own frames, and a direction
is a contrast of those means (e.g. style − neutral) at one source. The shipped
`steering/emotion_directions_mect60k_axes/` uses the GDN output of layer 9
(`source_index` 19, `L9.attn`) and folds the style directions into orthogonal
conceptual axes: valence, arousal, dominance, loudness, tempo, clarity and
articulation. `articulation` is the component enunciated and confused share, which
clarity (their difference) cancels: positive is slow, deliberate delivery, negative
quicker, looser speech. Its `calibration.json` carries per-voice doses.

Doses are raw multiples of the measured mean style shift; the usable window
is narrow (a style that responds at one dose can run away at twice that), which
is what the per-voice, per-pole calibration absorbs.

### Accent steering

`--accent-directions DIR` (`ACCENT_SERVE_DIRECTIONS` for `servers/sgl/scripts/serve.sh`)
loads `attnres_source` directions as accents: a request's `accent` /
`accent_strength` adds `strength × dose × direction` to the set's bank source on
every generated frame (dose from `DIR/calibration.json` per voice, 1 without
one or with `emotion_calibrated: false`). A `raw` set moves the speaker
embedding instead (see [Speaker-embedding accents](#speaker-embedding-accents)).
`/v2/capabilities` lists them under
`accents` and `accent_steering`. Emotion and accent sets may steer different
sources: the engine keeps one row-buffer slot per steered source and the
request delta carries one hidden-size slot per source (summed when they share
one).

The shipped `steering/accent_directions_mect60k/` (lm/mect_60k) holds Hindi only,
at L7.attn (`source_index` 15): accented English minus English, from
English-labelled clips that MMS-LID hears as most Hindi-like against native
English clips. The other accents did not work as one hidden vector (native speech
minus English mostly switches language before the accent moves); they stay on
`--accent-adapters` or the speaker-embedding directions below.

```bash
ACCENT_SERVE_DIRECTIONS=steering/accent_directions_mect60k servers/sgl/scripts/serve.sh
```

#### Speaker-embedding accents

A `raw` set moves the ownmect_s2 speaker embedding by `strength × dose × direction`,
rescaled to the embedding's norm as raw emotion directions are, and needs
`speaker_mode: embedding` (a reference-mode request with such an accent gets a
400). The direction is the mean over native L1 speakers minus the mean over native
English speakers of the embedding (clips averaged per recording first; gender and
age, `steering/speaker_attributes_mect60k/`, projected out). The embedding has no
text in it, so unlike the hidden native-minus-English direction it does not carry
a language switch. Strong doses trade voice identity for accent (the speaker partly
becomes a different one), where the LoRA adapters keep it. The shipped
`steering/speaker_accents_mect60k/` holds hin, spa, fra, por, deu, vie and pol;
Mandarin, Korean and Japanese barely move in the embedding alone and stay on
`--accent-adapters`.

```bash
EMOTION_SERVE_DIRECTIONS=steering/speaker_attributes_mect60k \
ACCENT_SERVE_DIRECTIONS=steering/speaker_accents_mect60k servers/sgl/scripts/serve.sh
```

#### East Asian accents: embedding and hidden together

Neither the speaker-embedding direction nor the hidden native − English direction
gives Mandarin, Korean or Japanese alone. Served together (the raw set as emotion
sliders, the hidden set as `--accent-directions`; the engine allows a raw emotion
set beside hidden accents) the two add up for Mandarin, at the price of the voice:
`steering/mandarin_mect60k/` (raw, `cmn`) and `steering/mandarin_hidden_mect60k/`
(L7.attn, `cmn`) are that pair (in the studio bundle as `mandarin` and
`mandarin_hidden`). Korean and Japanese stay on the LoRA adapters.

#### Regional English accents

`steering/english_accents_mect60k/` (raw) holds british, american, scottish, irish,
australian, indian and welsh, each relative to American English (american is
the reverse of british, for British voices). They are fitted from speakers with a
known accent, each contrast inside one corpus so its recording conditions cancel:
VCTK (accent column), VoxCeleb1 dev (nationality) and OpenSLR 83 English dialects
(contrasts against southern English, chained through british), gender and age
projected out. welsh also slows speech slightly.

#### Voice quality

`steering/voice_quality_mect60k/` (raw, served as emotion sliders) holds deep,
shrill, husky, raspy, soft and authoritative: ParaSpeechCaps' human speaker tags
(VoxCeleb / EARS / Expresso speakers on their original recordings), tagged minus
untagged speakers inside each corpus and gender, gender and age projected out. The
tags are noisy and the directions short, so a dose of 2 barely moves the voice;
useful doses run to about ×4.

#### Pitch level and range

`steering/pitch_mect60k/` (raw axes) holds pitch and pitch_range, fitted on
LibriTTS-R speakers with their measured F0 (pYIN): the least-squares slope of the
embedding on the speaker's log F0 (pitch) or F0 spread in semitones (pitch_range)
inside each gender, times the P10-P90 span (dose 1 = from a low-pitched to a
high-pitched speaker of the same gender), gender and age projected out. Gender holds
at ±2; pitch_range leaves the F0 level about where it was.

#### Recording channel

`steering/channel_mect60k/` (raw axis `clean`) is the within-speaker change of the
embedding with ParaSpeechCaps' recording-cleanliness tag (very noisy … very clean,
per clip): every VoxCeleb clip minus its speaker's mean clip, regressed on the tag
minus the speaker's mean tag (`"group_by": "clip", "center": "speaker"`), so the
direction is the channel, not who records in it. Positive doses also widen the
bandwidth and slow speech slightly. Useful range −2 … +4.

#### Speaking styles in hidden space (EARS)

`steering/ears_styles_mect60k/` (attnres_source, L9.attn like the emotion axes)
holds whisper, loud, tempo (fast − slow), pitch (high − low) and conversational
(spontaneous − read): EARS's speakers read the same passage and sentences in every
style and talk freely, so each direction is a within-speaker contrast
(`"center": "speaker"`) with no text or channel difference.

- conversational: +4 sounds spontaneous rather than read (more hesitation —
  pauses and prolongations, no inserted words — less enunciation, louder and
  flatter); −4 is the careful, quieter reading voice.
- whisper lowers voicing and level (×2 is strong; ×4 breaks speech); loud raises
  the level with the pitch rise of projected speech.
- tempo follows the dose on every voice (−2 … +2 is the usable range; −4
  collapses). EARS speakers raise their voice when they speed up and drop it when
  they slow down, and the direction carries that (level and F0 move with it); slow
  doses add pauses.

#### Expressiveness in hidden space

`steering/expressive_mect60k/` (attnres_source L9.attn, axis `expressive`) is the
within-speaker change of the bank with a clip's F0 spread: LibriTTS-R readers'
livelier and flatter clips. The direction is short, so doses run to ±16. Livelier
readings are also higher and slower in LibriTTS-R, and the direction keeps that;
the speaker-embedding `pitch_range` axis widens the spread at a constant F0 level,
so it is the cleaner control of the two.

#### Disfluency in hidden space

`steering/disfluency_mect60k/` (attnres_source L9.attn, axis `disfluent`):
DisfluencySpeech's speaker reads Switchboard turns with their fillers, repetitions
and restarts; each clip is teacher-forced with the FLUENT transcript, so the bank
sees hesitations the text does not ask for, and the direction is the regression on
how many. The hesitation is pauses, blocks and drawn-out words, not inserted
fillers; past +8 words start to drop.

#### Emotion: speaker embedding vs hidden space

EARS's speakers also read the same sentences in 22 emotions and neutral.
Within-speaker emotion − neutral gives angry, sad, happy (amusement), fearful,
disgusted and surprised (amazement) both in the speaker embedding
(`steering/speaker_emotions_mect60k/`, raw, gender and age projected out) and at
L9.attn (`steering/ears_emotions_mect60k/`, attnres_source). The speaker-embedding
directions move the perceived emotion most, at a larger identity cost and in
embedding mode only. The served emotion axes stay as they are; both EARS sets are
there to combine (e.g. as accent directions next to the axes).

#### Persona presets

`steering/speaker_traits_mect60k/` merges every speaker-embedding axis into one
served raw set (gender, age, the seven regional accents, six voice qualities,
pitch, pitch_range, clean) with six persona presets (`presets.json`), slider
weights at `emotion_strength` 1 (raw doses). A hidden style can ride along through
the accent field with `ears_styles_mect60k` as `--accent-directions` (its axes are
accents).

| preset | sliders |
|---|---|
| elderly Scottish narrator | age 1.5, scottish 1.5, deep 2, pitch_range −1 |
| young Australian | age −1.5, australian 1.5, pitch_range 1 |
| gravelly old man | gender −1, age 2, raspy 3, deep 2 |
| BBC newsreader | british 1.5, authoritative 3, clean 2 |
| soft-spoken Irish woman | gender 1, irish 1.5, soft 3 |
| Indian English caller | indian 1.5, clean −2 |

The components add up, except the Scottish and Irish accents, which weaken in a mix.
A persona is a different speaker by design.

```bash
EMOTION_SERVE_DIRECTIONS=steering/speaker_traits_mect60k \
ACCENT_SERVE_DIRECTIONS=steering/ears_styles_mect60k servers/sgl/scripts/serve.sh
```

#### The controls on the speech-editing checkpoint

The ZONOS3 Mini edit fine-tune of `lm/mect_60k` takes the same directions. This
runtime serves it natively (see Speech editing): plain TTS samples over codes, EOA
and pad only, so span-end is never sampled for an utterance. Every control moves its
measure in the same direction as on the release LM; most speaker-embedding controls
match it, while hidden-space controls get weaker at the same dose (and do less
damage).

#### More moods (EARS)

`steering/ears_moods_mect60k/` holds every EARS emotion against neutral in the same
space and spec style as `ears_emotions_mect60k` (attnres_source L9.attn,
`"group_by": "clip"`, `"center": "speaker"`; its six directions are identical):
adoring, confused, content, cute, desirous, disappointed, distressed, embarrassed,
ecstatic, guilty, interested, pained, proud, realizing, relieved and serene next to
angry, sad, happy, fearful, disgusted and surprised, plus `scared`, fear against the
other 21 emotions. The directions share one large "emotional reading vs neutral"
component, and most of them add that component and arousal without their specific
target.

Six moods ship as studio controls: angry, happy, sad, calm (`emotions.serene`),
scared (`emotions.scared`) and confused. In the studio bundle the `emotions` set is
`ears_moods_mect60k`, and each shipped mood is a unipolar, spannable `mood` control
whose safe range, range, effect line, identity curve and default dose come from
`steering/ears_moods_mect60k/eval/effects.json` (rendered on the edit checkpoint).
`ui.moods` places each one on the Mood pad at its default dose by its valence and
arousal shift.

At equal dose the release LM moves further than the edit checkpoint and breaks
sooner (about two thirds of the edit dose gives the same effect): base-model servers
using the bundle should keep angry and happy at ×2-3, sad at ×4-6, calm at ×4,
scared at ×8 and confused at ×2. To take an emotion out of a reference voice, use its
direction at a negative dose.

## Zonos Voice Studio

`/` serves Zonos Voice Studio (`studio/`: Vite + Preact + TypeScript, its committed production build in
`studio/dist/` served by both servers; see [studio/README.md](../studio/README.md)), a voice studio on top of every
direction set above:

- **Cast**: start from a library voice, an upload or recording, or a rolled voice; move
  it on the Who map (gender × age at measured positions, pitch on the wheel, an
  identity ring where the voice stops being itself), the accent atlas (drag a pin into
  a region), texture radar, room (phone ↔ studio), delivery and mood; persona
  presets; "voices near this one" variations; every change re-renders a preview and
  A/Bs it against the unsteered voice; save as a voice (the steered embedding, cached
  on the server) or to the cast.
- **Direct**: a script by lines; select words and paint delivery (projection, pace,
  read ↔ chatty, hesitation, liveliness, mood pad, emotion chips) as
  `control_spans`; takes per line, starred takes play and export as the script.
- **Scene**: cast members as characters, a dialogue rendered line by line, played on a
  timeline and exported as WAV.

It needs a server with a steering bundle; `steering/studio_bundle_mect60k.json`
holds 13 sets (identity, pitch, English and L1 accents, the Mandarin pair, texture,
room, the EARS delivery styles, expressiveness, disfluency, the emotion axes and the
EARS moods; set paths resolve relative to the bundle file), 24 UI controls with
tested ranges, identity curves and duration factors read from each set's evaluation,
15 accents, 8 personas, Who-map positions of the default voices and Mood-pad
landmarks of the six named moods (see More moods).

```bash
STEER_BUNDLE=steering/studio_bundle_mect60k.json servers/sgl/scripts/serve.sh
# open http://127.0.0.1:1919/
```

`--steer-bundle` (`zonos3.tts.steering`) serves all the sets at once: a request's
`controls` sum the raw directions into the speaker embedding (norm kept) and the
hidden ones into their Full AttnRes sources; `control_spans` add hidden controls on
the frames generated while those characters are spoken (character position ×
`control_span_rate`, stretched by the bundle's `timing` factors, 3-frame ramps; the
same seed gives identical audio before a span). `GET /v2/controls` returns the sets,
the directions (raw vectors included, so the client computes positions and the
steered embedding itself) and the UI metadata; `GET /v2/speakers/{id}/embedding`
returns a voice's embedding. With a bundle the server embeds every default voice at
startup on the GPU (`servers/sgl/scripts/serve.sh` sets `ZONOS3_SPEAKER_EMBEDDER_DEVICE=cuda`
when `STEER_BUNDLE` is set), and `POST /v2/speakers` accepts embedding files on
dual-path checkpoints (designed voices).

### Span steering by reading position

`control_span_follow: "reading"` places `control_spans` by where the model is reading instead of character position ×
`control_span_rate` (`zonos3.tts.span_follow`). The tokenizer maps each span's characters through text normalization
to the text tokens that spell them (SuperBPE offsets, every id of a pronunciation group, UTF-8 bytes on byte
checkpoints); while the request decodes, the LACI alignment heads of its language (`--laci-heads`; the probe of
`zonos3_sgl.tts.align_probe`, inside the decode CUDA graphs, no LACI retries unless `laci=true`) give one reading position
per step, smoothed (median of 3, monotone, at most +1 token per step). A span ramps in (3 frames) once the smoothed
position reaches its first token − `control_span_lead` (default 1) and out once it reaches the token after it +
`control_span_lag` (default 0). The readout reaches the host with the step's tokens, so a frame is decided on readouts
one or two steps old; every decision is recorded per frame, so the speaker-CFG twin steers exactly like its request and
a LACI rollback replays from its row. Works with speaker CFG (the probe reads the conditional request; the twin shares
its gate) and with `laci=true`. Without heads for the language the rate schedule runs; responses to requests with spans
carry `X-Control-Span-Follow: reading|rate`. Requests without `"reading"` run the rate-placed path unchanged.

Lead 1 (the default) places the span onset close to the spoken words; lead 2 steers more of the span's words and leaks
more into its neighbours; lag −1 loses coverage and +1 leaks. Whisper and anger carry on past a span by themselves (a
property of the model, not of the placement), so offsets tend to run late.

### Projects on the server

Studio projects can live on the server, shared by every browser that opens the studio link, and survive server
restarts and a new tunnel URL (`zonos3_sgl.server.projects`). `--studio-projects-dir` enables them; `servers/sgl/scripts/serve.sh`
passes `ZONOS3_STUDIO_PROJECTS` (default empty = projects stay in each browser's IndexedDB, the studio's offline
store). The directory holds `projects/<id>.json` (`{id, name, revision, updated_at, media_bytes, data}`, `data` = the
studio's project document) and `media/<sha256>` (content-addressed, written once, `media/<sha256>.type` its MIME type);
every write is a temp file renamed into place, under a lock (flock, so servers can share the directory). Limits:
`--studio-max-project-bytes` (20 MB) and `--studio-max-media-bytes` (2 GB).

```bash
ZONOS3_STUDIO_PROJECTS=<projects dir> STEER_BUNDLE=steering/studio_bundle_mect60k.json servers/sgl/scripts/serve.sh
```

- `GET /v2/projects` → `[{id, name, updated_at, revision, bytes, media_bytes, summary}]`, newest first (`summary` =
  the document's `data.summary`, `updated_at` = epoch milliseconds).
- `GET /v2/projects/{id}` → `{id, name, revision, updated_at, media_bytes, data}`.
- `PUT /v2/projects/{id}` `{name, data, base_revision}` → `{id, revision, updated_at}`. `base_revision` must be the
  stored revision (`null` for a new id), else 409 `{detail, revision}` (the stored one; `null` = deleted): two tabs
  never overwrite each other silently. A project's media are the keys of `data.media` (`{<sha256>: {type, size}}`);
  all must be stored, else 400 `{detail, missing: [sha256...]}`.
- `DELETE /v2/projects/{id}`; `POST /v2/projects/gc[?min_age_seconds=3600]` → `{removed, freed_bytes, kept}` removes media
  no project lists, older than the grace period (uploads of a save in flight are newer).
- `HEAD` / `GET /v2/media/{sha256}` (lowercase hex): the blob streamed from disk with its stored type,
  `ETag: "<sha256>"` (`If-None-Match` → 304) and `Cache-Control: … max-age=31536000, immutable` (`private` behind
  `ZONOS3_ACCESS_TOKEN`). `PUT /v2/media/{sha256}` raw bytes (type from `?type=` or `Content-Type`): streamed to a temp
  file while hashed, 400 `{detail, sha256: <actual>}` on a mismatch, 413 over the limit, 201 stored / 200 already
  there (body not read).
- `GET /v2/capabilities` → `projects: {enabled, max_media_bytes, max_project_bytes}`.

### Dubbing (Dub screen, `/v2/dub/*`)

The studio server also dubs (details in [laci_dub.md](laci_dub.md)): a video or audio reference → Qwen3-ASR transcript
and ForcedAligner words (or `ref_words` the studio already has) → segments at pauses with the SimAlign-projected
target text → per segment 10 LACI takes in neighbouring speaking-rate buckets, each held to the segment's length (LACI
duration target) → best fit placed on the timeline → optional AV-sync lip warp → MP4 mux. Every take is a
`/v2/generate` request on this server's scheduler (`laci=true` + duration target), so a dub uses the reference clone
(`voice: {mode: "source"}`) or any studio voice (`{mode: "studio", speaker_embedding_id, controls}`), per dub or per
segment, with per-segment direction (`segments[k].controls`) and pronunciation groups in target texts.

```bash
bash tools/setup_dub_align_env.sh   # once: .venv-dub-align (Qwen3-ASR/ForcedAligner, SimAlign) + .avsync (AV-sync)
STEER_BUNDLE=steering/studio_bundle_mect60k.json DUB=1 servers/sgl/scripts/serve.sh
```

`DUB=1` passes `--laci-heads` for the 9 target languages (the mect_60k heads in `laci_heads/`:
`zonos3_mini_mect60k_heads_zh.json` for `cmn`, `zonos3_mini_mect60k_heads.json` for the rest; `LACI_HEADS_DIR` /
`LACI_HEADS` override). The heads selected on the release LM also drive LACI on the edit checkpoint. With LACI off,
outputs do not depend on whether `--laci-heads` is passed.

#### Pace steering as a continuous speaking-rate bucket

The pace control (`delivery.tempo`, hidden L9.attn) can stand in for the speaking-rate condition row:

- **Every bucket has a tempo dose.** The doses that reproduce each of the row's ten buckets agree across English and
  French, at the same WER and identity as the buckets.
- **The dose is continuous and reaches further** than the ten buckets, in both directions; rate is monotone in the
  dose.
- **Leave the rate row unknown when steering.** With a bucket set, the row damps the steering.

#### Extended speaking-rate buckets

Dubs plan and retake over the row's ten buckets plus **tempo-steered buckets** that continue them: −1, −2, … below
bucket 0 and 10, 11, … above bucket 9, each the rate row unknown plus a `delivery.tempo` dose, only as far as measured
safe per target language. The kept scale is the bundle's `rate_extension` block (also
`steering/ears_styles_mect60k/eval/rate_extension.json`).

- **Nominal edges** continue the row's at its mean geometric step (×1.097): bucket 0 becomes [9.22, 10.12) B/s,
  −1 [8.41, 9.22), … and 9 becomes [21.21, 23.26), 10 [23.26, 25.52), …; a language's lowest / highest bucket stays
  open-ended.
- **Doses**: bucket −k / 9+k targets the achieved rate (vs the unsteered render) of the row's own achieved ladder
  continued k steps; the dose is interpolated on the measured dose curve of the language.
- **Kept buckets** are those whose achieved rate lands within one step of the target at an unchanged WER, collapse rate
  and identity, contiguous from the row:

| language | steered buckets |
|---|---|
| en | −7…−1, 10…12 |
| fr | −4…−1, 10…11 |
| de | −5…−1 |
| es | −4…−1, 10 |
| it | −4…−1 |
| pt | −3…−1, 10 |
| ja | −5…−1 |
| zh | −6…−1 |
| ko | −2…−1 |

- **Source voices use only the faster steered buckets** (`reference_sides: ["faster"]` in the bundle;
  `/v2/dub/capabilities` `buckets[].voice_modes`): on the slower side the source voice (reference mode, where the buckets
  were not validated) lost words or garbled. Studio voices use both sides.
- Languages without a steered bucket on a side, and servers whose bundle has no `rate_extension`, take exactly the
  row-only path.
