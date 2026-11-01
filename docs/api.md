# API reference

The HTTP endpoints of `python -m zonos3` (`scripts/serve.sh`). Feature-specific endpoints (speech editing, steering, studio, dubbing) are described in [guide.md](guide.md) and [laci_dub.md](laci_dub.md).

## `POST /tts/generate`

Full-featured TTS endpoint with streaming support.

**Request body:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | required | Text to synthesize |
| `language` | string | `en_us` | Text-normalization language: `en_us`, `en_gb`, `fr_fr`, `de`, `es`, `it`, `pt_br`, `ja`, `cmn`, `ko` |
| `text_normalization` | bool | `true` | Verbalize numbers, dates and currency before synthesis (`false` = raw byte tokenization) |
| `temperature` | float | `0.75` | Sampling temperature |
| `topk` | int | `80` | Top-k sampling |
| `top_p` | float | `0.69` | Nucleus (top-p) sampling threshold; `0` disables |
| `min_p` | float | `0.0` | Min-p probability filter; `0` disables |
| `max_tokens` | int \| null | model max | Maximum audio tokens. Omit or set `null` to use the model context limit; long prompts are clamped to remaining context. |
| `fade_out_ms` | float | `0.0` | Cosine fade-out applied to the audio tail; `0` disables |
| `repetition_window` | int | `96` | Recent generated frames to check per codebook; `0` disables |
| `repetition_penalty` | float | `2.4` | Per-codebook repetition penalty strength; `1.0` disables |
| `repetition_codebooks` | int | `6` | Number of codebooks from CB0 upward to penalize; negative means all |
| `seed` | int \| null | `null` | Random seed for reproducibility |
| `speaking_rate_enabled` | bool | `false` | Set `true` to use model speaking-rate conditioning when another speaking-rate field is present |
| `speaking_rate_bucket` | int \| null | `null` | Exact model speaking-rate bucket to prepend before text |
| `speaking_rate` | float \| null | `null` | Target speaking rate in cleaned UTF-8 bytes per second; mapped to a bucket |
| `speed` | float \| null | `null` | OpenAI-style multiplier; `1.0` maps to the model's neutral speaking-rate bucket |
| `quality_enabled` | bool | `true` | Enable quality-bin conditioning on supported models |
| `quality_buckets` | object \| list \| null | `{"trailing_silence_s": 3}` | Per-feature quality bucket indices (keyed by feature name, or a list in feature order) |
| `quality_values` | object \| list \| null | `null` | Raw quality metric values, mapped to buckets server-side (alternative to `quality_buckets`) |
| `clean_speaker_background` | bool | `false` | Mark the reference voice as having a clean background (supported models) |
| `accurate_mode` | bool | `true` | `true` = accurate mode (closer voice match), `false` = expressive mode |
| `emotion_enabled` | bool | `false` | Enable emotion-control conditioning (requires loaded emotion directions) |
| `emotion_sliders` | object \| null | `null` | Per-emotion weights, e.g. `{"happy": 1.0, "sad": 0.5}` (available names from `/tts/capabilities`) |
| `emotion_valence` | float | `0.0` | Valence axis (−1 negative … +1 positive) |
| `emotion_arousal` | float | `0.0` | Arousal axis (−1 calm … +1 excited) |
| `emotion_strength` | float | `1.0` | Multiplier on the calibrated strength; `1.0` = calibrated, higher exaggerates |
| `emotion_cfg_scale` | float | `1.0` | Emotion guidance; `1.0` = off, ~1.5 strongly amplifies emotion (best with expressive mode), ~2× compute |
| `accent` | string \| null | `null` | Accent adapter name from `/tts/capabilities` `accents` (server started with `--accent-adapters`); `null` serves the base model, an unknown name returns 400 |
| `accent_strength` | float | `1.0` | Adapter strength α: the request runs base + α × adapter |
| `controls` | object \| null | `null` | Steering-bundle controls (`--steer-bundle`): `{"<set>.<direction>": dose}`, ids from `GET /tts/controls`. Speaker-embedding (`raw`) directions need `speaker_mode` embedding; cannot be combined with the `emotion_*` fields |
| `control_spans` | list \| null | `null` | Hidden-space controls over text: `[{"start": 10, "end": 31, "controls": {...}}]`, character offsets into `text` as sent (nonverbal markup included), non-overlapping |
| `control_span_rate` | float \| null | `null` | Frames per character used to place the spans (default 1.0; measure it from an earlier take); nonverbal markers count as no characters (see [guide: Nonverbal markup and tag CFG](guide.md#nonverbal-markup-and-tag-cfg)) |
| `control_span_follow` | `"rate"` \| `"reading"` | `"rate"` | `"reading"`: the spans follow the model's reading position (LACI alignment heads of `language`, see [guide: Span steering by reading position](guide.md#span-steering-by-reading-position)) instead of `control_span_rate`; without heads for the language the rate schedule runs. The `X-Control-Span-Follow` response header (requests with spans) says which one ran |
| `control_span_lead` | float \| null | `null` | `"reading"`: a span ramps in once the reading is this many text tokens before its first token (0–8, default 1; 2 steers more of the span's words and leaks more) |
| `control_span_lag` | float \| null | `null` | `"reading"`: a span ramps out once the reading is this many tokens past its last token (−4–8, default 0) |
| `tag_cfg_scale` | float | `1.0` | Nonverbal-tag guidance (see [guide: Nonverbal markup and tag CFG](guide.md#nonverbal-markup-and-tag-cfg)): finite `1.0`–`4.0`, `1.0` = off; above `1.0` needs a recognized tag in `text` and `speaker_cfg_scale` = `emotion_cfg_scale` = 1, refused with `laci`; ~2× compute |
| `stream` | bool | `true` | Stream audio chunks |

On a phoneme-token checkpoint `text` may carry pronunciation groups,
U+E000 IPA U+E001 (see [guide: Pronunciation input](guide.md#pronunciation-input-phoneme-tokens)); a malformed group or a symbol
outside the phone inventory returns 400, as does a group sent to a checkpoint
without phoneme tokens. `/v1/audio/speech` `input` and `/tts/edit` `target_text`
follow the same rule. `text` may also carry nonverbal markup (`<laugh/>`,
`<shouting>…</shouting>`), read by NV fine-tunes and as plain text by other checkpoints.

**Response:** Raw PCM audio (`audio/pcm`, float32, 44.1 kHz, mono). Headers include `X-Audio-Sample-Rate`, `X-Audio-Channels`, `X-Audio-Format`.

## `POST /tts/phonemize`

Available when the server runs with `--tts-phoneme-lexicon`. Body:
`{"text": "...", "language": "en_us", "guess": true}` (a Text Norm language code,
or a lexicon code such as `rus`). Returns `{"language": "eng", "segments": [...]}`;
the segment texts concatenate back to the input, and each word segment has
`status` `ok` (with `ipa`), `guessed` (espeak-ng, with `ipa`), `ambiguous`,
`unknown`, `skipped` (Japanese/Chinese characters outside any multi-character
word) or, on a phoneme-token checkpoint, `unsupported` (a symbol the model has no
token for). `GET /tts/capabilities` reports `phonemize_enabled`,
`phonemize_languages`, `phonemize_guess_languages` and `phoneme_tokens`
(`null`, or the group markers and the phone symbols).

## `POST /v1/audio/speech`

OpenAI-compatible endpoint.

**Request body:**

```json
{
  "model": "zonos3",
  "input": "Hello world",
  "voice": "alloy",
  "response_format": "pcm"
}
```

For speaking-rate-enabled checkpoints, set `speaking_rate_enabled` to `true`
and use `speaking_rate_bucket` for exact bucket control, `speaking_rate` for
bytes-per-second control, or `speed` for OpenAI-style multiplier control.
`tag_cfg_scale` (same rules as on `/tts/generate`) is accepted here too.
