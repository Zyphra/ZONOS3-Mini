# API reference

Both servers — the CUDA server (`servers/sgl/scripts/serve.sh`, `python -m zonos3_sgl`) and the ggml server
(`servers/ggml`, `zonos3-server`) — implement one HTTP API. The schema of every request and response is
[servers/api/openapi.yaml](../servers/api/openapi.yaml) (bundled into `servers/api/openapi.json`); this page is the overview. Feature-specific
behavior (speech editing, steering, studio projects, plugins) is described in [guide.md](guide.md).
`servers/api/tests/conformance` checks a running server against the spec:
`uv run --package zonos3-api pytest servers/api/tests/conformance --base-url http://127.0.0.1:1919`.

## Routes

The native API lives under `/v2/`; the OpenAI-compatible routes stay under `/v1/`. Every other path belongs to Zonos
Voice Studio.

| Route | Purpose |
| --- | --- |
| `GET /v2/health` | `{"status": "ok"}` |
| `GET /v1/models` | OpenAI model list (one model, `id` `zonos3`) |
| `POST /v1/audio/speech` | OpenAI-compatible speech (`SpeechRequest`) |
| `POST /v2/generate` | Native TTS (`GenerateRequest`), streaming or buffered |
| `GET /v2/capabilities` | Feature flags of the loaded checkpoint and server |
| `GET /v2/controls` | The steering bundle's controls document, each set's `calibration` included (404 without `--steer-bundle`) |
| `GET\|POST /v2/speakers` | Default voices plus the session's cached voices (each with its `disabled_controls`); cache a clip or an embedding |
| `GET /v2/speakers/{id}/preview`, `/embedding` | A voice's audio; its speaker embedding |
| `POST /v2/phonemize` | Gold-lexicon IPA for the words of a text (404 without a lexicon) |
| `POST /v2/edit/transcribe`, `/v2/edit/align`, `/v2/edit` | Speech editing ([guide](guide.md#speech-editing)) |
| `/v2/<plugin>/*`, `/<plugin>` | Server plugins mounted with `--plugins` (listed in `plugins` of the capabilities; [guide](guide.md#plugins)) |
| `GET /v2/projects`, `POST /v2/projects/gc`, `GET\|PUT\|DELETE /v2/projects/{id}` | Studio projects ([guide](guide.md#projects-on-the-server)) |
| `GET\|HEAD\|PUT /v2/media/{sha256}` | Content-addressed studio media |

**Studio.** `GET /` returns the studio's `index.html` (`Cache-Control: no-store`); `GET /<path>` returns a file of the
studio build (`studio/dist`, `--studio-dir`), else 404 `{"detail": "Not Found"}`. Hashed `assets/` are immutable;
pages, `sw.js` and `manifest.webmanifest` are `no-store`; every other file (icons, `web-engine/`, `theme-boot.js`) is
`no-cache` (revalidated).
There is no SPA fallback (the studio routes with the URL hash), and `/v1/…`, `/v2/…` are never served from the studio
dir. Without a build, `GET /` is a 404 that names `pnpm --dir studio build`.

## Status codes

- **422** `{"detail": [{"loc": [...], "msg": ..., "type": ...}]}`: anything the schema expresses — types, enums,
  ranges, required fields and **unknown fields** (every request object is closed, so a misspelled or removed field is
  an error, not a no-op).
- **400** `{"detail": "..."}`: state- or semantics-dependent errors — an unknown voice, a feature this server or
  checkpoint lacks, a CFG combination, LACI without heads for the language, `stream: true` with `format: "wav"`.
- **404**: unknown resources on resource paths (speakers, projects, media) and features the server was not started
  with (`/v2/controls`, `/v2/phonemize`).
- Project errors add `missing`, `revision` or `sha256` to the 400 / 409 body.

## Headers

| Header | Direction | Meaning |
| --- | --- | --- |
| `X-TTS-Session-ID` | request | Scopes the speaker cache: `POST /v2/speakers` writes into the session (required there), reads see the session's voices plus the defaults |
| `X-Keepalive: 1` | request | On `/v2/edit*` and the POST paths a plugin lists: 200 at once, a space byte every 10 s, then the JSON body; errors arrive in the body as `{detail, status_code}` |
| `Authorization: Bearer <token>` | request | Needed when the server runs with `ZONOS3_ACCESS_TOKEN` (alternatives: the `zonos3_access` cookie, set once by `GET /?access_token=<token>`); a failure is 403 `Access denied` |
| `X-Audio-Sample-Rate`, `X-Audio-Channels`, `X-Audio-Format` | response | `44100`, `1`, `float32` on PCM audio |
| `X-Seed` | response | The seed used (drawn uniformly in 0..2³¹−1 when `seed` is null) |
| `X-Control-Span-Follow` | response | `rate` or `reading`: which schedule placed `control_spans` (sent when spans are set) |

CORS allows every origin and exposes the response headers above.

## `POST /v2/generate`

`text` is required; every other field has a default (see `GenerateRequest` in the spec). The common ones:

| Field | Default | Meaning |
| --- | --- | --- |
| `language` | `en_us` | Text-normalization language: `en_us`, `en_gb`, `fr_fr`, `de`, `es`, `it`, `pt_br`, `ja`, `cmn`, `ko` (the `server` rows of `model/src/zonos3/registries/languages.json`) |
| `text_normalization` | `true` | Verbalize numbers, dates and currency first |
| `stream` | `true` | Chunked PCM; `stream_decode` (`prefix` / `whole`, null = server default) picks the decode |
| `stream_codebooks`, `stream_codebooks_scope` | `1`, `first` | Prefix decode (every server): decode a frame once its first k codebooks arrived (missing residual codebooks left out), first audio up to 9 rows earlier; scope `first` (default) only for the first chunk, `all` for every chunk; k = n_codebooks waits for complete frames (defaults: `generation.json` `stream_codebooks`) |
| `format` | `pcm` | `pcm`: float32 LE mono 44.1 kHz; `wav`: buffered 16-bit WAV (samples NaN → 0, clipped to [-1, 1], × 32767, rounded half to even; the same encoder in every server), needs `stream: false` |
| `seed` | `null` | 0..4294967295; echoed in `X-Seed` |
| `temperature`, `topk`, `top_p`, `min_p` | `0.75`, `80`, `0.69`, `0` | Sampling (the defaults of `model/src/zonos3/registries/generation.json`) |
| `repetition_window`, `repetition_penalty`, `repetition_codebooks` | `96`, `2.4`, `6` | Per-codebook repetition penalty |
| `max_tokens` | `null` | Frame budget (null = the server's: the checkpoint's context on sgl, `--max` on the GGUF servers); shrunk to the context the prompt leaves |
| `speaker_audio_base64` / `speaker_embedding_base64` / `speaker_embedding_id` / `speaker_reference_codes` | `null` | The voice: a clip, a `.npy`/`.npz` embedding, a speaker id, or pre-encoded reference codes; `speaker_mode` picks embedding or raw-reference cloning on checkpoints with both |
| `speaker_blend_embedding_id_a`, `_b`, `speaker_blend_t` | `null` | Slerp between two voices |
| `speaker_cfg_scale` | `1.0` | Reference CFG (1 = off) |
| `condition_buckets` / `condition_values` | `null` | Condition rows (`loudness`, `fmax`, `speaking_rate`) by bucket or raw value |
| `silence_prefill_frames` | `null` | Codec-matched silence frames prefilled before the audio: null = the server default (3), 0 = off, at most 30 (more is a 400) |
| `accent`, `accent_strength` | `null`, `1.0` | Accent adapter from `/v2/capabilities` `accents` |
| `controls`, `control_spans`, `control_span_*` | `null` | Steering-bundle controls, whole-utterance or over character spans ([guide](guide.md#steering-controls)) |
| `controls_cfg_scale` | `1.0` | Guidance on the controls' hidden-space delta, embedding mode (1 = off; [guide](guide.md#steering-controls)) |
| `tag_cfg_scale` | `1.0` | Nonverbal-tag guidance, 1–4 ([guide](guide.md#nonverbal-markup-and-tag-cfg)) |
| `laci` | `false` | Alignment-guarded decoding ([features](features.md#laci-alignment-guarded-decoding)) |
| `headroom_db`, `fade_out_ms`, `decode_context` | registry, `0`, `false` | Output gain, tail fade, decoding behind the reference context |

On a phoneme-token checkpoint `text` may carry pronunciation groups, U+E000 IPA U+E001 (see
[guide: Pronunciation input](guide.md#pronunciation-input-phoneme-tokens)); a malformed group or a symbol outside the
phone inventory returns 400, as does a group sent to a checkpoint without phoneme tokens. `/v1/audio/speech` `input`
and `/v2/edit` `target_text` follow the same rule. `text` may also carry nonverbal markup (`<laugh/>`,
`<shouting>…</shouting>`), read by NV fine-tunes and as plain text by other checkpoints.

```bash
# Streamed float32 PCM with a default voice, played as it arrives
VOICE=$(curl -s http://127.0.0.1:1919/v2/speakers | python3 -c 'import json, sys; print(json.load(sys.stdin)["speakers"][0]["id"])')
curl -sN http://127.0.0.1:1919/v2/generate -H 'Content-Type: application/json' \
  -d "{\"text\": \"Hello from ZONOS3.\", \"speaker_embedding_id\": \"$VOICE\", \"seed\": 7}" \
  | ffplay -autoexit -f f32le -ar 44100 -ch_layout mono -

# One WAV file
curl -s http://127.0.0.1:1919/v2/generate -H 'Content-Type: application/json' -D - \
  -d "{\"text\": \"Hello from ZONOS3.\", \"speaker_embedding_id\": \"$VOICE\", \"stream\": false, \"format\": \"wav\"}" \
  --output hello.wav
```

## `POST /v1/audio/speech`

OpenAI-compatible: `model` (any value) and `input` are required. `voice` is a speaker id of `GET /v2/speakers` (an
unknown id is 400 `unknown voice '<v>'`, and it cannot be combined with any `speaker_*` field); `response_format` is
`pcm` (streamed float32, the default) or `wav` (buffered). Every `GenerateRequest` field except `text`, `stream` and
`format` is accepted as well; OpenAI's `speed` is not (422).

```bash
curl -s http://127.0.0.1:1919/v1/audio/speech -H 'Content-Type: application/json' \
  -d "{\"model\": \"zonos3\", \"input\": \"Hello world\", \"voice\": \"$VOICE\", \"response_format\": \"wav\"}" \
  --output speech.wav
```

## `POST /v2/phonemize`

Available when the server runs with `--tts-phoneme-lexicon`. Body:
`{"text": "...", "language": "en_us", "guess": true}` (a Text Norm language code,
or a lexicon code such as `rus`). Returns `{"language": "eng", "segments": [...]}`;
the segment texts concatenate back to the input, and each word segment has
`status` `ok` (with `ipa`), `guessed` (espeak-ng, with `ipa`), `ambiguous`,
`unknown`, `skipped` (Japanese/Chinese characters outside any multi-character
word) or, on a phoneme-token checkpoint, `unsupported` (a symbol the model has no
token for). `GET /v2/capabilities` reports `phonemize_enabled`,
`phonemize_languages`, `phonemize_guess_languages` and `phoneme_tokens`
(`null`, or the group markers and the phone symbols).
