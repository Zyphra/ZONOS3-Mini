# LACI dubbing in the studio server (ZONOS3 Mini)

Drop an audio or video file (0.5–180 s) on the studio's Dub screen, check the Qwen3-ASR transcript (or bring word
timings the studio already has), write a target transcript (any of the 9 target languages) and get a dub in the
reference voice or in studio voices. Synthesis matches timing by speaking-rate condition only; for video input an
optional post-processing step (AV-sync) then time-warps the dub onto the speaker's lips.

## Setup (once)

```bash
bash tools/setup_dub_align_env.sh   # .venv-dub-align: qwen-asr 0.0.6 (transformers 4.57.6) + simalign; downloads the weights;
                                      # AV-sync at a pinned commit in .avsync/ with its SyncNet and S3FD models (re-runnable)
```

## Run

One server: the studio server (`python -m zonos3_sgl`, `zonos3_sgl/server/api_server.py`) hosts `/v2/dub/*`. `DUB=1` in
`servers/sgl/scripts/serve.sh` passes `--laci-heads` for every dub target language (the heads selected on lm/mect_60k in
`laci_heads/`: `zonos3_mini_mect60k_heads_zh.json` for `cmn`, `zonos3_mini_mect60k_heads.json` for the rest;
`LACI_HEADS_DIR` / `LACI_HEADS` override) and points the server at the alignment worker and AV-sync
(`ZONOS3_DUB_ALIGN_PYTHON`, `ZONOS3_DUB_AVSYNC_HOME`, `ZONOS3_DUB_ALIGN_DEVICE`; defaults: the repo's `.venv-dub-align`,
`.avsync`, `cuda`):

```bash
STEER_BUNDLE=steering/studio_bundle_mect60k.json DUB=1 servers/sgl/scripts/serve.sh      # open http://127.0.0.1:1919/ -> Dub
# pronunciation groups in target texts: SERVE_MODEL=<phoneme checkpoint> PHONEME_LEXICON=<lexicon dir>
```

Every take is a `/v2/generate` request on the server's scheduler (`api_server._dub_synthesize`): `laci=true`, the
take's LACI duration target (`TTSTokenizeMsg.laci_target_frames` / `laci_adjust_rate` -> `zonos3.tts.laci.DurationTarget`
in the scheduler), its speaking-rate bucket in the condition rows (the others unknown), the release sampling defaults,
speaker CFG 1 (LACI's constraint), and its voice. The LACI record comes back on the request's final `TTSAudioReply`.
At most twice `--max-running-requests` takes are in flight, so studio requests interleave with a dub. The alignment
worker (`model/src/zonos3/workers/dub_align.py`, own venv) starts on the first call that needs it (transcribe / analyze /
generate / lips) and again after it died; it shares the server's GPU.

Voices and direction (`zonos3/dub/runtime.py`): `voice: {mode: "source"}` (default) clones the reference speaker from
its own audio (raw reference prefix codes of the clip from the first source-voice segment, up to 20 s);
`{mode: "studio", speaker_embedding_id, controls?}` is a studio library / saved voice in `speaker_mode=embedding` with
Cast-bench controls (raw + hidden doses; the request's `X-TTS-Session-ID` reaches the session's saved voices). Each
segment may override the voice (`segments[k].voice`) and add its own direction (`segments[k].controls`, summed per id
with the voice's controls, as `/v2/edit`'s `region_controls`); speaker-embedding (raw) controls need a studio voice.
Controls resolve through the same `_apply_request_controls` / `resolve_controls` path as `/v2/generate`. Target
texts may carry pronunciation groups (U+E000 IPA U+E001) on phoneme-token checkpoints: the TTS reads the group, the
forced aligner its IPA, and the rate counts its symbols as 1.4 letters each (`edit_align.token_chars`). Rows echo
`voice` (effective) and `controls` (the segment's own).

## Pipeline

1. FuseAlign (the server's forced-alignment worker, `model/src/zonos3/workers/fusealign.py`) times the reference words:
   the whitespace words of the transcript, or in ja / zh / yue one per character or Latin run
   (`zonos3.tts.align_text.align_tokens`); the caller's `ref_words` are used as they are (no ASR, no forced alignment of
   the reference). The transcript is split into segments at pauses >= the threshold.
2. SimAlign (mBERT, windowed beyond 510 wordpieces) word-aligns reference and target text; the alignment projects each
   reference segment onto a contiguous span of target words (`zonos3/dub/plan.py: project_segments`). Segment target
   texts stay editable.
3. Per segment, candidates are synthesized in the neighbouring speaking-rate buckets (offsets 0, -1, +1, -2, +2; two
   seeds each) under LACI; FuseAlign times each take's words (its unplaced words do not count), and the take whose
   speech duration fits the reference best is kept and placed so its speech starts at the reference segment start. An
   overrun pushes the next segment later (reported as `shift`).

   LACI also holds each take to its segment's length: every take carries a `zonos3.tts.laci.DurationTarget`
   (server: `laci_target_frames` on the take's request; offline: `TTSLLM.generate(laci=True, laci_duration=...)`). When
   the alignment heads' reading position reaches the last text
   token, which happens about 0.2 s before the speech ends, the elapsed frames over the segment's frames estimate the
   take's duration ratio. The check fails a take that reads its text more than 10 % + 0.2 s early, or that has not read
   it 10 % + 0.2 s past the target; the ratio of an unfinished take is extrapolated from how much of the text it has
   read. A failing take is rolled back to its first row and regenerated with a new seed in the speaking-rate bucket
   that ratio asks for (`RateBuckets.retry_bucket`; the continuation's condition rows), at most twice per take and
   within LACI's retry budget. Mid-text positions are not judged: per token they are too uneven. Each candidate
   reports its LACI `retries` by kind and its `final_bucket`.

   With a duration target LACI's content checks follow the segment too. An early EOS (truncation) regenerates one
   speaking-rate bucket slower rather than reseeding in the same bucket: in the fastest bucket the model often ends a
   short text within a few frames. And a stall is caught once the reading position has not advanced for three text
   bins at the target pace (at least 1 s), then rolled back to where it stopped like a LACI stall: LACI's own limit
   (three bins at the observed pace, at least 3 s) outlasts most segments, so a stall used to show only as a late
   "too slow" retake. This check reads the smoothed reading position, not LACI's text bins, which in a short text follow
   the alignment heads' average pace (dub takes read about twice as fast).

   The metrics count retries twice: `laci_retries` / `retries_by_kind` over the kept takes, and `all_retries` /
   `all_retries_by_kind` / `exhausted_takes` over every candidate the request synthesized, where most retries happen.
   Each segment row reports its kept take's `retries` (by kind), `final_bucket` and `fit` (speech length over the
   segment's), and `start` / `end` on the dub timeline; `metrics` add `synthesis_s`, `align_s` and `lipsync_s`.

   Each segment row has a speaking-rate **bucket** select: *auto* plans the candidates around the bucket of the text's
   rate (above); a set bucket puts all 10 candidates in that bucket, and the duration check then only records misses,
   it never moves the bucket. Generate and Regenerate both use it.

   **Extended speaking-rate scale** ([guide: Extended speaking-rate buckets](guide.md#extended-speaking-rate-buckets)): where the steering bundle's
   `rate_extension` measured them safe for the target language, the row's buckets 0-9 continue with tempo-steered
   buckets below 0 (−1, −2, …) and above 9 (10, 11, …): the rate row unknown plus a calibrated `delivery.tempo` dose,
   added to the take's own controls. The auto bucket, the candidate offsets, pace retakes and the one-slower truncation
   retake all move on the language's scale (`plan.RateScale`, `laci.RateBuckets.first`); a retake across the row's end
   switches the condition row (row bucket <-> unknown) and the take's steering delta (`laci_rate_*` on the take's
   request: the scale and the delta in every bucket). Rows report `bucket` / `final_bucket` on that scale. Languages
   without steered buckets (and servers whose bundle has no `rate_extension`) take exactly the row-only path.

   Regenerate makes new takes for one segment with the given seed, text and bucket, and optionally a new `voice` and
   `controls` (omitted: the segment's; `controls: {}` clears its own direction); an empty text silences the segment.
   The other segments keep their takes, and the timeline is placed again, so later segments can shift.
4. For video input `POST /v2/dub/mux/{dub_id}` copies the video stream and muxes the dub as the audio track (ffmpeg).
5. **Lip-sync** (video input, on by default): [AV-sync](https://github.com/Ryu1845/AV-sync) ("Dynamic Temporal
   Alignment of Speech to Lips", Halperin et al.) warps the finished dub onto the speaker's lips.
   - The server finds the video's first face track (S3FD) and computes the SyncNet lip embeddings
     (`POST /v2/dub/lips/{ref_id}`); generate / regenerate with `lipsync: true` need it first.
   - After each Generate / Regenerate the dub's SyncNet audio embeddings over the face track are aligned to the lips by
     AV-sync's banded dynamic programming (modality `va`, up to ±2 s), and the path is smoothed as in `align_audio.py`.
     WSOLA (`plan.wsola`: 23 ms Hann frames at 50 % overlap, each taken within ±11.6 ms of its nominal source position
     where it best continues the previous frame) applies the warp to the 44.1 kHz dub. It copies waveform segments, so
     the pitch and the level stay without any phase reconstruction. Outside the face track the warp holds the offset of
     its nearer end (`plan.LipMap`), so the dub stays continuous.
   - The server bounds the path before WSOLA (`plan.guard_lip_frames`, `LIP_GUARD_MAX_RATE` 1.5,
     `LIP_GUARD_MAX_SHIFT_S` 0.4): every shift within 0.4 s, the local playback rate within [1/1.5, 1.5], and no
     advance at the track's first frame. A dub's mouth shapes do not match the original video's (another language),
     and the unbounded DP path can then stand the dub still or skip words; the guard keeps the intelligibility of the
     unwarped dub and part of the sync gain. `lipsync.guard` reports the bounds.
   - The guarded warp is then kept only when it lowers the mean SyncNet distance of lips and dub over the track
     (`sync_distance_after` < `sync_distance_before` - `plan.LIP_MIN_SYNC_GAIN`, 0). Otherwise generate / regenerate
     return the unwarped dub with `lipsync.applied: false` and `lipsync.reason` ("no sync gain: SyncNet distance
     … unwarped, … warped"); the segments then carry `lip_start` / `lip_end` = their placed speech span and lip
     factors of 1.
   - The response's `lipsync` reports the track span, the largest and mean shift, the largest stretch and squeeze
     (`max_stretch`, `max_squeeze`) and the mean warp (`mean_warp`) over the track, and the mean SyncNet distance of
     the lips and the dub before and after the warp. Segments carry `lip_start` / `lip_end` and how much the warp
     time-scaled their speech (`plan.lip_warp`, local time scale over 50 ms), all factors with 1 = untouched:
     `lip_scale` (warped / unwarped length), `lip_stretch` (largest local slow-down, 1.5 = played 1.5x slower),
     `lip_squeeze` (largest local speed-up) and `lip_warp` (geometric mean of the local factor). Local factors are
     capped at 10x; the cap means the warp stood the dub still (or jumped over it) there. The words are in
     warped time; segment placement metrics (`start_error`, …) stay those of the unwarped timeline.
   - The glue lives in `model/src/zonos3/workers/dub_avsync.py`: AV-sync's two entry points are scripts, so their face tracking / crop
     and path post-processing are ported there with the same constants; everything else is imported from the checkout.

## HTTP API

`GET /v2/dub/capabilities` (target languages with LACI heads, rate edges and bucket count, `buckets` (the extended scale:
`[{index, label, rate, low, high, steered, languages, doses?}]`, nominal B/s midpoint and edges, the target languages
that have the bucket, the tempo dose per language of a steered one), `lipsync`, `voice_modes`,
`phoneme_groups`, limits, `enabled`; `/v2/capabilities` carries `dub`), `POST /v2/dub/ref?name=` (raw audio bytes, or a
video whose first audio track is taken), `/v2/dub/transcribe {ref_id, ref_lang?}` (text, language and the FuseAlign
`words`), `/v2/dub/analyze {ref_id, ref_text | ref_words, ref_lang, target_text, target_lang, min_pause_s}` (segments
with `bucket_auto`), `/v2/dub/generate {ref_id, target_lang, seed, segments: [{ref_start, ref_end, target_text, bucket?,
voice?, controls?}], voice?, lipsync}`, `/v2/dub/regenerate {dub_id, segment, target_text, seed, bucket, voice?, controls?,
lipsync}` (new takes for one segment, the others kept; returns a new dub), `POST /v2/dub/lips/{ref_id}?name=` (raw video
bytes -> face track for lip-sync), `POST /v2/dub/mux/{dub_id}?name=` (raw video bytes -> MP4). generate / regenerate read
`X-TTS-Session-ID` for the session's saved voices; every route but capabilities and mux accepts `X-Keepalive: 1`
([api.md](api.md#headers)). The schemas are in [api/openapi.yaml](../api/openapi.yaml). Errors: 400 invalid input, 404
unknown `ref_id` / `dub_id` (the server keeps the last 8 of each), 422 schema violations (unknown fields included), 503
alignment worker failure. Tests: `model/tests/test_dub_plan.py` (planning), `servers/sgl/tests/test_dub_runtime.py`
(voices, controls, regenerate, `ref_words`, groups).
