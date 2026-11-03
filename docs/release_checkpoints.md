# ZONOS3 Mini release checkpoints on Hugging Face
## Checkpoints from Hugging Face by default

The release checkpoints live in the private repo **Zyphra/ZONOS3_Mini_release_ckpts** (language models `lm/rate30_20k_r2grpo`
(the default) and `lm/mect_60k` (the previous default), the dual
codec `autoencoder/generator.pt` + its semantic teacher `autoencoder/teacher`, the two causal MR iSTFT decoder heads
`autoencoder/heads/{full_mr,keep2_plain}`, the `ownmect_s2` speaker encoder, registries, `manifest.json` with sha256s).
`zonos3.utils.release` resolves `hf://owner/repo/path` locations (files via `hf_hub_download`, folders via `snapshot_download`,
into `HF_HOME`; set `HF_TOKEN` or log in once while the repo is private). Defaults: `--model-path` =
`hf://Zyphra/ZONOS3_Mini_release_ckpts/lm/rate30_20k_r2grpo`; the registries in `model/src/zonos3/registries/`
(`codec_registry_keep2_plain.json` (the default: keep2 head), `codec_registry.json` (full-size head), `speaker_encoder_registry.json`) point at
the same repo (`ZONOS3_CODEC_REGISTRY` / `ZONOS3_SPEAKER_ENCODER_REGISTRY` override them); the tokenizer path in a
checkpoint may be relative to its folder. Local paths pass through unchanged.

The codec code is vendored in `model/src/zonos3/codec`.

The default LM is zonos3-rate30-20k-r2grpo (`lm/rate30_20k_r2grpo`, added in revision
`4d1d9efca3fb58ca880e028586d9f98bbc002288`): a sibling fine-tune of mect_60k's base (rate30 20k, tail DPO, round-2
GRPO, step 587) that differs from it only in the condition rows: `params.json` adds `condition_bucket_counts: [10, 10, 30]`,
so table i has counts[i]+1 rows (the last = unknown) and speaking rate has 29 edges (ids 0..29, 30 = unknown; loudness
and fmax 0..9, 10 = unknown). Its `conversion.json` (`minisgl-zonos2`, bf16) records the source checkpoint and step.

The previous release LM is srcmix-cond3-mect-60k (`lm/mect_60k`, uniform 10 buckets per table): `params.json` has `model_type: "zonos2"` (the checkpoint format
name), `audio_codec: "opus_dual_3n_600k_sched600k_r4096_step600000"`, `speaker_embedding_codec: "ownmect_s2"` (the MECT-B2
speaker encoder), `attention_pattern: "gdn3_global1"` and `residual_type: "full_attnres"` (the same for both LMs). Its
`conversion.json` records the source checkpoint and step that `tools/convert_lingua_checkpoint.py` converted. The
steering bundle's hidden-state directions, the emotion and accent directions and the LACI heads were fitted on mect_60k.
