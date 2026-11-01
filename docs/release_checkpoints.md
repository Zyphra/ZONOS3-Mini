# ZONOS3 Mini release checkpoints on Hugging Face
## Checkpoints from Hugging Face by default

The release checkpoints live in the private repo **Zyphra/ZONOS3_Mini_release_ckpts** (language model `lm/mect_60k`, the dual
codec `autoencoder/generator.pt` + its semantic teacher `autoencoder/teacher`, the two causal MR iSTFT decoder heads
`autoencoder/heads/{full_mr,keep2_plain}`, the `ownmect_s2` speaker encoder, registries, `manifest.json` with sha256s).
`zonos3.utils.release` resolves `hf://owner/repo/path` locations (files via `hf_hub_download`, folders via `snapshot_download`,
into `HF_HOME`; set `HF_TOKEN` or log in once while the repo is private). Defaults: `--model-path` =
`hf://Zyphra/ZONOS3_Mini_release_ckpts/lm/mect_60k`; `registries/codec_registry.json` (full-size head) /
`registries/codec_registry_keep2_plain.json` / `registries/speaker_encoder_registry.json` point at the same repo
(`ZONOS3_CODEC_REGISTRY` / `ZONOS3_SPEAKER_ENCODER_REGISTRY` override them); the tokenizer path in a checkpoint may be
relative to its folder. Local paths pass through unchanged.

The codec code is vendored in python/zonos3/codec.

The release LM is srcmix-cond3-mect-60k (`lm/mect_60k`): `params.json` has `model_type: "zonos2"` (the checkpoint format
name), `audio_codec: "opus_dual_3n_600k_sched600k_r4096_step600000"`, `speaker_embedding_codec: "ownmect_s2"` (the MECT-B2
speaker encoder), `attention_pattern: "gdn3_global1"` and `residual_type: "full_attnres"`. Its `conversion.json` records
the source checkpoint and step that `scripts/convert_lingua_checkpoint.py` converted.
