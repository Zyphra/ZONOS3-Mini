---
title: ZONOS3 Mini
sdk: gradio
sdk_version: 6.17.3
python_version: "3.12.12"
app_file: app.py
pinned: false
short_description: Multilingual TTS with voice cloning and steering controls
startup_duration_timeout: 1h
models:
  - Zyphra/ZONOS3_Mini_release_ckpts
---

# ZONOS3 Mini

The Zonos Voice Studio (Cast, Direct, Edit and the lite page) over the plain-PyTorch reference
implementation of ZONOS3 Mini, with every take and every speech edit generated on ZeroGPU.
Pronunciations can be typed in X-SAMPA (the release phoneme tokens), and "Phonemize selection"
looks words up in the gold pronunciation lexicon. Projects stay in your browser.

Words missing from the lexicon are guessed with espeak-ng (GPL-3.0, through `espeakng-loader`).

The first start downloads about 10.5 GB of weights and builds the text-normalization grammars,
so it takes several minutes.

Source: `servers/space/` in Zyphra/ZONOS3, deployed with `servers/space/deploy.py`.
