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

The Zonos Voice Studio (Cast, Direct, Scene and the lite page) over the plain-PyTorch reference
implementation of ZONOS3 Mini, with every take generated on ZeroGPU. Speech editing, dubbing and
server-side projects are off here; projects stay in your browser.

The first start downloads about 10.5 GB of weights and builds the text-normalization grammars,
so it takes several minutes.

Source: `space/` in Zyphra/ZONOS3, deployed with `space/deploy.py`.
