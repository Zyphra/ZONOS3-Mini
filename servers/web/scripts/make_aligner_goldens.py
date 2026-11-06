#!/usr/bin/env python3
"""Goldens for the in-browser FuseAlign port (harness/checks/aligner.ts, tests/aligner.test.ts).

Short 16 kHz clips cut from default_voices/ with their transcripts, aligned by the PyTorch reference
(fusealign_hub.load_aligner on CPU, batch size 1: FuseAlign fp32, Dia bf16) from the local
Zyphra/fusealign safetensors. Writes:
  harness/goldens/aligner/<clip>.s16    the exact 16 kHz input, int16 little-endian (the reference aligns s16 / 32768)
  harness/goldens/aligner/goldens.json  per clip: tokens and the reference word alignments
  tests/fixtures/aligner-host.json      torchaudio 44.1 -> 16 kHz resampling and fill_unplaced cases

Run from servers/web:  uv run --package zonos3-tools python scripts/make_aligner_goldens.py
"""
import argparse
import json
import os
import subprocess

import numpy as np
import torch
import torchaudio
from fusealign_hub import fill_unplaced, load_aligner
from fusealign_hub.aligner import CHECKPOINT_FILE, HF_REPO
from huggingface_hub import try_to_load_from_cache

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(WEB))

# (clip, source, start s, end s, tokens). Transcripts as heard; the "—" token is punctuation only, so it takes the
# zero-width placement at the previous word's end without reaching the model.
CLIPS = [
    ("american_male", "AmericanMale.mp3", 0.0, 10.65,
     "Hi, this is Brad Geiger. I'm an American cartoonist best known for my daily comic strip *Evil Inc.*, "
     "along with websites like Webcomics.com."),
    ("british_female", "BritishFemale.mp3", 0.0, 5.6,
     "Hello, I'm Julie Etchingham, — and I present News at Ten for ITV."),
    ("british_female_2", "BritishFemale.mp3", 6.6, 13.6,
     "One Ash Wednesday, a couple of years ago, I was looking for somewhere to go to mass before I read the "
     "ten o'clock news."),
]


def decode44k(path):
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-ac", "1", "-ar", "44100", "-f", "f32le",
                          "-"], check=True, capture_output=True).stdout
    return torch.from_numpy(np.frombuffer(raw, dtype="<f4").copy())


def r9(x):
    return float(f"{x:.9g}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None, help="fusealign-hub/1 safetensors (default: the local HF cache copy)")
    a = ap.parse_args()
    ckpt = a.checkpoint or try_to_load_from_cache(HF_REPO, CHECKPOINT_FILE)
    if not isinstance(ckpt, str):
        raise SystemExit(f"{HF_REPO}/{CHECKPOINT_FILE} is not in the Hugging Face cache; pass --checkpoint")
    out = os.path.join(WEB, "harness", "goldens", "aligner")
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(0)
    aligner = load_aligner(ckpt, device="cpu", batch_size=1)

    clips = []
    for name, src, start, end, text in CLIPS:
        wav16 = torchaudio.functional.resample(decode44k(os.path.join(REPO, "default_voices", src)), 44100, 16000)
        cut = wav16.numpy()[round(start * 16000):round(end * 16000)]
        s16 = np.clip(np.round(cut * 32768), -32768, 32767).astype("<i2")
        s16.tofile(os.path.join(out, name + ".s16"))
        pcm = torch.from_numpy(s16.astype(np.float32) / np.float32(32768))
        tokens = text.split()
        words = aligner.align(pcm, 16000, tokens)
        clips.append({
            "name": name, "file": name + ".s16", "samples": int(s16.size), "source": f"default_voices/{src}",
            "source_start": start, "tokens": tokens,
            "words": [{"word": w.word, "start": r9(w.start), "end": r9(w.end), "presence": r9(w.presence),
                       "placed": bool(w.placed)} for w in words],
        })
        print(name, f"{s16.size / 16000:.2f} s", f"{len(tokens)} tokens",
              " ".join(f"{w.word}[{w.start:.3f},{w.end:.3f}]" for w in words[:6]), "...")
    with open(os.path.join(out, "goldens.json"), "w") as f:
        json.dump({"reference": f"fusealign_hub.load_aligner({CHECKPOINT_FILE}, device='cpu', batch_size=1)",
                   "clips": clips}, f, indent=1, ensure_ascii=False)

    # Host-side pieces: torchaudio resampling (44.1 kHz and an odd rate) and fill_unplaced.
    rng = np.random.default_rng(7)
    resample = []
    for sr, n in ((44100, 1500), (22050, 700), (48000, 1200)):
        x = (rng.standard_normal(n) * 0.3).astype(np.float32)
        y = torchaudio.functional.resample(torch.from_numpy(x), sr, 16000).numpy()
        resample.append({"sr": sr, "x": [r9(v) for v in x], "y": [r9(v) for v in y]})
    fill_cases = [
        ([None, (0.5, 0.9), None, None, (2.0, 2.4), None], 3.0),
        ([None, None], 1.5),
        ([(0.1, 0.3), (0.2, 0.25), None], 1.0),
        ([], 2.0),
    ]
    fill = [{"intervals": [list(iv) if iv else None for iv in ivs], "duration": d,
             "filled": [list(iv) for iv in fill_unplaced(ivs, d)]} for ivs, d in fill_cases]
    with open(os.path.join(WEB, "tests", "fixtures", "aligner-host.json"), "w") as f:
        json.dump({"resample": resample, "fill_unplaced": fill}, f)
    print("goldens:", out)


if __name__ == "__main__":
    main()
