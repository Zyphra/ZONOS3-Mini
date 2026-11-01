#!/usr/bin/env python3
"""Emit the tiny steering bundle GGUF used by CI next to tests/fixtures/dummy-zonos3.gguf.

Matches the dummy backbone: n_embd 32, 4 layers (Full AttnRes sources 0..8), speaker_dim 192.
  who   raw  (192)   axes gender, age
  style hidden src 3 (32)  named whisper, axis tempo
  mood  hidden src 5 (32)  axis valence
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

UI = {
    "groups": [{"id": "who", "label": "Who", "hue": "c1"}, {"id": "delivery", "label": "Delivery", "hue": "c5"},
               {"id": "mood", "label": "Mood", "hue": "c6"}],
    "controls": [
        {"id": "gender", "label": "Gender", "type": "bipolar", "group": "who", "poles": ["masculine", "feminine"],
         "map": {"-": {"who.gender": -1}, "+": {"who.gender": 1}}, "range": [-3, 3], "safe": [-2, 2],
         "collapse": None, "spannable": False, "identity": [[0, 0.95], [2, 0.6]],
         "effect": "Shifts perceived speaker gender."},
        {"id": "pace", "label": "Pace", "type": "bipolar", "group": "delivery", "poles": ["slower", "faster"],
         "map": {"-": {"style.tempo": -1}, "+": {"style.tempo": 1}}, "range": [-3, 3], "safe": [-2, 2],
         "collapse": [-3, None], "spannable": True, "identity": None, "effect": "Speaking rate."},
        {"id": "whisper", "label": "Whisper", "type": "unipolar", "group": "delivery",
         "map": {"+": {"style.whisper": 1}}, "range": [0, 4], "safe": [0, 2], "collapse": None,
         "spannable": True, "identity": None, "effect": "Breathy, unvoiced delivery."},
        {"id": "valence", "label": "Valence", "type": "bipolar", "group": "mood", "poles": ["negative", "positive"],
         "map": {"-": {"mood.valence": -1}, "+": {"mood.valence": 1}}, "range": [-6, 6], "safe": [-4, 4],
         "collapse": None, "spannable": True, "identity": None, "effect": "Emotional valence."},
    ],
    "accents": [],
    "presets": {"Test persona": {"gender": 1, "pace": -1}},
    "preview_text": "Hello there.",
    "request_defaults": {"speaker_mode": "embedding", "speaker_cfg_scale": 1.5, "topk": 40},
    "identity_threshold": 0.8,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(HERE.parents[2] / "tools"))
    import convert_gguf

    rng = np.random.default_rng(1234)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        sets = {}

        def make_set(name, space, dim, scale, entries, source=None):
            d = tmp / name
            d.mkdir()
            manifest = {"dim": dim, "space": space, "directions": {}}
            if source is not None:
                manifest["source_index"] = source
            for dname, kind in entries:
                np.save(d / f"{dname}.npy", (rng.standard_normal(dim) * scale).astype(np.float32))
                manifest["directions"][dname] = {"file": f"{dname}.npy", **({"kind": kind} if kind else {})}
            (d / "manifest.json").write_text(json.dumps(manifest))
            sets[name] = name

        make_set("who", "raw", 192, 0.5, [("gender", "axis"), ("age", "axis")])
        make_set("style", "attnres_source", 32, 2.0, [("whisper", None), ("tempo", "axis")], source=3)
        make_set("mood", "attnres_source", 32, 2.0, [("valence", "axis")], source=5)
        bundle = tmp / "bundle.json"
        bundle.write_text(json.dumps({"name": "dummy-steer", "sets": sets,
                                      "timing": {"style.tempo": [[-2, 2.0], [0, 1.0], [2, 0.5]]}, "ui": UI}))
        convert_gguf.main(["steer", str(bundle), "-o", str(args.out)])


if __name__ == "__main__":
    main()
