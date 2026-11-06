"""python -m zonos3.reference --model <dir|hf://...> (--text T | --request req.json) -o out.wav [options]

Synthesizes one v2 GenerateRequest with the plain-PyTorch reference and writes 16-bit mono 44.1 kHz WAV.
``--dump-prompt ids.npy`` writes the prompt rows (float32 [rows, n_codebooks + 1]) and the ``ids.cond.json`` sidecar of
``zonos3-cli --build-prompt``; a CFG twin's rows go to ``ids.twin.npy`` / ``ids.twin.cond.json``. With
``--dump-prompt`` alone nothing is generated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from zonos3.generation import SAMPLE_RATE
from zonos3.tts.audio import write_pcm16_wav


def _cond_path(ids_path: Path) -> Path:
    return ids_path.with_suffix(".cond.json") if ids_path.suffix == ".npy" else Path(str(ids_path) + ".cond.json")


def _dump(path: Path, rows: torch.Tensor, cond: dict) -> None:
    np.save(path, rows.to(torch.float32).numpy())
    _cond_path(path).write_text(json.dumps(cond) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m zonos3.reference", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="LM checkpoint directory or hf:// folder")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--request", help="v2 GenerateRequest JSON file")
    parser.add_argument("-o", "--out", help="output WAV")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("f32", "bf16"), default="f32")
    parser.add_argument("--steer-bundle")
    parser.add_argument("--accent-adapters")
    parser.add_argument("--phoneme-rows", help="zonos-text-extra/1 rows (phoneme tokens) appended at load, after the "
                                               "vocabulary a speech-edit adapter grows")
    parser.add_argument("--laci-heads", action="append", default=[], metavar="LANG=PATH")
    parser.add_argument("--dump-prompt", metavar="IDS_NPY")
    args = parser.parse_args(argv)
    if not args.out and not args.dump_prompt:
        parser.error("-o/--out is required unless --dump-prompt is given")
    heads = {}
    for item in args.laci_heads:
        lang, sep, path = item.partition("=")
        if not sep or not lang or not path:
            parser.error(f"--laci-heads needs LANG=PATH, got {item!r}")
        heads[lang] = path
    request = {"text": args.text} if args.text is not None else json.loads(Path(args.request).read_text())

    from zonos3.reference.lm import Zonos3LM
    from zonos3.reference.synth import ReferenceAssets, plan_prompt, synthesize

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "f32" else torch.bfloat16
    lm = Zonos3LM.load(args.model, device=device, dtype=dtype)
    assets = ReferenceAssets.load(steer_bundle=args.steer_bundle, accent_adapters=args.accent_adapters,
                                  laci_heads=heads)
    assets.grow(lm)
    if args.phoneme_rows:
        from zonos3.models.text_extra import TextExtraRows

        lm.append_text_rows(TextExtraRows.load(args.phoneme_rows))
    if args.dump_prompt:
        plan = plan_prompt(lm, request, assets=assets, device=device)
        path = Path(args.dump_prompt)
        _dump(path, plan.cond.input_ids, plan.cond_json())
        if plan.twin is not None:
            twin = path.with_name(path.stem + ".twin" + (path.suffix or ".npy"))
            _dump(twin, plan.twin.input_ids, plan.cond_json(plan.twin))
        print(f"prompt: {len(plan.cond.input_ids)} rows -> {path}" + (f" (+ {plan.cfg_kind} twin)" if plan.twin is not None else ""))
    if args.out:
        result = synthesize(lm, request, assets=assets, device=device, dtype=dtype)
        write_pcm16_wav(args.out, result.pcm, SAMPLE_RATE)
        seconds = len(result.pcm) / SAMPLE_RATE
        print(f"{seconds:.2f} s, seed {result.seed}, {len(result.rows)} rows, eos frame {result.eos_frame} -> {args.out}")
        if result.laci is not None:
            print(json.dumps(result.laci))
    return 0


if __name__ == "__main__":
    sys.exit(main())
