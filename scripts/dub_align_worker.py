#!/usr/bin/env python3
"""Alignment worker of the dubbing server (runs in .venv-dub-align; see scripts/setup_dub_align_env.sh).

Reads one JSON request per stdin line ({"id", "op", ...}) and answers each with one JSON line on stdout, in order, until
stdin closes: {"id", "status": "ok", ...} or {"id", "status": "error", "error", "message", "invalid"} ("invalid": a
ValueError, i.e. the input is at fault). Models load once, before the first read; AV-sync's on its first use.

Ops:
  ping                                        -> {"aligner_languages": [lowercase names] | null, "avsync": bool}
  transcribe {audio, language|null}           -> {"text", "language"}                      (Qwen3-ASR)
  align {items: [{audio, text, language}]}    -> {"results": [{"items": [{text,start,end}]}]}  (Qwen3-ForcedAligner)
  word_align {src: [str], tgt: [str]}         -> {"links": [[i, j]]}   (SimAlign mBERT, BPE level, itermax; windowed)
  lips_prepare {video, out}                   -> {"start", "end", "track_frames", "video_frames", "tracks"}  (AV-sync)
  lips_sync {lips, audio, out,                -> {"start", "end", "rate", "src_frames", "dp_cost",
             max_rate?, max_shift_s?}            "sync_distance_before", "sync_distance_after"}             (AV-sync;
                                                  the optional bounds of the warp: plan.guard_lip_frames)
"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

import dub_avsync
from zonos3.dub import plan

ALIGN_CHUNK = 32


def read_audio(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    return wav, sr


def side_embeddings(emb, words):
    """Wordpiece embeddings of one side, stitched from overlapping windows; returns (X [wp, d], wordpiece -> word)."""
    counts = [len(emb.tokenizer.tokenize(w)) for w in words]
    windows = plan.word_windows(counts)
    owner = plan.window_of_words(windows, len(words))
    window_vecs = []
    for s, e in windows:
        window_vecs.append(emb.get_embed_list([words[s:e]])[0].float().cpu().numpy())
    rows, b2w = [], []
    for w, c in enumerate(counts):
        if c == 0:
            continue
        s, _ = windows[owner[w]]
        off = sum(counts[s:w])
        rows.append(window_vecs[owner[w]][off : off + c])
        b2w.extend([w] * c)
    if not rows:
        return None, []
    return np.concatenate(rows, axis=0), b2w


def word_align(emb, SentenceAligner, src, tgt):
    X, b2w_src = side_embeddings(emb, src)
    Y, b2w_tgt = side_embeddings(emb, tgt)
    if X is None or Y is None:
        return {"links": []}
    sim = SentenceAligner.get_similarity(X, Y) * plan.band_mask(len(X), len(Y))
    inter = SentenceAligner.iter_max(sim)
    links = sorted({(b2w_src[p], b2w_tgt[q]) for p, q in zip(*np.nonzero(inter))})
    return {"links": [[int(i), int(j)] for i, j in links]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--asr-model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--aligner-model", default="Qwen/Qwen3-ForcedAligner-0.6B")
    parser.add_argument("--word-aligner-model", default="google-bert/bert-base-multilingual-cased")
    parser.add_argument("--avsync-home", default=None, help="AV-sync checkout with its models in data/")
    args = parser.parse_args()

    # Keep stdout for responses only: library prints go to stderr.
    protocol = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    import simalign.simalign
    from qwen_asr import Qwen3ASRModel

    SentenceAligner = simalign.simalign.SentenceAligner
    asr = Qwen3ASRModel.from_pretrained(
        args.asr_model,
        forced_aligner=args.aligner_model,
        forced_aligner_kwargs=dict(dtype=torch.bfloat16, device_map=args.device),
        dtype=torch.bfloat16,
        device_map=args.device,
        max_inference_batch_size=32,
        max_new_tokens=1024,
    )
    emb = simalign.simalign.EmbeddingLoader(model=args.word_aligner_model, device=torch.device(args.device), layer=8)
    avsync = []  # the AV-sync models, loaded on first use

    def lips():
        if not avsync:
            if not args.avsync_home:
                raise RuntimeError("lip-sync is not set up (no --avsync-home)")
            avsync.append(dub_avsync.AvSync(args.avsync_home, args.device))
        return avsync[0]

    def handle(req):
        op = req["op"]
        if op == "ping":
            langs = asr.forced_aligner.get_supported_languages()
            return {
                "aligner_languages": None if langs is None else [x.lower() for x in langs],
                "avsync": bool(args.avsync_home) and dub_avsync.models_present(args.avsync_home),
            }
        if op == "transcribe":
            r = asr.transcribe(audio=read_audio(req["audio"]), language=req.get("language"))[0]
            return {"text": r.text, "language": r.language}
        if op == "align":
            items = req["items"]
            results = []
            for k in range(0, len(items), ALIGN_CHUNK):
                chunk = items[k : k + ALIGN_CHUNK]
                out = asr.forced_aligner.align(
                    audio=[read_audio(it["audio"]) for it in chunk],
                    text=[it["text"] for it in chunk],
                    language=[it["language"] for it in chunk],
                )
                results.extend(
                    {"items": [{"text": i.text, "start": float(i.start_time), "end": float(i.end_time)} for i in r.items]}
                    for r in out
                )
            return {"results": results}
        if op == "word_align":
            return word_align(emb, SentenceAligner, req["src"], req["tgt"])
        if op == "lips_prepare":
            return lips().prepare(req["video"], req["out"])
        if op == "lips_sync":
            return lips().sync(req["lips"], req["audio"], req["out"], req.get("max_rate"), req.get("max_shift_s"))
        raise ValueError(f"unknown op {op!r}")

    for line in sys.stdin:
        if not line.strip():
            continue
        rid = None
        try:
            req = json.loads(line)
            rid = req.get("id")
            with torch.no_grad():
                resp = handle(req)
            resp.update(id=rid, status="ok")
        except Exception as exc:  # noqa: BLE001 - one bad request must not kill the worker
            resp = {"id": rid, "status": "error", "error": repr(exc), "message": str(exc),
                    "invalid": isinstance(exc, ValueError)}
        protocol.write(json.dumps(resp) + "\n")


if __name__ == "__main__":
    main()
