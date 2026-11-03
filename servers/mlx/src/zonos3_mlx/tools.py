"""Debugging CLI of the MLX networks: decode, embed, ssl, encode, generate.

    python -m zonos3_mlx.tools decode dac.gguf codes.npy out.npy [--eos N]
    python -m zonos3_mlx.tools embed spk-encoder.gguf audio.wav emb.npy [--dump-dir D]
    python -m zonos3_mlx.tools ssl ssl-teacher.gguf (--audio A | --features F --mask M) --out D
    python -m zonos3_mlx.tools encode dac-encoder.gguf ssl-teacher.gguf audio.mp3 codes.npy
    python -m zonos3_mlx.tools generate zonos3-qat.gguf "Text." out.wav [--seed S] [--dump-codes codes.npy]

Outputs are the golden names of tools/dump_golden.py, for tools/compare_golden.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _save_dir(out: Path, arrays: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, value in arrays.items():
        np.save(out / f"{name}.npy", np.ascontiguousarray(value, dtype=np.float32))


def cmd_decode(args) -> int:
    from zonos3_mlx.models.codec_decoder import CodecDecoder

    stream = np.load(args.codes).astype(np.int64)
    if stream.ndim != 2 or stream.shape[1] != 10:
        raise SystemExit("codes must be the sheared [H, 10] stream")
    H = len(stream)
    E = min(args.eos if args.eos is not None else H, H - 9)
    if E <= 0:
        raise SystemExit("no frames to decode")
    codes = np.zeros((E, 10), dtype=np.int64)
    for c in range(10):
        rows = np.arange(E) + c
        codes[:, c] = np.where(rows < H, stream[np.minimum(rows, H - 1), c], 4097)
    audio = CodecDecoder.load(args.gguf).decode(np.clip(codes, 0, 4095))
    np.save(args.out, audio.astype(np.float32))
    print(f"decoded {E} frames -> {len(audio)} samples -> {args.out}")
    return 0


def _read_audio(path: str) -> np.ndarray:
    from zonos3_mlx.audio_io import decode_pcm

    return decode_pcm(Path(path).read_bytes())


def cmd_embed(args) -> int:
    from zonos3_mlx.models.speaker_encoder import SpeakerEncoder

    dump = {} if args.dump_dir else None
    emb = SpeakerEncoder.load(args.gguf).embed(_read_audio(args.audio), dump=dump)
    np.save(args.out, emb)
    if dump is not None:
        _save_dir(Path(args.dump_dir), dump)
    print(f"embedding norm {float(np.linalg.norm(emb)):.4f} -> {args.out}")
    return 0


def cmd_ssl(args) -> int:
    from zonos3_mlx.models.ssl_teacher import SSLTeacher

    ssl = SSLTeacher.load(args.gguf, dtype=args.dtype)
    if args.audio:
        feats, mask = ssl.features(_read_audio(args.audio)[: 60 * 44100])
    else:
        feats = np.load(args.features).astype(np.float32)
        mask = np.load(args.mask).astype(np.int32) if args.mask else np.ones(len(feats), dtype=np.int32)
    dump = {"features": feats, "attention_mask": mask.astype(np.float32)}
    ssl.forward(feats, mask, dump=dump)
    _save_dir(Path(args.out), dump)
    print(f"features {feats.shape}, valid {int(mask.sum())}/{len(mask)} -> {args.out}")
    return 0


def cmd_encode(args) -> int:
    from zonos3_mlx.audio_io import decode_pcm
    from zonos3_mlx.models.codec_encoder import MAX_SECONDS, CodecEncoder
    from zonos3_mlx.models.ssl_teacher import SSLTeacher

    encoder = CodecEncoder.load(args.encoder, SSLTeacher.load(args.ssl))
    pcm = decode_pcm(Path(args.audio).read_bytes(), max_seconds=MAX_SECONDS)
    codes = encoder.encode(pcm, max_seconds=MAX_SECONDS)
    np.save(args.out, codes.astype(np.int32))
    print(f"{codes.shape[0]} frames -> {args.out}")
    return 0


def cmd_generate(args) -> int:
    import soundfile as sf

    from zonos3_mlx.engine import Engine
    from zonos3_mlx.scheduler import Scheduler

    engine = Engine.load(args.model, dac=args.dac)
    scheduler = Scheduler(engine, batch=2)
    scheduler.start(warmup=False)
    try:
        request = {"text": args.text, "stream": False}
        if args.seed is not None:
            request["seed"] = args.seed
        if args.request:
            request.update(json.loads(args.request))
        result = scheduler.synthesize(request)
    finally:
        scheduler.stop()
    sf.write(args.out, result.pcm, 44100)
    if args.dump_codes:
        n_cb = engine.lm.config.n_codebooks
        rows = np.stack([np.asarray(r)[:n_cb] for r in result.rows]).astype(np.int32)
        np.save(args.dump_codes, rows)
    print(f"seed {result.seed}: {len(result.pcm) / 44100:.2f} s -> {args.out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m zonos3_mlx.tools", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("decode", help="sheared codes -> pre-gain 44.1 kHz PCM (npy)")
    d.add_argument("gguf"); d.add_argument("codes"); d.add_argument("out"); d.add_argument("--eos", type=int)
    d.set_defaults(fn=cmd_decode)
    e = sub.add_parser("embed", help="audio -> speaker embedding")
    e.add_argument("gguf"); e.add_argument("audio"); e.add_argument("out"); e.add_argument("--dump-dir")
    e.set_defaults(fn=cmd_embed)
    s = sub.add_parser("ssl", help="SSL teacher features and hidden states")
    s.add_argument("gguf")
    src = s.add_mutually_exclusive_group(required=True)
    src.add_argument("--audio"); src.add_argument("--features")
    s.add_argument("--mask"); s.add_argument("--out", required=True)
    s.add_argument("--dtype", choices=("f32", "bf16"), default="f32",
                   help="f32: the dump_golden.py --dtype f32 oracle; bf16: the reference encoder's recipe")
    s.set_defaults(fn=cmd_ssl)
    c = sub.add_parser("encode", help="audio -> ordinary [frames, 10] reference codes")
    c.add_argument("encoder"); c.add_argument("ssl"); c.add_argument("audio"); c.add_argument("out")
    c.set_defaults(fn=cmd_encode)
    g = sub.add_parser("generate", help="one in-process synthesis (Engine + Scheduler, no HTTP)")
    g.add_argument("model"); g.add_argument("text"); g.add_argument("out")
    g.add_argument("--dac", help="decoder GGUF (default: dac.gguf next to the model)")
    g.add_argument("--seed", type=int); g.add_argument("--dump-codes")
    g.add_argument("--request", help="extra GenerateRequest fields as JSON")
    g.set_defaults(fn=cmd_generate)
    p.add_argument("--cpu", action="store_true",
                   help="run MLX on the CPU device (the F32 oracles compare at CPU-accumulation precision)")
    args = p.parse_args(argv)
    if args.cpu:
        import mlx.core as mx

        mx.set_default_device(mx.cpu)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
