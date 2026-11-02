"""zonos3-mlx-server: the v2 HTTP API on Apple Silicon (MLX) over the ggml server's GGUF files.

python -m zonos3_mlx <model.gguf> [--dac dac.gguf] [--port 1919] [options]

dac.gguf, spk-encoder.gguf, dac-encoder.gguf, ssl-teacher.gguf, steer.gguf, emotion.gguf, accents.gguf and
accent-directions.gguf are auto-detected next to <model.gguf> when their flags are omitted (servers/ggml main.cpp).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from zonos3_mlx import paths


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m zonos3_mlx", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", help="backbone GGUF (zonos3-qat.gguf / zonos3-f16.gguf)")
    for flag, help_ in (("--dac", "codec decoder (dac.gguf)"),
                        ("--spk-encoder", "speaker encoder for speaker-embedding cloning of audio voices"),
                        ("--dac-encoder", "reference encoder (raw-reference cloning, speech editing; with --ssl-teacher)"),
                        ("--ssl-teacher", "SSL teacher of the reference encoder"),
                        ("--steer", "steering directions (steer.gguf): request controls and /v2/controls"),
                        ("--emotion-directions", "emotion directions (emotion.gguf)"),
                        ("--accent-adapters", "accent adapters (accents.gguf)"),
                        ("--accent-directions", "accent steering directions (accent-directions.gguf)")):
        p.add_argument(flag, default=None, help=help_)
    p.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=1919, help="port (default 1919)")
    p.add_argument("--max", type=int, default=2000, help="max frames per request (default 2000, ~133 s)")
    p.add_argument("--batch", type=int, default=4,
                   help="concurrent generation slots (default 4); a CFG request takes two")
    p.add_argument("--reference-idle-unload", type=float, default=60.0,
                   help="free the reference encoder + SSL teacher after N idle seconds (default 60; 0 keeps them)")
    p.add_argument("--stream-decode-default", choices=("prefix", "whole"),
                   default=os.environ.get("ZONOS3_STREAM_DECODE_DEFAULT", "prefix").strip().lower() or "prefix",
                   help="streaming decode of requests without stream_decode (default prefix: chunks of "
                        "ZONOS3_PREFIX_STREAM_CHUNK_FRAMES=15 frames behind a ZONOS3_PREFIX_STREAM_HOLD_BACK_FRAMES=3 "
                        "hold-back; whole: one exact chunk at EOS)")
    p.add_argument("--tts-default-voices-dir", default=str(paths.DEFAULT_VOICES),
                   help="default voices: audio files (+ <stem>.codes.npy reference codes) and .npy/.npz speaker "
                        "embeddings (default default_voices; empty disables)")
    p.add_argument("--python-src", default=str(paths.PYTHON_SRC),
                   help="directory holding the zonos3 package for the text-normalization worker (default model/src)")
    p.add_argument("--text-normalizer-python", default=None,
                   help="enable text normalization: PY runs <python-src>/zonos3/workers/textnorm.py (NeMo grammars, "
                        "needs pynini) as a persistent worker")
    p.add_argument("--laci-heads", nargs="+", default=None, metavar="LANG=PATH",
                   help="LACI alignment heads of a language (repeatable)")
    p.add_argument("--tts-phoneme-lexicon", default=None, help="gold pronunciation lexicon (LMDB) for /v2/phonemize")
    p.add_argument("--studio-projects-dir", default=None,
                   help="server-side studio projects and media (/v2/projects, /v2/media)")
    p.add_argument("--studio-max-project-bytes", type=int, default=20 * 1024**2)
    p.add_argument("--studio-max-media-bytes", type=int, default=2 * 1024**3)
    p.add_argument("--studio-dir", default=str(paths.STUDIO_DIR), help="studio production build served at /")
    p.add_argument("--log-level", default="info")
    return p


def parse_laci_heads(parser: argparse.ArgumentParser, entries) -> dict[str, str]:
    heads = {}
    for entry in entries or ():
        lang, sep, path = entry.partition("=")
        if not sep or not lang or not path:
            parser.error(f"--laci-heads takes LANG=PATH entries, got {entry!r}")
        path = os.path.expanduser(path)
        if not os.path.isfile(path):
            parser.error(f"--laci-heads {lang}: no file {path}")
        heads[lang.strip().lower().replace("-", "_")] = path
    return heads


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import uvicorn

    from zonos3_mlx.engine import Engine, TextNormalizer
    from zonos3_mlx.scheduler import Scheduler
    from zonos3_mlx.server.app import create_app
    from zonos3_mlx.server.state import ServerOptions, ServerState
    from zonos3_mlx.vocoder import prefix_settings

    try:
        prefix = prefix_settings()
    except ValueError as exc:
        parser.error(str(exc))
    laci = parse_laci_heads(parser, args.laci_heads)
    normalizer = None
    if args.text_normalizer_python:
        normalizer = TextNormalizer(args.text_normalizer_python, Path(args.python_src))
    engine = Engine.load(args.model, dac=args.dac, spk_encoder=args.spk_encoder, dac_encoder=args.dac_encoder,
                         ssl_teacher=args.ssl_teacher, steer=args.steer, emotion_directions=args.emotion_directions,
                         accent_adapters=args.accent_adapters, accent_directions=args.accent_directions,
                         laci_heads=laci, max_frames=args.max, normalizer=normalizer,
                         reference_idle_unload=args.reference_idle_unload)
    voices = Path(args.tts_default_voices_dir) if args.tts_default_voices_dir else None
    options = ServerOptions(model_path=args.model, batch=args.batch, max_frames=args.max,
                            stream_decode_default=args.stream_decode_default, default_voices_dir=voices,
                            studio_dir=Path(args.studio_dir) if args.studio_dir else None,
                            phoneme_lexicon=args.tts_phoneme_lexicon, text_normalization=normalizer is not None,
                            prefix=prefix)
    scheduler = Scheduler(engine, batch=args.batch)
    state = ServerState(engine, scheduler, options)
    if args.studio_projects_dir:
        from zonos3.server.projects import ProjectStore

        state.projects = ProjectStore(args.studio_projects_dir, args.studio_max_project_bytes,
                                      args.studio_max_media_bytes)
    if args.tts_phoneme_lexicon:
        from zonos3.tts.phonemes import PhonemeLexicon

        state.lexicon = PhonemeLexicon(args.tts_phoneme_lexicon)
    found = {k: Path(v).name for k, v in engine.paths.items() if v}
    logging.getLogger("zonos3_mlx").info("model %s; companions %s; batch %d; LACI %s", Path(args.model).name, found,
                                         args.batch, sorted(laci) or "off")
    scheduler.start()
    app = create_app(state)
    # Request logs would record the access_token query of a private server.
    uvicorn.run(app, host=args.host, port=args.port, access_log=not os.environ.get("ZONOS3_ACCESS_TOKEN"),
                log_level=args.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
