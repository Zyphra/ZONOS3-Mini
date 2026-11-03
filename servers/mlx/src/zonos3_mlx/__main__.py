"""zonos3-mlx-server: the v2 HTTP API on Apple Silicon (MLX) over the ggml server's GGUF files.

python -m zonos3_mlx <model.gguf> [--dac dac.gguf] [--port 1919] [options]

dac.gguf, spk-encoder.gguf, dac-encoder.gguf, ssl-teacher.gguf, steer.gguf and accents.gguf are auto-detected next to
<model.gguf> when their flags are omitted (servers/ggml main.cpp).
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from zonos3.generation import MAX_FRAMES

from zonos3_mlx import paths


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m zonos3_mlx", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", help="backbone GGUF (zonos3-qat.gguf / zonos3-f16.gguf)")
    for flag, help_ in (("--dac", "codec decoder (dac.gguf)"),
                        ("--spk-encoder", "speaker encoder for speaker-embedding cloning of audio voices"),
                        ("--dac-encoder", "reference encoder (raw-reference cloning, speech editing; with --ssl-teacher)"),
                        ("--ssl-teacher", "SSL teacher of the reference encoder"),
                        ("--steer", "steering bundle (steer.gguf): request controls and /v2/controls"),
                        ("--accent-adapters", "accent adapters (accents.gguf): request accent")):
        p.add_argument(flag, default=None, help=help_)
    p.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=1919, help="port (default 1919)")
    p.add_argument("--max", type=int, default=MAX_FRAMES,
                   help=f"max frames per request (default {MAX_FRAMES}, ~{MAX_FRAMES // 15} s at 15 frames/s)")
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
    p.add_argument("--worker-python", default=None, metavar="PY",
                   help="interpreter of the helper workers of model/src/zonos3/workers, each one persistent process: "
                        "text normalization (textnorm.py under <python-src>: NeMo grammars, needs pynini; off without "
                        "this flag) and the FuseAlign forced aligner of speech editing and dubbing (fusealign.py: needs "
                        "fusealign-hub; default this server's interpreter). Speech recognition (asr.py, Qwen3-ASR) and "
                        "the dubbing worker run in .venv-asr (tools/setup_asr_env.sh; ZONOS3_ASR_PYTHON)")
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


def fusealign_available(python: str) -> bool:
    """Whether the FuseAlign worker's interpreter can import ``fusealign_hub`` (third_party/fusealign-multilingual)."""
    from zonos3.tts.edit_align import fusealign_installed

    if python == sys.executable:
        return fusealign_installed()
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV")}  # as FuseAlignWorker runs it
    try:
        probe = subprocess.run([python, "-c", "import importlib.util, sys; "
                                "sys.exit(importlib.util.find_spec('fusealign_hub') is None)"],
                               env=env, capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import uvicorn
    from zonos3.tts.edit_align import FuseAlignWorker

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
    worker_python = args.worker_python or sys.executable
    normalizer = None
    if args.worker_python:
        normalizer = TextNormalizer(args.worker_python, Path(args.python_src))
    engine = Engine.load(args.model, dac=args.dac, spk_encoder=args.spk_encoder, dac_encoder=args.dac_encoder,
                         ssl_teacher=args.ssl_teacher, steer=args.steer, accent_adapters=args.accent_adapters,
                         laci_heads=laci, max_frames=args.max, normalizer=normalizer,
                         reference_idle_unload=args.reference_idle_unload)
    voices = Path(args.tts_default_voices_dir) if args.tts_default_voices_dir else None
    options = ServerOptions(model_path=args.model, batch=args.batch, max_frames=args.max,
                            stream_decode_default=args.stream_decode_default, default_voices_dir=voices,
                            studio_dir=Path(args.studio_dir) if args.studio_dir else None,
                            phoneme_lexicon=args.tts_phoneme_lexicon, text_normalization=normalizer is not None,
                            prefix=prefix, fusealign=fusealign_available(worker_python))
    scheduler = Scheduler(engine, batch=args.batch)
    state = ServerState(engine, scheduler, options, aligner=FuseAlignWorker(python=worker_python))
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
