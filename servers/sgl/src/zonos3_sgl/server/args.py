from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from zonos3.generation import SAMPLE_RATE
from zonos3.tts.laci import parse_laci_heads
from zonos3.utils.hf import cached_load_checkpoint_config

from zonos3_sgl.distributed import DistributedInfo
from zonos3_sgl.scheduler import SchedulerConfig
from zonos3_sgl.utils.logger import init_logger


@dataclass(frozen=True)
class ServerArgs(SchedulerConfig):
    server_host: str = "127.0.0.1"
    server_port: int = 1919
    num_tokenizer: int = 0
    silent_output: bool = False

    # TTS-specific settings
    tts_n_codebooks: int = 9
    tts_audio_pad_id: int = 1025
    tts_text_vocab: int | None = None
    tts_sample_rate: int = SAMPLE_RATE
    tts_audio_codec: str = "opus_dual_3n_600k_sched600k_r4096_step600000"
    tts_speaker_reference_max_seconds: float = 30.0
    tts_speaker_background_token_enabled: bool = False
    tts_accurate_mode_token_enabled: bool = False
    tts_speaking_rate_num_buckets: int = 0
    tts_speaking_rate_buckets: Tuple[str, ...] = ()
    tts_quality_num_buckets: int = 0
    tts_quality_features: Tuple[str, ...] = ()
    tts_quality_buckets: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    tts_quality_dropout: Dict[str, float] | None = None
    tts_default_voices_dir: str | None = None
    # Steering bundle (zonos3.tts.steering): direction sets driven together by a request's `controls` and
    # `control_spans`, plus the studio's UI metadata (/v2/controls).
    steer_bundle: str | None = None
    tts_phoneme_lexicon: str | None = None
    # Studio projects on the server (zonos3.server.projects: /v2/projects, /v2/media); None = disabled.
    studio_projects_dir: str | None = None
    studio_max_project_bytes: int = 20 * 1024**2
    studio_max_media_bytes: int = 2 * 1024**3
    # Production build of Zonos Voice Studio (studio/dist): / chooser, /studio, /lite (the studio serving rule of servers/api/openapi.yaml).
    studio_dir: str = str(Path(__file__).resolve().parents[5] / "studio" / "dist")
    # --plugins: server plugins (zonos3.server.plugins), each imported as zonos3_<name>.
    plugins: tuple[str, ...] = ()

    @property
    def share_tokenizer(self) -> bool:
        return self.num_tokenizer == 0

    @property
    def zmq_frontend_addr(self) -> str:
        return "ipc:///tmp/zonos3_3" + self._unique_suffix

    @property
    def zmq_tokenizer_addr(self) -> str:
        if self.share_tokenizer:
            return self.zmq_detokenizer_addr
        result = "ipc:///tmp/zonos3_4" + self._unique_suffix
        assert result != self.zmq_detokenizer_addr
        return result

    @property
    def tokenizer_create_addr(self) -> bool:
        return self.share_tokenizer

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return not self.share_tokenizer

    @property
    def frontend_create_tokenizer_link(self) -> bool:
        return not self.share_tokenizer

    @property
    def distributed_addr(self) -> str:
        return f"tcp://127.0.0.1:{self.server_port + 1}"


def parse_args(args: List[str], run_shell: bool = False) -> Tuple[ServerArgs, bool]:
    """
    Parse command line arguments and return an EngineConfig.

    Args:
        args: Command line arguments (e.g., sys.argv[1:])

    Returns:
        EngineConfig instance with parsed arguments
    """
    from zonos3_sgl.attention import validate_backend
    from zonos3_sgl.kvcache import SUPPORTED_CACHE_MANAGER

    def validate_moe_backend(backend: str) -> str:
        """Validate MoE backend argument."""
        # Accept any non-empty string as backend name
        # Specific backend validation can be added later as backends are implemented
        if not backend:
            from argparse import ArgumentTypeError

            raise ArgumentTypeError(f"MoE backend must be a non-empty string, got: {backend}")
        return backend

    parser = argparse.ArgumentParser(description="Zonos3 Server Arguments")

    from zonos3.utils.release import RELEASE_LM

    parser.add_argument(
        "--model-path",
        type=str,
        default=RELEASE_LM,
        help="Zonos3 TTS checkpoint: an hf://owner/repo/folder location (default: the ZONOS3 Mini release LM "
        f"{RELEASE_LM}; HF_TOKEN or a stored login while that repo is private), a Hugging Face repo id, or a local path.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type for model weights and activations. 'auto' will use FP16 for FP32/FP16 models and BF16 for BF16 models.",
    )

    parser.add_argument(
        "--tensor-parallel-size",
        "--tp-size",
        type=int,
        default=1,
        help="The tensor parallelism size.",
    )

    parser.add_argument(
        "--max-running-requests",
        type=int,
        dest="max_running_req",
        default=ServerArgs.max_running_req,
        help="The maximum number of running requests.",
    )

    parser.add_argument(
        "--max-seq-len-override",
        type=int,
        default=ServerArgs.max_seq_len_override,
        help="The maximum sequence length override.",
    )

    parser.add_argument(
        "--memory-ratio",
        type=float,
        default=ServerArgs.memory_ratio,
        help="The fraction of GPU memory to use for KV cache.",
    )

    assert ServerArgs.use_dummy_weight == False
    parser.add_argument(
        "--dummy-weight",
        action="store_true",
        dest="use_dummy_weight",
        help="Use dummy weights for testing.",
    )

    assert ServerArgs.use_pynccl == True
    parser.add_argument(
        "--disable-pynccl",
        action="store_false",
        dest="use_pynccl",
        help="Disable PyNCCL for tensor parallelism.",
    )

    parser.add_argument(
        "--host",
        type=str,
        dest="server_host",
        default=ServerArgs.server_host,
        help="The host address for the server.",
    )

    parser.add_argument(
        "--port",
        type=int,
        dest="server_port",
        default=ServerArgs.server_port,
        help="The port number for the server to listen on.",
    )

    parser.add_argument(
        "--cuda-graph-max-bs",
        "--graph",
        type=int,
        default=ServerArgs.cuda_graph_max_bs,
        help="The maximum batch size for CUDA graph capture. None means auto-tuning based on the GPU memory.",
    )

    parser.add_argument(
        "--num-tokenizer",
        "--tokenizer-count",
        type=int,
        default=ServerArgs.num_tokenizer,
        help="The number of tokenizer processes to launch. 0 means the tokenizer is shared with the detokenizer.",
    )

    parser.add_argument(
        "--max-prefill-length",
        "--max-extend-length",
        type=int,
        dest="max_extend_tokens",
        default=ServerArgs.max_extend_tokens,
        help="Chunk Prefill maximum chunk size in tokens.",
    )

    parser.add_argument(
        "--num-pages",
        "--num-tokens",
        dest="num_page_override",
        type=int,
        default=ServerArgs.num_page_override,
        help="Set the maximum number of pages for KVCache.",
    )

    parser.add_argument(
        "--attention-backend",
        "--attn",
        type=validate_backend,
        default=ServerArgs.attention_backend,
        help="The attention backend to use. If two backends are specified,"
        " the first one is used for prefill and the second one for decode.",
    )

    parser.add_argument(
        "--cache-type",
        type=str,
        default=ServerArgs.cache_type,
        choices=SUPPORTED_CACHE_MANAGER.supported_names(),
        help="The KV cache management strategy.",
    )

    parser.add_argument(
        "--eoa-policy",
        type=str,
        default=ServerArgs.eoa_policy,
        choices=["any", "codebook0"],
        help="Which sampled end-of-audio tokens stop a request: any codebook's, or "
        "codebook 0's only (lingua generate --eoa-policy codebook0).",
    )

    parser.add_argument(
        "--moe-backend",
        type=validate_moe_backend,
        default=ServerArgs.moe_backend,
        help="The MoE backend to use.",
    )

    parser.add_argument(
        "--shell-mode",
        action="store_true",
        help="Run the server in shell mode.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (shorthand for --log-level DEBUG).",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the log level (default: INFO). Overrides LOG_LEVEL env var.",
    )

    # TTS arguments
    parser.add_argument(
        "--tts-n-codebooks",
        type=int,
        default=ServerArgs.tts_n_codebooks,
        help="Number of audio codebooks for TTS (default: 9).",
    )

    parser.add_argument(
        "--tts-audio-pad-id",
        type=int,
        default=ServerArgs.tts_audio_pad_id,
        help="Audio padding token ID for TTS (default: 1025).",
    )

    parser.add_argument(
        "--tts-text-vocab",
        type=int,
        default=ServerArgs.tts_text_vocab,
        help="Text vocabulary size for TTS. If not set, auto-detected from model.",
    )

    parser.add_argument(
        "--tts-sample-rate",
        type=int,
        default=ServerArgs.tts_sample_rate,
        help=f"Audio sample rate for TTS output (default: {SAMPLE_RATE}, the API rate).",
    )

    parser.add_argument(
        "--tts-default-voices-dir",
        type=str,
        default=ServerArgs.tts_default_voices_dir,
        help="Directory of default speaker audio or .npy/.npz embedding files to pre-populate in the TTS UI.",
    )

    parser.add_argument(
        "--accent-adapters",
        type=str,
        default=ServerArgs.accent_adapters,
        help="Directory of accent adapters (<accent>.pt, lingua apps.zonos.accent_adapter). "
        "Requests then pick one with 'accent' and 'accent_strength' and share batches "
        "across accents. Single GPU (tensor parallel size 1) only.",
    )
    parser.add_argument(
        "--phoneme-rows",
        type=str,
        default=ServerArgs.phoneme_rows,
        help="Phoneme-token text rows (zonos-text-extra/1, lingua apps.zonos.text_extra) appended to the text table at "
        "load, after the vocabulary of a speech-edit adapter in --accent-adapters: pronunciation groups "
        "(U+E000 IPA U+E001) on a checkpoint without them. Single GPU only.",
    )
    parser.add_argument(
        "--steer-bundle",
        type=str,
        default=ServerArgs.steer_bundle,
        help="Steering bundle JSON (zonos3.tts.steering): direction sets served together; requests set "
        "'controls' {set.direction: dose} and 'control_spans'; /v2/controls describes them for the studio.",
    )

    parser.add_argument(
        "--tts-phoneme-lexicon",
        type=str,
        default=ServerArgs.tts_phoneme_lexicon,
        help="Gold pronunciation lexicon (LMDB directory built by lingua's "
        "apps.zonos.build_phoneme_lexicon). Enables POST /v2/phonemize and the UI's "
        "X-SAMPA pronunciation input; serve it only with phoneme-trained checkpoints.",
    )

    parser.add_argument(
        "--studio-dir",
        type=str,
        default=ServerArgs.studio_dir,
        help="Production build of Zonos Voice Studio: / chooser, /studio, /lite (default: <repo>/studio/dist; "
        "pnpm --dir studio build).",
    )
    parser.add_argument(
        "--plugins",
        nargs="+",
        default=[],
        metavar="NAME",
        help="server plugins: import zonos3_<NAME> and mount its routes (e.g. dub, from Zyphra/ZONOS3-dub)",
    )
    parser.add_argument(
        "--studio-projects-dir",
        type=str,
        default=ServerArgs.studio_projects_dir,
        help="Directory of the studio's shared projects (projects/<id>.json, content-addressed media/<sha256>): "
        "enables /v2/projects and /v2/media (zonos3.server.projects); created if missing. Unset: projects stay in the "
        "browser.",
    )
    parser.add_argument(
        "--studio-max-project-bytes",
        type=int,
        default=ServerArgs.studio_max_project_bytes,
        help="Largest stored studio project document (bytes).",
    )
    parser.add_argument(
        "--studio-max-media-bytes",
        type=int,
        default=ServerArgs.studio_max_media_bytes,
        help="Largest studio media blob (bytes); uploads stream to disk while hashed.",
    )

    parser.add_argument(
        "--laci-heads",
        nargs="+",
        metavar="LANG=PATH",
        default=None,
        help="LACI (zonos3.tts.laci): alignment-head spec JSON per language code, e.g. "
        "en_us=heads_en.json cmn=heads_zh.json. Requests then accept laci=true in those languages. Needs the 'fa' "
        "attention backend (chosen when --attention-backend is auto) and tensor parallel size 1.",
    )

    parser.add_argument(
        "--align-probe-max-text",
        type=int,
        default=None,
        help="Text tokens per request the decode CUDA graphs read for alignment probes (default 2048 with "
        "--laci-heads, else 0); a LACI request with a longer text decodes its steps eagerly.",
    )

    # Parse arguments
    kwargs = parser.parse_args(args).__dict__.copy()

    # Apply log level before any init_logger calls
    debug_mode = kwargs.pop("debug")
    log_level = kwargs.pop("log_level")
    if debug_mode:
        from zonos3_sgl.utils.logger import set_log_level

        set_log_level("DEBUG")
    elif log_level is not None:
        from zonos3_sgl.utils.logger import set_log_level

        set_log_level(log_level)

    # resolve some arguments
    run_shell |= kwargs.pop("shell_mode")
    if run_shell:
        kwargs["cuda_graph_max_bs"] = 1
        kwargs["max_running_req"] = 1
        kwargs["silent_output"] = True

    if kwargs["model_path"].startswith("~"):
        kwargs["model_path"] = os.path.expanduser(kwargs["model_path"])
    if kwargs.get("tts_default_voices_dir"):
        kwargs["tts_default_voices_dir"] = os.path.expanduser(kwargs["tts_default_voices_dir"])
    if kwargs.get("accent_adapters"):
        from zonos3.models.accent_format import list_accent_adapters

        kwargs["accent_adapters"] = os.path.expanduser(kwargs["accent_adapters"])
        list_accent_adapters(kwargs["accent_adapters"])
    if kwargs.get("phoneme_rows"):
        from zonos3.models.text_extra import TextExtraRows

        kwargs["phoneme_rows"] = os.path.expanduser(kwargs["phoneme_rows"])
        if TextExtraRows.load(kwargs["phoneme_rows"]).phonemes is None:
            parser.error(f"--phoneme-rows {kwargs['phoneme_rows']} holds no phoneme tokens")
    if kwargs.get("steer_bundle"):
        kwargs["steer_bundle"] = os.path.abspath(os.path.expanduser(kwargs["steer_bundle"]))
        from zonos3.tts.steering import load_bundle

        load_bundle(kwargs["steer_bundle"])  # fail at startup, not on the first request
    if kwargs.get("tts_phoneme_lexicon"):
        kwargs["tts_phoneme_lexicon"] = os.path.expanduser(kwargs["tts_phoneme_lexicon"])
        if not os.path.isfile(os.path.join(kwargs["tts_phoneme_lexicon"], "data.mdb")):
            parser.error(f"--tts-phoneme-lexicon {kwargs['tts_phoneme_lexicon']} has no data.mdb")
    if kwargs.get("studio_projects_dir"):
        kwargs["studio_projects_dir"] = os.path.abspath(os.path.expanduser(kwargs["studio_projects_dir"]))
        os.makedirs(kwargs["studio_projects_dir"], exist_ok=True)  # fail at startup, not on the first save

    DTYPE_MAP = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    checkpoint_config = cached_load_checkpoint_config(kwargs["model_path"])
    kwargs["tts_audio_codec"] = str(
        getattr(checkpoint_config, "audio_codec", "opus_dual_3n_600k_sched600k_r4096_step600000")
    )
    kwargs["tts_speaker_reference_max_seconds"] = float(
        getattr(checkpoint_config, "speaker_reference_max_seconds", 30.0)
    )

    if (dtype_str := kwargs["dtype"]) != "auto":
        kwargs["dtype"] = DTYPE_MAP[dtype_str]
    else:
        dtype_or_str = getattr(checkpoint_config, "dtype", "bfloat16")
        if isinstance(dtype_or_str, str):
            kwargs["dtype"] = DTYPE_MAP.get(dtype_or_str, torch.bfloat16)
        else:
            kwargs["dtype"] = dtype_or_str

    kwargs["tp_info"] = DistributedInfo(0, kwargs["tensor_parallel_size"])
    del kwargs["tensor_parallel_size"]

    # LACI: alignment-head specs per language; the probe needs the FA backend, every head on one rank, and a text
    # capacity for the decode graphs (zonos3.tts.laci).
    try:
        laci_heads = parse_laci_heads(kwargs.pop("laci_heads"))
    except ValueError as exc:
        parser.error(str(exc))
    kwargs["laci_heads"] = laci_heads
    if laci_heads:
        if kwargs["attention_backend"] == "auto":
            kwargs["attention_backend"] = "fa"
        elif kwargs["attention_backend"] != "fa":
            parser.error(f"--laci-heads needs --attention-backend fa (got {kwargs['attention_backend']!r})")
        if kwargs["tp_info"].size != 1:
            parser.error("--laci-heads needs --tensor-parallel-size 1")
        if kwargs["align_probe_max_text"] is None:
            kwargs["align_probe_max_text"] = 2048
    elif kwargs["align_probe_max_text"] is None:
        kwargs["align_probe_max_text"] = 0

    # Auto-detect TTS settings from model config if not provided
    if kwargs.get("tts_text_vocab") is None:
        text_vocab = getattr(checkpoint_config, "text_vocab", None)
        if text_vocab is not None:
            kwargs["tts_text_vocab"] = text_vocab

    # Auto-detect n_codebooks
    if kwargs.get("tts_n_codebooks") == ServerArgs.tts_n_codebooks:  # still default
        n_codebooks = getattr(checkpoint_config, "n_codebooks", None)
        if n_codebooks is not None:
            kwargs["tts_n_codebooks"] = n_codebooks

    # Auto-detect audio_pad_id
    if kwargs.get("tts_audio_pad_id") == ServerArgs.tts_audio_pad_id:  # still default
        audio_pad_id = getattr(checkpoint_config, "audio_pad_id", None)
        if audio_pad_id is not None:
            kwargs["tts_audio_pad_id"] = audio_pad_id

    speaking_rate_num_buckets = int(getattr(checkpoint_config, "speaking_rate_num_buckets", 0) or 0)
    if speaking_rate_num_buckets > 0:
        kwargs["tts_speaking_rate_num_buckets"] = speaking_rate_num_buckets
        kwargs["tts_speaking_rate_buckets"] = tuple(getattr(checkpoint_config, "speaking_rate_buckets", None) or ())

    if bool(getattr(checkpoint_config, "speaker_background_token_enabled", False)):
        kwargs["tts_speaker_background_token_enabled"] = True
        if bool(getattr(checkpoint_config, "accurate_mode_token_enabled", False)):
            kwargs["tts_accurate_mode_token_enabled"] = True

    quality_num_buckets = int(getattr(checkpoint_config, "quality_num_buckets", 0) or 0)
    if quality_num_buckets > 0:
        raw_quality_buckets = getattr(checkpoint_config, "quality_buckets", None) or {}
        raw_quality_features = getattr(checkpoint_config, "quality_features", None) or raw_quality_buckets.keys()
        quality_features = tuple(str(item) for item in (raw_quality_features or ()))
        kwargs["tts_quality_num_buckets"] = quality_num_buckets
        kwargs["tts_quality_features"] = quality_features
        kwargs["tts_quality_buckets"] = {
            str(feature): tuple(str(item) for item in (raw_quality_buckets.get(feature, ()) or ()))
            for feature in quality_features
        }
        kwargs["tts_quality_dropout"] = getattr(checkpoint_config, "quality_dropout", None)

    # Debug: print TTS settings
    logger = init_logger(__name__)
    logger.info(
        f"TTS settings: n_codebooks={kwargs.get('tts_n_codebooks')}, "
        f"text_vocab={kwargs.get('tts_text_vocab')}, "
        f"speaking_rate_num_buckets={kwargs.get('tts_speaking_rate_num_buckets', 0)}, "
        f"quality_num_buckets={kwargs.get('tts_quality_num_buckets', 0)}, "
        f"speaker_background_token={kwargs.get('tts_speaker_background_token_enabled', False)}, "
        f"accurate_mode_token={kwargs.get('tts_accurate_mode_token_enabled', False)}"
    )

    kwargs["plugins"] = tuple(kwargs["plugins"])
    result = ServerArgs(**kwargs)
    logger = init_logger(__name__)
    logger.info(f"Parsed arguments:\n{result}")
    return result, run_shell
