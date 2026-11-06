from __future__ import annotations

import logging
import multiprocessing as mp
import sys
from dataclasses import replace
from typing import TYPE_CHECKING

from zonos3.tts.conditioning import default_quality_buckets
from zonos3_sgl.distributed import DistributedInfo
from zonos3_sgl.utils.logger import init_logger

if TYPE_CHECKING:
    from .args import ServerArgs


def _run_scheduler(args: ServerArgs, ack_queue: mp.Queue[str]) -> None:
    import torch

    # Bind this process to its GPU before importing the model: importing the
    # model eagerly pulls in sgl_kernel, which creates the CUDA context, and it
    # must land on this rank's device rather than the default cuda:0.
    if torch.cuda.is_available():
        torch.cuda.set_device(args.tp_info.rank)

    from zonos3_sgl.scheduler import TTSScheduler

    with torch.inference_mode():
        scheduler = TTSScheduler(args)
        scheduler.sync_all_ranks()

        if args.tp_info.is_primary():
            ack_queue.put("Scheduler is ready")

        if args.silent_output:
            logging.disable(logging.INFO)

        try:
            scheduler.run_forever()
        except KeyboardInterrupt:
            logger = init_logger(__name__)
            if scheduler.tp_info.is_primary():
                print()  # for a clean newline after ^C
                logger.info("Scheduler exiting gracefully...")
            scheduler.shutdown()


def launch_server(run_shell: bool = False) -> None:
    from .api_server import run_api_server
    from .args import parse_args

    server_args, run_shell = parse_args(sys.argv[1:], run_shell)
    logger = init_logger(__name__, "initializer")

    def start_subprocess() -> None:
        import multiprocessing as mp

        from zonos3_sgl.tokenizer import tokenize_worker

        mp.set_start_method("spawn", force=True)

        world_size = server_args.tp_info.size
        # a multiprocessing queue to receive ack from subprocesses
        # so that we can guarantee all subprocesses are ready
        ack_queue: mp.Queue[str] = mp.Queue()

        for i in range(world_size):
            new_args = replace(
                server_args,
                tp_info=DistributedInfo(i, world_size),
            )
            mp.Process(
                target=_run_scheduler,
                args=(new_args, ack_queue),
                daemon=False,
                name=f"zonos3-TP{i}-scheduler",
            ).start()

        num_tokenizers = server_args.num_tokenizer

        # Legacy quality-conditioned checkpoints: every request gets the former default quality buckets.
        quality_bucket_counts = [
            len(server_args.tts_quality_buckets.get(feature, ()))
            for feature in server_args.tts_quality_features
        ]

        # Common tokenizer kwargs
        tokenizer_common_kwargs = {
            "backend_addr": server_args.zmq_backend_addr,
            "frontend_addr": server_args.zmq_frontend_addr,
            "local_bs": 1,
            "ack_queue": ack_queue,
            "n_codebooks": server_args.tts_n_codebooks,
            "codebook_sizes": list(server_args.model_config.codebook_sizes),
            "audio_pad_id": server_args.tts_audio_pad_id,
            "audio_codec": server_args.tts_audio_codec,
            "text_vocab": server_args.tts_text_vocab,
            "speaking_rate_num_buckets": server_args.tts_speaking_rate_num_buckets,
            "quality_bucket_counts": quality_bucket_counts,
            "quality_buckets": default_quality_buckets(server_args.tts_quality_features, quality_bucket_counts),
            "speaker_background_num_buckets": (
                2 if server_args.tts_speaker_background_token_enabled else 0
            ),
            "accurate_mode_num_buckets": (
                1
                if server_args.tts_speaker_background_token_enabled
                and server_args.tts_accurate_mode_token_enabled
                else 0
            ),
            "speaker_reference_max_seconds": (
                server_args.tts_speaker_reference_max_seconds
            ),
            "span_end_id": server_args.model_config.span_end_id,
            "condition_count": (
                int(getattr(server_args.model_config, "condition_count", 0) or 0)
                if getattr(server_args.model_config, "condition_tokens_enabled", False)
                else 0
            ),
            "condition_bucket_counts": list(server_args.model_config.condition_bucket_counts),
            "text_tokenizer_spec": server_args.text_tokenizer_spec,
        }

        # DeTokenizer, only 1
        mp.Process(
            target=tokenize_worker,
            kwargs={
                **tokenizer_common_kwargs,
                "addr": server_args.zmq_detokenizer_addr,
                "create": server_args.tokenizer_create_addr,
                "tokenizer_id": num_tokenizers,
            },
            daemon=False,
            name="zonos3-detokenizer-0",
        ).start()
        for i in range(num_tokenizers):
            mp.Process(
                target=tokenize_worker,
                kwargs={
                    **tokenizer_common_kwargs,
                    "addr": server_args.zmq_tokenizer_addr,
                    "create": server_args.tokenizer_create_addr,
                    "tokenizer_id": i,
                },
                daemon=False,
                name=f"zonos3-tokenizer-{i}",
            ).start()

        # Wait for acknowledgments from all worker processes:
        # - world_size schedulers (but only primary rank sends ack)
        # - num_tokenizers tokenizers
        # - 1 detokenizer
        # Total acks expected: 1 + num_tokenizers + 1 = num_tokenizers + 2
        for _ in range(num_tokenizers + 2):
            logger.info(ack_queue.get())

    run_api_server(server_args, start_subprocess, run_shell=run_shell)


if __name__ == "__main__":
    launch_server()

# CONDITION_ROWS_RUNTIME_20261016
