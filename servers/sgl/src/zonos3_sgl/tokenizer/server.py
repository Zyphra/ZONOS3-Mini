from __future__ import annotations

import multiprocessing as mp
import os
from types import SimpleNamespace
from typing import List, Sequence

import torch
from zonos3.condition_buckets import check_ids
from zonos3.generation import SAMPLE_RATE
from zonos3.tts import plan
from zonos3.tts.prompt import TTSPromptBuilder, TTSPromptConfig

from zonos3_sgl.message import (
    BaseBackendMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchTokenizerMsg,
    BatchTTSBackendMsg,
    BatchTTSFrontendMsg,
    BatchTTSTokenizerMsg,
    TTSAudioReply,
    TTSCodesReply,
    TTSDecodeCodesMsg,
    TTSDetokenizeMsg,
    TTSNormalizeMsg,
    TTSTokenizeMsg,
    TTSUserMsg,
)
from zonos3_sgl.utils.logger import init_logger
from zonos3_sgl.utils.mp import ZmqPullQueue, ZmqPushQueue


def _unwrap_msg(
    msg: BaseTokenizerMsg | BatchTokenizerMsg | BatchTTSTokenizerMsg,
) -> List[BaseTokenizerMsg]:
    if isinstance(msg, BatchTokenizerMsg):
        return msg.data
    if isinstance(msg, BatchTTSTokenizerMsg):
        return msg.data
    return [msg]


def _silence_rows(msg: TTSTokenizeMsg, *, builder: TTSPromptBuilder, vocoder_manager) -> torch.Tensor | None:
    """The codec-matched silence prefill rows ending a request's prompt (sheared rows after EOS; the LM continues):
    the server default unless the request names its own length (zonos3.tts.plan.silence_prefill_frames), none for
    a speech-edit fill (it continues real audio context)."""
    if msg.edit_codes is not None:
        return None
    frames = vocoder_manager.silence_prefill_frames_for(msg.silence_prefill_frames)
    return vocoder_manager.silence_prompt_rows(frames, builder.config) if frames > 0 else None


def _edit_layout(msg: TTSTokenizeMsg) -> SimpleNamespace | None:
    """A speech-edit fill's window (zonos3.tts.plan.text_rows ``edit``: its codes, masked spans and the fills so
    far); None for plain TTS."""
    if msg.edit_codes is None:
        return None
    return SimpleNamespace(codes=msg.edit_codes, spans=msg.edit_spans, fills=msg.edit_fills)


@torch.inference_mode()
def tokenize_worker(
    *,
    addr: str,
    create: bool,
    backend_addr: str,
    frontend_addr: str,
    local_bs: int,
    tokenizer_id: int = -1,
    ack_queue: mp.Queue[str] | None = None,
    n_codebooks: int = 9,
    codebook_sizes: list[int] | None = None,
    audio_pad_id: int = 1025,
    audio_codec: str,
    decoders: Sequence[str] | None = None,  # the served decoder ids (ServerArgs.tts_decoders; None = every one)
    text_vocab: int | None = None,
    speaking_rate_num_buckets: int = 0,
    quality_bucket_counts: List[int] | None = None,
    quality_buckets: List[int | None] | None = None,
    speaker_background_num_buckets: int = 0,
    accurate_mode_num_buckets: int = 0,
    speaker_reference_max_seconds: float = 30.0,
    span_end_id: int | None = None,
    condition_count: int = 0,
    condition_bucket_counts: Sequence[int] = (),
    text_tokenizer_spec: dict | None = None,  # the served spec (EngineConfig.text_tokenizer_spec)
    decodes_audio: bool = True,  # the detokenizer: the scheduler's rows come here (--num-tokenizer > 0: one of N + 1)
) -> None:
    send_backend = ZmqPushQueue(backend_addr, create=False, encoder=BaseBackendMsg.encoder)
    send_frontend = ZmqPushQueue(frontend_addr, create=False, encoder=BaseFrontendMsg.encoder)
    recv_listener = ZmqPullQueue(addr, create=create, decoder=BatchTokenizerMsg.decoder)
    assert local_bs > 0
    logger = init_logger(__name__, f"tokenizer_{tokenizer_id}")

    from zonos3.tokenizer.textnorm import load_normalizer, normalize_request_text

    from .vocoder import TTSVocoderManager

    if text_vocab is None:
        raise ValueError("TTS mode requires text_vocab from the model config.")
    text_normalizer = load_normalizer()
    tts_prompt_builder = TTSPromptBuilder(
        TTSPromptConfig(
            n_codebooks=n_codebooks,
            codebook_sizes=tuple(codebook_sizes) if codebook_sizes else None,
            audio_pad_id=audio_pad_id,
            text_vocab=text_vocab,
            text_tokenizer_spec=text_tokenizer_spec,
            speaking_rate_num_buckets=speaking_rate_num_buckets,
            quality_bucket_counts=tuple(quality_bucket_counts or ()),
            speaker_background_num_buckets=speaker_background_num_buckets,
            accurate_mode_num_buckets=accurate_mode_num_buckets,
        )
    )
    tts_vocoder_manager = TTSVocoderManager(
        n_codebooks=n_codebooks,
        audio_pad_id=audio_pad_id,
        audio_codec=audio_codec,
        decoders=decoders,
    )
    # The prefix codec stream (primed + CUDA-graphed) serves the rows of streaming requests, which only the detokenizer
    # receives: the tokenize-only processes of --num-tokenizer N would hold N more copies of the codec and its graphs.
    if decodes_audio:
        tts_vocoder_manager.warm_prefix_stream()
    # inline reference codes keep their first frames: the checkpoint's speaker_reference_max_seconds at the codec hop
    reference_limit = plan.reference_frame_limit(
        speaker_reference_max_seconds, tts_vocoder_manager.hop_length, SAMPLE_RATE
    )

    if ack_queue is not None:
        ack_queue.put(f"Tokenize server {tokenizer_id} is ready")
    # Delayed rows of in-flight edit-span generations, returned as codes on finish.
    span_rows: dict[int, list[list[int]]] = {}

    try:
        while True:
            pending_msg = _unwrap_msg(recv_listener.get())
            while len(pending_msg) < local_bs and not recv_listener.empty():
                pending_msg.extend(_unwrap_msg(recv_listener.get()))

            logger.debug(f"Received {len(pending_msg)} messages")

            tts_tokenize_msg = [m for m in pending_msg if isinstance(m, TTSTokenizeMsg)]
            tts_detokenize_msg = [m for m in pending_msg if isinstance(m, TTSDetokenizeMsg)]
            normalize_msgs = [m for m in pending_msg if isinstance(m, TTSNormalizeMsg)]
            decode_msgs = [m for m in pending_msg if isinstance(m, TTSDecodeCodesMsg)]
            frontend_replies = []
            for msg in decode_msgs:
                audio = tts_vocoder_manager.decode_all(
                    msg.codes.to(torch.int64)[None], apply_shear_up=False, decoder=msg.decoder
                )[0]
                frontend_replies.append(
                    TTSAudioReply(
                        uid=msg.uid,
                        audio_data=audio.float().cpu().numpy().tobytes(),
                        finished=True,
                    )
                )

            # Process TTS tokenize messages
            if len(tts_tokenize_msg) > 0:
                normalized_texts = [
                    normalize_request_text(msg.text, msg.language, text_normalizer if msg.text_normalization else None)
                    for msg in tts_tokenize_msg
                ]
                user_msgs = []
                for msg, text in zip(tts_tokenize_msg, normalized_texts, strict=True):
                    if msg.start_immediately:
                        text = plan.with_start_marker(text)
                    # The byte / SuperBPE text rows and what follows them (zonos3.tts.plan.text_rows): the edit layout
                    # of a speech-edit fill, else the silence prefill rows. ``quality_buckets``: the quality tokens of a
                    # legacy quality-conditioned checkpoint (zonos3.tts.conditioning.default_quality_buckets).
                    silence_rows = _silence_rows(msg, builder=tts_prompt_builder, vocoder_manager=tts_vocoder_manager)
                    edit = _edit_layout(msg)
                    text_rows = plan.text_rows(
                        tts_prompt_builder,
                        text,
                        silence=silence_rows,
                        edit=edit,
                        span_end_id=span_end_id,
                        quality_buckets=quality_buckets,
                    )
                    # Span steering by reading position: each span's characters -> the text tokens spelling them
                    # (zonos3.tts.span_follow).
                    follow_tokens = None
                    if getattr(msg, "steer_follow_chars", None) is not None and msg.edit_codes is None:
                        from zonos3.tts.span_follow import span_token_ranges

                        follow_tokens = [
                            list(r)
                            for r in span_token_ranges(
                                msg.text,
                                text,
                                [tuple(c) for c in msg.steer_follow_chars],
                                tts_prompt_builder.config.text_tokenizer_spec,
                            )
                        ]
                    # Condition rows (cond3 finetunes): one bucket id per row, every row unknown by default.
                    condition_values = None
                    requested_buckets = getattr(msg, "condition_buckets", None)
                    if condition_count > 0:
                        buckets = (
                            list(requested_buckets) if requested_buckets is not None else list(condition_bucket_counts)
                        )
                        condition_values = check_ids(buckets, condition_bucket_counts)
                    elif requested_buckets is not None:
                        raise ValueError("This checkpoint has no condition rows (loudness / fmax / speaking rate)")
                    # Reference prefix codes (inline in the request or encoded by the frontend), cut to the reference
                    # limit.
                    reference_codes = getattr(msg, "speaker_reference_codes", None)
                    if reference_codes is not None:
                        reference_codes = plan.truncate_reference(
                            torch.as_tensor(reference_codes, dtype=torch.int32), reference_limit
                        )
                    # [reference prefix | speaker slot (position 0, as in training)][condition rows] text rows
                    rows = plan.prompt_rows(
                        tts_prompt_builder,
                        text_rows,
                        condition_count=len(condition_values or ()),
                        reference_codes=reference_codes,
                        speaker_row=msg.speaker_embedding is not None,
                    )
                    t = rows.input_ids
                    reference_prefix_length = rows.reference_prefix_length
                    speaker_token_position = (
                        msg.speaker_token_position if msg.speaker_embedding is None else rows.speaker_row
                    )
                    # decode context / silence prefill (2026-10-17): tell the vocoder what precedes the generated rows
                    context_codes = plan.decode_context_codes(
                        reference_codes, bool(msg.decode_context), msg.decode_context_frames
                    )
                    # (the detokenizer is another process: the context rides on TTSUserMsg -> TTSReq -> the first detokenize row)
                    if silence_rows is not None or context_codes is not None or bool(msg.decode_context):
                        logger.info(
                            "decode context uid=%s: prefill rows %s, context frames %s (decode_context=%s, reference frames %s)",
                            msg.uid,
                            0 if silence_rows is None else int(silence_rows.shape[0]),
                            0 if context_codes is None else int(context_codes.shape[0]),
                            bool(msg.decode_context),
                            None if reference_codes is None else int(reference_codes.shape[0]),
                        )
                    if os.environ.get("LONG_TTS_EVAL") == "1":
                        mode = os.environ["LONG_TTS_REFERENCE_MODE"]
                        if mode == "none":
                            assert reference_prefix_length == 0 and speaker_token_position == -1
                            assert msg.speaker_embedding is None and msg.speaker_reference_codes is None
                            assert float(msg.sampling_params.speaker_cfg_scale) == 1.0
                        else:
                            assert reference_prefix_length > 0 and msg.speaker_embedding is None
                            assert float(msg.sampling_params.speaker_cfg_scale) == 2.0
                        assert len(t) + int(msg.sampling_params.max_tokens) + 1 <= 32768, (
                            "Long evaluation would truncate output for context capacity"
                        )
                        logger.info(
                            "LONG_INPUT_OK uid=%s mode=%s input_frames=%s reference_frames=%s max_tokens=%s cfg=%s",
                            msg.uid,
                            mode,
                            len(t),
                            reference_prefix_length,
                            msg.sampling_params.max_tokens,
                            msg.sampling_params.speaker_cfg_scale,
                        )
                    logger.debug("Tokenize uid=%d text='%s...' frames=%d", msg.uid, msg.text[:50], len(t))
                    user_msgs.append(
                        TTSUserMsg(
                            uid=msg.uid,
                            input_ids=t,
                            speaker_reference_prefix_length=reference_prefix_length,
                            sampling_params=msg.sampling_params,
                            speaker_embedding=msg.speaker_embedding,
                            speaker_token_position=speaker_token_position,
                            steering_delta=msg.steering_delta,
                            steer_schedule=msg.steer_schedule,
                            steer_release_delta=msg.steer_release_delta,
                            steer_release=msg.steer_release,
                            steer_follow=msg.language if follow_tokens is not None else None,
                            steer_follow_tokens=follow_tokens,
                            steer_follow_gate=msg.steer_follow_gate if follow_tokens is not None else None,
                            condition_values=condition_values,
                            condition_token_start=rows.condition_token_start,
                            decode_context_codes=None
                            if context_codes is None
                            else context_codes.to(torch.int32).contiguous(),
                            prefill_rows=None if silence_rows is None else silence_rows.to(torch.int32).contiguous(),
                            accent=msg.accent,
                            accent_strength=msg.accent_strength,
                            laci=msg.language if getattr(msg, "laci", False) else None,
                            laci_target_frames=msg.laci_target_frames,
                            laci_adjust_rate=msg.laci_adjust_rate,
                            laci_rate_edges=msg.laci_rate_edges,
                            laci_rate_first=msg.laci_rate_first,
                            laci_rate_bucket=msg.laci_rate_bucket,
                            laci_rate_deltas=msg.laci_rate_deltas,
                        )
                    )
                batch_output = BatchTTSBackendMsg(data=user_msgs)
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                send_backend.put(batch_output)

            # Process TTS detokenize (vocoder) messages; edit spans return codes.
            vocoder_msgs = []
            for msg in tts_detokenize_msg:
                if msg.end_token != "span":
                    vocoder_msgs.append(msg)
                elif not msg.finished:
                    span_rows.setdefault(msg.uid, []).append(msg.audio_codes)
                else:
                    rows = span_rows.pop(msg.uid, [])
                    frontend_replies.append(
                        TTSCodesReply(
                            uid=msg.uid,
                            codes=torch.tensor(rows, dtype=torch.int32).reshape(-1, n_codebooks),
                            eos_frame=msg.eos_frame,
                        )
                    )
            if vocoder_msgs:
                audio_chunks = tts_vocoder_manager.decode_frames(vocoder_msgs)
                # A non-final row that produced no audio (every row before EOS of a whole-decode request) needs no
                # reply: the frontend only reads audio bytes and the finished flag, and a reply per row per request
                # floods its event loop at high concurrency.
                frontend_replies.extend(
                    TTSAudioReply(uid=msg.uid, audio_data=audio, finished=msg.finished, laci=msg.laci)
                    for msg, audio in zip(vocoder_msgs, audio_chunks, strict=True)
                    if audio or msg.finished
                )
            if frontend_replies:
                batch_output = BatchTTSFrontendMsg(data=frontend_replies)
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                send_frontend.put(batch_output)

            # Texts of requests the frontend is still resolving (TTSNormalizeMsg): normalize them now so their
            # TTSTokenizeMsg finds the result memoized. After the vocoder work, which other streams wait on.
            if text_normalizer is not None:
                for msg in normalize_msgs:
                    normalize_request_text(msg.text, msg.language, text_normalizer)

    except KeyboardInterrupt:
        pass
