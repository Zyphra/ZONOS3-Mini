from __future__ import annotations
import hashlib
import collections

import multiprocessing as mp
import os
import threading
from typing import List

import torch
from zonos3.message import (
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
    TTSEncodeAudioMsg,
    TTSTokenizeMsg,
    TTSUserMsg,
)
from zonos3.tts.prompt import TTSPromptBuilder, TTSPromptConfig, strip_nonverbal_markup
from zonos3.utils import ZmqPullQueue, ZmqPushQueue, init_logger


def _unwrap_msg(
    msg: BaseTokenizerMsg | BatchTokenizerMsg | BatchTTSTokenizerMsg,
) -> List[BaseTokenizerMsg]:
    if isinstance(msg, BatchTokenizerMsg):
        return msg.data
    if isinstance(msg, BatchTTSTokenizerMsg):
        return msg.data
    return [msg]


_REFERENCE_CODE_CACHE: "collections.OrderedDict[tuple[str, float], torch.Tensor]" = collections.OrderedDict()
_REFERENCE_CODE_CACHE_SIZE = 64


def _encode_reference_cached(vocoder_manager, audio_bytes: bytes, max_seconds: float) -> torch.Tensor:
    """Reference codes memoized by the clip's bytes: a chat demo sends the same voice with every sentence."""
    key = (hashlib.sha1(audio_bytes).hexdigest(), float(max_seconds))
    codes = _REFERENCE_CODE_CACHE.get(key)
    if codes is None:
        codes = vocoder_manager.encode_reference_audio_bytes(audio_bytes, max_seconds=max_seconds)
        _REFERENCE_CODE_CACHE[key] = codes
        if len(_REFERENCE_CODE_CACHE) > _REFERENCE_CODE_CACHE_SIZE:
            _REFERENCE_CODE_CACHE.popitem(last=False)
    else:
        _REFERENCE_CODE_CACHE.move_to_end(key)
    return codes.clone()


def _tag_cfg_plain_text(msg: TTSTokenizeMsg, text: str) -> str | None:
    """The plain twin's text for tag CFG (tag_cfg_scale > 1.0): the request's already normalized text with its
    nonverbal markup stripped, so both branches read the same words; None (no twin) otherwise."""
    if float(msg.sampling_params.tag_cfg_scale) <= 1.0:
        return None
    return strip_nonverbal_markup(text)


def _text_prompt_rows(
    msg: TTSTokenizeMsg, text: str, *, builder: TTSPromptBuilder, vocoder_manager, span_end_id: int | None
) -> torch.Tensor:
    """The byte / SuperBPE text rows of a request and what follows them: the edit layout for speech-edit infill,
    else the codec-matched silence prefill rows (when on). Everything placed in front of these rows (condition rows,
    reference prefix, speaker slot) does not depend on the text."""
    from zonos3.tts.edit import edit_prompt_tokens

    # codec-silence prefill: the server default (ZONOS3_SILENCE_PREFILL_FRAMES, 3 = 0.2 s) unless the
    # request says prepend_silence false / silence_prefill_frames 0, or names its own length
    frames = vocoder_manager.silence_prefill_frames_for(msg.prepend_silence, getattr(msg, "silence_prefill_frames", None))
    prompt = builder.build(
        text,
        speaking_rate_bucket=msg.speaking_rate_bucket,
        quality_buckets=msg.quality_buckets,
    )
    if msg.edit_codes is not None:
        return edit_prompt_tokens(
            prompt, msg.edit_codes, msg.edit_spans, msg.edit_fills, builder.config, span_end_id=span_end_id
        )
    if frames > 0:
        # codec-matched silence frames open the audio stream (sheared rows after EOS; the LM continues)
        return torch.cat((prompt, vocoder_manager.silence_prompt_rows(frames, builder.config)), dim=0)
    return prompt


def _plain_twin_prompt(input_ids: torch.Tensor, text_rows: torch.Tensor, plain_text_rows: torch.Tensor) -> torch.Tensor:
    """The tag-CFG twin prompt: the request's final prompt with its text rows (the tail ``text_rows``) replaced by
    the plain text's rows; the leading speaker / reference / condition rows are shared."""
    lead = len(input_ids) - len(text_rows)
    if lead < 0 or not torch.equal(input_ids[lead:], text_rows):
        raise ValueError("Tag CFG: the prompt does not end with its text rows")
    return torch.cat((input_ids[:lead], plain_text_rows.to(input_ids.dtype)), dim=0)


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
    text_vocab: int | None = None,
    speaking_rate_num_buckets: int = 0,
    quality_bucket_counts: List[int] | None = None,
    speaker_background_num_buckets: int = 0,
    accurate_mode_num_buckets: int = 0,
    speaker_reference_max_seconds: float = 30.0,
    span_end_id: int | None = None,
    condition_count: int = 0,
    condition_n_buckets: int = 10,
    model_path: str = "",
) -> None:
    send_backend = ZmqPushQueue(backend_addr, create=False, encoder=BaseBackendMsg.encoder)
    send_frontend = ZmqPushQueue(frontend_addr, create=False, encoder=BaseFrontendMsg.encoder)
    recv_listener = ZmqPullQueue(addr, create=create, decoder=BatchTokenizerMsg.decoder)
    assert local_bs > 0
    logger = init_logger(__name__, f"tokenizer_{tokenizer_id}")

    from .textnorm import TTSTextNormalizer, collapse_line_breaks, normalization_enabled
    from .vocoder import TTSVocoderManager

    if text_vocab is None:
        raise ValueError("TTS mode requires text_vocab from the model config.")
    textnorm_enabled = normalization_enabled()
    text_normalizer = TTSTextNormalizer() if textnorm_enabled else None
    if text_normalizer is not None:
        # Compile/load the English grammars up front so the first request does
        # not pay the one-time WFST construction cost.
        threading.Thread(
            target=text_normalizer.warmup, args=(["en"],), daemon=True
        ).start()
    tts_prompt_builder = TTSPromptBuilder(
        TTSPromptConfig(
            n_codebooks=n_codebooks,
            codebook_sizes=tuple(codebook_sizes) if codebook_sizes else None,
            audio_pad_id=audio_pad_id,
            text_vocab=text_vocab,
            text_tokenizer_spec=__import__("zonos3.tts.superbpe", fromlist=["load_spec"]).load_spec(model_path),
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
            tts_detokenize_msg = [
                m for m in pending_msg if isinstance(m, TTSDetokenizeMsg)
            ]
            encode_msgs = [m for m in pending_msg if isinstance(m, TTSEncodeAudioMsg)]
            decode_msgs = [m for m in pending_msg if isinstance(m, TTSDecodeCodesMsg)]
            frontend_replies = []
            for msg in encode_msgs:
                try:
                    codes = tts_vocoder_manager.encode_reference_audio_bytes(
                        msg.audio, max_seconds=msg.max_seconds
                    )
                    frontend_replies.append(TTSCodesReply(uid=msg.uid, codes=codes))
                except (ValueError, RuntimeError) as exc:
                    frontend_replies.append(TTSCodesReply(uid=msg.uid, codes=None, error=str(exc)))
            for msg in decode_msgs:
                audio = tts_vocoder_manager.decode_all(
                    msg.codes.to(torch.int64)[None], apply_shear_up=False
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
                tensors = []
                plain_tensors = []
                normalized_texts = []
                for msg in tts_tokenize_msg:
                    text = collapse_line_breaks(msg.text)  # LINE_BREAKS_20261019: also when normalization is off
                    if text_normalizer is not None and msg.text_normalization:
                        text = text_normalizer.normalize(text, msg.language)
                    normalized_texts.append(text)
                for msg, text in zip(tts_tokenize_msg, normalized_texts, strict=True):
                    plain_prompt = None
                    prompt = _text_prompt_rows(
                        msg, text, builder=tts_prompt_builder, vocoder_manager=tts_vocoder_manager, span_end_id=span_end_id
                    )
                    plain_text = _tag_cfg_plain_text(msg, text)
                    if plain_text is not None:
                        plain_prompt = _text_prompt_rows(
                            msg, plain_text, builder=tts_prompt_builder, vocoder_manager=tts_vocoder_manager,
                            span_end_id=span_end_id,
                        )
                        logger.info(
                            "tag CFG uid=%s scale=%s: tagged text %r (%d rows), plain twin text %r (%d rows)",
                            msg.uid, msg.sampling_params.tag_cfg_scale, text, len(prompt), plain_text, len(plain_prompt),
                        )
                    tensors.append(prompt)
                    plain_tensors.append(plain_prompt)
                user_msgs = []
                for msg, t, plain_prompt, text in zip(
                    tts_tokenize_msg, tensors, plain_tensors, normalized_texts, strict=True
                ):
                    text_rows = t
                    # Span steering by reading position: each span's characters -> the text tokens spelling them
                    # (zonos3.tts.span_follow).
                    follow_tokens = None
                    if getattr(msg, "steer_follow_chars", None) is not None and msg.edit_codes is None:
                        from zonos3.tts.span_follow import span_token_ranges

                        follow_tokens = [list(r) for r in span_token_ranges(
                            msg.text, text, [tuple(c) for c in msg.steer_follow_chars],
                            tts_prompt_builder.config.text_tokenizer_spec)]
                    if (
                        msg.speaker_embedding is not None
                        and msg.speaker_reference_audio is not None
                    ):
                        raise ValueError(
                            "Speaker embeddings and raw reference audio are mutually exclusive."
                        )
                    # Condition rows (cond3 finetunes) lead the text; the reference prefix / speaker slot go in front.
                    condition_values = None
                    requested_buckets = getattr(msg, "condition_buckets", None)
                    if condition_count > 0:
                        buckets = list(requested_buckets) if requested_buckets is not None else [condition_n_buckets] * condition_count
                        if len(buckets) != condition_count or any(not 0 <= int(b) <= condition_n_buckets for b in buckets):
                            raise ValueError(
                                f"condition_buckets needs {condition_count} ids in [0, {condition_n_buckets}] ({condition_n_buckets} = unknown)"
                            )
                        condition_values = [int(b) for b in buckets]
                        t = torch.cat((tts_prompt_builder.condition_rows(condition_count, dtype=t.dtype, device=t.device), t), dim=0)
                    elif requested_buckets is not None:
                        raise ValueError("This checkpoint has no condition rows (loudness / fmax / speaking rate)")
                    reference_prefix_length = 0
                    reference_codes = getattr(msg, "speaker_reference_codes", None)
                    if reference_codes is not None:
                        reference_codes = torch.as_tensor(reference_codes, dtype=torch.int32)
                        max_frames = int(speaker_reference_max_seconds * 44100 // tts_vocoder_manager.hop_length)
                        reference_codes = reference_codes[:max_frames]
                    elif msg.speaker_reference_audio is not None:
                        reference_codes = _encode_reference_cached(
                            tts_vocoder_manager,
                            msg.speaker_reference_audio,
                            speaker_reference_max_seconds,
                        )
                    if reference_codes is not None:
                        reference_prefix = tts_prompt_builder.speaker_reference_prefix(
                            reference_codes
                        )
                        reference_prefix_length = len(reference_prefix)
                        t = torch.cat((reference_prefix, t), dim=0)
                    # decode context / silence prefill (2026-10-17): tell the vocoder what precedes the generated rows
                    silence_rows = None
                    frames = 0 if msg.edit_codes is not None else tts_vocoder_manager.silence_prefill_frames_for(
                        msg.prepend_silence, getattr(msg, "silence_prefill_frames", None)
                    )   # edit fills continue real audio context, never a silence prefill
                    if frames > 0:
                        silence_rows = tts_vocoder_manager.silence_prompt_rows(frames, tts_prompt_builder.config)
                    context_codes = None
                    if bool(getattr(msg, "decode_context", False)) and reference_codes is not None:
                        k = max(0, min(int(getattr(msg, "decode_context_frames", 30) or 0), int(reference_codes.shape[0])))
                        context_codes = reference_codes[int(reference_codes.shape[0]) - k:] if k > 0 else None
                    # (the detokenizer is another process: the context rides on TTSUserMsg -> TTSReq -> the first detokenize row)
                    if silence_rows is not None or context_codes is not None or bool(getattr(msg, "decode_context", False)):
                        logger.info("decode context uid=%s: prefill rows %s, context frames %s (decode_context=%s, reference frames %s)", msg.uid,
                                    0 if silence_rows is None else int(silence_rows.shape[0]), 0 if context_codes is None else int(context_codes.shape[0]),
                                    bool(getattr(msg, "decode_context", False)), None if reference_codes is None else int(reference_codes.shape[0]))
                    speaker_token_position = msg.speaker_token_position
                    if msg.speaker_embedding is not None:
                        # Canonical speaker slot is token position 0, matching training.
                        speaker_slot = tts_prompt_builder.speaker_slot(
                            dtype=t.dtype,
                            device=t.device,
                        )
                        t = torch.cat([speaker_slot, t], dim=0)
                        speaker_token_position = 0
                    condition_token_start = -1
                    if condition_values is not None:
                        condition_token_start = reference_prefix_length + (1 if msg.speaker_embedding is not None else 0)
                    if os.environ.get("LONG_TTS_EVAL") == "1":
                        mode = os.environ["LONG_TTS_REFERENCE_MODE"]
                        if mode == "none":
                            assert reference_prefix_length == 0 and speaker_token_position == -1
                            assert msg.speaker_embedding is None and msg.speaker_reference_audio is None
                            assert float(msg.sampling_params.speaker_cfg_scale) == 1.0
                        else:
                            assert reference_prefix_length > 0 and msg.speaker_embedding is None
                            assert float(msg.sampling_params.speaker_cfg_scale) == 2.0
                        assert len(t) + int(msg.sampling_params.max_tokens) + 1 <= 32768, "Long evaluation would truncate output for context capacity"
                        logger.info("LONG_INPUT_OK uid=%s mode=%s input_frames=%s reference_frames=%s max_tokens=%s cfg=%s", msg.uid, mode, len(t), reference_prefix_length, msg.sampling_params.max_tokens, msg.sampling_params.speaker_cfg_scale)
                    logger.debug(
                        "Tokenize uid=%d speaking_rate_bucket=%s quality_buckets=%s "
                        "text='%s...' frames=%d",
                        msg.uid,
                        msg.speaking_rate_bucket,
                        msg.quality_buckets,
                        msg.text[:50],
                        len(t),
                    )
                    user_msgs.append(
                        TTSUserMsg(
                            uid=msg.uid,
                            input_ids=t,
                            plain_input_ids=None if plain_prompt is None else _plain_twin_prompt(t, text_rows, plain_prompt),
                            speaker_reference_prefix_length=reference_prefix_length,
                            sampling_params=msg.sampling_params,
                            speaker_embedding=msg.speaker_embedding,
                            speaker_token_position=speaker_token_position,
                            speaker_emotion_delta=msg.speaker_emotion_delta,
                            steer_schedule=msg.steer_schedule,
                            steer_follow=msg.language if follow_tokens is not None else None,
                            steer_follow_tokens=follow_tokens,
                            steer_follow_gate=msg.steer_follow_gate if follow_tokens is not None else None,
                            clean_speaker_background=msg.clean_speaker_background,
                            accurate_mode=msg.accurate_mode,
                            condition_values=condition_values,
                            condition_token_start=condition_token_start,
                            decode_context_codes=None if context_codes is None else context_codes.to(torch.int32).contiguous(),
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
                batch_output = BatchTTSBackendMsg(
                    data=user_msgs
                )
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
                for receipt in tts_vocoder_manager.pop_prefix_receipts():
                    logger.info(
                        "prefix stream uid=%s: %s live chunks (%s/%s frames), live-vs-whole %s dB, max |d| %.4f, prefill %s, context %s",
                        receipt["uid"], receipt["chunks"], receipt["emitted_frames"], receipt["frames"],
                        receipt["live_vs_whole_rms_db"], receipt["live_vs_whole_max_abs"], receipt.get("prefill_frames", 0), receipt.get("context_frames", 0),
                    )
                frontend_replies.extend(
                    TTSAudioReply(uid=msg.uid, audio_data=audio, finished=msg.finished, laci=msg.laci)
                    for msg, audio in zip(vocoder_msgs, audio_chunks, strict=True)
                )
            if frontend_replies:
                batch_output = BatchTTSFrontendMsg(data=frontend_replies)
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                send_frontend.put(batch_output)

    except KeyboardInterrupt:
        pass

# CONDITION_ROWS_RUNTIME_20261016

# DECODE_CONTEXT_RUNTIME_20261017

# DECODE_CONTEXT_RUNTIME2_20261017

# SILENCE_DEFAULT_RUNTIME_20261017
