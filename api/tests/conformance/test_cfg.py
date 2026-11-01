"""Speaker and emotion CFG request rules."""

from __future__ import annotations

from .support import error, generate, needs, pcm


def test_speaker_cfg_with_a_voice(client, caps, default_voice):
    needs(caps["speaker_cfg_supported"], "speaker CFG is unsupported")
    pcm(generate(client, {"speaker_embedding_id": default_voice["id"], "speaker_cfg_scale": 1.5}))


def test_speaker_cfg_without_a_voice_is_400(client, caps):
    needs(caps["speaker_cfg_supported"], "speaker CFG is unsupported")
    error(generate(client, {"speaker_cfg_scale": 1.5}), 400)


def test_speaker_and_emotion_cfg_is_400(client, caps, default_voice):
    needs(caps["speaker_cfg_supported"] and caps["emotion_enabled"], "needs speaker CFG and emotion directions")
    error(generate(client, {"speaker_embedding_id": default_voice["id"], "speaker_cfg_scale": 1.5,
                            "emotion_cfg_scale": 1.5}), 400)
