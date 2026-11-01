"""Emotion sliders and emotion CFG."""

from __future__ import annotations

from .support import error, generate, needs, pcm


def _voice(caps, default_voice) -> dict:
    if "embedding" in caps["speaker_modes"]:
        return {"speaker_mode": "embedding", "speaker_embedding_id": default_voice["id"]}
    return {"speaker_embedding_id": default_voice["id"]} if caps["speaker_enabled"] else {}


def _name(caps) -> str:
    return (caps["emotion_names"] or caps["emotion_axes"])[0]


def test_sliders(client, caps, default_voice):
    needs(caps["emotion_enabled"], "the server has no emotion directions")
    pcm(generate(client, {**_voice(caps, default_voice), "emotion_enabled": True,
                          "emotion_sliders": {_name(caps): 0.5}}))


def test_emotion_cfg(client, caps, default_voice):
    needs(caps["emotion_enabled"], "the server has no emotion directions")
    pcm(generate(client, {**_voice(caps, default_voice), "emotion_enabled": True,
                          "emotion_sliders": {_name(caps): 0.5}, "emotion_cfg_scale": 1.5}))


def test_unknown_emotion_is_400(client, caps, default_voice):
    needs(caps["emotion_enabled"], "the server has no emotion directions")
    error(generate(client, {**_voice(caps, default_voice), "emotion_enabled": True,
                            "emotion_sliders": {"no_such_emotion": 0.5}}), 400)
