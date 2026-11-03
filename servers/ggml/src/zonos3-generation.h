// The generation registry model/src/zonos3/registries/generation.json (Python: zonos3.generation): the request
// defaults and limits every server shares, embedded at build time and parsed on first use.
#pragma once

struct zonos3_generation_registry {
    int sample_rate;                    // the API's audio rate (Hz)
    struct {                            // the v2 sampling defaults (api/openapi.yaml repeats them)
        float temperature, top_p, min_p, repetition_penalty;
        int topk, repetition_window;
        int repetition_codebooks;       // < 0 in a request means every codebook
    } sampling;
    double tag_cfg_min_scale, tag_cfg_max_scale;            // the tag-CFG scale range
    int silence_prefill_default_frames;                     // server default (ZONOS3_SILENCE_PREFILL_FRAMES overrides)
    int silence_prefill_max_frames;                         // the longest silence_prefill_frames a request may ask for
    int decode_context_frames;                              // decode_context_frames unset or 0
    int reference_max_frames;                               // the reference prefix; longer codes keep their first frames
    int max_frames;                                         // the default --max frame budget
    double reference_audio_max_seconds, upload_max_seconds, edit_max_seconds;   // the longest reference clip, upload
                                                                                // and speech-edit clip
};

const zonos3_generation_registry & zonos3_generation();
