// The long-form orchestration of the ASR (server/asr.h) and FuseAlign (server/fusealign.h) workers, shared by speech
// editing and dubbing: a port of model/src/zonos3/tts/edit_align.py (needs_anchors, chunk_utterances,
// transcribe_audio, align_words).
//
// A FuseAlign item whose audio is longer than LONG_FORM_SECONDS or whose tokens (joined with spaces) exceed
// MAX_TEXT_BYTES UTF-8 bytes is routed by approximate word times. They come from Qwen3-ASR on the recording's VAD
// chunks (the FuseAlign worker's `segments`): every chunk transcribed gives one utterance {text, start, end} (empty
// texts dropped), sent as the item's `utterances`; FuseAlign's built-in Whisper anchoring never runs. A transcript of
// a recording over LONG_FORM_SECONDS is made the same way, chunk by chunk, and its utterances anchor the alignments
// that follow. Worker failures are WorkerUnavailable (AsrError / FuseAlignError, 503) or std::invalid_argument (400),
// each with its final message.
#pragma once

#include "server/state.h"

#include <optional>
#include <string>
#include <vector>

namespace zs {

constexpr double LONG_FORM_SECONDS = 30.0;   // fusealign_hub.aligner.MAX_SECONDS: one FuseAlign forward
constexpr size_t MAX_TEXT_BYTES = 1022;      // fusealign_hub.longform.MAX_TEXT_BYTES

// The server's workers; AsrError / FuseAlignError when the server has none (no asr.py / no --worker-python).
AsrWorker & asr_worker(const ServerState & s);
FuseAligner & fusealign_worker(const ServerState & s);

// Whether FuseAlign aligns an item long-form: audio over LONG_FORM_SECONDS or tokens over MAX_TEXT_BYTES.
bool needs_anchors(double seconds, const std::vector<std::string> & tokens);

struct Transcript {
    std::string text;
    std::string language;   // as Qwen3-ASR reports it: the forced language, or the detected one(s), comma-joined
    std::optional<std::vector<FuseAlignUtterance>> utterances;   // the chunk transcripts of audio over LONG_FORM_SECONDS
};

// Qwen3-ASR transcript of a sound file of `seconds` (frames / sample rate), forced to `language` (a Qwen3-ASR name) or
// detected (nullopt). Up to LONG_FORM_SECONDS one recognition of the whole file; longer audio chunk by chunk: the
// chunk texts joined with " " ("" when the language, the forced one or else the first spoken chunk's first detected
// one, is Japanese / Chinese / Cantonese), the first spoken chunk's reported language ("" without speech), and the
// utterances.
Transcript transcribe_audio(const ServerState & s, const std::string & audio, double seconds,
                            const std::optional<std::string> & language);

struct AlignWordsItem {
    std::string audio;
    double seconds = 0;                   // the file's duration (frames / sample rate)
    std::vector<std::string> tokens;      // zonos3.tts.align_text.align_tokens
    std::optional<std::string> language;  // the Qwen3-ASR name anchors are recognized in; nullopt detects
    std::optional<std::vector<FuseAlignUtterance>> utterances;   // anchors already made for the audio
};

// FuseAlign words of the items, one per token. An item that needs_anchors carries utterances: its own, else the chunk
// transcripts of its audio (one segments and one transcribe call for every such item); the others carry none.
std::vector<std::vector<FuseAlignWord>> align_words(const ServerState & s, const std::vector<AlignWordsItem> & items);

}  // namespace zs
