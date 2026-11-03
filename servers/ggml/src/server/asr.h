// Qwen3-ASR speech recognition through model/src/zonos3/workers/asr.py: one persistent worker process per server in
// the ASR interpreter (ZONOS3_ASR_PYTHON, default <repo>/.venv-asr/bin/python, set up by tools/setup_asr_env.sh),
// started on first use, requests serialized, restarted once when it exits. Shared by speech editing (/v2/edit*) and
// dubbing (/v2/dub*); the Python twin is zonos3.tts.edit_align.AsrWorker.
#pragma once

#include "server/workers.h"

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace zs {

// "ASR is unavailable: ..." (503); input the worker rejects is a std::invalid_argument "ASR failed: ..." (400).
struct AsrError : WorkerUnavailable {
    using WorkerUnavailable::WorkerUnavailable;
};

struct AsrItem {
    std::string audio;                    // a sound file soundfile reads
    std::optional<double> start, end;     // the part of the file to recognize, in seconds
    std::optional<std::string> language;  // a Qwen3-ASR language name (AsrWorker::languages); nullopt detects it
};

struct AsrResult {
    std::string text;                     // stripped
    std::string language;                 // as Qwen3-ASR reports it ("" for an empty part)
};

class AsrWorker {
public:
    // python_src: the directory holding the zonos3 package. The interpreter is ZONOS3_ASR_PYTHON, else
    // <python_src>/../../.venv-asr/bin/python; ZONOS3_ASR_DEVICE (when set) is passed as --device.
    explicit AsrWorker(std::string python_src);
    // The interpreter exists (the speech_editing capability needs it).
    bool installed() const;
    // One result per item, in order (one batched request).
    std::vector<AsrResult> transcribe(const std::vector<AsrItem> & items);
    // The model id and the language names of the worker's ping (cached); start the worker.
    std::string model();
    std::vector<std::string> languages();
    void close();

private:
    nlohmann::json call(nlohmann::json request);   // m_ held
    void ping();                                     // m_ held

    std::string python_src_, python_;
    std::vector<std::string> argv_;
    std::mutex m_;
    std::unique_ptr<JsonLinesWorker> worker_;
    std::optional<std::string> model_;
    std::vector<std::string> languages_;
    long long next_id_ = 0;
};

}  // namespace zs
