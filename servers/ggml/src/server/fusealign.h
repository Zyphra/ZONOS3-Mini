// The FuseAlign forced aligner (fusealign_hub) through model/src/zonos3/workers/fusealign.py: one persistent worker
// process per server under --worker-python, started on first use, requests serialized, restarted once when it exits.
// Used for every word timing: speech editing (/v2/edit*) and dubbing (/v2/dub*). FuseAlign's own Whisper long-form
// anchoring never runs: a long item carries Qwen3-ASR utterances as its anchors (server/edit_align.h).
#pragma once

#include "server/workers.h"

#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace zs {

// "FuseAlign alignment is unavailable: ..." (503); input the worker rejects (a ValueError in it) is a
// std::invalid_argument "FuseAlign alignment failed: ..." (400).
struct FuseAlignError : WorkerUnavailable {
    using WorkerUnavailable::WorkerUnavailable;
};

// A timed utterance: the long-form anchors of an item (fusealign_hub.longform.guide_words).
struct FuseAlignUtterance {
    std::string text;
    double start = 0, end = 0;
};

struct FuseAlignItem {
    std::string audio;                 // a sound file soundfile reads
    std::vector<std::string> tokens;   // the words to place (zonos3.tts.align_text.align_tokens)
    std::optional<std::vector<FuseAlignUtterance>> utterances;   // long-form anchors (sent when set, even empty)
};

struct FuseAlignWord {
    std::string word;
    double start = 0, end = 0;
    double presence = 0;               // how surely the word is spoken
    bool placed = false;               // false: FuseAlign's fallback times
};

class FuseAligner {
public:
    // python: the --worker-python interpreter; python_src: the directory holding the zonos3 package.
    FuseAligner(std::string python, std::string python_src);
    // One word list per item, one word per token (an item without tokens gets none and is not sent).
    std::vector<std::vector<FuseAlignWord>> align(const std::vector<FuseAlignItem> & items);
    // Per audio file: silero speech regions packed into chunks of at most 25 s ([start, end] seconds), the ASR
    // chunks of long-form anchoring and of long transcriptions.
    std::vector<std::vector<std::pair<double, double>>> segments(const std::vector<std::string> & audios);
    // The checkpoint id of the worker's ping (Zyphra/fusealign/<file> or a local path); starts the worker.
    std::string model();
    void close();

private:
    nlohmann::json call(nlohmann::json request);   // m_ held

    std::vector<std::string> argv_;
    std::string python_src_;
    std::mutex m_;
    std::unique_ptr<JsonLinesWorker> worker_;
    std::optional<std::string> model_;
    long long next_id_ = 0;
};

}  // namespace zs
