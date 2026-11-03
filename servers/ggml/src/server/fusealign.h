// The FuseAlign forced aligner (fusealign_hub) through model/src/zonos3/workers/fusealign.py: one persistent worker
// process per server under --worker-python, started on first use, requests serialized, restarted once when it exits.
// Used for every word timing: speech editing (/v2/edit*) and dubbing (/v2/dub*).
#pragma once

#include "server/workers.h"

#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace zs {

// The worker cannot run or failed on its own (503); a request the worker rejects as invalid input (a ValueError in
// the worker) is a std::invalid_argument instead (400).
struct FuseAlignError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct FuseAlignItem {
    std::string audio;                 // a sound file soundfile reads
    std::vector<std::string> tokens;   // the words to place (zonos3.tts.align_text.align_tokens)
    std::string language;              // Whisper code for long-form anchoring; empty = null
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
    // The checkpoint id of the worker's ping (Zyphra/fusealign/<file> or a local path); starts the worker.
    std::string model();
    void close();

private:
    nlohmann::json call(nlohmann::json request);   // m_ held
    void start();                                    // m_ held

    std::string python_, python_src_;
    std::mutex m_;
    std::unique_ptr<JsonLinesWorker> worker_;
    std::string model_;
    long long next_id_ = 0;
};

}  // namespace zs
