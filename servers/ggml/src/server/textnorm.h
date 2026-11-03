// TTS text normalization through model/src/zonos3/workers/textnorm.py (zonos3.tokenizer.textnorm): one persistent
// worker process, requests serialized, restarted once when it exits.
#pragma once

#include "server/workers.h"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace zs {

class TextNormalizer {
public:
    // python: the --worker-python interpreter; python_src: the directory holding the zonos3 package.
    TextNormalizer(std::string python, std::string python_src);
    // The normalized text; throws WorkerError when the worker fails twice or reports an error.
    std::string normalize(const std::string & text, const std::string & language);
    void close();

private:
    std::string python_, python_src_;
    std::mutex m_;
    std::unique_ptr<JsonLinesWorker> worker_;
    long long next_id_ = 0;
};

}  // namespace zs
