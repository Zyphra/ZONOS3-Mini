// A persistent child process speaking JSON lines: one request object per stdin line, one response object per stdout
// line, in order (model/src/zonos3/workers/*.py). Used by text normalization, the MMS aligner (edit) and the dub
// alignment worker.
#pragma once

#include "json.hpp"

#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace zs {

struct WorkerError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

class JsonLinesWorker {
public:
    // argv[0] is the program (searched on PATH when it has no slash). `env` overrides the server's environment:
    // a value sets the variable, nullopt removes it. The child's stderr is the server's.
    JsonLinesWorker(std::vector<std::string> argv, std::map<std::string, std::optional<std::string>> env);
    ~JsonLinesWorker();
    JsonLinesWorker(const JsonLinesWorker &) = delete;
    JsonLinesWorker & operator=(const JsonLinesWorker &) = delete;

    // One request -> its response line. Starts the process on first use. Throws WorkerError when the process cannot
    // start, exits, or answers with a line that is not JSON. Thread-safe (calls are serialized).
    nlohmann::json call(const nlohmann::json & request);
    bool alive();
    // Close stdin and wait for the process to exit.
    void close();

private:
    void start_locked();
    void close_locked();
    bool read_line_locked(std::string & line);

    std::vector<std::string> argv_;
    std::map<std::string, std::optional<std::string>> env_;
    std::mutex m_;
    std::string pending_;          // bytes read past the last line
#ifdef _WIN32
    void * process_ = nullptr, * in_ = nullptr, * out_ = nullptr;
#else
    int pid_ = -1, in_ = -1, out_ = -1;
#endif
};

// Runs argv[0] (searched on PATH) with argv, stdin and stdout on the null device, and waits for it. Returns the exit
// code (-1 when it could not start or did not exit normally) with the child's stderr in `stderr_text`.
int run_process(const std::vector<std::string> & argv, std::string & stderr_text);

}  // namespace zs
