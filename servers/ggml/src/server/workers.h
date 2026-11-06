// Helper processes of the server (model/src/zonos3/workers/*.py): ChildProcess, a child with its stdin and stdout on
// pipes, and JsonLinesWorker, a persistent one speaking one JSON request per stdin line and one response per stdout
// line, in order (text normalization). The Python host (server/pyhost.h) multiplexes its own ChildProcess.
#pragma once

#include "json.hpp"

#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace zs {

struct WorkerError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// A started child process: stdin and stdout are pipes, stderr is the server's. write() and read_line() may run on
// different threads; every other call (and two concurrent writes) must be serialized by the owner.
class ChildProcess {
public:
    // argv[0] is the program (searched on PATH when it has no slash). `env` overrides the server's environment: a
    // value sets the variable, nullopt removes it. Throws WorkerError when the process cannot start.
    ChildProcess(const std::vector<std::string> & argv, const std::map<std::string, std::optional<std::string>> & env);
    ~ChildProcess();   // wait()
    ChildProcess(const ChildProcess &) = delete;
    ChildProcess & operator=(const ChildProcess &) = delete;

    // All of `data` to the child's stdin; false once stdin is closed or the child stopped reading.
    bool write(const std::string & data);
    // The next stdout line without its '\n'; false at end of stream.
    bool read_line(std::string & line);
    // Close the child's stdin (it sees end of input).
    void close_input();
    // Close stdin, wait for the child to exit and close stdout.
    void wait();

private:
    std::string pending_;          // bytes read past the last line
#ifdef _WIN32
    void * process_ = nullptr, * in_ = nullptr, * out_ = nullptr;
#else
    int pid_ = -1, in_ = -1, out_ = -1;
#endif
};

class JsonLinesWorker {
public:
    // See ChildProcess for `argv` and `env`.
    JsonLinesWorker(std::vector<std::string> argv, std::map<std::string, std::optional<std::string>> env);
    JsonLinesWorker(const JsonLinesWorker &) = delete;
    JsonLinesWorker & operator=(const JsonLinesWorker &) = delete;

    // One request -> its response line. Starts the process on first use. Throws WorkerError when the process cannot
    // start, exits, or answers with a line that is not JSON. Thread-safe (calls are serialized).
    nlohmann::json call(const nlohmann::json & request);
    // Close stdin and wait for the process to exit.
    void close();

private:
    std::vector<std::string> argv_;
    std::map<std::string, std::optional<std::string>> env_;
    std::mutex m_;
    std::unique_ptr<ChildProcess> process_;
};

// Runs argv[0] (searched on PATH) with argv, stdin and stdout on the null device, and waits for it. Returns the exit
// code (-1 when it could not start or did not exit normally) with the child's stderr in `stderr_text`.
int run_process(const std::vector<std::string> & argv, std::string & stderr_text);
// The same with the server environment plus `env` (a nullopt value removes the variable), as ChildProcess.
int run_process(const std::vector<std::string> & argv, const std::map<std::string, std::optional<std::string>> & env,
                std::string & stderr_text);

}  // namespace zs
