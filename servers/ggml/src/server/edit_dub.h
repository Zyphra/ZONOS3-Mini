// The edit/dub worker: speech editing (/v2/edit*) and dubbing (/v2/dub/*) run in Python, in
// <python_src>/zonos3/workers/edit_dub.py (zonos3.tts.edit_runtime.EditRuntime, zonos3.dub.runtime.DubRuntime and their
// Qwen3-ASR / FuseAlign workers), the same implementation the sgl and mlx servers run in-process.
//
// One persistent process under --worker-python, started on first use and again after it exited; `init` (the
// checkpoint's edit and dub facts) is its first request after every start. stdin / stdout carry JSON lines both ways,
// multiplexed: requests {"id", "op", ...} are answered by id, many in flight; while one runs, the worker calls back into
// the served model ({"call", "op", ...}: decode_audio, embed, encode, check, fill, decode, synthesize), and every call
// is answered on its own thread (the calls block on the scheduler). Audio crosses as file paths.
#pragma once

#include "json.hpp"

#include <atomic>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace zs {

struct ServerState;

class EditDubWorker {
public:
    explicit EditDubWorker(ServerState & s);
    ~EditDubWorker();   // close()
    EditDubWorker(const EditDubWorker &) = delete;
    EditDubWorker & operator=(const EditDubWorker &) = delete;

    // One request {"op": op, ...payload}: its result. Throws HttpError with the worker's status and detail when it
    // answers with an error, and 503 when the worker cannot start or exits before answering.
    nlohmann::json request(const std::string & op, nlohmann::json payload);
    // Ends the worker's input, waits for it to exit and for every engine call to finish.
    void close();

    // The dub's take slots: takes in flight across requests (twice the running-request cap) so requests interleave.
    void acquire_take_slot();
    void release_take_slot();

private:
    struct Run;
    std::shared_ptr<Run> ensure_run();   // start_m_ held
    nlohmann::json exchange(Run & run, const std::string & op, nlohmann::json payload);
    void read_loop(std::weak_ptr<Run> weak);
    void answer(std::shared_ptr<Run> run, nlohmann::json call);

    ServerState & s_;
    std::vector<std::string> argv_;
    std::map<std::string, std::optional<std::string>> env_;
    std::mutex start_m_;                // serializes (re)starts and their init
    std::shared_ptr<Run> run_;          // the current process
    std::atomic<int64_t> next_id_{0};
    std::mutex calls_m_;                // guards calls_
    std::condition_variable calls_cv_;
    int calls_ = 0;                     // engine calls running
    std::mutex slots_m_;
    std::condition_variable slots_cv_;
    int free_slots_;
};

// The interpreter of the worker's Qwen3-ASR worker exists: ZONOS3_ASR_PYTHON, else <python_src>/../../.venv-asr/bin/python
// (set up by tools/setup_asr_env.sh).
bool asr_installed(const ServerState & s);
// The speaking-rate condition row's bucket edges (UTF-8 bytes / s); empty without that row.
std::vector<double> speaking_rate_edges(const ServerState & s);
// zonos3.tts.edit_runtime.EditModel of the checkpoint ({frame_rate, n_codebooks, audio_pad_id, codebook_sizes,
// max_spans (0 when it cannot edit), speaker_row}): the worker's `init` and the edit capabilities.
nlohmann::json edit_model(const ServerState & s);

}  // namespace zs
