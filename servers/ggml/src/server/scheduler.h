// Continuous-batching scheduler: one worker thread owns the backbone and a zonos3_batch_ctx and only produces codes;
// DAC decode runs on a pool of lanes (each its own dac_model sharing the weights). HTTP handlers enqueue a ReqJob and
// read PCM blocks back through its channel.
#pragma once

#include "zonos3.h"
#include "zonos3-sampler.h"
#include "zonos3-steer.h"
#include "dac.h"
#include "zonos3-laci.h"
#include "zonos3-span-follow.h"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <optional>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <thread>
#include <vector>

namespace zs {

struct ServerState;

// A generation request resolved against the loaded model (generate.cpp builds it from a GenerateRequest).
struct GenReq {
    std::string text;                       // tokenized text (normalized when normalization ran)
    std::string raw_text;                   // the request's text; control spans index it
    std::string language;
    zonos3_sampling sp;
    int max_frames = 0;
    std::string speaker_mode;               // "embedding" | "reference" | "" (no speaker prompt)
    std::vector<int32_t> reference_codes;   // reference mode: ordinary [frames, n_codebooks]
    std::vector<float> speaker_embedding;   // embedding mode: [speaker_dim]
    std::string speaker_id;                 // the voice id (speaker_embedding_id), for calibrations
    std::vector<int32_t> condition_buckets; // [n_cond]; n_cond_buckets = unknown
    int silence_prefill = ZONOS3_DEFAULT_SILENCE_PREFILL;
    std::vector<int32_t> decode_context;    // decode_context: the reference's last frames, ordinary [K, n_codebooks]
    float cfg_scale = 1.0f;                 // speaker / tag / emotion CFG (1 = off; runs a twin slot)
    bool twin_steered = true;               // the twin decodes with the hidden steering delta (false: emotion CFG,
                                            // whose twin is the same prompt without the emotion / steering)
    int accent = -1;                        // accent adapter of model.accents (-1 = none), on both CFG branches
    float accent_strength = 0.0f;
    double headroom_db = -1.0;              // < 0: the decoder's own headroom
    double fade_out_ms = 0;
    bool stream = true;
    bool wav = false;
    bool prefix = false;                    // streaming with stream_decode=prefix
    zonos3_steering steering;
    // Speech-edit infill (edit_dub.cpp): the stop id (-1 = EOA; span-end for an infill), masked until min_frames frames
    // are generated (sgl min_frames); a codes-only job finishes with its generated rows in ReqJob::codes and decodes
    // nothing.
    int stop_id = -1;
    int min_frames = 0;
    bool codes_only = false;
    // LACI (laci=true; zonos3-laci.h): the language's alignment heads (null = off) and, for a dub take, its duration
    // target and extended speaking-rate scale (nullopt: the checkpoint's speaking-rate row) with the take's bucket on it
    // and its hidden steering delta in every bucket of the scale (rates.first..rates.last, steering layout; empty
    // vector = none). A rollback re-prefills the prompt plus the kept rows (zonos3_laci::Continuation).
    const zonos3_laci::LaciHeads * laci = nullptr;
    std::optional<zonos3_laci::DurationTarget> laci_duration;
    std::optional<zonos3_laci::RateBuckets> laci_rates;
    std::optional<int> laci_bucket;
    std::optional<std::vector<std::vector<float>>> laci_rate_deltas;
    // control_span_follow "reading" (zonos3-span-follow.h): the language's heads (null = the character-rate schedule
    // of steering.spans), each steering span's text-span token range [a, b) and the follow gate.
    const zonos3_laci::LaciHeads * follow = nullptr;
    std::vector<std::pair<int, int>> follow_tokens;
    double follow_lead = 1.0, follow_lag = 0.0;
};

// A DAC decode unit handed from the backbone worker to the job's lane, with a snapshot of the code rows (the worker
// keeps appending to job->codes). WINDOW stream-decodes frames up to f_hi (prefix streaming); FINALIZE does the exact
// whole decode (emitting what the stream has not), then finishes the channel. `drop` finishes without decoding
// (client gone); `fail` finishes as failed.
struct ReqJob;
struct DacTask {
    std::shared_ptr<ReqJob> job;
    std::vector<int32_t> codes;   // delayed rows [H, n_codebooks], decode-context rows first
    int H = 0;
    enum Kind { WINDOW, FINALIZE } kind = WINDOW;
    int f_hi = 0;
    bool drop = false;
    bool fail = false;
};

struct DacLane {
    dac_model dac;
    std::thread th;
    std::mutex m;
    std::condition_variable cv;
    std::deque<DacTask> q;
};

struct ReqJob {
    // request (handler-filled, read-only to the worker)
    GenReq gr;
    zonos3_prompt prompt;
    std::vector<float> idf;              // prompt ids [n0 * W] as floats
    int n0 = 0;
    zonos3_prompt twin;                  // speaker-CFG twin prompt (empty without CFG)
    std::vector<float> idf_twin;
    int ctx_frames = 0;                  // decode_context frames decoded ahead of the stream (gr.decode_context)
    int64_t skip_samples = 0;            // decoded audio never emitted: decode context + silence prefill
    float gain = 1.0f;                   // decoder headroom, applied by the lane

    // runtime (worker-owned)
    zonos3_slot slot;
    std::unique_ptr<zonos3_sampler> smp;
    std::vector<float> logits;
    zonos3_slot twin_slot;
    int twin_index = -1;
    std::vector<float> twin_logits, guided;
    std::vector<int32_t> codes;          // delayed audio-stream rows (silence prefill first)
    int eos_frame = -1;
    int emitted = 0;                     // prefix streaming: frames handed to the lane
    int lane = 0;
    dac_stream dst;                      // lane-owned incremental decoder state (prefix streaming)
    int64_t streamed_samples = 0;        // lane-owned: raw samples the stream decode produced

    std::chrono::steady_clock::time_point t_decode0;
    double decode_step_ms = 0.0;
    // Alignment probe (LACI / span follow; attach_probe, then worker-owned). The session owns the stream: only drained
    // rows reach `codes`; a rollback re-prefills the slot in place.
    zonos3_align_probe probe;            // heads and text span of the conditional prompt
    std::unique_ptr<zonos3_follow::SpanFollower> follower;
    std::unique_ptr<zonos3_laci::LaciSession> laci;
    zonos3_laci::ProbeControl probe_ctl;
    bool readout_ready = false;          // slot.readout holds the last decode step's, not yet observed
    int history_rows = 0;                // generated rows before the current attempt (steering / follower frames)
    int attempt_max = 0;                 // the current attempt's row budget (max_frames, then Continuation::max_tokens)
    std::string laci_record;             // LaciSession.record() as JSON once final (empty: not a LACI job)

    // channel: DAC lane -> handler
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::vector<float>> pcm;
    bool finished = false;
    bool failed = false;
    bool client_dropped = false;
};

// Prefix streaming (sgl TTSVocoderManager): a chunk once ZONOS3_PREFIX_STREAM_CHUNK_FRAMES new frames are decodable
// beyond ZONOS3_PREFIX_STREAM_HOLD_BACK_FRAMES held-back frames.
struct PrefixStreamSettings {
    int chunk_frames = 15;
    int hold_back_frames = 3;
    int context_frames = 450;
};

void start_scheduler(ServerState & s);   // DAC lanes + backbone worker; waits until the worker is ready
void stop_scheduler(ServerState & s);
void submit_job(ServerState & s, const std::shared_ptr<ReqJob> & job);
// Whole-decode the job's delayed code rows (job->codes; eos_frame < 0 = every complete frame) on a DAC lane without
// the backbone; the PCM arrives on the job's channel like a generation's.
void submit_decode(ServerState & s, const std::shared_ptr<ReqJob> & job);

// The alignment probe of a job whose prompt is built (gr.laci / gr.follow): the probed text span, the span follower and
// the LACI session. Throws std::invalid_argument (a 400) when the request cannot run it (text longer than the probe's
// capacity, LaciSession's own checks).
void attach_probe(const ServerState & s, ReqJob & job);
}  // namespace zs
