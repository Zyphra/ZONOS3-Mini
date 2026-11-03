// The server's shared state: loaded models and companions, the speaker store, the scheduler and feature settings.
#pragma once

#include "server/edit_dub.h"
#include "server/scheduler.h"
#include "server/textnorm.h"

#include "dac-encoder.h"
#include "dac.h"
#include "spk-encoder.h"
#include "ssl-encoder.h"
#include "zonos3-accent.h"
#include "zonos3-emotion.h"
#include "zonos3-steer.h"
#include "zonos3.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

class zonos3_phoneme_lexicon;   // zonos3-phonemes.h

namespace zs {

class ProjectStore;

// A voice: a default voice (default_voices/) or one cached in a session (POST /v2/speakers).
struct Speaker {
    std::string id, label;
    std::string source_type;                  // "audio" | "embedding"
    std::optional<std::string> original_name;
    std::optional<std::string> transcript;    // default voices: the <voice>.txt sidecar
    std::string path, codes_path;             // default voices: the file and an audio voice's <stem>.codes.npy
    std::vector<int32_t> codes;               // reference codes, ordinary [frames, 10] (lazy for audio voices)
    std::vector<float> embedding;             // [speaker_dim] (lazy for audio voices)
    std::vector<float> pcm;                   // 44.1 kHz audio of an audio voice (preview, lazy compute)
    double created = 0, mtime = 0, codes_mtime = 0;
    bool is_default = false;
};

struct ServerState {
    // ---- models
    zonos3_model model;
    std::string model_path;
    dac_model dac;                         // owns the DAC weights shared by every lane
    std::string dac_name;                  // dac.gguf file name (capabilities.decoder)
    dac_encoder_model encoder;
    ssl_model teacher;
    std::string encoder_path, teacher_path;
    bool reference_models_loaded = false;
    spk_model spk;
    std::string spk_path;
    bool spk_loaded = false;
    int reference_idle_unload_s = 60;
    std::chrono::steady_clock::time_point reference_last_use;   // guarded by reference_compute_mtx
    std::string steer_path;
    bool steer_loaded = false;
    zonos3_steer_bundle steer;
    // Silence prefill (sgl _silence_prefill_spec): ZONOS3_SILENCE_PREFILL_FRAMES (default 3) frames for a request
    // without silence_prefill_frames, unless ZONOS3_SILENCE_PREFILL_DEFAULT is 0/false/no/off.
    int silence_prefill_frames = ZONOS3_DEFAULT_SILENCE_PREFILL;
    bool silence_prefill_on = true;
    std::string controls_doc;               // GET /v2/controls

    // ---- settings
    int max_frames = 2000;
    int context_capacity = 0;
    int batch_slots = 8;
    int dac_threads = 4;
    bool use_gpu = false;
    bool prof = false;
    std::string stream_decode_default = "prefix";
    PrefixStreamSettings prefix;
    std::string studio_dir;
    std::string default_voice_dir = "default_voices";
    std::string python_src;                 // directory holding the zonos3 package (model/src or <exe>/python)
    std::string worker_python;              // --worker-python: interpreter of <python_src>/zonos3/workers
    std::unique_ptr<TextNormalizer> text_normalizer;   // set when --worker-python is given
    std::unique_ptr<EditDubWorker> edit_dub;           // speech editing and dubbing; set when --worker-python is given
    bool fusealign_installed = false;                  // the --worker-python interpreter has fusealign_hub
    // Companions of later features (emotion.gguf, accents.gguf, accent-directions.gguf, --laci-heads LANG=PATH,
    // --tts-phoneme-lexicon DIR); recorded here so every module sees the configuration.
    std::string emotion_directions_path, accent_adapters_path, accent_directions_path, phoneme_lexicon_dir;
    // Emotion directions the backbone can consume (sgl _emotion_directions_for_model); null = emotion control off.
    std::shared_ptr<zonos3_directions> emotion;
    std::shared_ptr<zonos3_directions> accent_directions;   // --accent-directions (attnres_source or raw)
    std::shared_ptr<zonos3_accents> accent_adapters;         // --accent-adapters, attached as model.accents
    // Full AttnRes bank sources that take a hidden delta (sgl _steering_sources: emotion + accent directions + the
    // steering bundle's hidden sets), ascending: the layout of every request's zonos3_steering.
    std::vector<int> steer_sources;
    std::shared_ptr<zonos3_phoneme_lexicon> phoneme_lexicon;   // --tts-phoneme-lexicon, opened at startup (phonemize.cpp)
    std::map<std::string, std::string> laci_heads;   // --laci-heads: language -> heads JSON path
    // LACI / span-follow alignment heads by language (zonos3_laci::LaciHeads::load of laci_heads, checked against the
    // backbone's attention heads at startup) and the probe's text capacity (zonos3_batch_features::probe_max_text:
    // text rows a probed request may hold; 0 without heads).
    std::map<std::string, zonos3_laci::LaciHeads> laci;
    int probe_max_text = 0;
    std::unique_ptr<ProjectStore> projects;          // --studio-projects-dir; null = disabled

    // ---- scheduler (scheduler.cpp)
    std::thread worker;
    std::atomic<bool> stop{false};
    std::mutex sched_mtx;
    std::condition_variable sched_cv;
    std::deque<std::shared_ptr<ReqJob>> incoming;
    bool worker_ready = false, worker_failed = false;
    std::vector<DacLane> dac_lanes;
    std::atomic<uint64_t> next_lane{0};      // the backbone worker and submit_decode (handler threads) pick lanes

    // ---- speakers (speakers.cpp)
    std::mutex reference_mtx;              // guards sessions / default_speakers
    std::mutex reference_compute_mtx;      // serializes the native encoders (not reentrant)
    std::map<std::string, std::map<std::string, Speaker>> sessions;   // session -> id -> voice
    std::map<std::string, Speaker> default_speakers;                   // id -> voice
    std::map<std::string, double> default_skipped;                     // unreadable embedding files already warned
    std::thread reference_reaper;
    std::mutex reference_reaper_mtx;
    std::condition_variable reference_cv;

    ~ServerState();
};

}  // namespace zs
