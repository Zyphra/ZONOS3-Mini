// Dubbing (/v2/dub/*): a port of model/src/zonos3/dub/runtime.py (DubRuntime) and the /v2/dub routes of
// zonos3_sgl/server/api_server.py over dub-plan (zonos3/dub/plan.py). The references and dubs live in two small LRU
// stores; the dubbing worker (<python-src>/zonos3/workers/dub_align.py in the .venv-asr interpreter: SimAlign word
// links, AV-sync) is a JsonLinesWorker started on first use and again after it died; transcripts come from the
// server's ASR worker (server/asr.h: Qwen3-ASR) and word timings from its FuseAlign worker (server/fusealign.h), both
// through server/edit_align.h (Qwen3-ASR anchors past 30 s); every take is a LACI /v2/generate request on the
// scheduler (plan_laci_take) with the take's duration target.
#include "compat.h"

#include "server/dub.h"

#include "server/capabilities.h"
#include "server/edit_align.h"
#include "server/generate.h"
#include "server/http.h"
#include "server/phonemize.h"
#include "server/schema.h"
#include "server/speakers.h"
#include "server/util.h"
#include "server/workers.h"

#include "align-text.h"
#include "audio-decode.h"
#include "dub-plan.h"
#include "zonos3-edit.h"
#include "zonos3-markup.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <list>
#include <mutex>
#include <set>
#include <thread>

namespace zs {

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

constexpr int STORE_CAPACITY = 8;
constexpr size_t MAX_REF_WORDS = 20000;
const char * const GROUP_OPEN = "\xEE\x80\x80";   // U+E000
const char * const GROUP_CLOSE = "\xEE\x80\x81";  // U+E001

// runtime.py error classes: UnknownId -> 404, AlignWorkerError -> 503, the ASR / FuseAlign WorkerUnavailable -> 503,
// ValueError (std::invalid_argument) -> 400, RuntimeError (std::runtime_error) -> 500 (_dub_call).
struct UnknownId : std::exception {
    const char * what() const noexcept override { return "unknown id"; }
};
struct AlignWorkerError : std::runtime_error {
    using std::runtime_error::runtime_error;
};
[[noreturn]] void value_error(const std::string & msg) { throw std::invalid_argument(msg); }

bool set(const json & j, const char * key) { return j.is_object() && j.contains(key) && !j[key].is_null(); }

std::string fmt(const char * f, double v) {
    char buf[64];
    snprintf(buf, sizeof buf, f, v);
    return buf;
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// The last `n` bytes of `s`, starting on a UTF-8 boundary (Python's str[-n:] of decoded text).
std::string tail_text(const std::string & s, size_t n) {
    size_t i = s.size() > n ? s.size() - n : 0;
    while (i < s.size() && ((unsigned char) s[i] & 0xC0) == 0x80) ++i;
    return s.substr(i);
}

// ------------------------------------------------------------------------------------------------------------ files

// A scratch directory removed with its contents (tempfile.TemporaryDirectory).
struct TempDir {
    fs::path path;
    explicit TempDir(const char * prefix) {
        path = fs::temp_directory_path() / (std::string(prefix) + random_hex(16));
        fs::create_directories(path);
    }
    ~TempDir() { std::error_code ec; fs::remove_all(path, ec); }
    TempDir(const TempDir &) = delete;
    TempDir & operator=(const TempDir &) = delete;
    std::string file(const std::string & name) const { return (path / name).string(); }
};

// soundfile.write(path, audio, sr, subtype="FLOAT"): a mono IEEE-float WAV.
void write_float_wav(const std::string & path, const std::vector<float> & audio, int sr) {
    std::vector<uint8_t> out;
    auto bytes = [&](const void * p, size_t n) { const uint8_t * b = (const uint8_t *) p; out.insert(out.end(), b, b + n); };
    auto w32 = [&](uint32_t v) { bytes(&v, 4); };
    auto w16 = [&](uint16_t v) { bytes(&v, 2); };
    const uint32_t data = (uint32_t) (audio.size() * 4);
    out.reserve(44 + data);
    bytes("RIFF", 4); w32(36 + data); bytes("WAVE", 4);
    bytes("fmt ", 4); w32(16); w16(3); w16(1); w32((uint32_t) sr); w32((uint32_t) sr * 4); w16(4); w16(32);
    bytes("data", 4); w32(data);
    bytes(audio.data(), audio.size() * 4);
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + path);
    const bool ok = fwrite(out.data(), 1, out.size(), f) == out.size();
    if (fclose(f) != 0 || !ok) throw std::runtime_error("cannot write " + path);
}

// A WAV file (PCM 8/16/24/32-bit or IEEE float 32/64, plain or extensible) as mono float32, channels averaged
// (soundfile.read(..., always_2d=True).mean(axis=1)). False when it is not such a file.
bool read_wav(const std::string & path, std::vector<float> & audio, int & sr) {
    std::string d;
    if (!read_file(path, d) || d.size() < 12 || d.compare(0, 4, "RIFF") || d.compare(8, 4, "WAVE")) return false;
    auto u16 = [&](size_t o) { return (uint32_t) (uint8_t) d[o] | (uint32_t) (uint8_t) d[o + 1] << 8; };
    auto u32 = [&](size_t o) { return u16(o) | u16(o + 2) << 16; };
    uint32_t format = 0, channels = 0, bits = 0;
    size_t data_off = 0, data_len = 0;
    bool have_fmt = false, have_data = false;
    for (size_t o = 12; o + 8 <= d.size();) {
        const std::string id = d.substr(o, 4);
        size_t len = u32(o + 4);
        const size_t body = o + 8;
        if (id == "fmt " && body + 16 <= d.size()) {
            format = u16(body); channels = u16(body + 2); sr = (int) u32(body + 4); bits = u16(body + 14);
            if (format == 0xFFFE && len >= 26 && body + 26 <= d.size()) format = u16(body + 24);
            have_fmt = true;
        } else if (id == "data") {
            data_off = body;
            data_len = std::min(len, d.size() - body);
            have_data = true;
            break;
        }
        o = body + len + (len & 1);
    }
    if (!have_fmt || !have_data || channels == 0 || sr <= 0) return false;
    const size_t width = bits / 8;
    if (!((format == 1 && (bits == 8 || bits == 16 || bits == 24 || bits == 32)) || (format == 3 && (bits == 32 || bits == 64))))
        return false;
    const size_t frames = data_len / (width * channels);
    audio.assign(frames, 0.0f);
    const char * p = d.data() + data_off;
    for (size_t i = 0; i < frames; ++i) {
        double acc = 0.0;
        for (size_t c = 0; c < channels; ++c, p += width) {
            double v;
            if (format == 3 && bits == 32) { float x; memcpy(&x, p, 4); v = x; }
            else if (format == 3) { double x; memcpy(&x, p, 8); v = x; }
            else if (bits == 8) v = ((int) (uint8_t) p[0] - 128) / 128.0;
            else if (bits == 16) { int16_t x; memcpy(&x, p, 2); v = x / 32768.0; }
            else if (bits == 24) {
                int32_t x = (int32_t) ((uint32_t) (uint8_t) p[0] << 8 | (uint32_t) (uint8_t) p[1] << 16 | (uint32_t) (uint8_t) p[2] << 24);
                v = (x >> 8) / 8388608.0;
            } else { int32_t x; memcpy(&x, p, 4); v = x / 2147483648.0; }
            acc += v;
        }
        audio[i] = (float) (acc / channels);
    }
    return true;
}

// Whether libsndfile (sgl's soundfile.read) reads the file natively: WAV / FLAC / Ogg / MP3 / AIFF, by magic bytes.
bool sound_file(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return false;
    unsigned char h[12] = {0};
    const size_t n = fread(h, 1, sizeof h, f);
    fclose(f);
    if (n < 4) return false;
    if (n >= 12 && !memcmp(h, "RIFF", 4) && !memcmp(h + 8, "WAVE", 4)) return true;
    if (n >= 12 && !memcmp(h, "FORM", 4) && (!memcmp(h + 8, "AIFF", 4) || !memcmp(h + 8, "AIFC", 4))) return true;
    if (!memcmp(h, "fLaC", 4) || !memcmp(h, "OggS", 4) || !memcmp(h, "ID3", 3)) return true;
    return h[0] == 0xFF && (h[1] & 0xE0) == 0xE0 && (h[1] & 0x06) != 0;   // an MPEG audio frame (layer I-III)
}

std::string ffmpeg() { return audio_ffmpeg_path().string(); }

// runtime.py decode_upload: mono float32 audio of an uploaded file -- a sound file at its own rate, or the first
// audio stream of a video at 44.1 kHz (ffmpeg).
std::vector<float> decode_upload(const std::string & path, int & sr) {
    TempDir tmp("dub-");
    const std::string wav = tmp.file("audio.wav");
    std::vector<std::string> argv{ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-i", path, "-vn", "-map", "0:a:0"};
    if (sound_file(path)) {
        argv.insert(argv.end(), {"-c:a", "pcm_f32le", wav});   // native rate and channels (averaged below)
    } else {
        argv.insert(argv.end(), {"-ac", "1", "-ar", std::to_string(dubplan::SAMPLE_RATE), "-c:a", "pcm_f32le", wav});
    }
    std::string err;
    std::vector<float> audio;
    if (run_process(argv, err) != 0 || !read_wav(wav, audio, sr))
        value_error("upload an audio file or a video with an audio track (" + tail_text(strip(err), 200) + ")");
    return audio;
}

// ------------------------------------------------------------------------------------------------------- the stores

struct Lips {
    std::shared_ptr<TempDir> dir;   // removed with the last reference / dub using it
    std::string path;
    double start = 0, end = 0;
};

struct Ref {
    std::vector<float> audio;
    int sr = 0;
    double duration = 0;
    std::mutex m;                   // guards lips
    std::shared_ptr<Lips> lips;
};

struct Take {
    std::vector<float> audio;
    double s0 = 0, s1 = 0;
    json items;                     // FuseAlign words [{word, start, end}]
    int attempts = 0;
    json retries;                   // LACI retries by kind
};

struct Dub {
    double duration = 0;
    std::string target_lang, ref_id;
    std::optional<std::vector<int32_t>> ref_codes;   // source voice prompt codes, [frames, n_codebooks]
    std::map<int, std::shared_ptr<const Take>> takes;
    std::shared_ptr<Lips> lips;
    json segments = json::array();
    std::vector<uint8_t> wav;
};

// runtime.py _Store: a small LRU of uploaded references / generated dubs.
template <class T>
class Store {
public:
    std::string put(std::shared_ptr<T> value) {
        const std::string key = random_hex(32);
        std::lock_guard<std::mutex> lk(m_);
        items_.emplace_back(key, std::move(value));
        while (items_.size() > (size_t) STORE_CAPACITY) items_.pop_front();
        return key;
    }
    std::shared_ptr<T> get(const std::string & key) {
        std::lock_guard<std::mutex> lk(m_);
        for (auto it = items_.begin(); it != items_.end(); ++it)
            if (it->first == key) {
                items_.splice(items_.end(), items_, it);
                return items_.back().second;
            }
        throw UnknownId();
    }

private:
    std::mutex m_;
    std::list<std::pair<std::string, std::shared_ptr<T>>> items_;
};

// --------------------------------------------------------------------------------------------- model description

// The speaking-rate condition row's bucket edges (UTF-8 bytes / s); empty without that row. The GGUF stores them as
// float32: each is read back as the shortest decimal that round-trips (the checkpoint's value, as sgl reports it).
std::vector<double> rate_edges(const ServerState & s) {
    const auto & hp = s.model.hp;
    std::vector<double> out;
    for (size_t i = 0; i < hp.cond_names.size() && i < hp.cond_edges.size(); ++i)
        if (hp.cond_names[i] == "speaking_rate")
            for (float v : hp.cond_edges[i]) out.push_back(float_decimal(v));
    return out;
}

// model/src/zonos3/dub/runtime.py WorkerSettings.from_env: the dubbing worker's interpreter (the ASR worker's:
// ZONOS3_ASR_PYTHON, else <repo>/.venv-asr), AV-sync home, device and word aligner.
struct WorkerSettings {
    std::string python, avsync_home, device;
    std::string word_aligner_model = "google-bert/bert-base-multilingual-cased";

    static WorkerSettings from_env(const ServerState & s) {
        const fs::path repo = fs::path(s.python_src) / ".." / "..";
        auto env = [](const char * k, const std::string & d) {
            const char * v = getenv(k);
            return v && *v ? std::string(v) : d;
        };
        WorkerSettings w;
        w.python = env("ZONOS3_ASR_PYTHON", (repo / ".venv-asr" / "bin" / "python").lexically_normal().string());
        w.avsync_home = env("ZONOS3_DUB_AVSYNC_HOME", (repo / ".avsync").lexically_normal().string());
        w.device = env("ZONOS3_ASR_DEVICE", "cuda");
        return w;
    }
    bool installed() const { std::error_code ec; return fs::is_regular_file(python, ec); }
    bool avsync_ready() const {
        std::error_code ec;
        const fs::path data = fs::path(avsync_home) / "data";   // zonos3/workers/dub_avsync.models_present
        return fs::is_regular_file(data / "syncnet_v2.model", ec) && fs::is_regular_file(data / "sfd_face.pth", ec);
    }
};

// Voice modes of the dub (runtime.py VOICE_MODES order): the reference clone and studio voices.
bool source_voice(const ServerState & s) {
    const auto m = checkpoint_speaker_modes(s);
    return std::find(m.begin(), m.end(), "reference") != m.end();
}
bool studio_voices(const ServerState & s) {
    const auto m = checkpoint_speaker_modes(s);
    return std::find(m.begin(), m.end(), "embedding") != m.end();
}
std::vector<std::string> voice_modes(const ServerState & s) {
    std::vector<std::string> out;
    if (source_voice(s)) out.push_back("source");
    if (studio_voices(s)) out.push_back("studio");
    return out;
}

const dubplan::TargetLanguage * target_language(const std::string & code) {
    for (const auto & t : dubplan::TARGET_LANGUAGES) if (t.code == code) return &t;
    return nullptr;
}
std::string ref_language_name(const std::string & code) {
    for (const auto & [c, n] : dubplan::REF_LANGUAGES) if (c == code) return n;
    return "";
}

// ------------------------------------------------------------------------------------------------ request shapes

// runtime.py merge_controls: steering doses summed per control id, zero doses dropped; null when nothing is left.
json merge_controls(std::initializer_list<const json *> maps, const std::string & where) {
    std::vector<std::pair<std::string, double>> out;   // insertion order
    for (const json * m : maps) {
        if (!m || m->is_null()) continue;
        if (!m->is_object()) value_error(where + ": controls must map control ids to doses");
        for (auto it = m->begin(); it != m->end(); ++it) {
            if (!it.value().is_number()) value_error(where + ": dose of " + py_repr(it.key()) + " must be a number");
            const double v = it.value().get<double>();
            if (!std::isfinite(v)) value_error(where + ": dose of " + py_repr(it.key()) + " must be finite");
            auto e = std::find_if(out.begin(), out.end(), [&](const auto & p) { return p.first == it.key(); });
            if (e == out.end()) out.emplace_back(it.key(), v);
            else e->second += v;
        }
    }
    json r = json::object();
    for (const auto & [k, v] : out) if (v != 0.0) r[k] = v;
    return r.empty() ? json() : r;
}

// runtime.py normalize_voice: null (inherit), {"mode": "source", "controls"} or
// {"mode": "studio", "speaker_embedding_id", "controls"}.
json normalize_voice(const json & raw, const std::string & where) {
    if (raw.is_null()) return nullptr;
    if (!raw.is_object()) value_error(where + ": voice must be an object");
    std::vector<std::string> unknown;
    for (auto it = raw.begin(); it != raw.end(); ++it)
        if (it.key() != "mode" && it.key() != "speaker_embedding_id" && it.key() != "controls") unknown.push_back(it.key());
    if (!unknown.empty()) {
        std::string list = "[";
        for (size_t i = 0; i < unknown.size(); ++i) list += (i ? ", " : "") + py_repr(unknown[i]);
        value_error(where + ": unknown voice field(s) " + list + "]");
    }
    const std::string mode = set(raw, "mode") && !raw["mode"].get<std::string>().empty() ? raw["mode"].get<std::string>() : "source";
    const json controls = merge_controls({raw.contains("controls") ? &raw["controls"] : nullptr}, where + ".controls");
    if (mode == "source") {
        if (set(raw, "speaker_embedding_id"))
            value_error(where + ": a source voice clones the reference; speaker_embedding_id needs mode 'studio'");
        return {{"mode", "source"}, {"controls", controls}};
    }
    if (mode == "studio") {
        const std::string sid = set(raw, "speaker_embedding_id") && raw["speaker_embedding_id"].is_string()
                                    ? strip(raw["speaker_embedding_id"].get<std::string>()) : "";
        if (sid.empty()) value_error(where + ": a studio voice needs speaker_embedding_id");
        return {{"mode", "studio"}, {"speaker_embedding_id", sid}, {"controls", controls}};
    }
    value_error(where + ": voice mode must be one of ['source', 'studio']");
}

// runtime.py normalize_ref_words: word timings the caller already has, in time order, inside the reference.
json normalize_ref_words(const json & raw, double duration) {
    if (!raw.is_array() || raw.empty()) value_error("ref_words must be a non-empty list of {word, start, end}");
    if (raw.size() > MAX_REF_WORDS) value_error("ref_words: at most " + std::to_string(MAX_REF_WORDS) + " words");
    json words = json::array();
    double prev = 0.0;
    for (size_t k = 0; k < raw.size(); ++k) {
        const json & w = raw[k];
        const std::string at = "ref_words[" + std::to_string(k) + "]";
        if (!w.is_object()) value_error(at + " must be an object {word, start, end}");
        const std::string text = set(w, "word") && w["word"].is_string() ? strip(w["word"].get<std::string>()) : "";
        if (!(set(w, "start") && w["start"].is_number() && set(w, "end") && w["end"].is_number()))
            value_error(at + ": start and end must be numbers (seconds)");
        const double start = w["start"].get<double>(), end = w["end"].get<double>();
        if (text.empty()) value_error(at + ": empty word");
        if (!(std::isfinite(start) && std::isfinite(end) && 0.0 <= start && start <= end && end <= duration + 0.05))
            value_error(at + ": need 0 <= start <= end <= " + fmt("%.2f", duration));
        if (start < prev - 1e-6) value_error(at + ": words must be in time order");
        prev = start;
        words.push_back({{"word", text}, {"start", start}, {"end", end}});
    }
    return words;
}

// Splits on pronunciation groups (U+E000 IPA U+E001), keeping them (re.split with a capturing _GROUP_RE).
std::vector<std::string> split_groups(const std::string & text) {
    std::vector<std::string> parts;
    size_t pos = 0, from = 0;
    for (;;) {
        const size_t open = text.find(GROUP_OPEN, from);
        if (open == std::string::npos) break;
        size_t close = std::string::npos;
        for (size_t i = open + 3; i < text.size();) {
            if (!text.compare(i, 3, GROUP_CLOSE)) { close = i; break; }
            if (!text.compare(i, 3, GROUP_OPEN)) break;
            utf8_next(text, i);
        }
        if (close == std::string::npos) { from = open + 3; continue; }
        parts.push_back(text.substr(pos, open - pos));
        parts.push_back(text.substr(open, close + 3 - open));
        pos = from = close + 3;
    }
    parts.push_back(text.substr(pos));
    return parts;
}
bool is_group(const std::string & p) { return p.size() >= 6 && !p.compare(0, 3, GROUP_OPEN); }

// runtime.py spoken_text: nonverbal markup removed (its enclosed words kept), each pronunciation group by its IPA.
std::string spoken_text(const std::string & text) {
    std::string out;
    for (const auto & p : split_groups(zonos3_strip_nonverbal_markup(text))) out += is_group(p) ? p.substr(3, p.size() - 6) : p;
    return out;
}

// runtime.py text_bytes: UTF-8 bytes of a target text for its speaking rate (a group counts its IPA symbols in
// letter-equivalents, markers count nothing).
int text_bytes(const std::string & text) {
    double total = 0.0;
    for (const auto & p : split_groups(zonos3_strip_nonverbal_markup(text)))
        if (!p.empty()) total += is_group(p) ? zonos3_edit::token_chars(p) : (double) p.size();
    return (int) std::nearbyint(total);
}

// --------------------------------------------------------------------------------------------- alignment worker

class AlignWorker {
public:
    // One zonos3/workers/dub_align.py process (word alignment and lip-sync); call() is thread-safe (the
    // JsonLinesWorker serializes calls).
    AlignWorker(const ServerState & s, const WorkerSettings & w) {
        if (!w.installed()) throw AlignWorkerError("alignment interpreter not found: " + w.python + " (run tools/setup_asr_env.sh)");
        std::vector<std::string> argv{w.python, (fs::path(s.python_src) / "zonos3" / "workers" / "dub_align.py").string(),
                                      "--device", w.device, "--word-aligner-model", w.word_aligner_model};
        if (w.avsync_ready()) { argv.push_back("--avsync-home"); argv.push_back(w.avsync_home); }
        std::map<std::string, std::optional<std::string>> env{{"PYTHONPATH", s.python_src},
                                                              {"PYTHONDONTWRITEBYTECODE", std::string("1")},
                                                              {"VIRTUAL_ENV", std::nullopt}};
        const char * offline = getenv("HF_HUB_OFFLINE");
        if (!offline) env["HF_HUB_OFFLINE"] = std::string("1");
        worker_ = std::make_unique<JsonLinesWorker>(std::move(argv), std::move(env));
    }
    bool alive() { return worker_->alive(); }
    json call(const std::string & op, json payload) {
        payload["id"] = ++next_id_;
        payload["op"] = op;
        json resp;
        try {
            resp = worker_->call(payload);
        } catch (const WorkerError & e) {
            throw AlignWorkerError(std::string("alignment worker exited (") + e.what() + ")");
        }
        if (!(resp.contains("status") && resp["status"] == "ok")) {
            auto text = [&](const char * k) { return resp.contains(k) && resp[k].is_string() ? resp[k].get<std::string>() : std::string(); };
            if (resp.contains("invalid") && resp["invalid"].is_boolean() && resp["invalid"].get<bool>())
                value_error(!text("message").empty() ? text("message") : text("error"));   // the input is at fault: 400
            throw AlignWorkerError(text("error").empty() ? "unknown error" : text("error"));
        }
        return resp;
    }
    void close() { worker_->close(); }

private:
    std::unique_ptr<JsonLinesWorker> worker_;
    std::atomic<int64_t> next_id_{0};
};

// ------------------------------------------------------------------------------------------------- dub runtime

// Bounds the dub takes in flight across requests (sgl _DUB_TAKE_SLOTS: twice the running-request cap) so studio
// requests interleave.
class Slots {
public:
    explicit Slots(int n) : free_(std::max(1, n)) {}
    void acquire() { std::unique_lock<std::mutex> lk(m_); cv_.wait(lk, [&] { return free_ > 0; }); --free_; }
    void release() { { std::lock_guard<std::mutex> lk(m_); ++free_; } cv_.notify_one(); }
private:
    std::mutex m_;
    std::condition_variable cv_;
    int free_;
};

struct Job {
    int bucket = 0;
    int64_t seed = 0;
    int segment = 0, slot = 0;
    std::string text;
    int max_tokens = 0;
    double target_frames = 0;
    bool pinned = false;
};

struct SynthTake {
    std::vector<float> audio;       // empty = no audio
    int attempts = 0;
    json retries = json::object();  // non-ignored failures by kind
    bool exhausted = false;
    std::string end;
    bool capped = false;
    json final_bucket;
};

struct Stats {
    int candidates = 0;
    std::vector<std::pair<std::string, int>> retries;   // collections.Counter, insertion order
    int exhausted = 0;
    double synthesis_s = 0, align_s = 0;
};

void counter_add(std::vector<std::pair<std::string, int>> & c, const json & by_kind) {
    for (auto it = by_kind.begin(); it != by_kind.end(); ++it) {
        auto e = std::find_if(c.begin(), c.end(), [&](const auto & p) { return p.first == it.key(); });
        if (e == c.end()) c.emplace_back(it.key(), it.value().get<int>());
        else e->second += it.value().get<int>();
    }
}
json counter_json(const std::vector<std::pair<std::string, int>> & c) {
    json j = json::object();
    for (const auto & [k, v] : c) if (v > 0) j[k] = v;
    return j;
}

double mean(const std::vector<double> & xs) {
    double t = 0;
    for (double x : xs) t += x;
    return xs.empty() ? 0.0 : t / (double) xs.size();
}

class DubRuntime {
public:
    explicit DubRuntime(ServerState & s) : s_(s), settings_(WorkerSettings::from_env(s)), slots_(2 * s.batch_slots) {
        edges_ = rate_edges(s);
        n_buckets_ = (int) s.model.hp.n_cond_buckets;
        row_scale_.edges = edges_;
        if (s.steer_loaded)
            for (const auto & d : s.steer.dirs) if (!d.hidden) raw_controls_.insert(d.id);
        build_rate_scales();
    }

    Store<Ref> refs;
    Store<Dub> dubs;

    // ---- capabilities ------------------------------------------------------------------------------------------

    json capabilities() {
        std::vector<std::string> laci = laci_targets();
        json target_languages = json::array();
        for (const auto & t : dubplan::TARGET_LANGUAGES)
            if (std::find(laci.begin(), laci.end(), t.code) != laci.end())
                target_languages.push_back({{"code", t.code}, {"label", t.label}, {"tts_language", t.tts}});
        json ref_languages = json::array();
        for (const auto & [c, n] : dubplan::REF_LANGUAGES) ref_languages.push_back({{"code", c}, {"name", n}});
        const bool running = running_;
        // the ASR and FuseAlign workers' pings (they start the workers); "" when one cannot run
        const std::string asr = asr_model();
        const std::string aligner = aligner_model();
        json offsets = json::array();
        for (int o : dubplan::CANDIDATE_BUCKET_OFFSETS) offsets.push_back(o);
        return {
            {"available", settings_.installed() && !asr.empty() && !aligner.empty() && !laci.empty() && !edges_.empty()},
            {"ref_languages", ref_languages},
            {"target_languages", target_languages},
            {"max_ref_seconds", dubplan::MAX_REF_SECONDS},
            {"max_video_bytes", dubplan::MAX_VIDEO_BYTES},
            {"max_segments", dubplan::MAX_SEGMENTS},
            {"default_min_pause_s", dubplan::DEFAULT_MIN_PAUSE_S},
            {"min_pause_range", json::array({dubplan::MIN_PAUSE_RANGE[0], dubplan::MIN_PAUSE_RANGE[1]})},
            {"sample_rate", dubplan::SAMPLE_RATE},
            {"frame_rate", dubplan::FRAME_RATE},
            {"rate_edges", edges_},
            {"n_buckets", n_buckets_},
            {"buckets", buckets(laci)},
            {"candidate_bucket_offsets", offsets},
            {"candidate_seeds", dubplan::CANDIDATE_SEEDS},
            {"model", fs::path(s_.model_path).filename().string()},
            {"asr_model", asr},
            {"aligner_model", aligner},
            {"word_aligner", settings_.word_aligner_model},
            {"align_worker", {{"installed", settings_.installed()}, {"running", running}}},
            {"lipsync", settings_.installed() && settings_.avsync_ready()},
            {"voice_modes", voice_modes(s_)},
            {"segment_controls", true},
            {"phoneme_groups", !phoneme_capabilities(s_)["phoneme_tokens"].is_null()},
            {"ref_words", true},
        };
    }

    // ---- references --------------------------------------------------------------------------------------------

    json add_ref(const std::string & path) {
        auto ref = std::make_shared<Ref>();
        ref->audio = decode_upload(path, ref->sr);
        ref->duration = (double) ref->audio.size() / ref->sr;
        if (!(dubplan::MIN_REF_SECONDS <= ref->duration && ref->duration <= dubplan::MAX_REF_SECONDS))
            value_error("reference must be " + fmt("%g", dubplan::MIN_REF_SECONDS) + "\xE2\x80\x93" + fmt("%g", dubplan::MAX_REF_SECONDS) +
                        " s, got " + fmt("%.1f", ref->duration) + " s");
        const double duration = ref->duration;
        const int sr = ref->sr;
        const std::string ref_id = refs.put(std::move(ref));
        return {{"ref_id", ref_id}, {"duration", duration}, {"sample_rate", sr}};
    }

    // Qwen3-ASR transcript of the reference (transcribe_audio: chunk-wise past 30 s) and its words (FuseAlign, anchored
    // by the transcript's chunks) in the given language, else the detected one: Qwen3-ASR's (past 30 s the first
    // spoken chunk's), its first name when it reports several.
    json transcribe(const std::string & ref_id, const std::optional<std::string> & ref_lang) {
        auto ref = refs.get(ref_id);
        check_ref_lang(ref_lang);
        TempDir tmp("dub-");
        const std::string wav = tmp.file("ref.wav");
        write_float_wav(wav, ref->audio, ref->sr);
        const Transcript out = transcribe_audio(s_, wav, ref->duration,
                                                ref_lang ? std::optional<std::string>(ref_language_name(*ref_lang)) : std::nullopt);
        const std::string name = strip(out.language.substr(0, out.language.find(',')));
        std::optional<std::string> lang = ref_lang;
        if (!lang)
            for (const auto & [c, n] : dubplan::REF_LANGUAGES) if (n == name) { lang = c; break; }
        const std::string text = strip(out.text);
        json words = json::array();
        if (lang && !text.empty()) words = public_words(align({{wav, ref->duration, text, *lang, out.utterances}})[0]);
        return {{"text", text}, {"language", lang ? json(*lang) : json()}, {"language_name", name}, {"words", words}};
    }

    void check_lips_ready() const {
        if (!(settings_.installed() && settings_.avsync_ready()))
            value_error("lip-sync (AV-sync) is not set up on this server; run tools/setup_asr_env.sh");
    }

    // Face track and lip embeddings of the reference's video (AV-sync), kept with the reference.
    json prepare_lips(const std::string & ref_id, const std::string & video) {
        check_lips_ready();
        auto ref = refs.get(ref_id);
        auto lips = std::make_shared<Lips>();
        lips->dir = std::make_shared<TempDir>("dub-lips-");
        lips->path = lips->dir->file("lips.pt");
        const json r = call("lips_prepare", {{"video", video}, {"out", lips->path}});
        lips->start = r["start"].get<double>();
        lips->end = r["end"].get<double>();
        {
            std::lock_guard<std::mutex> lk(ref->m);
            ref->lips = lips;
        }
        return {{"start", r["start"]}, {"end", r["end"]}, {"track_frames", r["track_frames"]},
                {"video_frames", r["video_frames"]}, {"tracks", r["tracks"]}};
    }

    // ---- analysis ----------------------------------------------------------------------------------------------

    json analyze(const json & req) {
        auto ref = refs.get(req["ref_id"].get<std::string>());
        const std::string ref_lang = set(req, "ref_lang") ? req["ref_lang"].get<std::string>() : "en";
        const std::string target_lang = set(req, "target_lang") ? req["target_lang"].get<std::string>() : "fr";
        check_ref_lang(ref_lang);
        if (!target_language(target_lang)) value_error("unsupported target language: " + target_lang);
        const std::string target_text = strip(set(req, "target_text") ? req["target_text"].get<std::string>() : "");
        if (target_text.empty()) value_error("the target transcript is required");
        const double min_pause = set(req, "min_pause_s") ? req["min_pause_s"].get<double>() : dubplan::DEFAULT_MIN_PAUSE_S;
        if (!(dubplan::MIN_PAUSE_RANGE[0] <= min_pause && min_pause <= dubplan::MIN_PAUSE_RANGE[1]))
            value_error("min_pause_s must be within " + fmt("%g", dubplan::MIN_PAUSE_RANGE[0]) + "\xE2\x80\x93" +
                        fmt("%.1f", dubplan::MIN_PAUSE_RANGE[1]) + " s");

        json words;
        if (set(req, "ref_words")) {
            words = normalize_ref_words(req["ref_words"], ref->duration);
        } else {
            const std::string ref_text = strip(set(req, "ref_text") ? req["ref_text"].get<std::string>() : "");
            if (ref_text.empty()) value_error("send the reference transcript (ref_text) or its word timings (ref_words)");
            TempDir tmp("dub-");
            const std::string wav = tmp.file("ref.wav");
            write_float_wav(wav, ref->audio, ref->sr);
            words = public_words(align({{wav, ref->duration, ref_text, ref_lang, std::nullopt}})[0]);
            if (words.empty()) value_error("the forced aligner returned no words; check the transcript and its language");
        }

        const auto units = dubplan::text_units(target_text, target_lang);
        if (units.empty()) value_error("the target transcript has no words");
        json src = json::array(), tgt = json::array();
        for (const auto & w : words) src.push_back(w["word"]);
        for (const auto & u : units) tgt.push_back(spoken_text(u.text));
        const json links = call("word_align", {{"src", src}, {"tgt", tgt}})["links"];

        std::vector<dubplan::TimedWord> timed;
        for (const auto & w : words) timed.push_back({w["start"].get<double>(), w["end"].get<double>()});
        const auto ranges = dubplan::ref_segments(timed, min_pause);
        if ((int) ranges.size() > dubplan::MAX_SEGMENTS) value_error("too many pauses; raise the pause threshold");
        std::vector<std::pair<double, double>> spans;
        for (const auto & [a, b] : ranges) {
            double lo = INFINITY, hi = -INFINITY;
            for (int i = a; i < b; ++i) { lo = std::min(lo, timed[i].start); hi = std::max(hi, timed[i].end); }
            spans.emplace_back(lo, hi);
        }
        std::vector<std::pair<int, int>> link_pairs;
        for (const auto & l : links) link_pairs.emplace_back(l[0].get<int>(), l[1].get<int>());
        const auto ubytes = dubplan::unit_bytes(target_text, units);
        const auto breaks = dubplan::break_after(target_text, units);
        const auto projected = dubplan::project_segments(ranges, spans, link_pairs, ubytes, breaks);
        const dubplan::RateScale & auto_scale = rate_scale(target_lang, source_voice(s_) ? "source" : "studio");
        json segments = json::array();
        for (size_t k = 0; k < ranges.size() && k < projected.size(); ++k) {
            const auto [wa, wb] = ranges[k];
            const auto [ua, ub] = projected[k];
            const auto [st, en] = spans[k];
            const std::string text = dubplan::span_text(target_text, units, ua, ub);
            std::vector<std::string> ref_words;
            for (int i = wa; i < wb; ++i) ref_words.push_back(words[i]["word"].get<std::string>());
            segments.push_back({{"index", k}, {"ref_start", st}, {"ref_end", en}, {"ref_word_range", {wa, wb}},
                                {"target_unit_range", {ua, ub}}, {"ref_text", dubplan::join_words(ref_words, ref_lang)},
                                {"target_text", text},
                                {"bucket_auto", !text.empty() && en > st ? json(auto_scale.bucket(text_bytes(text) / (en - st))) : json()}});
        }
        json warnings = json::array();
        if (links.empty()) warnings.push_back("The word aligner found no links; segments are split by length");
        json target_units = json::array();
        for (const auto & u : units) target_units.push_back({{"text", u.text}, {"start", u.start}, {"end", u.end}});
        return {{"duration", ref->duration}, {"ref_words", words}, {"target_units", target_units}, {"links", links},
                {"segments", segments}, {"warnings", warnings}};
    }

    // ---- generation --------------------------------------------------------------------------------------------

    json generate(const json & req, const std::optional<std::string> & session) {
        const std::string ref_id = req["ref_id"].get<std::string>();
        auto ref = refs.get(ref_id);
        const std::string target_lang = req["target_lang"].get<std::string>();
        const int64_t seed = set(req, "seed") ? req["seed"].get<int64_t>() : 0;
        if (!target_language(target_lang)) value_error("unsupported target language: " + target_lang);
        if (seed < 0) value_error("seed must be >= 0");
        const json & raw = req["segments"];
        if (!(1 <= raw.size() && raw.size() <= (size_t) dubplan::MAX_SEGMENTS))
            value_error("need 1\xE2\x80\x93" + std::to_string(dubplan::MAX_SEGMENTS) + " segments");
        const json default_voice = normalize_voice(req.value("voice", json()), "voice");
        const double duration = ref->duration;
        std::vector<dubplan::RefSegment> segs;
        std::vector<std::pair<json, json>> voices;   // (voice, own controls)
        double prev_end = 0.0;
        for (size_t k = 0; k < raw.size(); ++k) {
            const json & sj = raw[k];
            const double a = sj["ref_start"].get<double>(), b = sj["ref_end"].get<double>();
            if (!(0 <= a && a < b && b <= duration + 0.05))
                value_error("segment " + std::to_string(k) + ": need 0 <= ref_start < ref_end <= " + fmt("%.2f", duration));
            if (a < prev_end - 1e-6) value_error("segment " + std::to_string(k) + ": segments must be sorted and non-overlapping");
            prev_end = b;
            dubplan::RefSegment seg;
            seg.ref_start = a;
            seg.ref_end = b;
            seg.text = strip(sj["target_text"].get<std::string>());
            if (set(sj, "bucket")) seg.bucket = sj["bucket"].get<int>();
            segs.push_back(seg);
            json voice = normalize_voice(sj.value("voice", json()), "segments[" + std::to_string(k) + "].voice");
            if (voice.is_null()) voice = default_voice;
            voices.push_back(segment_voice((int) k, voice, sj.value("controls", json())));
        }
        if (std::none_of(segs.begin(), segs.end(), [](const auto & sg) { return !sg.text.empty(); }))
            value_error("every segment is silent; enter target text");

        const bool lipsync = req.value("lipsync", false);
        auto dub = std::make_shared<Dub>();
        dub->duration = duration;
        dub->target_lang = target_lang;
        dub->ref_id = ref_id;
        dub->lips = lips_of(ref_id, lipsync);
        for (size_t k = 0; k < segs.size(); ++k)
            dub->segments.push_back(segment_entry((int) k, segs[k], seed, voices[k].first, voices[k].second,
                                                  rate_scale(target_lang, voices[k].first["mode"].get<std::string>())));
        const auto began = std::chrono::steady_clock::now();
        std::vector<dubplan::RefSegment> source_segs;
        for (size_t k = 0; k < segs.size(); ++k) if (voices[k].first["mode"] == "source") source_segs.push_back(segs[k]);
        ensure_voice_prompt(*dub, ref->audio, ref->sr, source_segs);
        std::vector<int> ks;
        for (size_t k = 0; k < segs.size(); ++k) if (!segs[k].text.empty()) ks.push_back((int) k);
        const Stats stats = render(*dub, ks, session);
        return assemble(dub, stats, seconds_since(began));
    }

    json regenerate(const json & req, const std::optional<std::string> & session) {
        auto old = dubs.get(req["dub_id"].get<std::string>());
        const int64_t seed = set(req, "seed") ? req["seed"].get<int64_t>() : 0;
        const int64_t k = req["segment"].get<int64_t>();
        if (seed < 0) value_error("seed must be >= 0");
        if (!(0 <= k && k < (int64_t) old->segments.size()))
            value_error("segment must be in [0, " + std::to_string(old->segments.size()) + ")");
        const json & seg = old->segments[(size_t) k];
        const std::string text = strip(req["target_text"].get<std::string>());
        json voice = normalize_voice(req.value("voice", json()), "voice");
        if (voice.is_null()) voice = seg["voice"];
        const json own = set(req, "controls") ? req["controls"] : seg["controls"];
        const auto [v, controls] = segment_voice((int) k, voice, own);
        const bool lipsync = req.value("lipsync", false);
        std::shared_ptr<Lips> lips;
        if (lipsync) lips = old->lips ? old->lips : lips_of(old->ref_id, true);
        auto dub = std::make_shared<Dub>(*old);   // segments / takes copied
        dub->wav.clear();
        dub->lips = lips;
        dubplan::RefSegment rs;
        rs.ref_start = seg["ref_start"].get<double>();
        rs.ref_end = seg["ref_end"].get<double>();
        rs.text = text;
        if (set(req, "bucket")) rs.bucket = req["bucket"].get<int>();
        json entry = segment_entry((int) k, rs, seed, v, controls, rate_scale(old->target_lang, v["mode"].get<std::string>()));
        dub->segments[(size_t) k] = entry;
        dub->takes.erase((int) k);
        if (text.empty() && dub->takes.empty()) value_error("every segment would be silent; enter target text");
        const auto began = std::chrono::steady_clock::now();
        if (v["mode"] == "source" && !dub->ref_codes) {
            auto ref = refs.get(old->ref_id);   // UnknownId (404): the reference left the store
            dubplan::RefSegment span;
            span.ref_start = entry["ref_start"].get<double>();
            span.ref_end = entry["ref_end"].get<double>();
            ensure_voice_prompt(*dub, ref->audio, ref->sr, {span});
        }
        const Stats stats = render(*dub, text.empty() ? std::vector<int>{} : std::vector<int>{(int) k}, session);
        return assemble(dub, stats, seconds_since(began));
    }

    // ---- video -------------------------------------------------------------------------------------------------

    // Replace the video's audio with the dub (video stream copied; re-encoded if the copy fails).
    void mux(const std::string & dub_id, const std::string & video, const std::string & out) {
        auto dub = dubs.get(dub_id);
        const std::string dub_path = (fs::path(video).parent_path() / "dub.wav").string();
        {
            FILE * f = fopen(dub_path.c_str(), "wb");
            if (!f) throw std::runtime_error("cannot write " + dub_path);
            const bool ok = fwrite(dub->wav.data(), 1, dub->wav.size(), f) == dub->wav.size();
            if (fclose(f) != 0 || !ok) throw std::runtime_error("cannot write " + dub_path);
        }
        const std::vector<std::string> base{ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-i", video, "-i", dub_path};
        const std::vector<std::string> tail{"-map", "0:v:0", "-map", "1:a:0"};
        const std::vector<std::string> audio{"-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out};
        auto cat = [](std::initializer_list<std::vector<std::string>> parts) {
            std::vector<std::string> r;
            for (const auto & p : parts) r.insert(r.end(), p.begin(), p.end());
            return r;
        };
        std::string err;
        if (run_process(cat({base, tail, {"-c:v", "copy"}, audio}), err) == 0) return;
        if (run_process(cat({base, tail, {"-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"}, audio}), err) != 0)
            value_error("ffmpeg could not mux this video: " + tail_text(err, 400));
    }

private:
    ServerState & s_;
    WorkerSettings settings_;
    Slots slots_;
    std::vector<double> edges_;
    int n_buckets_ = 0;
    dubplan::RateScale row_scale_;
    std::set<std::string> raw_controls_;
    std::string rate_direction_;
    // voice mode -> TTS language -> its scale with tempo-steered buckets (only languages that have some)
    std::map<std::string, std::map<std::string, dubplan::RateScale>> rate_scales_;
    std::mutex start_mtx_;                // serializes worker (re)starts
    std::mutex state_mtx_;                // guards worker_
    std::shared_ptr<AlignWorker> worker_;
    std::atomic<bool> running_{false};    // the worker answered and has not failed since

    // sgl _dub_rate_scales: the steering bundle's rate_extension per voice mode (source voices only on its
    // reference_sides); none when it was measured against other row edges.
    void build_rate_scales() {
        if (!s_.steer_loaded || !s_.steer.rate_extension) return;
        const auto & ext = *s_.steer.rate_extension;
        bool same = ext.row_edges.size() == edges_.size();
        for (size_t i = 0; same && i < edges_.size(); ++i) same = std::fabs(ext.row_edges[i] - edges_[i]) <= 1e-6;
        if (!same) {
            fprintf(stderr, "steering bundle rate_extension was measured on other speaking-rate edges than the checkpoint's: "
                            "dubs plan in the row buckets only\n");
            return;
        }
        rate_direction_ = ext.direction;
        auto side = [&](int b) {
            const std::string name = b < 0 ? "slower" : "faster";
            return std::find(ext.reference_sides.begin(), ext.reference_sides.end(), name) != ext.reference_sides.end();
        };
        for (const auto & [lang, doses] : ext.doses) {
            std::map<int, double> source;
            for (const auto & [b, d] : doses) if (side(b)) source[b] = d;
            if (!doses.empty()) rate_scales_["studio"][lang] = dubplan::RateScale::extended(edges_, ext.below, ext.above, doses);
            if (!source.empty()) rate_scales_["source"][lang] = dubplan::RateScale::extended(edges_, ext.below, ext.above, source);
        }
    }

    // The target codes LACI serves, sorted.
    std::vector<std::string> laci_targets() const {
        const auto served = laci_languages(s_);
        std::vector<std::string> out;
        for (const auto & t : dubplan::TARGET_LANGUAGES)
            if (std::find(served.begin(), served.end(), t.tts) != served.end()) out.push_back(t.code);
        std::sort(out.begin(), out.end());
        return out;
    }

    const dubplan::RateScale & rate_scale(const std::string & target_lang, const std::string & mode) const {
        const auto m = rate_scales_.find(mode);
        if (m == rate_scales_.end()) return row_scale_;
        const auto * t = target_language(target_lang);
        const auto l = t ? m->second.find(t->tts) : m->second.end();
        return l == m->second.end() ? row_scale_ : l->second;
    }

    // DubRuntime.buckets: every bucket of the target languages' scales.
    json buckets(const std::vector<std::string> & laci) const {
        json out = json::array();
        if (edges_.empty()) return out;
        std::vector<std::pair<std::pair<std::string, std::string>, const dubplan::RateScale *>> scales;   // dict order
        for (const auto & v : voice_modes(s_))
            for (const auto & c : laci) scales.push_back({{v, c}, &rate_scale(c, v)});
        if (scales.empty()) scales.push_back({{"", ""}, &row_scale_});
        int lo = INT32_MAX, hi = INT32_MIN;
        for (const auto & [vc, sc] : scales) { lo = std::min(lo, sc->first); hi = std::max(hi, sc->last()); }
        for (int b = lo; b <= hi; ++b) {
            const dubplan::RateScale * inner = nullptr;
            int best = INT32_MIN;
            std::set<std::string> languages, modes;
            std::vector<std::pair<std::pair<std::string, std::string>, const dubplan::RateScale *>> having;
            for (const auto & e : scales) {
                const auto * sc = e.second;
                if (!(sc->first <= b && b <= sc->last())) continue;
                having.push_back(e);
                const int room = std::min(b - sc->first, sc->last() - b);
                if (room > best) { best = room; inner = sc; }
                if (!e.first.second.empty()) languages.insert(e.first.second);
                if (!e.first.first.empty()) modes.insert(e.first.first);
            }
            if (!inner) continue;
            const int i = b - inner->first;
            const bool steered = !(0 <= b && b < n_buckets_);
            json entry = {{"index", b}, {"label", std::to_string(b)}, {"rate", inner->rate(b)},
                          {"low", i > 0 ? json(inner->edges[i - 1]) : json()},
                          {"high", i < (int) inner->edges.size() ? json(inner->edges[i]) : json()},
                          {"steered", steered}, {"languages", languages}, {"voice_modes", modes}};
            if (steered) {
                std::sort(having.begin(), having.end(), [](const auto & x, const auto & y) { return x.first < y.first; });
                json doses = json::object();
                for (const auto & [vc, sc] : having) {
                    const auto d = sc->doses.find(b);
                    if (d != sc->doses.end()) doses[vc.second] = d->second;
                }
                entry["doses"] = doses;
            }
            out.push_back(entry);
        }
        return out;
    }

    // ---- worker calls ------------------------------------------------------------------------------------------

    std::shared_ptr<AlignWorker> ensure_worker() {
        std::lock_guard<std::mutex> start(start_mtx_);
        std::shared_ptr<AlignWorker> w;
        {
            std::lock_guard<std::mutex> lk(state_mtx_);
            w = worker_;
        }
        if (w && w->alive()) return w;
        running_ = false;
        if (w) w->close();
        w = std::make_shared<AlignWorker>(s_, settings_);
        {
            std::lock_guard<std::mutex> lk(state_mtx_);
            worker_ = w;
        }
        w->call("ping", json::object());
        running_ = true;
        return w;
    }

    json call(const std::string & op, json payload) {
        auto w = ensure_worker();
        try {
            return w->call(op, std::move(payload));
        } catch (const AlignWorkerError &) {
            running_ = false;
            throw;
        }
    }

    // FuseAlign word timings of each (audio of `seconds`, text, Whisper-style REF_LANGUAGES code, utterances: the
    // long-form anchors of a transcript of the audio): the words of zonos3.tts.align_text.align_tokens, per item
    // [{word, start, end, placed}]. Long-form items get Qwen3-ASR anchors (align_words).
    struct AlignJob {
        std::string audio;
        double seconds = 0;
        std::string text, lang;
        std::optional<std::vector<FuseAlignUtterance>> utterances;
    };
    std::vector<json> align(const std::vector<AlignJob> & jobs) {
        std::vector<AlignWordsItem> items;
        for (const auto & j : jobs) {
            const std::string name = ref_language_name(j.lang);   // P.REF_LANGUAGES.get(code)
            items.push_back({j.audio, j.seconds, aligntext::align_tokens(j.text, j.lang),
                             name.empty() ? std::nullopt : std::optional<std::string>(name), j.utterances});
        }
        std::vector<json> out;
        for (const auto & words : align_words(s_, items)) {
            json item = json::array();
            for (const auto & w : words)
                item.push_back({{"word", w.word}, {"start", w.start}, {"end", w.end}, {"placed", w.placed}});
            out.push_back(std::move(item));
        }
        return out;
    }

    // The API's word records [{word, start, end}] of aligned words.
    static json public_words(const json & aligned) {
        json out = json::array();
        for (const auto & w : aligned) out.push_back({{"word", w["word"]}, {"start", w["start"]}, {"end", w["end"]}});
        return out;
    }

    // The FuseAlign checkpoint (the worker's ping; starts it); empty when the worker cannot run.
    std::string aligner_model() {
        try {
            return fusealign_worker(s_).model();
        } catch (const WorkerUnavailable & e) {
            fprintf(stderr, "zonos3-server: %s\n", e.what());
            return "";
        }
    }

    // The ASR model (the worker's ping; starts it); empty when the worker cannot run.
    std::string asr_model() {
        try {
            return asr_worker(s_).model();
        } catch (const WorkerUnavailable & e) {
            fprintf(stderr, "zonos3-server: %s\n", e.what());
            return "";
        }
    }

    static void check_ref_lang(const std::optional<std::string> & ref_lang) {
        if (!ref_lang) return;
        for (const auto & [c, n] : dubplan::REF_LANGUAGES) if (c == *ref_lang) return;
        value_error("unsupported reference language: " + *ref_lang);
    }

    std::shared_ptr<Lips> lips_of(const std::string & ref_id, bool lipsync) {
        if (!lipsync) return nullptr;
        std::shared_ptr<Lips> lips;
        try {
            auto ref = refs.get(ref_id);
            std::lock_guard<std::mutex> lk(ref->m);
            lips = ref->lips;
        } catch (const UnknownId &) {}
        if (!lips) value_error("lip-sync needs the video's face track: POST /v2/dub/lips/{ref_id} with the video first");
        return lips;
    }

    // ---- generation steps --------------------------------------------------------------------------------------

    // A segment's effective voice and its own controls (speaker-embedding controls need a studio voice).
    std::pair<json, json> segment_voice(int k, json voice, const json & own) {
        if (voice.is_null()) voice = {{"mode", "source"}, {"controls", nullptr}};
        const std::string mode = voice["mode"].get<std::string>();
        if (mode == "source" && !source_voice(s_))
            value_error("segment " + std::to_string(k) + ": this checkpoint has no reference-clone path; choose a studio voice");
        if (mode == "studio" && !studio_voices(s_))
            value_error("segment " + std::to_string(k) + ": this checkpoint has no speaker-embedding row for studio voices");
        const json controls = merge_controls({&own}, "segments[" + std::to_string(k) + "].controls");
        const json merged = merge_controls({&voice["controls"], &controls}, "controls");
        std::vector<std::string> raw;
        if (merged.is_object())
            for (auto it = merged.begin(); it != merged.end(); ++it) if (raw_controls_.count(it.key())) raw.push_back(it.key());
        std::sort(raw.begin(), raw.end());
        if (!raw.empty() && mode != "studio") {
            std::string list;
            for (size_t i = 0; i < raw.size(); ++i) list += (i ? ", " : "") + raw[i];
            value_error("segment " + std::to_string(k) + ": controls " + list + " move the speaker embedding and need a studio voice");
        }
        return {voice, controls};
    }

    // Reference codes of the source voice: the clip from the first source-voice segment, up to VOICE_PROMPT_MAX_S.
    void ensure_voice_prompt(Dub & dub, const std::vector<float> & audio, int sr, const std::vector<dubplan::RefSegment> & source_segs) {
        if (dub.ref_codes || source_segs.empty()) return;
        const auto [a, b] = dubplan::voice_prompt_span(source_segs, dub.duration);
        const size_t i0 = std::min(audio.size(), (size_t) std::max(0.0, std::floor(a * sr)));
        const size_t i1 = std::min(audio.size(), (size_t) std::max(0.0, std::floor(b * sr)));
        std::vector<float> clip(audio.begin() + (ptrdiff_t) i0, audio.begin() + (ptrdiff_t) std::max(i0, i1));
        if (sr != dubplan::SAMPLE_RATE && !clip.empty()) {   // the codec encodes 44.1 kHz audio
            TempDir tmp("dub-");
            const std::string wav = tmp.file("voice.wav");
            write_float_wav(wav, clip, sr);
            clip = audio_decode_file(wav.c_str(), dubplan::SAMPLE_RATE, (int) dubplan::VOICE_PROMPT_MAX_S + 1);
            if (clip.empty()) value_error("could not resample the voice prompt");
        }
        const size_t cap = (size_t) (dubplan::VOICE_PROMPT_MAX_S * dubplan::SAMPLE_RATE);
        if (clip.size() > cap) clip.resize(cap);
        if (clip.empty()) value_error("the voice prompt clip is empty");
        dub.ref_codes = encode_audio_codes(s_, clip, (int) dubplan::VOICE_PROMPT_MAX_S);
    }

    // The response entry of one segment, its candidate plan included (placement fields filled later).
    json segment_entry(int k, const dubplan::RefSegment & seg, int64_t seed, const json & voice, const json & controls,
                       const dubplan::RateScale & scale) {
        if (seg.bucket && !(scale.first <= *seg.bucket && *seg.bucket <= scale.last()))
            value_error("segment " + std::to_string(k) + ": bucket must be in [" + std::to_string(scale.first) + ", " +
                        std::to_string(scale.last()) + "] for this target language");
        const double d = seg.ref_end - seg.ref_start;
        const int nbytes = text_bytes(seg.text);
        const bool has = !seg.text.empty();
        json entry = {{"index", k}, {"ref_start", seg.ref_start}, {"ref_end", seg.ref_end}, {"target_text", seg.text},
                      {"status", has ? "ok" : "silent"}, {"seed", seed}, {"bytes", nbytes},
                      {"rate", has ? json(nbytes / d) : json()}, {"bucket", nullptr}, {"pinned", seg.bucket.has_value()},
                      {"voice", voice}, {"controls", controls}, {"candidates", json::array()}, {"chosen", nullptr},
                      {"final_bucket", nullptr}, {"retries", nullptr}, {"fit", nullptr}, {"speech_start", nullptr},
                      {"speech_end", nullptr}, {"shift", nullptr}, {"start_error", nullptr}, {"end_error", nullptr},
                      {"duration_error", nullptr}, {"lip_start", nullptr}, {"lip_end", nullptr}, {"lip_scale", nullptr},
                      {"lip_stretch", nullptr}, {"lip_squeeze", nullptr}, {"lip_warp", nullptr}};
        if (has) {
            std::vector<dubplan::Candidate> plan;
            if (!seg.bucket) {
                const int b = scale.bucket(nbytes / d);
                entry["bucket"] = b;
                plan = dubplan::candidate_plan(b, scale, seed, k);
            } else {
                entry["bucket"] = *seg.bucket;
                plan = dubplan::pinned_candidate_plan(*seg.bucket, seed, k);
            }
            for (const auto & c : plan)
                entry["candidates"].push_back({{"bucket", c.bucket}, {"offset", c.offset}, {"seed", c.seed}, {"speech_s", nullptr},
                                               {"attempts", nullptr}, {"retries", nullptr}, {"final_bucket", nullptr},
                                               {"end", nullptr}, {"capped", nullptr}, {"score", nullptr}});
        }
        return entry;
    }

    // Every job's take under LACI on the scheduler, each with its segment's duration target (sgl _synthesize and
    // _dub_synthesize): every request is planned before the first is queued (HTTP 400 for a bad voice / control /
    // text); at most the take slots are in flight.
    std::vector<SynthTake> synthesize(const Dub & dub, const std::vector<Job> & jobs, const std::optional<std::string> & session) {
        const std::string tts_lang = target_language(dub.target_lang)->tts;
        const int K = (int) s_.model.hp.n_codebooks;
        json ref_codes = nullptr;
        if (dub.ref_codes) {
            ref_codes = json::array();
            for (size_t f = 0; f + (size_t) K <= dub.ref_codes->size(); f += (size_t) K)
                ref_codes.push_back(std::vector<int32_t>(dub.ref_codes->begin() + (ptrdiff_t) f, dub.ref_codes->begin() + (ptrdiff_t) f + K));
        }
        std::vector<std::shared_ptr<ReqJob>> planned;
        for (const Job & j : jobs) {
            const json & e = dub.segments[(size_t) j.segment];
            const json & voice = e["voice"];
            const std::string mode = voice["mode"].get<std::string>();
            const dubplan::RateScale & scale = rate_scale(dub.target_lang, mode);
            const bool steered_scale = !scale.doses.empty();
            // sgl _dub_take_request: a tempo-steered bucket leaves the rate row unknown
            const json row = steered_scale && scale.doses.count(j.bucket) ? json() : json(j.bucket);
            LaciTake take;
            take.request = {{"text", j.text}, {"language", tts_lang}, {"seed", j.seed}, {"max_tokens", j.max_tokens},
                            {"stream", false}, {"laci", true}, {"speaker_cfg_scale", 1.0},
                            {"condition_buckets", {{"speaking_rate", row}}},
                            {"controls", merge_controls({&voice["controls"], &e["controls"]}, "controls")}};
            if (mode == "source") {
                take.request["speaker_mode"] = "reference";
                take.request["speaker_reference_codes"] = ref_codes;
            } else {
                take.request["speaker_mode"] = "embedding";
                take.request["speaker_embedding_id"] = voice["speaker_embedding_id"];
            }
            take.duration.frames = j.target_frames;
            take.duration.adjust_rate = !j.pinned;
            if (steered_scale)
                take.scale = LaciTake::Scale{scale.edges, scale.first, j.bucket, scale.doses, rate_direction_};
            planned.push_back(plan_laci_take(s_, take, session));
        }

        std::vector<TakeResult> results(planned.size());
        std::vector<std::exception_ptr> errors(planned.size());
        std::atomic<size_t> next{0};
        std::atomic<bool> failed{false};
        auto run = [&] {
            for (;;) {
                const size_t i = next++;
                if (i >= planned.size()) return;
                if (failed) continue;   // a take failed: the rest are never queued
                slots_.acquire();
                try {
                    submit_job(s_, planned[i]);
                    results[i] = wait_take(planned[i]);
                } catch (...) {
                    errors[i] = std::current_exception();
                    failed = true;
                }
                slots_.release();
            }
        };
        const size_t n_threads = std::min(planned.size(), (size_t) std::max(1, 2 * s_.batch_slots));
        std::vector<std::thread> threads;
        for (size_t t = 0; t < n_threads; ++t) threads.emplace_back(run);
        for (auto & t : threads) t.join();
        for (auto & e : errors) if (e) std::rethrow_exception(e);

        std::vector<SynthTake> out;
        for (auto & r : results) {
            const json & rec = r.laci;
            SynthTake t;
            t.audio = std::move(r.pcm);
            t.attempts = rec["attempts"].get<int>();
            std::vector<std::pair<std::string, int>> retries;
            for (const auto & f : rec["failures"]) {
                const std::string kind = f["kind"].get<std::string>();
                if (!f["ignored"].get<bool>()) counter_add(retries, {{kind, 1}});
                else if (kind != "pace_fast" && kind != "pace_slow") t.capped = true;   // laci.PACE_KINDS
            }
            t.retries = counter_json(retries);
            t.exhausted = rec["capped"].get<bool>();
            t.end = rec["end"].get<std::string>();
            t.final_bucket = rec["rate_bucket"];
            out.push_back(std::move(t));
        }
        return out;
    }

    // Synthesize, align and choose the takes of segments `ks` into dub.takes (sgl _render).
    Stats render(Dub & dub, const std::vector<int> & ks, const std::optional<std::string> & session) {
        Stats stats;
        if (ks.empty()) return stats;
        json & info = dub.segments;
        std::vector<Job> jobs;
        for (int k : ks) {
            const json & e = info[(size_t) k];
            const double d = e["ref_end"].get<double>() - e["ref_start"].get<double>();
            int slot = 0;
            for (const auto & c : e["candidates"]) {
                Job j;
                j.bucket = c["bucket"].get<int>();
                j.seed = c["seed"].get<int64_t>();
                j.segment = k;
                j.slot = slot++;
                j.text = e["target_text"].get<std::string>();
                j.max_tokens = dubplan::max_tokens_for(d);
                j.target_frames = d * dubplan::FRAME_RATE;
                j.pinned = e["pinned"].get<bool>();
                jobs.push_back(std::move(j));
            }
        }
        auto t0 = std::chrono::steady_clock::now();
        std::vector<SynthTake> takes = synthesize(dub, jobs, session);
        stats.synthesis_s = seconds_since(t0);
        stats.candidates = (int) jobs.size();
        for (const auto & t : takes) {
            counter_add(stats.retries, t.retries);
            stats.exhausted += t.exhausted;
        }

        t0 = std::chrono::steady_clock::now();
        std::vector<size_t> valid;
        for (size_t i = 0; i < takes.size(); ++i) if (!takes[i].audio.empty()) valid.push_back(i);
        std::map<size_t, json> aligned;
        {
            TempDir tmp("dub-");
            std::vector<AlignJob> reqs;
            for (size_t i : valid) {
                const std::string path = tmp.file("take" + std::to_string(i) + ".wav");
                write_float_wav(path, takes[i].audio, dubplan::SAMPLE_RATE);
                reqs.push_back({path, (double) takes[i].audio.size() / dubplan::SAMPLE_RATE, spoken_text(jobs[i].text),
                                dub.target_lang, std::nullopt});
            }
            if (!reqs.empty()) {
                const auto res = align(reqs);
                for (size_t n = 0; n < valid.size() && n < res.size(); ++n) aligned[valid[n]] = res[n];
            }
        }
        stats.align_s = seconds_since(t0);
        std::map<size_t, std::pair<double, double>> spans;
        for (size_t i : valid) {
            std::vector<dubplan::TimedWord> items;
            for (const auto & it : aligned[i])
                items.push_back({it["start"].get<double>(), it["end"].get<double>(), it["placed"].get<bool>()});
            const auto sp = dubplan::speech_span(items, takes[i].audio.data(), takes[i].audio.size(), dubplan::SAMPLE_RATE);
            if (sp && sp->second > sp->first) spans[i] = *sp;
        }

        std::vector<int> nonempty;
        for (size_t j = 0; j < info.size(); ++j) if (info[j]["status"] == "ok") nonempty.push_back((int) j);
        for (int k : ks) {
            json & e = info[(size_t) k];
            const double d = e["ref_end"].get<double>() - e["ref_start"].get<double>();
            double room = INFINITY;
            for (int j : nonempty)
                if (j > k) { room = info[(size_t) j]["ref_start"].get<double>() - dubplan::MIN_GAP_S - e["ref_start"].get<double>(); break; }
            std::vector<size_t> mine;
            for (size_t i = 0; i < jobs.size(); ++i) if (jobs[i].segment == k) mine.push_back(i);
            std::vector<std::optional<dubplan::TakeResult>> cands;
            for (size_t i : mine) {
                const auto sp = spans.find(i);
                if (sp == spans.end()) { cands.push_back(std::nullopt); continue; }
                const SynthTake & t = takes[i];
                const double s0 = sp->second.first, s1 = sp->second.second;
                dubplan::TakeResult c{s1 - s0, t.capped, t.end};
                cands.push_back(c);
                const auto score = dubplan::candidate_score(c, d, room);
                json & cand = e["candidates"][(size_t) jobs[i].slot];
                cand["speech_s"] = s1 - s0;
                cand["attempts"] = t.attempts;
                cand["retries"] = t.retries;
                cand["final_bucket"] = t.final_bucket;
                cand["end"] = t.end;
                cand["capped"] = t.capped;
                cand["score"] = score ? json(*score) : json();
            }
            const auto pick = dubplan::choose_candidate(cands, d, room);
            if (!pick) throw std::runtime_error("segment " + std::to_string(k) + ": no usable take");
            const size_t i = mine[(size_t) *pick];
            e["chosen"] = jobs[i].slot;
            SynthTake & t = takes[i];
            const auto [s0, s1] = spans[i];
            e["final_bucket"] = t.final_bucket;
            e["retries"] = t.retries;
            e["fit"] = (s1 - s0) / d;
            auto kept = std::make_shared<Take>();
            kept->audio = t.audio;
            kept->s0 = s0;
            kept->s1 = s1;
            kept->items = aligned[i];
            kept->attempts = t.attempts;
            kept->retries = t.retries;
            dub.takes[k] = std::move(kept);
        }
        return stats;
    }

    // Place every chosen take on the timeline, lip-sync it when asked, store the dub and build the response.
    json assemble(const std::shared_ptr<Dub> & dub, const Stats & stats, double generation_s) {
        json & info = dub->segments;
        std::vector<int> nonempty;
        for (const auto & [k, t] : dub->takes) nonempty.push_back(k);
        std::vector<dubplan::PlaceItem> items;
        for (int k : nonempty) {
            const Take & t = *dub->takes[k];
            items.push_back({info[(size_t) k]["ref_start"].get<double>(), info[(size_t) k]["ref_end"].get<double>(),
                             (double) t.audio.size() / dubplan::SAMPLE_RATE, t.s0, t.s1});
        }
        const auto [placed, timeline_len] = dubplan::place_segments(items, dub->duration);

        const int sr = dubplan::SAMPLE_RATE;
        const int fade = std::max(1, (int) (dubplan::FADE_S * sr));
        std::vector<std::pair<int64_t, std::vector<float>>> chunks;
        for (size_t n = 0; n < nonempty.size(); ++n) {
            const auto & take = dub->takes[nonempty[n]]->audio;
            const auto & p = placed[n];
            const int64_t len = (int64_t) take.size();
            auto clampi = [&](double x) { return std::clamp((int64_t) std::nearbyint(x), (int64_t) 0, len); };
            const int64_t a = clampi(p.crop_start * sr), b = std::max(a, clampi(p.crop_end * sr));
            std::vector<float> piece(take.begin() + a, take.begin() + b);
            const int nf = std::min(fade, (int) (piece.size() / 2));
            for (int q = 0; q < nf; ++q) {
                const float r = nf > 1 ? (float) ((double) q / (nf - 1)) : 0.0f;
                piece[(size_t) q] *= r;
                piece[piece.size() - 1 - (size_t) q] *= r;
            }
            chunks.emplace_back((int64_t) std::nearbyint(p.place_start * sr), std::move(piece));
        }
        int64_t total = (int64_t) std::nearbyint(timeline_len * sr);
        for (const auto & [pos, pc] : chunks) total = std::max(total, pos + (int64_t) pc.size());
        std::vector<float> mix((size_t) total, 0.0f);
        for (const auto & [pos, pc] : chunks)
            for (size_t q = 0; q < pc.size(); ++q) if (pos + (int64_t) q >= 0) mix[(size_t) pos + q] += pc[q];
        std::optional<dubplan::LipMap> lip_map;
        json lipsync = nullptr;
        double lipsync_s = 0.0;
        if (dub->lips) {
            const auto t0 = std::chrono::steady_clock::now();
            lip_sync(*dub->lips, mix, lip_map, lipsync);
            lipsync_s = seconds_since(t0);
        }
        for (float & v : mix) v = std::clamp(v, -1.0f, 1.0f);
        total = (int64_t) mix.size();
        dub->wav = dac_wav_bytes(mix, sr);

        auto warp = [&](double t) { return lip_map ? lip_map->out_time(t) : t; };
        json words = json::array();
        std::vector<double> start_err, end_err, dur_err;
        double max_shift = -INFINITY;
        for (size_t n = 0; n < nonempty.size(); ++n) {
            const int k = nonempty[n];
            const auto & p = placed[n];
            json & e = info[(size_t) k];
            const Take & t = *dub->takes[k];
            e["start"] = p.speech_start;
            e["end"] = p.speech_end;
            e["speech_start"] = p.speech_start;
            e["speech_end"] = p.speech_end;
            e["shift"] = p.shift;
            e["start_error"] = p.start_error;
            e["end_error"] = p.end_error;
            e["duration_error"] = p.duration_error;
            e["lip_start"] = lipsync.is_null() ? json() : json(warp(p.speech_start));
            e["lip_end"] = lipsync.is_null() ? json() : json(warp(p.speech_end));
            if (lip_map) {
                const auto w = dubplan::lip_warp(*lip_map, p.speech_start, p.speech_end);
                e["lip_scale"] = w.lip_scale;
                e["lip_stretch"] = w.lip_stretch;
                e["lip_squeeze"] = w.lip_squeeze;
                e["lip_warp"] = w.lip_warp;
            } else if (!lipsync.is_null()) {
                e["lip_scale"] = e["lip_stretch"] = e["lip_squeeze"] = e["lip_warp"] = 1.0;
            }
            for (const auto & it : t.items)
                words.push_back({{"word", it["word"]}, {"start", warp(p.speech_start + (it["start"].get<double>() - t.s0))},
                                 {"end", warp(p.speech_start + (it["end"].get<double>() - t.s0))}, {"segment", k}});
            start_err.push_back(std::fabs(p.start_error));
            end_err.push_back(std::fabs(p.end_error));
            dur_err.push_back(std::fabs(p.duration_error));
            max_shift = std::max(max_shift, p.shift);
        }
        int laci_retries = 0;
        std::vector<std::pair<std::string, int>> by_kind;
        for (int k : nonempty) {
            laci_retries += dub->takes[k]->attempts;
            counter_add(by_kind, dub->takes[k]->retries);
        }
        int all_retries = 0;
        for (const auto & [kind, n] : stats.retries) all_retries += n;
        const json metrics = {
            {"mean_abs_start_error_ms", 1000 * mean(start_err)},
            {"mean_abs_end_error_ms", 1000 * mean(end_err)},
            {"mean_abs_duration_error_ms", 1000 * mean(dur_err)},
            {"max_shift_ms", 1000 * max_shift},
            {"laci_retries", laci_retries},
            {"retries_by_kind", counter_json(by_kind)},
            {"all_retries", all_retries},
            {"all_retries_by_kind", counter_json(stats.retries)},
            {"exhausted_takes", stats.exhausted},
            {"candidates", stats.candidates},
            {"generation_s", generation_s},
            {"synthesis_s", stats.synthesis_s},
            {"align_s", stats.align_s},
            {"lipsync_s", lipsync_s},
        };
        json segments = json::array();
        for (const auto & e : info) {
            json out = e;
            out["k"] = e["index"];
            segments.push_back(std::move(out));
        }
        const std::string audio_b64 = b64_encode(dub->wav.data(), dub->wav.size());
        const std::string dub_id = dubs.put(dub);
        return {{"dub_id", dub_id}, {"audio_base64", audio_b64}, {"sample_rate", sr}, {"duration", (double) total / sr},
                {"segments", segments}, {"words", words}, {"metrics", metrics}, {"lipsync", lipsync}};
    }

    // AV-sync's warp of the dub timeline onto the lips of the video, bounded by the guard and kept only when it lowers
    // the mean SyncNet distance by more than LIP_MIN_SYNC_GAIN (sgl _lip_sync).
    void lip_sync(const Lips & lips, std::vector<float> & mix, std::optional<dubplan::LipMap> & lip_map, json & info) {
        const int sr = dubplan::SAMPLE_RATE;
        TempDir tmp("dub-");
        const std::string src = tmp.file("dub.wav"), out = tmp.file("synced.wav");
        std::vector<float> padded = mix;
        const int64_t want = (int64_t) std::ceil(lips.end * sr);   // the warp reads the whole face track
        if (want > (int64_t) padded.size()) padded.resize((size_t) want, 0.0f);
        write_float_wav(src, padded, sr);
        const json r = call("lips_sync", {{"lips", lips.path}, {"audio", src}, {"out", out},
                                          {"max_rate", dubplan::LIP_GUARD_MAX_RATE}, {"max_shift_s", dubplan::LIP_GUARD_MAX_SHIFT_S}});
        std::vector<float> synced;
        int synced_sr = 0;
        if (!read_wav(out, synced, synced_sr)) throw std::runtime_error("cannot read the lip-synced audio");
        const auto m = dubplan::LipMap::from_frames(r["start"].get<double>(), r["rate"].get<double>(),
                                                    r["src_frames"].get<std::vector<double>>());
        double max_abs = 0.0, sum_abs = 0.0;
        const auto shifts = m.shifts();
        for (double x : shifts) { max_abs = std::max(max_abs, std::fabs(x)); sum_abs += std::fabs(x); }
        const auto track = dubplan::lip_warp(m, m.src_t.front(), m.src_t.back());
        const double before = r["sync_distance_before"].get<double>(), after = r["sync_distance_after"].get<double>();
        info = {{"start", r["start"]}, {"end", r["end"]}, {"max_shift_ms", 1000 * max_abs},
                {"mean_abs_shift_ms", 1000 * (shifts.empty() ? 0.0 : sum_abs / (double) shifts.size())},
                {"max_stretch", track.lip_stretch}, {"max_squeeze", track.lip_squeeze}, {"mean_warp", track.lip_warp},
                {"sync_distance_before", r["sync_distance_before"]}, {"sync_distance_after", r["sync_distance_after"]},
                {"guard", {{"max_rate", dubplan::LIP_GUARD_MAX_RATE}, {"max_shift_s", dubplan::LIP_GUARD_MAX_SHIFT_S}}}};
        const bool applied = before - after > dubplan::LIP_MIN_SYNC_GAIN;
        info["applied"] = applied;
        if (!applied) {
            info["reason"] = "no sync gain: SyncNet distance " + fmt("%.3f", before) + " unwarped, " + fmt("%.3f", after) + " warped";
            return;
        }
        mix = std::move(synced);
        lip_map = m;
    }
};

std::mutex runtime_mtx;
std::unique_ptr<DubRuntime> runtime_ptr;
const ServerState * runtime_owner = nullptr;

// The server's DubRuntime (created on first use; its alignment worker starts with the first call that needs it). A
// server started again in the same process (zonos3-app) gets a new one.
DubRuntime & runtime(ServerState & s) {
    std::lock_guard<std::mutex> lk(runtime_mtx);
    if (!runtime_ptr || runtime_owner != &s) {
        runtime_ptr = std::make_unique<DubRuntime>(s);
        runtime_owner = &s;
    }
    return *runtime_ptr;
}

// sgl _dub_ready.
DubRuntime & dub_ready(ServerState & s) {
    if (!dub_available(s))
        fail(400, "Dubbing is not enabled on this server: it needs --laci-heads, a checkpoint with a speaking-rate "
                  "condition row and --worker-python (the FuseAlign forced aligner)");
    return runtime(s);
}

// sgl _dub_call: the runtime's errors as HTTP statuses.
template <class F>
auto dub_call(F && f) -> decltype(f()) {
    try {
        return f();
    } catch (const HttpError &) {
        throw;
    } catch (const UnknownId &) {
        fail(404, "unknown ref_id/dub_id; upload or generate again");
    } catch (const AlignWorkerError & e) {
        fail(503, std::string("alignment worker: ") + e.what());
    } catch (const WorkerUnavailable & e) {   // the ASR / FuseAlign worker is unavailable (the message says which)
        fail(503, e.what());
    } catch (const std::invalid_argument & e) {
        fail(400, e.what());
    } catch (const std::runtime_error & e) {
        fail(500, e.what());
    }
}

const std::set<std::string> UPLOAD_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".wav", ".flac", ".mp3", ".ogg"};

// sgl _dub_upload: the request body (a file named `name`) in `dir`; 413 past the dub's upload cap.
std::string dub_upload(const httplib::Request & req, const std::string & name, const TempDir & dir) {
    if ((int64_t) req.body.size() > dubplan::MAX_VIDEO_BYTES) fail(413, "upload too large");
    const std::string suffix = lower_ascii(fs::path(name).extension().string());
    const std::string path = dir.file("input" + (UPLOAD_SUFFIXES.count(suffix) ? suffix : std::string(".bin")));
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) fail(500, "cannot write a temporary upload file");
    const bool ok = fwrite(req.body.data(), 1, req.body.size(), f) == req.body.size();
    if (fclose(f) != 0 || !ok) fail(500, "cannot write a temporary upload file");
    return path;
}

std::string param(const httplib::Request & req, const char * key, const char * d) {
    return req.has_param(key) ? req.get_param_value(key) : std::string(d);
}

// re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("._") or "video"
std::string download_stem(const std::string & name) {
    const std::string stem = fs::path(name).stem().string();
    std::string out;
    bool run = false;
    for (char c : stem) {
        const bool ok = isalnum((unsigned char) c) || c == '.' || c == '_' || c == '-';
        if (ok) { out += c; run = false; }
        else if (!run) { out += '_'; run = true; }
    }
    const size_t a = out.find_first_not_of("._"), b = out.find_last_not_of("._");
    out = a == std::string::npos ? "" : out.substr(a, b - a + 1);
    return out.empty() ? "video" : out;
}

}  // namespace

// sgl _dub_available, plus the FuseAlign worker (--worker-python) every dub's word timings need.
bool dub_available(const ServerState & s) { return s.aligner && !laci_languages(s).empty() && !rate_edges(s).empty(); }

json dub_capabilities(const ServerState & s) {
    json caps = runtime(const_cast<ServerState &>(s)).capabilities();
    caps["enabled"] = dub_available(s);
    return caps;
}

void register_dub_routes(httplib::Server & svr, ServerState & s) {
    svr.Get("/v2/dub/capabilities", guarded([&s](const httplib::Request &, httplib::Response & res) {
        send_json(res, dub_capabilities(s));
    }));
    svr.Post("/v2/dub/ref", keepalive([&s](const httplib::Request & req, httplib::Response & res) {
        DubRuntime & rt = dub_ready(s);
        TempDir tmp("dub-");
        const std::string path = dub_upload(req, param(req, "name", "audio"), tmp);
        send_json(res, dub_call([&] { return rt.add_ref(path); }));
    }));
    svr.Post(R"(/v2/dub/lips/([^/]+))", keepalive([&s](const httplib::Request & req, httplib::Response & res) {
        DubRuntime & rt = dub_ready(s);
        const std::string ref_id = req.matches[1];
        send_json(res, dub_call([&] {
            rt.refs.get(ref_id);   // 404 before the upload is stored
            TempDir tmp("dub-");
            const std::string video = dub_upload(req, param(req, "name", "video"), tmp);
            return rt.prepare_lips(ref_id, video);
        }));
    }));
    svr.Post("/v2/dub/transcribe", keepalive([&s](const httplib::Request & req, httplib::Response & res) {
        const json body = schema::parse_body(req, "DubTranscribeRequest");
        DubRuntime & rt = dub_ready(s);
        const std::optional<std::string> lang = set(body, "ref_lang") ? std::optional<std::string>(body["ref_lang"].get<std::string>()) : std::nullopt;
        send_json(res, dub_call([&] { return rt.transcribe(body["ref_id"].get<std::string>(), lang); }));
    }));
    svr.Post("/v2/dub/analyze", keepalive([&s](const httplib::Request & req, httplib::Response & res) {
        const json body = schema::parse_body(req, "DubAnalyzeRequest");
        DubRuntime & rt = dub_ready(s);
        send_json(res, dub_call([&] { return rt.analyze(body); }));
    }));
    svr.Post("/v2/dub/generate", keepalive([&s](const httplib::Request & req, httplib::Response & res) {
        const json body = schema::parse_body(req, "DubGenerateRequest");
        DubRuntime & rt = dub_ready(s);
        const auto session = session_id(req);
        send_json(res, dub_call([&] { return rt.generate(body, session); }));
    }));
    svr.Post("/v2/dub/regenerate", keepalive([&s](const httplib::Request & req, httplib::Response & res) {
        const json body = schema::parse_body(req, "DubRegenerateRequest");
        DubRuntime & rt = dub_ready(s);
        const auto session = session_id(req);
        send_json(res, dub_call([&] { return rt.regenerate(body, session); }));
    }));
    // The video with the dub as its audio track (raw video bytes in, MP4 out).
    svr.Post(R"(/v2/dub/mux/([^/]+))", guarded([&s](const httplib::Request & req, httplib::Response & res) {
        DubRuntime & rt = dub_ready(s);
        const std::string dub_id = req.matches[1];
        const std::string name = param(req, "name", "video");
        const std::string stem = download_stem(name);
        std::string data;
        dub_call([&] {
            rt.dubs.get(dub_id);   // 404 before the upload is stored
            TempDir tmp("dub-");
            const std::string video = dub_upload(req, name, tmp);
            const std::string out = tmp.file("out.mp4");
            rt.mux(dub_id, video, out);
            if (!read_file(out, data)) throw std::runtime_error("ffmpeg wrote no video");
            return 0;
        });
        res.set_header("Content-Disposition", "attachment; filename=\"" + stem + "_dub.mp4\"");
        res.set_content(data, "video/mp4");
    }));
}

}  // namespace zs
