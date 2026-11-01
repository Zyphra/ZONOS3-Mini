#include "compat.h"

#include "server/edit.h"

#include "server/generate.h"
#include "server/http.h"
#include "server/schema.h"
#include "server/speakers.h"
#include "server/util.h"
#include "server/workers.h"

#include "zonos3-edit.h"
#include "zonos3-markup.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <list>
#include <random>
#include <thread>

namespace zs {

using json = nlohmann::json;
namespace fs = std::filesystem;
namespace E = zonos3_edit;

// sgl _EDIT_ALIGN_MAX_SECONDS / _EDIT_MAX_SECONDS: the MMS alignment limit bounds every edit.
static constexpr double EDIT_ALIGN_MAX_SECONDS = 150.0;
static constexpr double EDIT_MAX_SECONDS = EDIT_ALIGN_MAX_SECONDS;
static constexpr int EDIT_DECODE_MAX_SECONDS = 600;   // uploads are decoded this far, so the length check sees them
static const char * GROUP_OPEN = "\xEE\x80\x80";      // U+E000

namespace {

// ---------------------------------------------------------------------------------------------- request helpers

bool set(const json & j, const char * key) { return j.contains(key) && !j[key].is_null(); }
double num(const json & j, const char * key, double d) { return set(j, key) ? j[key].get<double>() : d; }
std::string str(const json & j, const char * key, const std::string & d) { return set(j, key) ? j[key].get<std::string>() : d; }

[[noreturn]] void bad(const std::exception & e) { fail(400, e.what()); }

double frame_rate(const ServerState & s) { return (double) s.dac.sample_rate / dac_samples_per_frame(s.dac); }

int edit_max_spans(const ServerState & s) { return std::max(0, zonos3_edit_marker_rows(s.model)); }

// sgl _require_editing.
void require_editing(const ServerState & s) {
    if (!zonos3_supports_editing(s.model))
        fail(400, "The loaded checkpoint does not support speech editing (needs span_end_id and marker rows).");
}

// A scratch directory for the MMS worker's audio files, removed with its contents.
struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() / ("zonos3-edit-" + random_hex(16));
        fs::create_directories(path);
    }
    ~TempDir() { std::error_code ec; fs::remove_all(path, ec); }
    std::string file(const std::string & name) const { return (path / name).string(); }
};

// sgl _write_edit_audio: decode the upload, check its length and store it as WAV for MMS.
struct EditAudio { std::vector<float> pcm; std::string path; double duration = 0; };
EditAudio write_edit_audio(const std::vector<uint8_t> & bytes, const TempDir & tmp, double max_seconds) {
    EditAudio a;
    a.pcm = decode_audio_bytes(bytes, EDIT_DECODE_MAX_SECONDS);
    a.duration = (double) a.pcm.size() / 44100.0;
    if (a.duration > max_seconds) {
        char msg[160];
        snprintf(msg, sizeof msg, "Audio is %.1f s long; at most %.0f s are supported here.", a.duration, max_seconds);
        fail(400, msg);
    }
    a.path = tmp.file("original.wav");
    if (!dac_write_wav(a.path.c_str(), a.pcm, 44100)) fail(500, "cannot write a temporary audio file");
    return a;
}

// ------------------------------------------------------------------------------------------------------ MMS workers

// MMS workers by worker language, least recently used first; at most ZONOS3_MMS_MAX_WORKERS (default 2) stay resident
// (each holds MMS-1B-all on the GPU). A worker that failed is replaced on the next request (sgl _mms_worker).
std::mutex mms_mtx;
std::list<std::pair<std::string, std::shared_ptr<JsonLinesWorker>>> mms_workers;

std::shared_ptr<JsonLinesWorker> mms_worker(const ServerState & s, const std::string & language) {
    const char * root = getenv("ZSA_ROOT");
    if (!root || !*root)
        throw WorkerError("Speech editing needs ZSA_ROOT: a z-speech-alignment checkout "
                          "(git@github.com:Zyphra/z-speech-alignment.git) with its uv environment (run `uv sync` in it); "
                          "ZSA_PYTHON overrides the interpreter.");
    std::lock_guard<std::mutex> lk(mms_mtx);
    for (auto it = mms_workers.begin(); it != mms_workers.end(); ++it)
        if (it->first == language) {
            mms_workers.splice(mms_workers.end(), mms_workers, it);
            return mms_workers.back().second;
        }
    const char * zsa_python = getenv("ZSA_PYTHON");
    std::vector<std::string> argv{zsa_python && *zsa_python ? std::string(zsa_python) : (fs::path(root) / ".venv/bin/python").string(),
                                  (fs::path(s.python_src) / "zonos3" / "workers" / "zsa_mms.py").string(), "--lang", language};
    const char * device = getenv("ZONOS3_MMS_DEVICE");
    if (device && *device) { argv.push_back("--device"); argv.push_back(device); }
    auto worker = std::make_shared<JsonLinesWorker>(
        argv, std::map<std::string, std::optional<std::string>>{{"PYTHONPATH", std::string(root)},
                                                                {"HF_HUB_OFFLINE", std::string("1")},
                                                                {"PYTHONDONTWRITEBYTECODE", std::string("1")},
                                                                {"VIRTUAL_ENV", std::nullopt}});
    mms_workers.emplace_back(language, worker);
    const char * cap_env = getenv("ZONOS3_MMS_MAX_WORKERS");
    const size_t cap = (size_t) std::max(1, cap_env && *cap_env ? atoi(cap_env) : 2);
    while (mms_workers.size() > cap) {
        auto old = mms_workers.front().second;
        mms_workers.pop_front();
        std::thread([old] { old->close(); }).detach();   // an in-flight call finishes first (the worker's lock)
    }
    return worker;
}

void drop_mms_worker(const std::string & language, const std::shared_ptr<JsonLinesWorker> & worker) {
    std::lock_guard<std::mutex> lk(mms_mtx);
    for (auto it = mms_workers.begin(); it != mms_workers.end(); ++it)
        if (it->first == language && it->second == worker) { mms_workers.erase(it); break; }
    std::thread([worker] { worker->close(); }).detach();
}

// sgl _mms_run: 503 when the worker cannot run, 400 when it answers an error.
std::vector<json> mms_run(const ServerState & s, const std::vector<json> & requests, const std::string & language) {
    std::vector<json> responses;
    std::shared_ptr<JsonLinesWorker> worker;
    try {
        worker = mms_worker(s, language);
        for (const json & r : requests) responses.push_back(worker->call(r));
    } catch (const WorkerError & e) {
        if (worker) drop_mms_worker(language, worker);
        fail(503, std::string("MMS alignment is unavailable: ") + e.what());
    }
    for (size_t i = 0; i < requests.size(); ++i)
        if (responses[i].contains("error")) {
            const json & err = responses[i]["error"];
            fail(400, "MMS " + requests[i]["op"].get<std::string>() + " failed: " + (err.is_string() ? err.get<std::string>() : err.dump()));
        }
    return responses;
}

json mms_one(const ServerState & s, const json & request, const std::string & language) {
    return mms_run(s, {request}, language)[0];
}

// sgl _edit_mms_language: the MMS worker language of a request's `language` (400 for one MMS does not serve).
std::string edit_mms_language(const json & body) {
    const std::string language = resolve_language(body);
    const auto code = E::mms_language(language);
    if (!code) fail(400, E::mms_language_error(language));
    return *code;
}

std::vector<std::vector<std::string>> token_lists(const json & j) {
    std::vector<std::vector<std::string>> out;
    for (const auto & l : j) out.push_back(l.get<std::vector<std::string>>());
    return out;
}

std::vector<E::AlignedWord> aligned_words(const json & response) {
    std::vector<E::AlignedWord> out;
    for (const auto & w : response.at("words")) {
        E::AlignedWord a;
        a.word = w.at("word").get<std::string>();
        a.start = w.at("start").get<double>();
        a.end = w.at("end").get<double>();
        if (set(w, "confidence")) a.confidence = w["confidence"].get<double>();
        a.tokens = w.at("tokens").get<std::vector<std::string>>();
        out.push_back(std::move(a));
    }
    return out;
}

json public_words(const std::vector<E::AlignedWord> & words) {
    json out = json::array();
    for (const auto & w : words)
        out.push_back({{"word", w.word}, {"start", w.start}, {"end", w.end},
                       {"confidence", w.confidence ? json(*w.confidence) : json()}, {"tokens", w.tokens}});
    return out;
}

json seconds_pair(double a, double b) { return json::array({a, b}); }

// ---------------------------------------------------------------------------------------------------- generation

// Region steering: controls + region_controls[k] summed per id (sgl _edit_region_doses), checked against the bundle.
std::optional<std::vector<std::map<std::string, double>>> edit_region_doses(const ServerState & s, const json & body,
                                                                            bool use_speaker_row) {
    const bool has_spans = set(body, "spans");
    if (set(body, "region_controls")) {
        if (!has_spans) throw std::invalid_argument("region_controls needs explicit spans (one entry per span).");
        const size_t n = body["region_controls"].size(), m = body["spans"].size();
        if (n != m)
            throw std::invalid_argument("region_controls has " + std::to_string(n) + " entries for " + std::to_string(m) +
                                        " spans; send one per span (null for none).");
    }
    const size_t regions = has_spans ? body["spans"].size() : 1;
    std::vector<std::map<std::string, double>> doses;
    bool any = false;
    for (size_t k = 0; k < regions; ++k) {
        std::vector<std::pair<std::string, double>> merged;   // insertion order, like the Python dict
        auto add = [&](const json & controls) {
            if (controls.is_null()) return;
            for (auto it = controls.begin(); it != controls.end(); ++it) {
                auto at = std::find_if(merged.begin(), merged.end(), [&](const auto & p) { return p.first == it.key(); });
                if (at == merged.end()) merged.emplace_back(it.key(), it.value().get<double>());
                else at->second += it.value().get<double>();
            }
        };
        if (set(body, "controls")) add(body["controls"]);
        if (set(body, "region_controls")) add(body["region_controls"][k]);
        std::map<std::string, double> region;
        for (const auto & [id, v] : merged) if (v != 0.0) region[id] = v;
        any = any || !region.empty();
        doses.push_back(std::move(region));
    }
    if (!any) return std::nullopt;
    if (!s.steer_loaded)
        throw std::invalid_argument("This server has no steering bundle (place steer.gguf next to the backbone or pass "
                                    "--steer); `controls` are unavailable.");
    std::vector<std::string> raw;
    for (const auto & region : doses) {
        std::vector<float> probe(s.steer.speaker_dim, 0.0f);
        zonos3_steering scratch;
        std::string err;
        if (!zonos3_steer_resolve(s.steer, region, {}, 0, ZONOS3_STEER_FRAMES_PER_CHAR, &probe, scratch, err))
            throw std::invalid_argument(err);
        for (const auto & [id, v] : region) {
            const auto & d = s.steer.dirs[s.steer.index.at(id)];
            if (!d.hidden && std::find(raw.begin(), raw.end(), id) == raw.end()) raw.push_back(id);
        }
    }
    std::sort(raw.begin(), raw.end());
    if (!raw.empty() && !use_speaker_row) {
        std::string ids;
        for (const auto & id : raw) ids += (ids.empty() ? "" : ", ") + id;
        throw std::invalid_argument("Controls " + ids + " move the speaker embedding and need the speaker row "
                                    "(speaker_embedding true or unset, on a speaker-embedding checkpoint).");
    }
    return doses;
}

struct RegionSteering {
    std::vector<float> speaker_embedding;   // empty = no speaker row
    zonos3_steering steering;
};

// One span-end-terminated infill generation through the scheduler: the generated delayed rows (without the finishing
// row, as sgl hands them to extract_fill) and the aligned eos frame (-1 = none).
struct Generated { std::vector<int32_t> rows; int eos_frame = -1; };
Generated generate_fill(ServerState & s, const std::string & text, const RegionSteering & steer,
                        const std::vector<int32_t> & condition_ids, const std::vector<int32_t> & window_codes,
                        const std::vector<E::Span> & window_spans, const std::vector<std::vector<int32_t>> & window_fills,
                        const zonos3_sampling & sp, int max_frames, int min_frames) {
    const auto & hp = s.model.hp;
    const int K = (int) hp.n_codebooks, W = K + 1;
    auto job = std::make_shared<ReqJob>();
    zonos3_prompt_options opt;
    if (!steer.speaker_embedding.empty()) opt.speaker_embedding = steer.speaker_embedding.data();
    opt.condition_buckets = condition_ids;
    opt.silence_prefill_frames = 0;
    std::string err;
    if (!zonos3_build_prompt(s.model, text, opt, job->prompt, &err))
        fail(400, err.empty() ? "could not build the edit prompt" : err);
    E::Layout layout;
    layout.n_codebooks = K;
    layout.audio_pad_id = (int) hp.audio_pad_id;
    layout.text_vocab = (int) hp.text_vocab;
    layout.span_end_id = hp.span_end_id;
    try {
        job->prompt.ids = E::edit_prompt_tokens(job->prompt.ids, window_codes, window_spans, window_fills, layout);
    } catch (const std::invalid_argument & e) {
        bad(e);
    }
    job->prompt.n_rows = (int) (job->prompt.ids.size() / W);
    if (job->prompt.n_rows + max_frames > s.context_capacity)
        fail(400, "the edit prompt and max_span_seconds exceed the slot context capacity (" +
                  std::to_string(s.context_capacity) + " rows)");
    job->idf.assign(job->prompt.ids.begin(), job->prompt.ids.end());
    job->n0 = job->prompt.n_rows;
    GenReq & gr = job->gr;
    gr.sp = sp;
    gr.max_frames = max_frames;
    gr.stream = false;
    gr.steering = steer.steering;
    gr.stop_id = hp.span_end_id;
    gr.min_frames = min_frames;
    gr.codes_only = true;
    submit_job(s, job);
    Generated out;
    {
        std::unique_lock<std::mutex> lk(job->m);
        job->cv.wait(lk, [&] { return job->finished; });
        if (job->failed) fail(500, "generation failed");
        out.rows = job->codes;
        out.eos_frame = job->eos_frame;
    }
    if (out.rows.size() >= (size_t) K) out.rows.resize(out.rows.size() - K);   // the finishing row is not emitted
    return out;
}

// The whole decode of ordinary codes [frames, K] on a DAC lane (sgl TTSDecodeCodesMsg).
std::vector<float> decode_codes(ServerState & s, const std::vector<int32_t> & frames) {
    const int K = (int) s.model.hp.n_codebooks;
    const int n = (int) (frames.size() / K), H = n + K - 1;
    auto job = std::make_shared<ReqJob>();
    job->codes.assign((size_t) H * K, (int32_t) s.model.hp.audio_pad_id);
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < K; ++c)
            if (r - c >= 0 && r - c < n) job->codes[(size_t) r * K + c] = frames[(size_t) (r - c) * K + c];
    job->gain = dac_output_gain(s.dac, -1.0);
    submit_decode(s, job);
    std::vector<float> audio;
    std::unique_lock<std::mutex> lk(job->m);
    job->cv.wait(lk, [&] { return job->finished; });
    if (job->failed) fail(500, "decoding the edited codes failed");
    for (auto & b : job->pcm) audio.insert(audio.end(), b.begin(), b.end());
    return audio;
}

// ---------------------------------------------------------------------------------------------------- handlers

json handle_transcribe(ServerState & s, const json & body) {
    require_editing(s);
    TempDir tmp;
    const EditAudio audio = write_edit_audio(decode_b64_field(body, "audio_base64"), tmp, EDIT_ALIGN_MAX_SECONDS);
    const json response = mms_one(s, {{"id", "original"}, {"op", "transcribe"}, {"audio", audio.path}}, edit_mms_language(body));
    return {{"text", response.at("text")}, {"duration", audio.duration}, {"words", public_words(aligned_words(response))}};
}

json handle_align(ServerState & s, const json & body) {
    require_editing(s);
    const std::string text = strip(body.at("text").get<std::string>());
    if (text.empty()) fail(400, "text is empty.");
    TempDir tmp;
    const EditAudio audio = write_edit_audio(decode_b64_field(body, "audio_base64"), tmp, EDIT_ALIGN_MAX_SECONDS);
    const json response = mms_one(s, {{"id", "original"}, {"op", "align"}, {"audio", audio.path}, {"text", text}},
                                  edit_mms_language(body));
    return {{"text", text}, {"duration", audio.duration}, {"words", public_words(aligned_words(response))}};
}

// POST /v2/edit (sgl tts_edit): regenerate the edited regions of recorded speech; unedited codec frames are kept
// exactly. Each region is one span-end-terminated generation over [speaker row][condition rows][text][unmasked
// context][earlier fills][marker]; a clip longer than max_context_seconds is cut into word-snapped windows.
json handle_edit(ServerState & s, const json & body) {
    require_editing(s);
    if (num(body, "tag_cfg_scale", 1.0) != 1.0)
        fail(400, "tag_cfg_scale is not supported by /v2/edit (speech-edit requests run without tag CFG); use 1.0.");
    const auto & hp = s.model.hp;
    const int max_spans = edit_max_spans(s);
    const int K = (int) hp.n_codebooks;
    const double fr = frame_rate(s);
    // Nonverbal markup is text for the model only: markers (canonical, no inner whitespace) ride in the prompt text and
    // never reach MMS (word diff, alignment, verify).
    const std::string target_text = E::canonical_markup(strip(body.at("target_text").get<std::string>()));
    const std::string plain_target_text = zonos3_strip_nonverbal_markup(target_text);
    const bool has_spans = set(body, "spans");
    const double min_fill_ratio = num(body, "min_fill_ratio", 0.7);
    const double max_span_seconds = num(body, "max_span_seconds", 30.0);
    const bool verify = set(body, "verify") && body["verify"].get<bool>();
    TempDir tmp;

    std::optional<std::vector<std::map<std::string, double>>> region_doses;
    std::string language, mms_lang;
    bool use_speaker_row;
    std::vector<int32_t> condition_ids;
    std::vector<uint8_t> upload;
    EditAudio original;
    try {
        if (target_text.empty()) throw std::invalid_argument("target_text is empty.");
        const std::string groups = zonos3_check_phoneme_groups(s.model.tok, target_text);
        if (!groups.empty()) throw std::invalid_argument(groups);
        if (set(body, "original_words") && strip(str(body, "original_text", "")).empty())
            throw std::invalid_argument("original_words needs the original_text they spell.");
        const bool speaker_row_capable = hp.speaker_dim > 0;
        use_speaker_row = set(body, "speaker_embedding") ? body["speaker_embedding"].get<bool>() : speaker_row_capable;
        region_doses = edit_region_doses(s, body, use_speaker_row);
        language = resolve_language(body);
        const auto code = E::mms_language(language);
        if (!code) throw std::invalid_argument(E::mms_language_error(language));
        mms_lang = *code;
        if (use_speaker_row && !speaker_row_capable)
            throw std::invalid_argument("speaker_embedding needs a checkpoint with a speaker-embedding row.");
        condition_ids = resolve_condition_buckets(s, body);
        upload = decode_b64_field(body, "audio_base64");
        original = write_edit_audio(upload, tmp, EDIT_MAX_SECONDS);
    } catch (const std::invalid_argument & e) {
        bad(e);
    }
    std::vector<float> speaker_embedding;
    if (use_speaker_row) speaker_embedding = embed_audio(s, original.pcm, (int) EDIT_MAX_SECONDS);
    const std::vector<int32_t> codes = encode_audio_codes(s, original.pcm, (int) EDIT_MAX_SECONDS);
    const int total_frames = (int) (codes.size() / K);

    // Longer clips are windowed and every fill gets a minimum length; both read word timings of the original (the
    // caller's original_words, else an MMS alignment) and the target words' zsa tokens. Without original_text, a
    // short clip with explicit spans is not transcribed: its regions' own lengths are expected.
    const bool windowed = total_frames / fr > num(body, "max_context_seconds", 16.0);
    const std::vector<std::string> target_words = E::split_words(target_text);
    const E::TargetNormalize target_norm = E::target_normalize_request(target_text);
    std::optional<std::string> original_text;
    if (set(body, "original_text")) original_text = zonos3_strip_nonverbal_markup(body["original_text"].get<std::string>());
    const bool has_original_text = original_text && !original_text->empty();
    std::optional<std::vector<E::AlignedWord>> words;
    std::vector<std::vector<std::string>> normalized;   // [target tokens, tokens of target word 0, ...]
    try {
        if (set(body, "original_words")) {
            const json & given = body["original_words"];
            std::vector<size_t> missing;
            json texts = json::array({*original_text});
            for (const auto & t : target_norm.texts) texts.push_back(t);
            for (size_t k = 0; k < given.size(); ++k)
                if (!set(given[k], "tokens")) { missing.push_back(k); texts.push_back(given[k].at("word")); }
            const auto lists = token_lists(mms_one(s, {{"id", "original_words"}, {"op", "normalize"}, {"normalize", texts}},
                                                   mms_lang).at("normalized"));
            const size_t nt = target_norm.texts.size();
            if (lists.size() != 1 + nt + missing.size()) fail(503, "MMS alignment is unavailable: malformed normalize answer");
            std::vector<E::WordIn> ws;
            std::vector<std::vector<std::string>> word_tokens;
            size_t m = 0;
            for (size_t k = 0; k < given.size(); ++k) {
                E::WordIn w;
                w.word = given[k].at("word").get<std::string>();
                w.start = given[k].at("start").get<double>();
                w.end = given[k].at("end").get<double>();
                if (set(given[k], "confidence")) w.confidence = given[k]["confidence"].get<double>();
                ws.push_back(std::move(w));
                word_tokens.push_back(set(given[k], "tokens") ? given[k]["tokens"].get<std::vector<std::string>>()
                                                              : lists[1 + nt + m++]);
            }
            words = E::edit_words(ws, word_tokens, lists[0], original.duration);
            normalized = target_norm.assemble({lists.begin() + 1, lists.begin() + 1 + nt});
        } else if (!has_spans || windowed || (min_fill_ratio > 0 && has_original_text)) {
            json request = {{"id", "original"}, {"op", has_original_text ? "align" : "transcribe"},
                            {"audio", original.path}, {"normalize", target_norm.texts}};
            if (has_original_text) request["text"] = *original_text;
            const json response = mms_one(s, request, mms_lang);
            normalized = target_norm.assemble(token_lists(response.at("normalized")));
            if (!has_original_text) original_text = response.at("text").get<std::string>();
            words = aligned_words(response);
        }
    } catch (const std::invalid_argument & e) {
        bad(e);
    }
    // verify with pronunciation groups: the reference's tokens, each group a wildcard word
    std::optional<std::vector<std::string>> verify_reference;
    if (verify && target_text.find(GROUP_OPEN) != std::string::npos) {
        if (words) verify_reference = normalized.at(0);
        else
            verify_reference = target_norm.assemble(token_lists(
                mms_one(s, {{"id", "target"}, {"op", "normalize"}, {"normalize", target_norm.texts}}, mms_lang).at("normalized")))[0];
    }

    std::vector<E::Span> spans;
    std::vector<E::EditWindow> windows;
    std::vector<double> expected;
    std::vector<int> min_frames;
    int max_tokens = 0;
    std::vector<RegionSteering> steering;
    try {
        if (has_spans) {
            std::vector<std::pair<double, double>> pairs;
            for (const auto & p : body["spans"]) pairs.emplace_back(p.at(0).get<double>(), p.at(1).get<double>());
            spans = E::seconds_to_spans(pairs, total_frames, fr);
        } else {
            spans = E::diff_spans(*words, normalized.at(0), total_frames, fr, num(body, "margin_seconds", 0.08), 5,
                                  max_spans).spans;
        }
        E::validate_edit_spans(spans, total_frames, max_spans);
        const std::vector<std::vector<std::string>> target_word_tokens =
            words ? std::vector<std::vector<std::string>>(normalized.begin() + 1, normalized.end())
                  : std::vector<std::vector<std::string>>{};
        windows = {E::whole_clip_window(spans, total_frames, target_text)};
        if (windowed)
            windows = E::plan_edit_windows(spans, *words, target_words, target_word_tokens, total_frames, fr,
                                           num(body, "context_seconds", 4.0), target_text);
        if (words) expected = E::expected_fill_seconds(spans, *words, target_words, target_word_tokens, fr);
        else for (const auto & sp : spans) expected.push_back(sp.length / fr);
        // The span-end token is masked until a fill reaches min_fill_ratio of its expected length.
        for (double seconds : expected)
            min_frames.push_back(std::min((int) (min_fill_ratio * seconds * fr), (int) (max_span_seconds * fr)));
        max_tokens = std::min((int) (max_span_seconds * fr) + K + 1, s.max_frames);   // _resolve_tts_max_tokens
        // Per region: (speaker-row embedding, hidden steering delta). Without explicit spans the request's controls
        // apply to every diffed region alike.
        for (size_t k = 0; k < spans.size(); ++k) {
            RegionSteering r;
            r.speaker_embedding = speaker_embedding;
            if (region_doses) {
                std::string err;
                if (!zonos3_steer_resolve(s.steer, (*region_doses)[has_spans ? k : 0], {},
                                          zonos3_steer_text_chars(target_text), ZONOS3_STEER_FRAMES_PER_CHAR,
                                          r.speaker_embedding.empty() ? nullptr : &r.speaker_embedding, r.steering, err))
                    throw std::invalid_argument(err);
            }
            steering.push_back(std::move(r));
        }
    } catch (const std::invalid_argument & e) {
        bad(e);
    }

    uint32_t base_seed;
    if (set(body, "seed")) base_seed = (uint32_t) body["seed"].get<uint64_t>();
    else {   // random.randrange(2**31 - 8)
        static std::mutex rng_m;
        static std::mt19937_64 rng(std::random_device{}());
        std::lock_guard<std::mutex> lk(rng_m);
        base_seed = (uint32_t) std::uniform_int_distribution<uint64_t>(0, (1ull << 31) - 9)(rng);
    }
    zonos3_sampling base;
    base.temperature = (float) num(body, "temperature", 0.75);
    base.top_k = set(body, "topk") ? body["topk"].get<int>() : 80;
    base.top_p = (float) num(body, "top_p", 0.69);
    base.min_p = (float) num(body, "min_p", 0.0);
    base.rep_penalty = (float) num(body, "repetition_penalty", 2.4);
    base.rep_window = set(body, "repetition_window") ? body["repetition_window"].get<int>() : 96;
    base.rep_codebooks = set(body, "repetition_codebooks") ? body["repetition_codebooks"].get<int>() : 6;
    if (base.rep_codebooks < 0) base.rep_codebooks = K;
    base.greedy = base.temperature <= 0.0f;
    const bool normalize = !set(body, "text_normalization") || body["text_normalization"].get<bool>();
    std::vector<std::string> window_texts;
    for (const auto & w : windows) window_texts.push_back(normalize_text(s, w.text, language, normalize));
    std::vector<int> codebook_sizes(K, (int) hp.codebook_size);
    const int sr = s.dac.sample_rate;

    auto edit_take = [&](int k) -> json {
        zonos3_sampling sp = base;
        sp.seed = base_seed + (uint32_t) k;
        std::vector<std::vector<int32_t>> fills(spans.size());
        std::vector<std::optional<bool>> stopped(spans.size());
        for (size_t wi = 0; wi < windows.size(); ++wi) {
            const E::EditWindow & window = windows[wi];
            const std::vector<int32_t> window_codes(codes.begin() + (size_t) window.start * K, codes.begin() + (size_t) window.end * K);
            const std::vector<E::Span> window_spans = window.spans(spans);
            std::vector<std::vector<int32_t>> window_fills;
            for (int region : window.regions) {
                const Generated g = generate_fill(s, window_texts[wi], steering[region], condition_ids, window_codes,
                                                  window_spans, window_fills, sp, max_tokens, min_frames[region]);
                E::Fill fill = E::extract_fill(g.rows, g.eos_frame, (int) hp.audio_pad_id, codebook_sizes);
                window_fills.push_back(fill.frames);
                fills[region] = std::move(fill.frames);
                stopped[region] = fill.ended;
            }
        }
        std::vector<int32_t> spliced;
        try {
            spliced = E::splice_edit(codes, K, spans, fills);
        } catch (const std::invalid_argument & e) {
            bad(e);
        }
        const std::vector<float> edited = decode_codes(s, spliced);
        const double duration = (double) edited.size() / sr;
        json verify_json = nullptr, take_words = nullptr;
        if (verify) {
            const std::string edited_path = tmp.file("take" + std::to_string(k) + ".wav");
            if (!dac_write_wav(edited_path.c_str(), edited, sr)) fail(500, "cannot write a temporary audio file");
            const std::string id = "take" + std::to_string(k);
            std::vector<json> aligned;
            if (!verify_reference) {
                std::vector<json> requests{{{"id", id}, {"op", "transcribe"}, {"audio", edited_path}, {"reference", plain_target_text}}};
                if (duration <= EDIT_ALIGN_MAX_SECONDS)
                    requests.push_back({{"id", id + ":words"}, {"op", "align"}, {"audio", edited_path}, {"text", plain_target_text}});
                auto responses = mms_run(s, requests, mms_lang);
                const json & asr = responses[0];
                verify_json = {{"asr_text", asr.at("text")},
                               {"wer", asr.at("word_errors").get<double>() / std::max(1, asr.at("ref_words").get<int>())},
                               {"cer", asr.at("cer")}};
                aligned.assign(responses.begin() + 1, responses.end());
            } else {
                // Groups are wildcard words: any one recognized word matches; the take's words come from aligning the
                // target with each group replaced by the word heard.
                const json asr = mms_one(s, {{"id", id}, {"op", "transcribe"}, {"audio", edited_path}}, mms_lang);
                std::vector<std::string> hyp;
                for (const auto & w : asr.at("words")) for (const auto & t : w.at("tokens")) hyp.push_back(t.get<std::string>());
                const E::Verify scored = E::wildcard_verify(*verify_reference, hyp);
                json groups = json::array();
                std::vector<std::optional<std::string>> heard;
                for (const auto & g : scored.groups) {
                    groups.push_back({{"ipa", g.ipa}, {"heard", g.heard ? json(*g.heard) : json()}});
                    heard.push_back(g.heard);
                }
                verify_json = {{"asr_text", asr.at("text")},
                               {"wer", (double) scored.word_errors / std::max(1, scored.ref_words)},
                               {"cer", scored.cer}, {"groups", groups}};
                const std::string heard_text = E::fill_groups(plain_target_text, heard);
                if (!heard_text.empty() && duration <= EDIT_ALIGN_MAX_SECONDS)
                    aligned = mms_run(s, {{{"id", id + ":words"}, {"op", "align"}, {"audio", edited_path}, {"text", heard_text}}}, mms_lang);
            }
            if (!aligned.empty()) take_words = public_words(aligned_words(aligned[0]));
        }
        const std::vector<uint8_t> wav = dac_wav_bytes(edited, sr);
        json edited_regions = json::array(), fill_seconds = json::array(), stopped_json = json::array();
        int shift = 0;
        for (size_t r = 0; r < spans.size(); ++r) {
            const int n = (int) (fills[r].size() / K);
            edited_regions.push_back(seconds_pair((spans[r].start + shift) / fr, (spans[r].start + shift + n) / fr));
            shift += n - spans[r].length;
            fill_seconds.push_back(n / fr);
            stopped_json.push_back(stopped[r] ? json(*stopped[r]) : json());
        }
        return {{"seed", (uint64_t) base_seed + (uint64_t) k}, {"audio_base64", b64_encode(wav.data(), wav.size())},
                {"duration", duration}, {"edited_regions", edited_regions}, {"fill_seconds", fill_seconds},
                {"stopped", stopped_json}, {"verify", verify_json}, {"words", take_words}};
    };

    const int num_takes = set(body, "num_takes") ? body["num_takes"].get<int>() : 1;
    std::vector<std::future<json>> futures;
    for (int k = 0; k < num_takes; ++k) futures.push_back(std::async(std::launch::async, edit_take, k));
    json takes = json::array();
    for (auto & f : futures) f.wait();
    for (auto & f : futures) takes.push_back(f.get());   // the first take's error, after every take settled

    json spans_json = json::array(), contexts = nullptr, expected_json = json::array();
    for (const auto & sp : spans) spans_json.push_back(seconds_pair(sp.start / fr, (sp.start + sp.length) / fr));
    if (windowed) {
        contexts = json::array();
        for (const auto & w : windows) contexts.push_back(seconds_pair(w.start / fr, w.end / fr));
    }
    for (double e : expected) expected_json.push_back(e);
    return {{"sample_rate", sr}, {"original_text", original_text ? json(*original_text) : json()},
            {"target_text", target_text}, {"words", words ? public_words(*words) : json()}, {"spans", spans_json},
            {"contexts", contexts}, {"expected_seconds", expected_json}, {"takes", takes}};
}

}  // namespace

json edit_capabilities(const ServerState & s) {
    const bool model = zonos3_supports_editing(s.model);
    const char * root = getenv("ZSA_ROOT");
    return {{"speech_editing", model && root && *root}, {"edit_max_spans", model ? edit_max_spans(s) : 0},
            {"edit_frame_rate", model ? json(frame_rate(s)) : json()},
            {"edit_max_seconds", EDIT_MAX_SECONDS}, {"edit_align_max_seconds", EDIT_ALIGN_MAX_SECONDS}};
}

void register_edit_routes(httplib::Server & svr, ServerState & s) {
    svr.Post("/v2/edit/transcribe", keepalive([&s](const httplib::Request & req, httplib::Response & res) {
        send_json(res, handle_transcribe(s, schema::parse_body(req, "EditTranscribeRequest")));
    }));
    svr.Post("/v2/edit/align", keepalive([&s](const httplib::Request & req, httplib::Response & res) {
        send_json(res, handle_align(s, schema::parse_body(req, "EditAlignRequest")));
    }));
    svr.Post("/v2/edit", keepalive([&s](const httplib::Request & req, httplib::Response & res) {
        send_json(res, handle_edit(s, schema::parse_body(req, "EditRequest")));
    }));
}

}  // namespace zs
