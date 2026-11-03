#include "compat.h"

#include "server/edit.h"

#include "server/edit_align.h"
#include "server/generate.h"
#include "server/http.h"
#include "server/schema.h"
#include "server/speakers.h"
#include "server/util.h"

#include "align-text.h"
#include "zonos3-edit.h"
#include "zonos3-markup.h"
#include "zonos3-textmatch.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <future>
#include <random>

namespace zs {

using json = nlohmann::json;
namespace fs = std::filesystem;
namespace E = zonos3_edit;

// sgl _EDIT_ALIGN_MAX_SECONDS / _EDIT_MAX_SECONDS: the transcribe / align limit bounds every edit.
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

// A scratch directory for the workers' audio files, removed with its contents.
struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() / ("zonos3-edit-" + random_hex(16));
        fs::create_directories(path);
    }
    ~TempDir() { std::error_code ec; fs::remove_all(path, ec); }
    std::string file(const std::string & name) const { return (path / name).string(); }
};

// sgl _write_edit_audio: decode the upload, check its length and store it as WAV for the workers.
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

// --------------------------------------------------------------------------------------------- ASR and FuseAlign

// sgl _edit_worker: an ASR / FuseAlign call (server/edit_align.h); 503 when a worker is unavailable, 400 when the input
// is at fault, with the workers' clients' messages.
template <class F>
auto worker_call(F && f) -> decltype(f()) {
    try {
        return f();
    } catch (const WorkerUnavailable & e) {
        fail(503, e.what());
    } catch (const std::invalid_argument & e) {
        bad(e);
    }
}

// sgl _edit_language: (Whisper-style code, Qwen3-ASR name) of a request's `language` (E::edit_language over the ASR
// worker's languages; 400 for one editing does not serve).
using EditLanguage = std::pair<std::string, std::string>;
EditLanguage edit_language_of(const ServerState & s, const json & body) {
    const std::string language = resolve_language(body);
    const auto supported = worker_call([&] { return asr_worker(s).languages(); });
    try {
        return E::edit_language(language, supported);
    } catch (const std::invalid_argument & e) {
        bad(e);
    }
}

// sgl _edit_transcribe: Qwen3-ASR transcript of a clip in the edit's language (chunk-wise past 30 s).
Transcript edit_transcribe(const ServerState & s, const std::string & audio, double seconds, const EditLanguage & language) {
    return worker_call([&] { return transcribe_audio(s, audio, seconds, language.second); });
}

// sgl _edit_aligned_words (edit_align.edit_aligned_words): the words of `text` in the clip with their edit tokens,
// FuseAlign over align_tokens(text, code); `utterances` are the long-form anchors of a transcript of the same clip.
std::vector<E::AlignedWord> edit_aligned_words(const ServerState & s, const std::string & audio, double seconds,
                                               const std::string & text, const EditLanguage & language,
                                               const std::optional<std::vector<FuseAlignUtterance>> & utterances = std::nullopt) {
    return worker_call([&] {
        const std::vector<std::string> words = aligntext::align_tokens(text, language.first);
        std::vector<FuseAlignWord> aligned;
        if (!words.empty()) aligned = align_words(s, {{audio, seconds, words, language.second, utterances}})[0];
        std::vector<E::WordIn> ws;
        for (const auto & w : aligned) ws.push_back({w.word, w.start, w.end, w.presence});
        std::vector<std::vector<std::string>> tokens;
        for (const auto & w : words) tokens.push_back(zonos3_textmatch::tokens(w));
        return E::aligned_words(ws, tokens);
    });
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

// Qwen3-ASR recognizes the text (chunk-wise past 30 s); FuseAlign places its words, anchored by the transcript's chunks.
json handle_transcribe(ServerState & s, const json & body) {
    require_editing(s);
    TempDir tmp;
    const EditAudio audio = write_edit_audio(decode_b64_field(body, "audio_base64"), tmp, EDIT_ALIGN_MAX_SECONDS);
    const EditLanguage language = edit_language_of(s, body);
    const Transcript transcript = edit_transcribe(s, audio.path, audio.duration, language);
    return {{"text", transcript.text}, {"duration", audio.duration},
            {"words", public_words(edit_aligned_words(s, audio.path, audio.duration, transcript.text, language,
                                                      transcript.utterances))}};
}

json handle_align(ServerState & s, const json & body) {
    require_editing(s);
    const std::string text = strip(body.at("text").get<std::string>());
    if (text.empty()) fail(400, "text is empty.");
    TempDir tmp;
    const EditAudio audio = write_edit_audio(decode_b64_field(body, "audio_base64"), tmp, EDIT_ALIGN_MAX_SECONDS);
    return {{"text", text}, {"duration", audio.duration},
            {"words", public_words(edit_aligned_words(s, audio.path, audio.duration, text, edit_language_of(s, body)))}};
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
    // never reach the workers (word diff, alignment, verify).
    const std::string target_text = E::canonical_markup(strip(body.at("target_text").get<std::string>()));
    const std::string plain_target_text = zonos3_strip_nonverbal_markup(target_text);
    const EditLanguage edit_lang = edit_language_of(s, body);   // every ASR and FuseAlign call of the edit
    const bool has_spans = set(body, "spans");
    const double min_fill_ratio = num(body, "min_fill_ratio", 0.7);
    const double max_span_seconds = num(body, "max_span_seconds", 30.0);
    const bool verify = set(body, "verify") && body["verify"].get<bool>();
    TempDir tmp;

    std::optional<std::vector<std::map<std::string, double>>> region_doses;
    std::string language;
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
    // caller's original_words, else a FuseAlign alignment) and the target words' edit tokens. Without original_text, a
    // short clip with explicit spans is not transcribed: its regions' own lengths are expected.
    const bool windowed = total_frames / fr > num(body, "max_context_seconds", 16.0);
    const std::vector<std::string> target_words = E::split_words(target_text);
    // [target tokens, tokens of each target word]; a pronunciation group is one target token (target_tokens).
    const std::vector<std::vector<std::string>> normalized = E::target_tokens(target_text);
    std::optional<std::string> original_text;
    if (set(body, "original_text")) original_text = zonos3_strip_nonverbal_markup(body["original_text"].get<std::string>());
    const bool has_original_text = original_text && !original_text->empty();
    std::optional<std::vector<E::AlignedWord>> words;
    if (set(body, "original_words")) {
        const json & given = body["original_words"];
        std::vector<E::WordIn> ws;
        std::vector<std::vector<std::string>> word_tokens;
        for (const auto & g : given) {
            E::WordIn w;
            w.word = g.at("word").get<std::string>();
            w.start = g.at("start").get<double>();
            w.end = g.at("end").get<double>();
            if (set(g, "confidence")) w.confidence = g["confidence"].get<double>();
            word_tokens.push_back(set(g, "tokens") ? g["tokens"].get<std::vector<std::string>>() : zonos3_textmatch::tokens(w.word));
            ws.push_back(std::move(w));
        }
        try {
            words = E::edit_words(ws, word_tokens, zonos3_textmatch::tokens(*original_text), original.duration);
        } catch (const std::invalid_argument & e) {
            bad(e);
        }
    } else if (!has_spans || windowed || (min_fill_ratio > 0 && has_original_text)) {
        // the original's words: Qwen3-ASR recognizes the text when it is not given (its chunks anchor the alignment
        // of a long clip), FuseAlign places it
        std::optional<std::vector<FuseAlignUtterance>> utterances;
        if (!has_original_text) {
            Transcript transcript = edit_transcribe(s, original.path, original.duration, edit_lang);
            original_text = std::move(transcript.text);
            utterances = std::move(transcript.utterances);
        }
        words = edit_aligned_words(s, original.path, original.duration, *original_text, edit_lang, utterances);
    }
    // verify with pronunciation groups: the reference's tokens, each group a wildcard word
    std::optional<std::vector<std::string>> verify_reference;
    if (verify && target_text.find(GROUP_OPEN) != std::string::npos) verify_reference = normalized.at(0);

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
            // The take's transcript (Qwen3-ASR) and its anchors for the alignment of the take's words.
            const Transcript transcript = edit_transcribe(s, edited_path, duration, edit_lang);
            if (!verify_reference) {
                const auto ref = zonos3_textmatch::tokens(plain_target_text);
                const int word_errors = zonos3_textmatch::word_errors(ref, zonos3_textmatch::tokens(transcript.text));
                verify_json = {{"asr_text", transcript.text},
                               {"wer", (double) word_errors / std::max<double>(1, (double) ref.size())},
                               {"cer", zonos3_textmatch::char_error_rate(plain_target_text, transcript.text)}};
                take_words = public_words(edit_aligned_words(s, edited_path, duration, plain_target_text, edit_lang,
                                                             transcript.utterances));
            } else {
                // Groups are wildcard words: any one recognized word matches; the take's words come from aligning the
                // target with each group replaced by the word heard.
                const E::Verify scored = E::wildcard_verify(*verify_reference, zonos3_textmatch::tokens(transcript.text));
                json groups = json::array();
                std::vector<std::optional<std::string>> heard;
                for (const auto & g : scored.groups) {
                    groups.push_back({{"ipa", g.ipa}, {"heard", g.heard ? json(*g.heard) : json()}});
                    heard.push_back(g.heard);
                }
                verify_json = {{"asr_text", transcript.text},
                               {"wer", (double) scored.word_errors / std::max(1, scored.ref_words)},
                               {"cer", scored.cer}, {"groups", groups}};
                const std::string heard_text = E::fill_groups(plain_target_text, heard);
                if (!heard_text.empty())
                    take_words = public_words(edit_aligned_words(s, edited_path, duration, heard_text, edit_lang,
                                                                 transcript.utterances));
            }
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

// speech_editing: an edit checkpoint, the ASR interpreter (ZONOS3_ASR_PYTHON / .venv-asr) and FuseAlign (fusealign_hub
// in the --worker-python interpreter).
json edit_capabilities(const ServerState & s) {
    const bool model = zonos3_supports_editing(s.model);
    return {{"speech_editing", model && s.asr && s.asr->installed() && s.aligner && s.fusealign_installed},
            {"edit_max_spans", model ? edit_max_spans(s) : 0}, {"edit_frame_rate", model ? json(frame_rate(s)) : json()},
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
