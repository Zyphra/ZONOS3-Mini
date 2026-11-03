// Pure planning logic of the LACI dubbing tool: a port of model/src/zonos3/dub/plan.py (same constants, same
// decisions, same error messages). Text offsets are Unicode code-point offsets (Python str indices); text is UTF-8.
// Functions that raise ValueError in Python throw std::invalid_argument with the same message.
#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace dubplan {

struct TargetLanguage {
    std::string code;
    std::string label;
    std::string tts;  // zonos3 TTS language for text normalization
};

// code (Whisper's, as FuseAlign takes it) -> Qwen3-ASR language name (dict order)
inline const std::vector<std::pair<std::string, std::string>> REF_LANGUAGES = {
    {"en", "English"}, {"fr", "French"}, {"de", "German"}, {"es", "Spanish"}, {"it", "Italian"},
    {"pt", "Portuguese"}, {"ja", "Japanese"}, {"zh", "Chinese"}, {"ko", "Korean"}, {"ru", "Russian"},
    {"yue", "Cantonese"},
};
inline const std::vector<TargetLanguage> TARGET_LANGUAGES = {
    {"en", "English", "en_us"}, {"fr", "French", "fr_fr"}, {"de", "German", "de"},
    {"es", "Spanish", "es"},    {"it", "Italian", "it"},   {"pt", "Portuguese", "pt_br"},
    {"ja", "Japanese", "ja"},   {"zh", "Chinese", "cmn"},  {"ko", "Korean", "ko"},
};
inline const std::array<const char *, 3> UNSPACED = {"ja", "zh", "yue"};
constexpr double MAX_REF_SECONDS = 180.0;  // the longest reference a dub takes
constexpr double MIN_REF_SECONDS = 0.5;    // qwen_asr MIN_ASR_INPUT_SECONDS
constexpr int64_t MAX_VIDEO_BYTES = int64_t(1) << 30;
constexpr double DEFAULT_MIN_PAUSE_S = 0.30;
constexpr double MIN_PAUSE_RANGE[2] = {0.10, 2.00};
constexpr int MAX_SEGMENTS = 120;
constexpr std::array<int, 5> CANDIDATE_BUCKET_OFFSETS = {0, -1, 1, -2, 2};
constexpr int CANDIDATE_SEEDS = 2;
constexpr double PRE_PAD_S = 0.04, POST_PAD_S = 0.06, MIN_GAP_S = 0.05, FADE_S = 0.01;
constexpr double VOICE_PROMPT_MAX_S = 20.0;
constexpr double FRAME_RATE = 15.0;
constexpr int SAMPLE_RATE = 44100;
inline const char *const BREAK_PUNCT = ",.;:!?…、。，；：！？";  // UTF-8
constexpr int WINDOW_MAX_WP = 510;  // mBERT positions minus [CLS]/[SEP]
constexpr double WINDOW_STRIDE_FRACTION = 0.75;
constexpr int BAND_MIN_WP = 40;
constexpr double BAND_FRACTION = 0.2;
// Local time scale of the lip-sync warp is measured over this span (about a phone); capped at LIP_SCALE_MAX.
constexpr double LIP_SCALE_WINDOW_S = 0.05;
constexpr double LIP_SCALE_MAX = 10.0;
// Bounds of the lip-sync warp the server applies (guard_lip_frames): local playback rate and shift.
constexpr double LIP_GUARD_MAX_RATE = 1.5;
constexpr double LIP_GUARD_MAX_SHIFT_S = 0.4;
// The warp is kept only when it lowers the dub's mean SyncNet distance to the lips by more than this.
constexpr double LIP_MIN_SYNC_GAIN = 0.0;
// WSOLA at 44.1 kHz: 23 ms Hann frames at 50 % overlap, each taken within ±11.6 ms of its nominal source position.
constexpr int WSOLA_WINDOW = 1024;
constexpr int WSOLA_TOLERANCE = 512;
constexpr double WSOLA_DRIFT_PENALTY = 0.05;

bool unspaced(const std::string &lang);  // lang in UNSPACED

struct RefSegment {
    double ref_start = 0.0;
    double ref_end = 0.0;
    std::string text;
    std::optional<int> bucket;  // speaking-rate bucket set by the user; nullopt = from the text's rate
};

struct TextUnit {
    std::string text;  // UTF-8
    int start = 0;     // code-point offsets, [start, end)
    int end = 0;
};

std::vector<TextUnit> text_units(const std::string &text, const std::string &lang);
std::vector<int> unit_bytes(const std::string &text, const std::vector<TextUnit> &units);
std::vector<bool> break_after(const std::string &text, const std::vector<TextUnit> &units);
std::string span_text(const std::string &text, const std::vector<TextUnit> &units, int a, int b);
std::string join_words(const std::vector<std::string> &words, const std::string &lang);

struct TimedWord {
    double start = 0.0;
    double end = 0.0;
    bool placed = true;  // FuseAlign placed the word (false: its fallback times; speech_span ignores it)
};
using Range = std::pair<int, int>;  // half-open

std::vector<Range> ref_segments(const std::vector<TimedWord> &words, double min_pause);
std::vector<Range> project_segments(const std::vector<Range> &ref_ranges,
                                    const std::vector<std::pair<double, double>> &ref_spans,
                                    const std::vector<std::pair<int, int>> &links, const std::vector<int> &unit_bytes,
                                    const std::vector<bool> &break_after);

// Bucket i, first <= i <= last(), spans [edges[i - first - 1], edges[i - first]) in UTF-8 bytes / s; doses holds the
// tempo dose of every steered bucket.
struct RateScale {
    std::vector<double> edges;
    int first = 0;
    std::map<int, double> doses;

    static RateScale extended(const std::vector<double> &row_edges, const std::vector<double> &below,
                              const std::vector<double> &above, const std::map<int, double> &doses);
    int last() const { return first + int(edges.size()); }
    int bucket(double rate) const;
    double rate(int bucket) const;
    bool operator==(const RateScale &o) const { return edges == o.edges && first == o.first && doses == o.doses; }
};

struct Candidate {
    int bucket = 0;
    int offset = 0;
    int64_t seed = 0;
};
std::vector<Candidate> candidate_plan(int bucket, const RateScale &scale, int64_t seed, int segment_index);
std::vector<Candidate> pinned_candidate_plan(int bucket, int64_t seed, int segment_index);
int max_tokens_for(double duration_s);
std::pair<double, double> voice_prompt_span(const std::vector<RefSegment> &segments, double duration);
// The span of the placed words with end > start, else the RMS-based speech extent of `audio`.
std::optional<std::pair<double, double>> speech_span(const std::vector<TimedWord> &items, const float *audio,
                                                     size_t n_audio, int sr);

struct TakeResult {
    double speech_s = 0.0;
    bool capped = false;
    std::string end;
};
std::optional<double> candidate_score(const std::optional<TakeResult> &c, double ref_dur, double room);
std::optional<int> choose_candidate(const std::vector<std::optional<TakeResult>> &cands, double ref_dur, double room);

struct PlaceItem {
    double ref_start = 0.0, ref_end = 0.0, take_len = 0.0, s0 = 0.0, s1 = 0.0;
};
struct Placed {
    double crop_start, crop_end, place_start, speech_start, speech_end, shift, start_error, end_error, duration_error;
};
std::pair<std::vector<Placed>, double> place_segments(const std::vector<PlaceItem> &items, double ref_duration);

std::vector<Range> word_windows(const std::vector<int> &counts);
std::vector<int> window_of_words(const std::vector<Range> &windows, int n);
std::vector<uint8_t> band_mask(int w1, int w2);  // row-major [w1, w2], 1 = inside the band

// numpy.interp (xp increasing; clamps to fp's ends outside).
double interp(double x, const std::vector<double> &xp, const std::vector<double> &fp);

// AV-sync's time warp of a dub: at output time out_t[i] the warped dub plays the dub's time src_t[i] (seconds, both
// increasing). Outside [out_t[0], out_t[-1]] the offset of the nearer end holds.
struct LipMap {
    std::vector<double> out_t;
    std::vector<double> src_t;

    static LipMap from_frames(double start, double rate, const std::vector<double> &src_frames);
    double src_time(double t) const;  // dub time played at output time t
    double out_time(double t) const;  // output time at which the dub's time t plays
    std::vector<double> shifts() const;
};

std::vector<double> guard_lip_frames(const std::vector<double> &src_frames, double rate, double max_rate,
                                     double max_shift_s);

struct LipWarp {
    double lip_scale, lip_stretch, lip_squeeze, lip_warp;
};
LipWarp lip_warp(const LipMap &m, double src_start, double src_end);

// Time-warp x without changing its pitch: output sample t plays about source sample src_of(t).
std::vector<double> wsola(const std::vector<double> &x, const std::function<double(double)> &src_of, int64_t n_out,
                          int window = WSOLA_WINDOW, int tolerance = WSOLA_TOLERANCE);

}  // namespace dubplan
