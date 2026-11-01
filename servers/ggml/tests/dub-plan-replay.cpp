// dub-plan-replay: run one src/dub-plan.h function on a JSON request {"fn", "args"} read from stdin and print the
// result as JSON (tuples and numpy arrays as nested lists). Errors print {"error": <Python exception class>,
// "message": ...}. Parity against model/src/zonos3/dub/plan.py: servers/ggml/tests/parity/test_ggml_dub_plan.py.
// Numbers may be given as "inf" / "-inf" / "nan" strings (JSON has no non-finite literals).
#include "dub-plan.h"

#include "json.hpp"

#include <cmath>
#include <iostream>
#include <iterator>
#include <stdexcept>

using json = nlohmann::json;
using namespace dubplan;

namespace {

double num(const json &j) {
    if (j.is_string()) {
        const auto s = j.get<std::string>();
        if (s == "inf") return INFINITY;
        if (s == "-inf") return -INFINITY;
        if (s == "nan") return NAN;
        throw std::runtime_error("bad number: " + s);
    }
    return j.get<double>();
}

std::vector<double> nums(const json &j) {
    std::vector<double> out;
    for (const auto &v : j) out.push_back(num(v));
    return out;
}

std::vector<TextUnit> units_of(const json &j) {
    std::vector<TextUnit> out;
    for (const auto &u : j) out.push_back({u.value("text", std::string()), u.at("start").get<int>(), u.at("end").get<int>()});
    return out;
}

std::vector<TimedWord> words_of(const json &j) {
    std::vector<TimedWord> out;
    for (const auto &w : j) out.push_back({num(w.at("start")), num(w.at("end"))});
    return out;
}

std::vector<Range> ranges_of(const json &j) {
    std::vector<Range> out;
    for (const auto &r : j) out.emplace_back(r.at(0).get<int>(), r.at(1).get<int>());
    return out;
}

json ranges_json(const std::vector<Range> &rs) {
    json out = json::array();
    for (const auto &[a, b] : rs) out.push_back({a, b});
    return out;
}

RateScale scale_of(const json &j) {
    RateScale s;
    s.edges = nums(j.at("edges"));
    s.first = j.value("first", 0);
    if (j.contains("doses"))
        for (const auto &d : j.at("doses")) s.doses[d.at(0).get<int>()] = num(d.at(1));
    return s;
}

json scale_json(const RateScale &s) {
    json doses = json::array();
    for (const auto &[b, d] : s.doses) doses.push_back({b, d});
    return {{"edges", s.edges}, {"first", s.first}, {"last", s.last()}, {"doses", doses}};
}

LipMap map_of(const json &j) { return {nums(j.at("out_t")), nums(j.at("src_t"))}; }

std::optional<TakeResult> take_of(const json &j) {
    if (j.is_null()) return std::nullopt;
    return TakeResult{num(j.at("speech_s")), j.value("capped", false), j.value("end", std::string())};
}

json candidates_json(const std::vector<Candidate> &cs) {
    json out = json::array();
    for (const auto &c : cs) out.push_back({{"bucket", c.bucket}, {"offset", c.offset}, {"seed", c.seed}});
    return out;
}

json warp_json(const LipWarp &w) {
    return {{"lip_scale", w.lip_scale}, {"lip_stretch", w.lip_stretch}, {"lip_squeeze", w.lip_squeeze},
            {"lip_warp", w.lip_warp}};
}

json opt_pair(const std::optional<std::pair<double, double>> &p) {
    if (!p) return nullptr;
    return {p->first, p->second};
}

json scalar_or_list(const json &t, const std::function<double(double)> &f) {
    if (!t.is_array()) return f(num(t));
    json out = json::array();
    for (const auto &v : t) out.push_back(f(num(v)));
    return out;
}

std::function<double(double)> src_of_of(const json &j) {
    if (j.contains("affine")) {  // t * scale + offset
        const double scale = num(j["affine"].at(0)), offset = num(j["affine"].at(1));
        return [=](double t) { return t * scale + offset; };
    }
    if (j.contains("interp")) {  // np.interp(t / sr, xp, fp) * sr
        const auto xp = nums(j["interp"].at("xp")), fp = nums(j["interp"].at("fp"));
        const double sr = num(j["interp"].at("sr"));
        return [=](double t) { return interp(t / sr, xp, fp) * sr; };
    }
    if (j.contains("lipmap")) {  // lip_map.src_time(t / sr) * sr (dub_avsync)
        const LipMap m = map_of(j["lipmap"]);
        const double sr = num(j["lipmap"].at("sr"));
        return [=](double t) { return m.src_time(t / sr) * sr; };
    }
    throw std::runtime_error("bad src_of");
}

json constants() {
    json tl = json::object();
    for (const auto &t : TARGET_LANGUAGES) tl[t.code] = {t.label, t.tts};
    json rl = json::object();
    for (const auto &[c, n] : REF_LANGUAGES) rl[c] = n;
    return {
        {"REF_LANGUAGES", rl},
        {"TARGET_LANGUAGES", tl},
        {"UNSPACED", std::vector<std::string>(UNSPACED.begin(), UNSPACED.end())},
        {"MAX_REF_SECONDS", MAX_REF_SECONDS},
        {"MIN_REF_SECONDS", MIN_REF_SECONDS},
        {"MAX_VIDEO_BYTES", MAX_VIDEO_BYTES},
        {"DEFAULT_MIN_PAUSE_S", DEFAULT_MIN_PAUSE_S},
        {"MIN_PAUSE_RANGE", {MIN_PAUSE_RANGE[0], MIN_PAUSE_RANGE[1]}},
        {"MAX_SEGMENTS", MAX_SEGMENTS},
        {"CANDIDATE_BUCKET_OFFSETS", CANDIDATE_BUCKET_OFFSETS},
        {"CANDIDATE_SEEDS", CANDIDATE_SEEDS},
        {"PRE_PAD_S", PRE_PAD_S},
        {"POST_PAD_S", POST_PAD_S},
        {"MIN_GAP_S", MIN_GAP_S},
        {"FADE_S", FADE_S},
        {"VOICE_PROMPT_MAX_S", VOICE_PROMPT_MAX_S},
        {"FRAME_RATE", FRAME_RATE},
        {"SAMPLE_RATE", SAMPLE_RATE},
        {"BREAK_PUNCT", BREAK_PUNCT},
        {"WINDOW_MAX_WP", WINDOW_MAX_WP},
        {"WINDOW_STRIDE_FRACTION", WINDOW_STRIDE_FRACTION},
        {"BAND_MIN_WP", BAND_MIN_WP},
        {"BAND_FRACTION", BAND_FRACTION},
        {"LIP_SCALE_WINDOW_S", LIP_SCALE_WINDOW_S},
        {"LIP_SCALE_MAX", LIP_SCALE_MAX},
        {"LIP_GUARD_MAX_RATE", LIP_GUARD_MAX_RATE},
        {"LIP_GUARD_MAX_SHIFT_S", LIP_GUARD_MAX_SHIFT_S},
        {"LIP_MIN_SYNC_GAIN", LIP_MIN_SYNC_GAIN},
        {"WSOLA_WINDOW", WSOLA_WINDOW},
        {"WSOLA_TOLERANCE", WSOLA_TOLERANCE},
        {"WSOLA_DRIFT_PENALTY", WSOLA_DRIFT_PENALTY},
    };
}

json run(const std::string &fn, const json &a) {
    if (fn == "constants") return constants();
    if (fn == "text_units") {
        json out = json::array();
        for (const auto &u : text_units(a.at("text"), a.at("lang")))
            out.push_back({{"text", u.text}, {"start", u.start}, {"end", u.end}});
        return out;
    }
    if (fn == "unit_bytes") return unit_bytes(a.at("text"), units_of(a.at("units")));
    if (fn == "break_after") return break_after(a.at("text"), units_of(a.at("units")));
    if (fn == "span_text") return span_text(a.at("text"), units_of(a.at("units")), a.at("a"), a.at("b"));
    if (fn == "join_words") return join_words(a.at("words").get<std::vector<std::string>>(), a.at("lang"));
    if (fn == "ref_segments") return ranges_json(ref_segments(words_of(a.at("words")), num(a.at("min_pause"))));
    if (fn == "project_segments") {
        std::vector<std::pair<double, double>> spans;
        for (const auto &s : a.at("ref_spans")) spans.emplace_back(num(s.at(0)), num(s.at(1)));
        std::vector<std::pair<int, int>> links;
        for (const auto &l : a.at("links")) links.emplace_back(l.at(0).get<int>(), l.at(1).get<int>());
        return ranges_json(project_segments(ranges_of(a.at("ref_ranges")), spans, links,
                                            a.at("unit_bytes").get<std::vector<int>>(),
                                            a.at("break_after").get<std::vector<bool>>()));
    }
    if (fn == "RateScale") return scale_json(scale_of(a));
    if (fn == "RateScale.extended") {
        std::map<int, double> doses;
        for (const auto &d : a.at("doses")) doses[d.at(0).get<int>()] = num(d.at(1));
        return scale_json(RateScale::extended(nums(a.at("row_edges")), nums(a.at("below")), nums(a.at("above")), doses));
    }
    if (fn == "RateScale.bucket") return scale_of(a.at("scale")).bucket(num(a.at("rate")));
    if (fn == "RateScale.rate") return scale_of(a.at("scale")).rate(a.at("bucket").get<int>());
    if (fn == "candidate_plan")
        return candidates_json(candidate_plan(a.at("bucket"), scale_of(a.at("scale")), a.at("seed").get<int64_t>(),
                                              a.at("segment_index")));
    if (fn == "pinned_candidate_plan")
        return candidates_json(pinned_candidate_plan(a.at("bucket"), a.at("seed").get<int64_t>(), a.at("segment_index")));
    if (fn == "max_tokens_for") return max_tokens_for(num(a.at("duration_s")));
    if (fn == "voice_prompt_span") {
        std::vector<RefSegment> segs;
        for (const auto &s : a.at("segments")) segs.push_back({num(s.at("ref_start")), num(s.at("ref_end")), "", {}});
        const auto [x, y] = voice_prompt_span(segs, num(a.at("duration")));
        return {x, y};
    }
    if (fn == "speech_span") {
        std::vector<float> audio;
        for (const auto &v : a.at("audio")) audio.push_back(float(num(v)));
        return opt_pair(speech_span(words_of(a.at("items")), audio.data(), audio.size(), a.at("sr")));
    }
    if (fn == "candidate_score") {
        const auto s = candidate_score(take_of(a.at("c")), num(a.at("ref_dur")), num(a.at("room")));
        return s ? json(*s) : json(nullptr);
    }
    if (fn == "choose_candidate") {
        std::vector<std::optional<TakeResult>> cands;
        for (const auto &c : a.at("cands")) cands.push_back(take_of(c));
        const auto i = choose_candidate(cands, num(a.at("ref_dur")), num(a.at("room")));
        return i ? json(*i) : json(nullptr);
    }
    if (fn == "place_segments") {
        std::vector<PlaceItem> items;
        for (const auto &it : a.at("items"))
            items.push_back({num(it.at("ref_start")), num(it.at("ref_end")), num(it.at("take_len")), num(it.at("s0")),
                             num(it.at("s1"))});
        const auto [placed, timeline] = place_segments(items, num(a.at("ref_duration")));
        json out = json::array();
        for (const auto &p : placed)
            out.push_back({{"crop_start", p.crop_start}, {"crop_end", p.crop_end}, {"place_start", p.place_start},
                           {"speech_start", p.speech_start}, {"speech_end", p.speech_end}, {"shift", p.shift},
                           {"start_error", p.start_error}, {"end_error", p.end_error},
                           {"duration_error", p.duration_error}});
        return {out, timeline};
    }
    if (fn == "word_windows") return ranges_json(word_windows(a.at("counts").get<std::vector<int>>()));
    if (fn == "window_of_words") return window_of_words(ranges_of(a.at("windows")), a.at("n"));
    if (fn == "band_mask") {
        const int w1 = a.at("w1"), w2 = a.at("w2");
        const auto m = band_mask(w1, w2);
        json out = json::array();
        for (int p = 0; p < w1; ++p) {
            json row = json::array();
            for (int q = 0; q < w2; ++q) row.push_back(m[size_t(p) * w2 + q] != 0);
            out.push_back(row);
        }
        return out;
    }
    if (fn == "interp") {
        const auto xp = nums(a.at("xp")), fp = nums(a.at("fp"));
        return scalar_or_list(a.at("x"), [&](double x) { return interp(x, xp, fp); });
    }
    if (fn == "LipMap.from_frames") {
        const auto m = LipMap::from_frames(num(a.at("start")), num(a.at("rate")), nums(a.at("src_frames")));
        return {{"out_t", m.out_t}, {"src_t", m.src_t}};
    }
    if (fn == "LipMap.src_time") {
        const auto m = map_of(a.at("map"));
        return scalar_or_list(a.at("t"), [&](double t) { return m.src_time(t); });
    }
    if (fn == "LipMap.out_time") {
        const auto m = map_of(a.at("map"));
        return scalar_or_list(a.at("t"), [&](double t) { return m.out_time(t); });
    }
    if (fn == "LipMap.shifts") return map_of(a.at("map")).shifts();
    if (fn == "guard_lip_frames")
        return guard_lip_frames(nums(a.at("src_frames")), num(a.at("rate")), num(a.at("max_rate")),
                                num(a.at("max_shift_s")));
    if (fn == "lip_warp") return warp_json(lip_warp(map_of(a.at("map")), num(a.at("src_start")), num(a.at("src_end"))));
    if (fn == "wsola")
        return wsola(nums(a.at("x")), src_of_of(a.at("src_of")), a.at("n_out").get<int64_t>(),
                     a.value("window", WSOLA_WINDOW), a.value("tolerance", WSOLA_TOLERANCE));
    throw std::runtime_error("unknown fn: " + fn);
}

}  // namespace

int main(int argc, char **argv) {
    if (argc > 1 && std::string(argv[1]) == "--smoke") {  // ctest: a few fixed answers from model/tests/test_dub_plan.py
        const auto units = text_units("我用iPhone15手机", "zh");
        const bool ok = units.size() == 5 && units[2].text == "iPhone15" &&
                        project_segments({{0, 1}, {1, 2}}, {{0, 1}, {1, 4}}, {}, std::vector<int>(8, 1),
                                         std::vector<bool>(8, true)) == std::vector<Range>{{0, 2}, {2, 8}} &&
                        word_windows({0, 0}) == std::vector<Range>{{0, 2}};
        std::cout << (ok ? "PASS" : "FAIL") << "\n";
        return ok ? 0 : 1;
    }
    try {
        const json req = json::parse(std::string(std::istreambuf_iterator<char>(std::cin), {}));
        std::cout << run(req.at("fn"), req.value("args", json::object())).dump() << "\n";
    } catch (const std::invalid_argument &e) {
        std::cout << json{{"error", "ValueError"}, {"message", e.what()}}.dump() << "\n";
    } catch (const std::out_of_range &e) {
        std::cout << json{{"error", "IndexError"}, {"message", e.what()}}.dump() << "\n";
    }
    return 0;
}
