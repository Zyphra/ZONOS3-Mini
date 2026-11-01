// laci-replay: drive the C++ ports of zonos3.tts.laci / zonos3.tts.span_follow from a JSON trace on stdin and print
// the result JSON on stdout. servers/ggml/tests/parity/test_ggml_laci_replay.py drives the Python modules the same
// way and compares; `laci-replay --trace T --expect FILE` compares against a stored result (ctest) and prints PASS.
//
// mode "laci" (default): one LaciSession driven like model/tests/test_laci.py `run_attempt` across attempts.
//   {text_len, heads_json: <laci_heads JSON object> | "<path>", frames_per_token?: prior override, cfg?: {LaciConfig
//    fields}, n_prefill?: 0, n_codebooks?: 10, max_tokens?: 4000, seed?: 0 | null,
//    duration_target?: {frames, adjust_rate?} | null, rates?: {index, edges, n_buckets, first?} | null,
//    condition_values?: [int] | null, bucket?: int | null, rate_deltas?: [[float] | null] | null,
//    follower?: {ranges, deltas, cfg?, origin?} | null,
//    attempts: [{steps: [[pos, m], ...], eos_at?: step | null, eos_lag?: 0}]   (the last one repeats as needed)
//    -- or the single-attempt shorthand steps / eos_at / eos_lag at the top level}
//   Rows are width 1 holding a tag: prefill -1, attempt k's prefill-sampled row k + 1, its step s row
//   (k + 1) * 100000 + s.
// mode "span_follow": {ranges, deltas, cfg?, origin?, ops: [["observe", frame, pos] | ["on_step", step, pos] |
//   ["weights", frame] | ["delta", frame] | ["truncate", frame] | ["position"]]}
// mode "span_ranges": {text, final, spans, markers, offsets, bpe?: "words" | null, phonemes?: {open_marker,
//   close_marker} | null}
#include "zonos3-laci.h"
#include "zonos3-span-follow.h"

#include "json.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using nlohmann::json;
namespace L = zonos3_laci;
namespace F = zonos3_follow;

static std::u32string utf32(const std::string & s) {
    std::u32string out;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = (unsigned char)s[i];
        const int n = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
        char32_t cp = n == 1 ? c : n == 2 ? (c & 0x1F) : n == 3 ? (c & 0x0F) : (c & 0x07);
        for (int k = 1; k < n; k++) cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3F);
        out.push_back(cp);
        i += n;
    }
    return out;
}

template <typename T>
static std::optional<T> opt(const json & j, const char * key) {
    const auto it = j.find(key);
    if (it == j.end() || it->is_null()) return std::nullopt;
    return it->get<T>();
}

static json jopt(const std::optional<int> & v) { return v ? json(*v) : json(nullptr); }

static std::vector<F::Range> ranges_of(const json & j) {
    std::vector<F::Range> out;
    for (const auto & r : j) out.push_back({r.at(0).get<int>(), r.at(1).get<int>()});
    return out;
}

static F::FollowConfig follow_cfg(const json & j) {
    F::FollowConfig c;
    if (j.is_object()) {
        c.lead = j.value("lead", c.lead);
        c.lag = j.value("lag", c.lag);
        c.smooth = j.value("smooth", c.smooth);
        c.ramp = j.value("ramp", c.ramp);
    }
    return c;
}

static F::SpanFollower make_follower(const json & j) {
    return F::SpanFollower(ranges_of(j.at("ranges")), j.at("deltas").get<std::vector<std::vector<float>>>(),
                           follow_cfg(j.value("cfg", json())), j.value("origin", 0));
}

static json mask_json(const std::optional<L::Windows> & m) {
    if (!m) return nullptr;
    json out = json::array();
    for (const auto & w : *m) out.push_back({w.head.layer, w.head.head, w.lo, w.hi});
    return out;
}

static json tags(const L::LaciSession & s, L::RowRange r) {
    json out = json::array();
    for (int i = r.begin; i < r.end; i++) out.push_back(s.stream()[i][0]);
    return out;
}

static json run_laci(const json & t) {
    const int text_len = t.at("text_len").get<int>();
    L::SessionParams p;
    const json & hj = t.at("heads_json");
    p.heads = hj.is_string() ? L::LaciHeads::load(hj.get<std::string>()) : L::LaciHeads::parse(hj.dump(), "heads_json");
    if (auto fpt = opt<double>(t, "frames_per_token")) p.heads.frames_per_token = *fpt;
    if (t.contains("cfg")) {
        const json & c = t["cfg"];
        p.cfg.bin_tokens = c.value("bin_tokens", p.cfg.bin_tokens);
        p.cfg.cover_frac = c.value("cover_frac", p.cfg.cover_frac);
        p.cfg.skip_frac = c.value("skip_frac", p.cfg.skip_frac);
        p.cfg.stall_factor = c.value("stall_factor", p.cfg.stall_factor);
        p.cfg.min_stall_frames = c.value("min_stall_frames", p.cfg.min_stall_frames);
        p.cfg.tail_frames = c.value("tail_frames", p.cfg.tail_frames);
        p.cfg.rollback_margin_frames = c.value("rollback_margin_frames", p.cfg.rollback_margin_frames);
        p.cfg.max_retries = c.value("max_retries", p.cfg.max_retries);
        p.cfg.pace_window_bins = c.value("pace_window_bins", p.cfg.pace_window_bins);
        p.cfg.warmup_bins = c.value("warmup_bins", p.cfg.warmup_bins);
    }
    const int n_prefill = t.value("n_prefill", 0);
    p.prefill_rows.assign(n_prefill, std::vector<int32_t>{-1});
    p.seed = t.contains("seed") ? opt<int64_t>(t, "seed") : std::optional<int64_t>(0);
    p.max_tokens = t.value("max_tokens", 4000);
    p.n_codebooks = t.value("n_codebooks", 10);
    if (t.contains("duration_target") && !t["duration_target"].is_null()) {
        const json & d = t["duration_target"];
        p.duration = d.is_number() ? L::DurationTarget{d.get<double>(), true}
                                   : L::DurationTarget{d.at("frames").get<double>(), d.value("adjust_rate", true)};
    }
    if (t.contains("rates") && !t["rates"].is_null()) {
        const json & r = t["rates"];
        p.rates = L::RateBuckets{r.at("index").get<int>(), r.at("edges").get<std::vector<double>>(),
                                 r.at("n_buckets").get<int>(), r.value("first", 0)};
    }
    p.condition_values = opt<std::vector<int>>(t, "condition_values");
    p.bucket = opt<int>(t, "bucket");
    if (t.contains("rate_deltas") && !t["rate_deltas"].is_null()) {
        std::vector<std::vector<float>> d;
        for (const auto & e : t["rate_deltas"]) d.push_back(e.is_null() ? std::vector<float>{} : e.get<std::vector<float>>());
        p.rate_deltas = std::move(d);
    }
    std::optional<F::SpanFollower> follower;
    if (t.contains("follower") && !t["follower"].is_null()) {
        follower.emplace(make_follower(t["follower"]));
        p.follower = &*follower;
    }
    json attempts = t.contains("attempts") ? t["attempts"]
                                           : json::array({{{"steps", t.at("steps")}, {"eos_at", t.value("eos_at", json())},
                                                           {"eos_lag", t.value("eos_lag", 0)}}});
    const int max_attempts = t.value("max_attempts", 64);
    const int C = p.n_codebooks;

    L::LaciSession s(std::move(p));
    json out_attempts = json::array();
    for (int k = 0;; k++) {
        if (k >= max_attempts) throw std::runtime_error("more than max_attempts attempts");
        const json & a = attempts[std::min<size_t>(k, attempts.size() - 1)];
        const int tag = k + 1;
        const std::optional<int> eos_at = opt<int>(a, "eos_at");
        const int eos_lag = a.value("eos_lag", 0);
        L::ProbeControl probe;
        probe.text_len = text_len;
        probe.mask = s.probe_mask();
        json rec = {{"mask", mask_json(probe.mask)}};
        json changes = json::array(), drained = json::array(), fweights = json::array();
        s.append({tag}, std::nullopt);
        for (const auto & v : tags(s, s.drain())) drained.push_back(v);
        std::optional<int> countdown;
        int steps_run = 0;
        std::optional<int> stop_step;
        const json & steps = a.at("steps");
        for (int step = 1; step <= (int)steps.size(); step++) {
            const int pos = steps[step - 1].at(0).get<int>();
            const double m = steps[step - 1].at(1).get<double>();
            if (follower) fweights.push_back(follower->weights(s.attempt_start() + step - s.n_prefill()));
            const json before = mask_json(probe.mask);
            s.on_step(probe, step, pos, m);
            steps_run = step;
            const json after = mask_json(probe.mask);
            if (after != before) changes.push_back({step, after});
            std::optional<int> eos;
            if (eos_at && step >= *eos_at) {
                eos = *eos_at - eos_lag;
                if (!countdown) countdown = C + 1;
            }
            s.append({tag * 100000 + step}, eos);
            if (countdown) *countdown -= 1;
            if (probe.stop || (countdown && *countdown == 0)) {
                if (probe.stop) stop_step = step;
                break;
            }
            for (const auto & v : tags(s, s.drain())) drained.push_back(v);
        }
        rec["mask_changes"] = changes;
        rec["steps_run"] = steps_run;
        rec["stop_step"] = jopt(stop_step);
        rec["drained"] = drained;
        if (follower) rec["follower_weights"] = fweights;
        const auto cont = s.finish_attempt();
        if (cont) {
            json cv = cont->condition_values ? json(*cont->condition_values) : json(nullptr);
            rec["continuation"] = {{"n_rows", cont->n_rows}, {"history_rows", cont->history_rows},
                                   {"seed", cont->seed ? json(*cont->seed) : json(nullptr)},
                                   {"max_tokens", cont->max_tokens}, {"condition_values", cv},
                                   {"swap_delta", cont->swap_delta},
                                   {"speaker_emotion_delta", cont->speaker_emotion_delta
                                                                 ? json(*cont->speaker_emotion_delta) : json(nullptr)},
                                   {"probe_mask", mask_json(s.probe_mask())}};
        } else {
            rec["continuation"] = nullptr;
        }
        out_attempts.push_back(rec);
        if (!cont) break;
    }
    json out = {{"attempts", out_attempts}};
    out["final_drain"] = tags(s, s.drain(true));
    out["eos_frame"] = jopt(s.eos_frame());
    out["end"] = {s.end()->first, jopt(s.end()->second)};
    out["record"] = json::parse(s.record().to_json());
    out["stream"] = tags(s, {s.n_prefill(), (int)s.stream().size()});
    return out;
}

static json run_span_follow(const json & t) {
    F::SpanFollower f = make_follower(t);
    json results = json::array();
    for (const auto & op : t.at("ops")) {
        const std::string name = op.at(0).get<std::string>();
        if (name == "observe") {
            f.observe(op.at(1).get<int>(), op.at(2).get<int>());
            results.push_back(f.position());
        } else if (name == "on_step") {
            f.on_step(op.at(1).get<int>(), op.at(2).get<int>());
            results.push_back(f.position());
        } else if (name == "weights") {
            results.push_back(f.weights(op.at(1).get<int>()));
        } else if (name == "delta") {
            std::vector<float> d;
            results.push_back(f.delta(op.at(1).get<int>(), d) ? json(d) : json(nullptr));
        } else if (name == "truncate") {
            f.truncate(op.at(1).get<int>());
            results.push_back(f.position());
        } else if (name == "position") {
            results.push_back(f.position());
        } else {
            throw std::runtime_error("unknown span_follow op " + name);
        }
    }
    return {{"results", results}};
}

// Test tokenizer for the SuperBPE path: tokens are the matches of \s*\S+|\s+ (one per word, leading space attached).
static std::vector<F::Range> word_offsets(const std::u32string & s) {
    std::vector<F::Range> out;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = i;
        while (j < s.size() && F::py_isspace(s[j])) j++;
        if (j == s.size()) {
            out.push_back({(int)i, (int)j});
            break;
        }
        while (j < s.size() && !F::py_isspace(s[j])) j++;
        out.push_back({(int)i, (int)j});
        i = j;
    }
    return out;
}

static json run_span_ranges(const json & t) {
    const std::u32string text = utf32(t.at("text").get<std::string>()), fin = utf32(t.at("final").get<std::string>());
    F::BpeOffsets bpe;
    if (t.value("bpe", json()).is_string()) bpe = word_offsets;
    std::optional<F::PhonemeMarkers> ph;
    if (t.contains("phonemes") && !t["phonemes"].is_null())
        ph = F::PhonemeMarkers{utf32(t["phonemes"].at("open_marker").get<std::string>()),
                               utf32(t["phonemes"].at("close_marker").get<std::string>())};
    const auto offsets = t.at("offsets").get<std::vector<int>>();
    json ops = json::array();
    for (const auto & o : F::diff_opcodes(text, fin)) ops.push_back({o.tag, o.i1, o.i2, o.j1, o.j2});
    const auto tok = F::token_char_ranges(fin, bpe, ph);
    json tr = json::array(), sr = json::array();
    for (const auto & [a, b] : tok) tr.push_back({a, b});
    for (const auto & [a, b] : F::span_token_ranges(text, fin, ranges_of(t.at("spans")), tok, ranges_of(t.at("markers"))))
        sr.push_back({a, b});
    return {{"opcodes", ops}, {"map_starts", F::map_offsets(text, fin, offsets)},
            {"map_ends", F::map_offsets(text, fin, offsets, true)}, {"token_ranges", tr}, {"span_ranges", sr}};
}

int main(int argc, char ** argv) {
    std::string expect, trace;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--expect" && i + 1 < argc) expect = argv[++i];
        else if (a == "--trace" && i + 1 < argc) trace = argv[++i];
        else {
            fprintf(stderr, "usage: %s [--trace trace.json (default: stdin)] [--expect result.json]\n", argv[0]);
            return 2;
        }
    }
    try {
        std::stringstream ss;
        if (trace.empty()) {
            ss << std::cin.rdbuf();
        } else {
            std::ifstream tf(trace, std::ios::binary);
            if (!tf) throw std::runtime_error("cannot read " + trace);
            ss << tf.rdbuf();
        }
        const json t = json::parse(ss.str());
        const std::string mode = t.value("mode", "laci");
        json out;
        if (mode == "laci") out = run_laci(t);
        else if (mode == "span_follow") out = run_span_follow(t);
        else if (mode == "span_ranges") out = run_span_ranges(t);
        else throw std::runtime_error("unknown mode " + mode);
        if (expect.empty()) {
            printf("%s\n", out.dump().c_str());
            return 0;
        }
        std::ifstream f(expect);
        if (!f) throw std::runtime_error("cannot read " + expect);
        const json want = json::parse(f);
        if (want != out) {
            fprintf(stderr, "FAIL: result differs from %s\n%s\n", expect.c_str(), json::diff(want, out).dump(1).c_str());
            return 1;
        }
        printf("PASS\n");
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "laci-replay: %s\n", e.what());
        return 1;
    }
}
