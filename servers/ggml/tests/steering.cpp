// Steering: steer.gguf loader (incl. set calibrations), control resolution math, per-frame hidden delta and
// decode-graph injection.
#include "zonos3.h"
#include "zonos3-steer.h"
#include "model-paths.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <optional>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(cond, label) do { if (!(cond)) { fprintf(stderr, "FAIL: %s (%s:%d)\n", label, __FILE__, __LINE__); exit(1); } } while (0)

static bool contains(const std::string & s, const char * sub) { return s.find(sub) != std::string::npos; }
static bool near(double a, double b, double tol) { return std::fabs(a - b) <= tol; }

static zonos3_steer_direction make_dir(const char * set, const char * name, bool hidden, int source, std::vector<float> v) {
    zonos3_steer_direction d;
    d.set = set; d.name = name; d.id = std::string(set) + "." + name;
    d.hidden = hidden; d.source = source; d.vec = std::move(v);
    return d;
}

static void test_loader(const zonos3_model & model, const std::string & steer_path, const std::string & backbone_path) {
    zonos3_steer_bundle b;
    std::string err;
    CHECK(zonos3_steer_load(b, steer_path.c_str(), model, err), ("loader: dummy-steer.gguf loads: " + err).c_str());
    CHECK((b.sources == std::vector<int>{3, 5}), "loader: sources {3,5}");
    const char * want[] = {"who.gender", "who.age", "style.whisper", "style.tempo", "mood.valence"};
    CHECK(b.dirs.size() == 5, "loader: 5 directions");
    for (int i = 0; i < 5; i++) CHECK(b.dirs[i].id == want[i], "loader: direction order");
    CHECK(b.dirs[0].vec.size() == 192 && b.dirs[2].vec.size() == 32, "loader: vector dims");
    CHECK(b.timing.count("style.tempo") && b.timing["style.tempo"].size() == 3, "loader: timing table");
    CHECK(!zonos3_steer_controls_json(b, 192).empty(), "loader: controls json");

    // Set calibrations (scripts/make-dummy-steer.py): who has none; style and mood disable a control of VOICE.
    const std::string voice = "default_19139adcad9351f1";
    CHECK(!b.sets[0].calibration && b.sets[1].calibration && b.sets[2].calibration, "calibration: per-set presence");
    const zonos3_steer_calibration & mood = *b.sets[2].calibration;
    CHECK(mood.strength("", "valence") == 1.5 && mood.strength(voice, "valence") == 1.5 &&
          mood.strength("x", "other") == 1.25, "calibration: default / global_default lookup");
    CHECK(b.sets[1].calibration->strength(voice, "whisper") == 0.0, "calibration: by_speaker lookup");
    CHECK((zonos3_steer_disabled_controls(b, voice) == std::vector<std::string>{"style.whisper", "mood.valence-"}),
          "calibration: disabled_controls of the calibrated voice");
    CHECK(zonos3_steer_disabled_controls(b, "spk_other").empty() && zonos3_steer_disabled_controls(b, "").empty(),
          "calibration: no disabled_controls for other voices");
    const std::string doc = zonos3_steer_controls_json(b, 192);
    CHECK(contains(doc, "\"who\":{\"space\":\"raw\",\"modes\":[\"embedding\"]}"), "controls json: no calibration for who");
    CHECK(contains(doc, "\"source_index\":3,\"calibration\":{\"default\":{\"tempo\":0.5},\"by_speaker\":"
                        "{\"default_19139adcad9351f1\":{\"whisper\":0.0}},\"global_default\":1.0}}"),
          "controls json: style calibration document");
    CHECK(contains(doc, "\"source_index\":5,\"calibration\":{\"default\":{\"valence\":1.5},\"by_speaker\":"
                        "{\"default_19139adcad9351f1\":{\"valence-\":0.0}},\"global_default\":1.25}}"),
          "controls json: mood calibration document");

    // Obsolete schema: a steer.gguf without zonos3_steer.set.calibrations is refused.
    {
        ggml_context * data = nullptr;
        gguf_context * g = gguf_init_from_file(steer_path.c_str(), {false, &data});
        CHECK(g, "loader: reread dummy-steer.gguf");
        gguf_context * out = gguf_init_empty();
        gguf_set_kv(out, g);
        CHECK(gguf_remove_key(out, "zonos3_steer.set.calibrations") >= 0, "loader: strip calibrations key");
        for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) gguf_add_tensor(out, ggml_get_tensor(data, gguf_get_tensor_name(g, i)));
        const std::string old = "steering-no-calibrations.tmp.gguf";   // the test's working directory
        CHECK(gguf_write_to_file(out, old.c_str(), false), "loader: write stripped steer file");
        gguf_free(out);
        gguf_free(g);
        ggml_free(data);
        zonos3_steer_bundle stale;
        const bool loaded = zonos3_steer_load(stale, old.c_str(), model, err);
        std::remove(old.c_str());
        CHECK(!loaded && contains(err, "missing or mistyped zonos3_steer.* metadata"), "loader: missing set.calibrations refused");
    }
    // Steer then hold: zonos3_steer.steer_release (absent from the dummy) loads into b.steer_release; its exempt ids must
    // be hidden directions; /v2/controls reports it.
    CHECK(!b.steer_release && contains(doc, "\"steer_release\":null"), "release: none without the key");
    CHECK(zonos3_steer_release_json(nullptr) == "null", "release: null without a bundle");
    auto with_release = [&](const char * value, zonos3_steer_bundle & out) {
        ggml_context * data = nullptr;
        gguf_context * g = gguf_init_from_file(steer_path.c_str(), {false, &data});
        CHECK(g, "release: reread dummy-steer.gguf");
        gguf_context * o = gguf_init_empty();
        gguf_set_kv(o, g);
        gguf_set_val_str(o, "zonos3_steer.steer_release", value);
        for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) gguf_add_tensor(o, ggml_get_tensor(data, gguf_get_tensor_name(g, i)));
        const std::string path = "steering-release.tmp.gguf";
        CHECK(gguf_write_to_file(o, path.c_str(), false), "release: write steer file");
        gguf_free(o);
        gguf_free(g);
        ggml_free(data);
        const bool loaded = zonos3_steer_load(out, path.c_str(), model, err);
        std::remove(path.c_str());
        return loaded;
    };
    {
        zonos3_steer_bundle r;
        CHECK(with_release("{\"onset_seconds\": 4.0, \"ramp_seconds\": 1.0, \"hold\": 0.25, \"exempt\": [\"style.tempo\"]}", r),
              ("release: loads " + err).c_str());
        CHECK(r.steer_release && r.steer_release->onset_seconds == 4.0 && r.steer_release->ramp_seconds == 1.0 &&
              r.steer_release->hold == 0.25 && (r.steer_release->exempt == std::vector<std::string>{"style.tempo"}),
              "release: parsed");
        CHECK(zonos3_steer_release_json(&r) == "{\"onset_seconds\":4.0,\"ramp_seconds\":1.0,\"hold\":0.25,\"exempt\":[\"style.tempo\"]}",
              "release: document");
        CHECK(contains(zonos3_steer_controls_json(r, 192), "\"steer_release\":{\"onset_seconds\":4.0,"), "release: /v2/controls");
        CHECK(with_release("", r) && !r.steer_release, "release: \"\" = none");
        CHECK(with_release("{\"onset_seconds\": 4, \"ramp_seconds\": 1, \"hold\": 0.5}", r) && r.steer_release &&
              r.steer_release->exempt.empty(), "release: exempt optional");
        CHECK(!with_release("{\"onset_seconds\": 4, \"ramp_seconds\": 1, \"hold\": 0.25, \"exempt\": [\"who.gender\"]}", r) &&
              contains(err, "exempts 'who.gender', which is not a hidden direction"), "release: raw exempt refused");
        CHECK(!with_release("{\"onset_seconds\": 4, \"hold\": 0.25}", r) && contains(err, "bad steer_release JSON"),
              "release: missing key refused");
        CHECK(!with_release("{\"onset_seconds\": 4, \"ramp_seconds\": 1, \"hold\": 0.25, \"extra\": 1}", r), "release: extra key refused");
        CHECK(!with_release("{\"onset_seconds\": 4, \"ramp_seconds\": 1, \"hold\": 1.5}", r) && contains(err, "steer_hold must be in [0, 1]"),
              "release: hold > 1 refused");
        CHECK(!with_release("{\"onset_seconds\": -1, \"ramp_seconds\": 1, \"hold\": 0.5}", r), "release: negative onset refused");
    }
    zonos3_steer_bundle bad;
    CHECK(!zonos3_steer_load(bad, backbone_path.c_str(), model, err), "loader: backbone is not a steer file");
    CHECK(contains(err, "not a valid zonos3 steering GGUF"), "loader: invalid-file message");
}

static zonos3_steer_bundle make_bundle() {
    zonos3_steer_bundle b;
    b.name = "mem"; b.speaker_dim = 3; b.hidden_size = 2;
    b.sets = {{"who", false, -1}, {"accent", false, -1}, {"deep", true, 15}, {"style", true, 19}};
    b.dirs.push_back(make_dir("who", "gender", false, -1, {0, 2, 0}));
    b.dirs.push_back(make_dir("accent", "british", false, -1, {0, 0, 1}));
    b.dirs.push_back(make_dir("deep", "cmn", true, 15, {0, 3}));
    b.dirs.push_back(make_dir("style", "whisper", true, 19, {1, 0}));
    b.dirs.push_back(make_dir("style", "tempo", true, 19, {-1, 1}));
    b.timing["style.tempo"] = {{-1, 2}, {0, 1}, {1, 0.5}};
    zonos3_steer_index(b);
    return b;
}

static void test_resolve() {
    const zonos3_steer_bundle b = make_bundle();
    CHECK((b.sources == std::vector<int>{15, 19}), "resolve: bundle sources");
    std::string err;
    zonos3_steering st;

    std::vector<float> emb = {4, 0, 0};
    CHECK(zonos3_steer_resolve(b, {{"who.gender", 1.5}, {"accent.british", 4}}, {}, 10, 0, &emb, st, err), err.c_str());
    const double k = 4.0 / std::sqrt(41.0);
    CHECK(near(emb[0], 4 * k, 1e-6) && near(emb[1], 3 * k, 1e-6) && near(emb[2], 4 * k, 1e-6), "resolve: raw embedding move");
    CHECK(near(std::sqrt((double) emb[0] * emb[0] + emb[1] * emb[1] + emb[2] * emb[2]), 4.0, 1e-6), "resolve: norm kept");
    CHECK(!st.hidden_active(), "resolve: raw only has no hidden delta");

    CHECK(zonos3_steer_resolve(b, {{"style.whisper", 2}, {"deep.cmn", 1}}, {}, 10, 0, nullptr, st, err), err.c_str());
    CHECK((st.hidden == std::vector<float>{0, 3, 2, 0}), "resolve: hidden layout [K*D]");
    CHECK((st.sources == std::vector<int>{15, 19}) && st.hidden_size == 2, "resolve: steering layout");

    // Span at the temp-adjusted pace: global tempo +1 -> x0.5, span tempo -1 -> x2.
    zonos3_control_span sp;
    sp.start = 4; sp.end = 8; sp.controls = {{"style.tempo", -1}};
    CHECK(zonos3_steer_resolve(b, {{"style.tempo", 1}}, {sp}, 10, 2.0, nullptr, st, err), err.c_str());
    CHECK(st.spans.size() == 1 && st.spans[0].start == 4 && st.spans[0].end == 12 && st.spans[0].ramp == 3, "resolve: span frames");
    CHECK((st.spans[0].delta == std::vector<float>{0, 0, 1, -1}), "resolve: span delta");

    // numpy round(): halves go to even.
    zonos3_control_span half;
    half.start = 5; half.end = 7; half.controls = {{"style.whisper", 1}};
    CHECK(zonos3_steer_resolve(b, {}, {half}, 10, 0.5, nullptr, st, err), err.c_str());
    CHECK(st.spans.size() == 1 && st.spans[0].start == 2 && st.spans[0].end == 4, "resolve: half-to-even rounding");

    // Ramp weights.
    zonos3_steering ramp;
    ramp.sources = {15, 19}; ramp.hidden_size = 2;
    ramp.spans.push_back({10, 20, 3, {0, 0, 1, 0}});
    std::vector<float> out;
    ramp.delta_at(7, out);  CHECK(out.empty(), "delta_at: 7 cleared");
    ramp.delta_at(8, out);  CHECK(out.size() == 4 && near(out[2], 1.0 / 3, 1e-6), "delta_at: 8 -> 1/3");
    ramp.delta_at(10, out); CHECK(near(out[2], 1, 1e-6), "delta_at: 10 -> 1");
    ramp.delta_at(20, out); CHECK(near(out[2], 1, 1e-6), "delta_at: 20 -> 1");
    ramp.delta_at(21, out); CHECK(near(out[2], 2.0 / 3, 1e-6), "delta_at: 21 -> 2/3");
    ramp.delta_at(23, out); CHECK(out.empty(), "delta_at: 23 cleared");

    // Errors.
    CHECK(!zonos3_steer_resolve(b, {{"nope.x", 1}}, {}, 10, 0, nullptr, st, err) && contains(err, "Unknown control 'nope.x'"), "error: unknown id");
    CHECK(!zonos3_steer_resolve(b, {{"who.gender", 1}, {"accent.british", 4}}, {}, 10, 0, nullptr, st, err) &&
          contains(err, "Controls accent.british, who.gender move the speaker embedding"), "error: raw without embedding");
    zonos3_control_span raw; raw.start = 0; raw.end = 2; raw.controls = {{"who.gender", 1}};
    CHECK(!zonos3_steer_resolve(b, {}, {raw}, 10, 0, nullptr, st, err) && contains(err, "cannot vary inside a request"), "error: raw in span");
    zonos3_control_span a, c;
    a.start = 0; a.end = 4; a.controls = {{"style.whisper", 1}};
    c.start = 3; c.end = 6; c.controls = {{"style.whisper", 1}};
    CHECK(!zonos3_steer_resolve(b, {}, {a, c}, 10, 0, nullptr, st, err) && contains(err, "overlap"), "error: overlap");
    zonos3_control_span out_of; out_of.start = 8; out_of.end = 12; out_of.controls = {{"style.whisper", 1}};
    CHECK(!zonos3_steer_resolve(b, {}, {out_of}, 10, 0, nullptr, st, err) && contains(err, "outside the text (0..10)"), "error: outside text");
}

// Steer then hold: the request schedule, the exempt split, the weight and the per-frame delta.
static void test_release() {
    zonos3_steer_bundle b = make_bundle();
    std::string err;
    std::optional<zonos3_steer_release_frames> rel;
    const double nan = std::nan("");

    // Request fields without the bundle key: unset or 1 = constant; below 1 refused.
    CHECK(zonos3_steer_request_release(&b, nan, nan, nan, 15.0, rel, err) && !rel, "request: no key, unset");
    CHECK(zonos3_steer_request_release(&b, 1.0, 2.0, nan, 15.0, rel, err) && !rel, "request: no key, hold 1");
    CHECK(!zonos3_steer_request_release(&b, 0.5, nan, nan, 15.0, rel, err) &&
          err == "steer_hold below 1 needs a steering bundle with steer_release; this server's bundle has none.", "request: no key, hold < 1");
    CHECK(zonos3_steer_request_release(nullptr, nan, nan, nan, 15.0, rel, err) && !rel, "request: no bundle, unset");
    CHECK(!zonos3_steer_request_release(nullptr, 0.0, nan, nan, 15.0, rel, err), "request: no bundle, hold < 1");

    // With the key: defaults, overrides, rounding (Python round: halves to even), hold 1 = constant.
    b.steer_release = zonos3_steer_release{4.0, 1.0, 0.25, {"style.tempo"}};
    CHECK(zonos3_steer_request_release(&b, nan, nan, nan, 15.0, rel, err) && rel && rel->onset == 60 && rel->ramp == 15 &&
          rel->hold == 0.25, "request: bundle defaults in frames");
    CHECK(zonos3_steer_request_release(&b, 0.5, 0.1, 0.0, 15.0, rel, err) && rel && rel->onset == 2 && rel->ramp == 0 &&
          rel->hold == 0.5, "request: overrides");
    CHECK(zonos3_steer_request_release(&b, nan, 2.5, 3.5, 1.0, rel, err) && rel && rel->onset == 2 && rel->ramp == 4,
          "request: half-to-even frames (2.5 -> 2, 3.5 -> 4)");
    CHECK(zonos3_steer_request_release(&b, 1.0, nan, nan, 15.0, rel, err) && !rel, "request: hold 1 = constant");

    // The split: exempt (style.tempo) and every hidden control without a schedule stay in `hidden`.
    const zonos3_steer_release_frames frames{4, 4, 0.25};
    zonos3_steering st, flat;
    CHECK(zonos3_steer_resolve(b, {{"style.whisper", 2}, {"deep.cmn", 1}, {"style.tempo", 1}}, {}, 10, 0, nullptr, flat, err), err.c_str());
    CHECK(flat.release.empty() && (flat.hidden == std::vector<float>{0, 3, 1, 1}), "split: none without a schedule");
    CHECK(zonos3_steer_resolve(b, {{"style.whisper", 2}, {"deep.cmn", 1}, {"style.tempo", 1}}, {}, 10, 0, nullptr, st, err, &frames),
          err.c_str());
    CHECK((st.hidden == std::vector<float>{0, 0, -1, 1}) && (st.release == std::vector<float>{0, 3, 2, 0}), "split: exempt vs released");
    CHECK(st.release_frames.onset == 4 && st.release_frames.ramp == 4 && st.release_frames.hold == 0.25, "split: frames");
    zonos3_steering only_exempt;
    CHECK(zonos3_steer_resolve(b, {{"style.tempo", 1}}, {}, 10, 0, nullptr, only_exempt, err, &frames), err.c_str());
    CHECK(only_exempt.release.empty() && (only_exempt.hidden == std::vector<float>{0, 0, -1, 1}), "split: nothing to release");
    zonos3_steering only_released;
    CHECK(zonos3_steer_resolve(b, {{"deep.cmn", 2}}, {}, 10, 0, nullptr, only_released, err, &frames), err.c_str());
    CHECK(only_released.hidden.empty() && only_released.hidden_active(), "split: released only is still hidden-active");

    // release_weight: 1 before onset, the linear ramp, hold after; ramp 0 is a step.
    CHECK(st.release_weight(0) == 1.0 && st.release_weight(3) == 1.0, "weight: before onset");
    CHECK(st.release_weight(4) == 1.0 && st.release_weight(5) == 1.0 - 0.75 / 4 && st.release_weight(6) == 0.625 &&
          st.release_weight(7) == 1.0 - 0.75 * 3 / 4, "weight: ramp");
    CHECK(st.release_weight(8) == 0.25 && st.release_weight(1000) == 0.25, "weight: hold");
    zonos3_steering step = st;
    step.release_frames = {4, 0, 0.0};
    CHECK(step.release_weight(3) == 1.0 && step.release_weight(4) == 0.0, "weight: ramp 0 = a step");

    // delta_at = hidden + w * release (+ spans).
    std::vector<float> out;
    st.delta_at(2, out);
    CHECK((out == std::vector<float>{0, 3, 1, 1}), "delta: before onset = the constant push");
    st.delta_at(6, out);
    CHECK((out == std::vector<float>{0, 0.625f * 3, -1 + 0.625f * 2, 1}), "delta: mid ramp");
    st.delta_at(9, out);
    CHECK((out == std::vector<float>{0, 0.75f, -0.5f, 1}), "delta: hold");
    only_released.delta_at(20, out);
    CHECK((out == std::vector<float>{0, 1.5f, 0, 0}), "delta: released only, no constant part");
    zonos3_steering zero = only_released;
    zero.release_frames = {0, 0, 0.0};
    zero.delta_at(5, out);
    CHECK(out.size() == 4 && out[1] == 0.0f, "delta: hold 0 keeps a zero delta");
    st.base_at(9, out);
    CHECK((out == std::vector<float>{0, 0.75f, -0.5f, 1}), "base_at: hidden + w * release");
    zonos3_steering spanned = st;
    spanned.spans.push_back({10, 20, 3, {1, 0, 0, 0}});
    spanned.delta_at(12, out);
    CHECK((out == std::vector<float>{1, 0.75f, -0.5f, 1}), "delta: a span adds on top");

    // Hold 1 is never a release: resolve leaves the constant layout of before (bit identity).
    CHECK(zonos3_steer_request_release(&b, 1.0, nan, nan, 15.0, rel, err) && !rel, "hold 1: no schedule");
    zonos3_steering same;
    CHECK(zonos3_steer_resolve(b, {{"style.whisper", 2}, {"deep.cmn", 1}, {"style.tempo", 1}}, {}, 10, 0, nullptr, same, err,
                               rel ? &*rel : nullptr) && same.hidden == flat.hidden && same.release.empty(), "hold 1: constant delta");
}

struct run_result { std::vector<std::vector<float>> logits; };  // per step: slot 0 then slot 1

static run_result run_decode(const zonos3_model & m, const zonos3_prompt & p, const zonos3_prompt & twin,
                             const std::vector<int> & sources, const std::vector<float> & steer0, int steps = 4) {
    const int cap = std::max(p.n_rows, twin.n_rows) + steps + 4;
    zonos3_batch_ctx bc;
    CHECK(zonos3_batch_init(bc, m, 2, cap, sources), "graph: batch_init");
    const size_t stride = (size_t) m.hp.audio_vocab * m.hp.n_codebooks;
    std::vector<float> l0(stride), l1(stride);
    std::vector<float> ids0(p.ids.begin(), p.ids.end()), ids1(twin.ids.begin(), twin.ids.end());
    CHECK(zonos3_batch_slot_prefill(bc, 0, ids0.data(), p.n_rows, p.cond, l0.data()), "graph: prefill 0");
    CHECK(zonos3_batch_slot_prefill(bc, 1, ids1.data(), twin.n_rows, twin.cond, l1.data()), "graph: prefill 1");
    zonos3_slot s0, s1;
    s0.index = 0; s0.active = true; s0.n_past = p.n_rows; s0.prefill = p.prefill_frames;
    s1.index = 1; s1.active = true; s1.n_past = twin.n_rows; s1.prefill = p.prefill_frames;
    std::vector<int32_t> next(m.hp.n_codebooks + 1, 1);
    next.back() = (int32_t) m.hp.text_vocab;
    s0.next_ids = s1.next_ids = next;
    s0.steer = steer0;
    run_result r;
    for (int i = 0; i < steps; i++) {
        CHECK(zonos3_batch_step(bc, {&s0, &s1}), "graph: decode step");
        std::vector<float> both(2 * stride);
        memcpy(both.data(), zonos3_batch_slot_logits(bc, 0), stride * sizeof(float));
        memcpy(both.data() + stride, zonos3_batch_slot_logits(bc, 1), stride * sizeof(float));
        r.logits.push_back(std::move(both));
    }
    zonos3_batch_free(bc);
    return r;
}

static bool same_bits(const run_result & a, const run_result & b, int slot) {
    const size_t stride = a.logits[0].size() / 2;
    for (size_t i = 0; i < a.logits.size(); i++)
        if (memcmp(a.logits[i].data() + slot * stride, b.logits[i].data() + slot * stride, stride * sizeof(float))) return false;
    return true;
}
static float max_diff(const run_result & a, const run_result & b, int slot) {
    const size_t stride = a.logits[0].size() / 2;
    float d = 0;
    for (size_t i = 0; i < a.logits.size(); i++)
        for (size_t j = 0; j < stride; j++)
            d = std::max(d, std::fabs(a.logits[i][slot * stride + j] - b.logits[i][slot * stride + j]));
    return d;
}

// One slot decoding `steps` frames with a per-step delta (zonos3_steering::delta_at of the step, like the scheduler).
static std::vector<std::vector<float>> run_steps(const zonos3_model & m, const zonos3_prompt & p, const std::vector<int> & sources,
                                                 const std::function<void(int, std::vector<float> &)> & steer, int steps) {
    zonos3_batch_ctx bc;
    CHECK(zonos3_batch_init(bc, m, 1, p.n_rows + steps + 4, sources), "release graph: batch_init");
    const size_t stride = (size_t) m.hp.audio_vocab * m.hp.n_codebooks;
    std::vector<float> l(stride), ids(p.ids.begin(), p.ids.end());
    CHECK(zonos3_batch_slot_prefill(bc, 0, ids.data(), p.n_rows, p.cond, l.data()), "release graph: prefill");
    zonos3_slot s;
    s.index = 0; s.active = true; s.n_past = p.n_rows; s.prefill = p.prefill_frames;
    s.next_ids.assign(m.hp.n_codebooks + 1, 1);
    s.next_ids.back() = (int32_t) m.hp.text_vocab;
    std::vector<std::vector<float>> out;
    for (int i = 0; i < steps; i++) {
        steer(i, s.steer);
        CHECK(zonos3_batch_step(bc, {&s}), "release graph: decode step");
        const float * x = zonos3_batch_slot_logits(bc, 0);
        out.emplace_back(x, x + stride);
    }
    zonos3_batch_free(bc);
    return out;
}

static void test_graph(zonos3_model & m, const zonos3_steer_bundle & b, const std::string & spk_path) {
    std::vector<float> speaker;
    std::string err;
    CHECK(mp_load_speaker_embedding(spk_path, speaker, err), err.c_str());
    zonos3_prompt_options opt;
    opt.speaker_embedding = speaker.data();
    zonos3_prompt prompt, twin;
    CHECK(zonos3_build_prompt(m, "steer test", opt, prompt), "graph: prompt");
    CHECK(zonos3_build_cfg_twin(m, prompt, twin), "graph: cfg twin");

    const run_result plain = run_decode(m, prompt, twin, {}, {});
    const run_result zero = run_decode(m, prompt, twin, {3, 5}, {});
    CHECK(same_bits(plain, zero, 0) && same_bits(plain, zero, 1), "graph: zero delta is bit-identical");

    const size_t D = m.hp.n_embd;
    auto delta_for = [&](const char * id, int slot) {
        std::vector<float> d(2 * D, 0.0f);
        const auto & v = b.dirs[b.index.at(id)].vec;
        std::copy(v.begin(), v.end(), d.begin() + slot * D);
        return d;
    };
    const run_result s3 = run_decode(m, prompt, twin, {3, 5}, delta_for("style.tempo", 0));
    CHECK(max_diff(zero, s3, 0) > 1e-4f, "graph: source 3 delta moves slot 0 logits");
    CHECK(same_bits(zero, s3, 1), "graph: unsteered twin slot is untouched");
    const run_result s5 = run_decode(m, prompt, twin, {3, 5}, delta_for("mood.valence", 1));
    CHECK(max_diff(zero, s5, 0) > 1e-4f, "graph: source 5 delta moves slot 0 logits");
    CHECK(same_bits(zero, s5, 1), "graph: unsteered twin slot is untouched (source 5)");

    // Validation.
    zonos3_batch_ctx bc;
    CHECK(!zonos3_batch_init(bc, m, 2, 64, {9}), "validation: source beyond bank");
    CHECK(!zonos3_batch_init(bc, m, 2, 64, {5, 3}), "validation: descending sources");
    CHECK(!zonos3_batch_init(bc, m, 2, 64, {3, 3}), "validation: duplicate sources");
    CHECK(zonos3_batch_init(bc, m, 2, std::max(prompt.n_rows, twin.n_rows) + 8, {3, 5}), "validation: valid sources");
    std::vector<float> ids(prompt.ids.begin(), prompt.ids.end()), l(m.hp.audio_vocab * m.hp.n_codebooks);
    CHECK(zonos3_batch_slot_prefill(bc, 0, ids.data(), prompt.n_rows, prompt.cond, l.data()), "validation: prefill");
    zonos3_slot s;
    s.index = 0; s.active = true; s.n_past = prompt.n_rows;
    s.next_ids.assign(m.hp.n_codebooks + 1, 1);
    s.next_ids.back() = (int32_t) m.hp.text_vocab;
    s.steer.assign(7, 0.0f);
    CHECK(!zonos3_batch_step(bc, {&s}), "validation: wrong delta size rejected");
    zonos3_batch_free(bc);

    // Steer then hold in the decode graph: a constant push (style.whisper, source 3) released from step 2 over 2 steps to
    // 0.25 matches the same deltas set by hand, step for step, and leaves the steps before onset bit-identical to the
    // constant push.
    zonos3_steering constant;
    constant.sources = {3, 5};
    constant.hidden_size = (int) D;
    constant.hidden = delta_for("style.whisper", 0);
    zonos3_steering released = constant;
    released.release = released.hidden;
    released.hidden.clear();
    released.release_frames = {2, 2, 0.25};
    const int steps = 6;
    const auto c = run_steps(m, prompt, {3, 5}, [&](int f, std::vector<float> & d) { constant.delta_at(f, d); }, steps);
    const auto r = run_steps(m, prompt, {3, 5}, [&](int f, std::vector<float> & d) { released.delta_at(f, d); }, steps);
    const float w[] = {1, 1, 1, 0.625f, 0.25f, 0.25f};
    const auto h = run_steps(m, prompt, {3, 5}, [&](int f, std::vector<float> & d) {
        d = constant.hidden;
        for (auto & v : d) v *= w[f];
    }, steps);
    for (int i = 0; i < steps; i++)
        CHECK(!memcmp(r[i].data(), h[i].data(), r[i].size() * sizeof(float)), "release graph: per-step delta = w(f) x push");
    for (int i = 0; i < 3; i++)
        CHECK(!memcmp(r[i].data(), c[i].data(), r[i].size() * sizeof(float)), "release graph: before the release = constant push");
    float diff = 0;
    for (size_t j = 0; j < r[4].size(); j++) diff = std::max(diff, std::fabs(r[4][j] - c[4][j]));
    CHECK(diff > 1e-4f, "release graph: the held push differs from the constant one");
}

int main(int argc, char ** argv) {
    if (argc != 3) { fprintf(stderr, "usage: steering MODEL.gguf STEER.gguf\n"); return 2; }
    zonos3_model model;
    if (!zonos3_model_load(model, argv[1], false)) return 1;
    test_loader(model, argv[2], argv[1]);
    test_resolve();
    test_release();
    zonos3_steer_bundle b;
    std::string err;
    CHECK(zonos3_steer_load(b, argv[2], model, err), err.c_str());
    test_graph(model, b, mp_parent_dir(argv[1]) + "/speaker-embedding.npy");
    zonos3_model_free(model);
    printf("PASS\n");
    return 0;
}
