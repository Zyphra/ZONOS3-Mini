// Graph features: alignment probe (readout + masked heads), accent adapters, emotion directions.
//
//   graph-features --selftest <dummy-zonos3.gguf> <dummy-accents.gguf> <dummy-emotion.gguf> [--gpu]
//       Invariants of the three features (ctest on CPU; --gpu runs them on the F16-KV / FlashAttention profile).
//   graph-features --emotion <directions.gguf> --request <request.json>
//       Prints {"embedding": [...]|null, "hidden": [...]|null} (or {"error": msg}) for one apply_emotion /
//       emotion_hidden_delta request, the parity driver of tests/parity/test_ggml_graph.py. Request keys: base (float
//       list), sliders ([[name, weight], ...] in order), valence, arousal, strength, preserve_norm, strict, speaker,
//       calibrated (fold calibration.json first, as the sgl server's calibrated mode).
#include "zonos3.h"
#include "zonos3-accent.h"
#include "zonos3-emotion.h"
#include "json.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using nlohmann::json;

#define CHECK(cond, label) do { if (!(cond)) { fprintf(stderr, "FAIL: %s (%s:%d)\n", label, __FILE__, __LINE__); exit(1); } } while (0)

namespace {

double max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) return INFINITY;
    double d = 0.0;
    for (size_t i = 0; i < a.size(); ++i) d = std::max(d, (double) std::fabs(a[i] - b[i]));
    return d;
}

struct slot_trace {
    std::vector<std::vector<float>> logits;     // prefill last row, then every decode step
    std::vector<zonos3_probe_readout> readouts; // every decode step (probe enabled)
};

// Prefills `n_slots` copies of `prompt` and teacher-forces `rows` through all of them; `setup(slot)` picks each
// slot's accent / probe (the prefill uses the slot's accent).
std::vector<slot_trace> decode(const zonos3_model & model, const zonos3_prompt & prompt,
                               const std::vector<std::vector<int32_t>> & rows, int n_slots,
                               const std::function<void(zonos3_slot &)> & setup,
                               const zonos3_batch_features & features = {}, const std::vector<int> & steer_sources = {},
                               const std::vector<float> & steer = {}) {
    const int av = (int) model.hp.audio_vocab, ncb = (int) model.hp.n_codebooks;
    zonos3_batch_ctx bc;
    CHECK(zonos3_batch_init(bc, model, n_slots, prompt.n_rows + (int) rows.size() + 4, steer_sources, features), "batch init");
    std::vector<zonos3_slot> slots(n_slots);
    std::vector<slot_trace> out(n_slots);
    std::vector<float> ids(prompt.ids.begin(), prompt.ids.end()), logits((size_t) av * ncb);
    for (int i = 0; i < n_slots; ++i) {
        zonos3_slot & s = slots[i];
        s.index = i; s.active = true; s.n_past = prompt.n_rows; s.prefill = prompt.prefill_frames;
        setup(s);
        zonos3_conditioning cond = prompt.cond;
        cond.accent = s.accent; cond.accent_strength = s.accent_strength;
        CHECK(zonos3_batch_slot_prefill(bc, i, ids.data(), prompt.n_rows, cond, logits.data()), "prefill");
        out[i].logits.push_back(logits);
    }
    std::vector<zonos3_slot *> active;
    for (auto & s : slots) active.push_back(&s);
    for (const auto & row : rows) {
        for (auto & s : slots) { s.next_ids = row; s.steer = steer; }
        CHECK(zonos3_batch_step(bc, active), "decode step");
        for (int i = 0; i < n_slots; ++i) {
            const float * l = zonos3_batch_slot_logits(bc, i);
            out[i].logits.emplace_back(l, l + (size_t) av * ncb);
            if (slots[i].probe.enabled()) out[i].readouts.push_back(slots[i].readout);
        }
    }
    zonos3_batch_free(bc);
    return out;
}

std::vector<std::vector<int32_t>> random_rows(const zonos3_model & model, int n) {
    std::mt19937 rng(7);
    std::vector<std::vector<int32_t>> rows(n, std::vector<int32_t>(model.hp.n_codebooks + 1));
    for (auto & r : rows) {
        for (uint32_t c = 0; c < model.hp.n_codebooks; ++c) r[c] = (int32_t) (rng() % model.hp.codebook_size);
        r[model.hp.n_codebooks] = (int32_t) model.hp.text_vocab;
    }
    return rows;
}

double trace_diff(const slot_trace & a, const slot_trace & b) {
    double d = 0.0;
    for (size_t i = 0; i < std::min(a.logits.size(), b.logits.size()); ++i) d = std::max(d, max_abs_diff(a.logits[i], b.logits[i]));
    return a.logits.size() == b.logits.size() ? d : INFINITY;
}

void test_accents(const zonos3_model & model, const zonos3_prompt & prompt, const std::vector<std::vector<int32_t>> & rows) {
    const zonos3_accents & bank = *model.accents;
    CHECK(bank.n == 2 && bank.names[0] == "alpha" && bank.names[1] == "beta", "accents: names in slot order");
    CHECK(bank.K == 4 && bank.R == 4, "accents: stacked ranks");
    CHECK(zonos3_accent_index(bank, "beta") == 1 && zonos3_accent_index(bank, "gamma") == -1, "accents: index lookup");
    CHECK(!bank.experts.empty() && bank.router_bias.size() >= 2, "accents: expert factors and router shifts");

    // Uncached forward: strength 0 (and no accent) is the plain model, bit for bit.
    std::vector<float> ids(prompt.ids.begin(), prompt.ids.end()), plain, zero, shifted;
    zonos3_conditioning cond = prompt.cond;
    CHECK(zonos3_logits(model, ids.data(), prompt.n_rows, cond, plain), "accents: plain logits");
    cond.accent = 0; cond.accent_strength = 0.0f;
    CHECK(zonos3_logits(model, ids.data(), prompt.n_rows, cond, zero), "accents: zero-strength logits");
    CHECK(max_abs_diff(plain, zero) == 0.0, "accents: zero strength is exact");
    cond.accent_strength = 1.0f;
    CHECK(zonos3_logits(model, ids.data(), prompt.n_rows, cond, shifted), "accents: adapter logits");
    CHECK(max_abs_diff(plain, shifted) > 1e-4, "accents: an adapter changes the logits");
    cond.accent = 5;
    CHECK(!zonos3_logits(model, ids.data(), prompt.n_rows, cond, shifted), "accents: unknown adapter refused");

    // Batched: a slot without an accent beside accent slots matches a plain batch; the accent slot's prefill
    // matches the uncached forward.
    const auto plain_run = decode(model, prompt, rows, 1, [](zonos3_slot &) {});
    const auto mixed = decode(model, prompt, rows, 3, [](zonos3_slot & s) {
        if (s.index == 0) { s.accent = 0; s.accent_strength = 1.0f; }
        if (s.index == 2) { s.accent = 1; s.accent_strength = -0.7f; }
    });
    CHECK(trace_diff(mixed[1], plain_run[0]) < 1e-5, "accents: rows without an accent are unaffected");
    CHECK(trace_diff(mixed[0], plain_run[0]) > 1e-4 && trace_diff(mixed[2], plain_run[0]) > 1e-4, "accents: decode shifts");
    const size_t row = (size_t) model.hp.audio_vocab * model.hp.n_codebooks;
    cond.accent = 0; cond.accent_strength = 1.0f;
    CHECK(zonos3_logits(model, ids.data(), prompt.n_rows, cond, shifted), "accents: adapter logits");
    std::vector<float> last(shifted.end() - row, shifted.end());
    CHECK(max_abs_diff(mixed[0].logits[0], last) < 1e-4, "accents: batched prefill == uncached forward");
    // Decode == uncached forward over prompt + rows (teacher forcing).
    std::vector<float> full(ids);
    for (const auto & r : rows) full.insert(full.end(), r.begin(), r.end());
    CHECK(zonos3_logits(model, full.data(), prompt.n_rows + (int) rows.size(), cond, shifted), "accents: full forward");
    std::vector<float> final_row(shifted.end() - row, shifted.end());
    CHECK(max_abs_diff(mixed[0].logits.back(), final_row) < 1e-4, "accents: decode == uncached forward");
    // Zero strength in decode is the plain graph's arithmetic.
    const auto zero_run = decode(model, prompt, rows, 1, [](zonos3_slot & s) { s.accent = 1; s.accent_strength = 0.0f; });
    CHECK(trace_diff(zero_run[0], plain_run[0]) == 0.0, "accents: zero strength decode is exact");
    printf("accents: ok\n");
}

void test_probe(const zonos3_model & model, const zonos3_prompt & prompt, const std::vector<std::vector<int32_t>> & rows) {
    std::vector<int32_t> pids(prompt.ids.begin(), prompt.ids.end());
    int ts = -1, tl = 0;
    CHECK(zonos3_locate_text_span(model, pids.data(), prompt.n_rows, ts, tl), "probe: text span located");
    CHECK(tl >= 4, "probe: text span long enough");
    int global = -1;
    for (uint32_t L = 0; L < model.hp.n_layer; ++L) if (!model.layers[L].is_gdn) { global = (int) L; break; }
    CHECK(global >= 0, "probe: model has a global layer");
    const int nh = (int) model.hp.n_head;
    zonos3_align_probe probe;
    probe.text_start = ts; probe.text_len = tl;
    for (int h = 0; h < nh; ++h) probe.heads.push_back({global, h});
    zonos3_batch_features f;
    f.probe_max_text = tl + 3;

    const auto plain_run = decode(model, prompt, rows, 1, [](zonos3_slot &) {});
    const auto probed = decode(model, prompt, rows, 1, [&](zonos3_slot & s) { s.probe = probe; }, f);
    CHECK(trace_diff(probed[0], plain_run[0]) == 0.0, "probe: the readout leaves the logits untouched");
    CHECK(probed[0].readouts.size() == rows.size(), "probe: one readout per decode step");
    for (const auto & r : probed[0].readouts)
        CHECK(r.argmax >= 0 && r.argmax < tl && r.centroid >= 0.0f && r.centroid <= (float) (tl - 1), "probe: readout range");

    // An open window on every head: the masked recompute equals attention, and the readout is unchanged.
    zonos3_align_probe open = probe;
    for (int h = 0; h < nh; ++h) open.windows.push_back({global, h, -5, tl + 5});
    const auto opened = decode(model, prompt, rows, 1, [&](zonos3_slot & s) { s.probe = open; }, f);
    CHECK(trace_diff(opened[0], plain_run[0]) < 1e-5, "probe: open windows keep the attention output");
    for (size_t i = 0; i < rows.size(); ++i)
        CHECK(opened[0].readouts[i].argmax == probed[0].readouts[i].argmax &&
              std::fabs(opened[0].readouts[i].centroid - probed[0].readouts[i].centroid) < 1e-5, "probe: open-window readout");

    // A narrow window [2, 3] on every head: the readout lives inside it and the attention output changes.
    zonos3_align_probe narrow = probe;
    for (int h = 0; h < nh; ++h) narrow.windows.push_back({global, h, 2, 3});
    const auto mixed = decode(model, prompt, rows, 3, [&](zonos3_slot & s) {
        if (s.index == 1) s.probe = narrow;
        if (s.index == 2) s.probe = probe;
    }, f);
    for (const auto & r : mixed[1].readouts)
        CHECK(r.argmax >= 2 && r.argmax <= 3 && r.centroid >= 2.0f - 1e-4f && r.centroid <= 3.0f + 1e-4f, "probe: windowed readout");
    // The dummy backbone's 0.02-scale weights keep the effect small; unaffected rows are bit-exact.
    CHECK(trace_diff(mixed[1], plain_run[0]) > 1e-7, "probe: masked heads change the attention output");
    CHECK(trace_diff(mixed[0], plain_run[0]) == 0.0 && trace_diff(mixed[2], plain_run[0]) == 0.0,
          "probe: other slots are unaffected by a masked slot");
    for (size_t i = 0; i < rows.size(); ++i)
        CHECK(mixed[2].readouts[i].argmax == probed[0].readouts[i].argmax &&
              std::fabs(mixed[2].readouts[i].centroid - probed[0].readouts[i].centroid) < 1e-4, "probe: batched readout");

    // Probe + accent in one graph.
    const auto both = decode(model, prompt, rows, 2, [&](zonos3_slot & s) {
        s.probe = narrow;
        if (s.index == 1) { s.accent = 0; s.accent_strength = 1.0f; }
    }, f);
    CHECK(trace_diff(both[0], mixed[1]) < 1e-5, "probe: masked slot beside an accent slot");

    // Refusals: a span longer than the batch limit, a GDN head, a batch without probe support.
    zonos3_batch_ctx bc;
    CHECK(zonos3_batch_init(bc, model, 1, prompt.n_rows + 8), "probe: plain batch");
    std::vector<float> ids(prompt.ids.begin(), prompt.ids.end()), logits((size_t) model.hp.audio_vocab * model.hp.n_codebooks);
    CHECK(zonos3_batch_slot_prefill(bc, 0, ids.data(), prompt.n_rows, prompt.cond, logits.data()), "probe: prefill");
    zonos3_slot s; s.index = 0; s.active = true; s.n_past = prompt.n_rows; s.next_ids = rows[0]; s.probe = probe;
    std::vector<zonos3_slot *> active{&s};
    CHECK(!zonos3_batch_step(bc, active), "probe: refused without probe support");
    zonos3_batch_free(bc);
    printf("probe: ok\n");
}

void test_emotion(const zonos3_model & model, const std::string & path, const zonos3_prompt & base_prompt) {
    zonos3_directions d;
    std::string err;
    CHECK(zonos3_directions_load(d, path.c_str(), err), err.c_str());
    CHECK(d.space == "raw" && d.dim == 192 && d.names.size() == 3 && d.axes.size() == 2 && d.calibrated, "emotion: loader");
    CHECK(d.calibration.strength("default_0123456789abcdef", "happy") == 0.0 && d.calibration.strength("", "angry") == 2.0 &&
          d.calibration.strength("x", "sad") == 1.25, "emotion: calibration lookup");
    CHECK(d.calibration.pole_strength("default_0123456789abcdef", "valence", -1.0) == 0.25 &&
          d.calibration.pole_strength("default_0123456789abcdef", "valence", 1.0) == 1.25, "emotion: pole strength");
    const auto off = d.calibration.disabled("default_0123456789abcdef", d.controls());
    CHECK(off.size() == 1 && off[0] == "happy", "emotion: disabled controls");

    std::vector<float> base(192);
    for (int i = 0; i < 192; ++i) base[i] = std::sin(0.1 * i) + 0.5f;
    std::vector<float> out;
    CHECK(zonos3_emotion_apply(d, base, {{"sad", 0.5}, {"angry", -1.0}}, 0.25, 0.0, 1.5, true, true, out, err), err.c_str());
    double nb = 0, no = 0, dot = 0;
    for (int i = 0; i < 192; ++i) {
        const double want = base[i] + 1.5 * (0.5 * d.named[1][i] - 1.0 * d.named[2][i] + 0.25 * d.axis[0][i]);
        nb += (double) base[i] * base[i]; no += (double) out[i] * out[i]; dot += want * out[i];
    }
    CHECK(std::fabs(std::sqrt(no) - std::sqrt(nb)) < 1e-4, "emotion: norm preserved");
    double nw = 0;
    for (int i = 0; i < 192; ++i) {
        const double want = base[i] + 1.5 * (0.5 * d.named[1][i] - 1.0 * d.named[2][i] + 0.25 * d.axis[0][i]);
        nw += want * want;
    }
    CHECK(dot / (std::sqrt(nw) * std::sqrt(no)) > 1.0 - 1e-9, "emotion: direction of the raw move");
    CHECK(!zonos3_emotion_apply(d, base, {{"joy", 1.0}}, 0, 0, 1, true, true, out, err) &&
          err.rfind("Unknown emotion direction 'joy'. Available: ['happy', 'sad', 'angry', 'valence', 'arousal'].", 0) == 0,
          "emotion: strict unknown name");
    CHECK(zonos3_emotion_apply(d, base, {{"joy", 1.0}}, 0, 0, 1, true, false, out, err) && out == base, "emotion: lenient skip");

    // Hidden spaces: proj deltas join the speaker row, attnres_source deltas a bank source.
    zonos3_directions h = d;
    h.space = "proj";
    std::vector<float> delta;
    CHECK(zonos3_emotion_hidden_delta(h, {{"happy", 1.0}}, 0, 0, 2.0, true, delta, err) && delta.size() == 192, "emotion: hidden delta");
    CHECK(zonos3_emotion_apply(h, base, {{"happy", 1.0}}, 0, 0, 2.0, true, true, out, err) && out == base, "emotion: hidden apply is a no-op");
    zonos3_prompt_options opt;
    std::vector<float> spk(model.hp.speaker_dim, 0.1f);
    opt.speaker_embedding = spk.data();
    zonos3_prompt p;
    CHECK(zonos3_build_prompt(model, "Hello there.", opt, p), "emotion: speaker prompt");
    std::vector<float> ids(p.ids.begin(), p.ids.end()), a, b;
    CHECK(zonos3_logits(model, ids.data(), p.n_rows, p.cond, a), "emotion: plain logits");
    zonos3_conditioning c = p.cond;
    c.speaker_delta.assign(model.hp.n_embd, 0.0f);
    CHECK(zonos3_logits(model, ids.data(), p.n_rows, c, b) && max_abs_diff(a, b) == 0.0, "emotion: zero proj delta is exact");
    for (uint32_t i = 0; i < model.hp.n_embd; ++i) c.speaker_delta[i] = 0.05f * std::cos((float) i);
    CHECK(zonos3_logits(model, ids.data(), p.n_rows, c, b) && max_abs_diff(a, b) > 1e-5, "emotion: proj delta moves the logits");
    c.speaker_row = -1;
    CHECK(!zonos3_logits(model, ids.data(), p.n_rows, c, b), "emotion: proj delta needs a speaker row");

    std::vector<float> layout;
    std::vector<float> hid(model.hp.n_embd, 0.02f);
    CHECK(zonos3_steer_add_delta({1, 3}, (int) model.hp.n_embd, 3, hid, layout) && layout.size() == 2 * model.hp.n_embd &&
          layout[0] == 0.0f && layout[model.hp.n_embd] == 0.02f, "emotion: steering layout");
    CHECK(!zonos3_steer_add_delta({1, 3}, (int) model.hp.n_embd, 2, hid, layout), "emotion: unknown source refused");
    // Accent steering on a direction set: raw moves the embedding (calibrated dose), hidden scales the direction.
    std::vector<float> moved = base, hv;
    CHECK(zonos3_accent_steering(d, "angry", 0.5, "", true, &moved, hv, err) && hv.empty() && moved != base,
          "emotion: raw accent steering");
    CHECK(zonos3_emotion_apply(d, base, {{"angry", 0.5 * 2.0}}, 0, 0, 1, true, true, out, err) && out == moved,
          "emotion: raw accent steering == apply_emotion at strength * dose");
    CHECK(!zonos3_accent_steering(d, "angry", 0.5, "", true, nullptr, hv, err), "emotion: raw accent steering needs an embedding");
    moved = base;
    CHECK(zonos3_accent_steering(d, "happy", 1.0, "default_0123456789abcdef", true, &moved, hv, err) && moved == base,
          "emotion: a zero dose disables accent steering");
    CHECK(zonos3_accent_steering(h, "sad", 2.0, "", false, nullptr, hv, err) && hv.size() == 192 && hv[5] == 2.0f * h.named[1][5],
          "emotion: hidden accent steering");
    const auto rows = random_rows(model, 3);
    const auto plain_run = decode(model, base_prompt, rows, 1, [](zonos3_slot &) {}, {}, {1, 3});
    const auto steered = decode(model, base_prompt, rows, 1, [](zonos3_slot &) {}, {}, {1, 3}, layout);
    CHECK(trace_diff(plain_run[0], steered[0]) > 1e-6, "emotion: attnres_source delta steers decode");
    printf("emotion: ok\n");
}

int selftest(const char * model_path, const char * accents_path, const char * emotion_path, bool gpu) {
    zonos3_model model;
    CHECK(zonos3_model_load(model, model_path, gpu), "load backbone");
    zonos3_accents bank;
    std::string err;
    CHECK(zonos3_accents_load(bank, accents_path, model, err), err.c_str());
    zonos3_accents wrong;
    CHECK(!zonos3_accents_load(wrong, emotion_path, model, err), "accents: a directions file is refused");
    model.accents = &bank;
    zonos3_prompt_options opt;
    zonos3_prompt prompt;
    CHECK(zonos3_build_prompt(model, "The quick brown fox jumps.", opt, prompt), "prompt");
    const auto rows = random_rows(model, 5);
    test_accents(model, prompt, rows);
    test_probe(model, prompt, rows);
    test_emotion(model, emotion_path, prompt);
    model.accents = nullptr;
    zonos3_accents_free(bank);
    zonos3_model_free(model);
    printf("PASS\n");
    return 0;
}

int emotion_request(const char * path, const char * request_path) {
    zonos3_directions d;
    std::string err;
    json out;
    if (!zonos3_directions_load(d, path, err)) {
        out["error"] = err;
        printf("%s\n", out.dump().c_str());
        return 1;
    }
    std::ifstream in(request_path);
    std::stringstream ss;
    ss << in.rdbuf();
    const json req = json::parse(ss.str());
    zonos3_emotion_sliders sliders;
    for (const auto & s : req.value("sliders", json::array())) sliders.emplace_back(s.at(0).get<std::string>(), s.at(1).get<double>());
    double valence = req.value("valence", 0.0), arousal = req.value("arousal", 0.0);
    if (req.value("calibrated", false)) zonos3_emotion_calibrate(d, req.value("speaker", std::string()), sliders, valence, arousal);
    const double strength = req.value("strength", 1.0);
    const bool strict = req.value("strict", false), preserve = req.value("preserve_norm", true);
    std::vector<float> base = req.value("base", std::vector<float>{}), emb, hidden;
    out["embedding"] = nullptr;
    out["hidden"] = nullptr;
    if (!zonos3_emotion_hidden_delta(d, sliders, valence, arousal, strength, strict, hidden, err) ||
        (!d.hidden() && !base.empty() &&
         !zonos3_emotion_apply(d, base, sliders, valence, arousal, strength, preserve, strict, emb, err))) {
        out = json{{"error", err}};
    } else {
        if (!d.hidden() && !base.empty()) out["embedding"] = emb;
        if (!hidden.empty()) out["hidden"] = hidden;
    }
    printf("%s\n", out.dump(-1, ' ', false, json::error_handler_t::replace).c_str());
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc == 5 && !strcmp(argv[1], "--selftest")) return selftest(argv[2], argv[3], argv[4], false);
    if (argc == 6 && !strcmp(argv[1], "--selftest") && !strcmp(argv[5], "--gpu")) return selftest(argv[2], argv[3], argv[4], true);
    if (argc == 5 && !strcmp(argv[1], "--emotion") && !strcmp(argv[3], "--request")) return emotion_request(argv[2], argv[4]);
    fprintf(stderr, "usage: %s --selftest <model.gguf> <accents.gguf> <emotion.gguf> [--gpu]\n"
                    "       %s --emotion <directions.gguf> --request <request.json>\n", argv[0], argv[0]);
    return 2;
}
