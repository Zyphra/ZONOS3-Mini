// Graph features: alignment probe (readout + masked heads), accent adapters.
//
//   graph-features --selftest <dummy-zonos3.gguf> <dummy-accents.gguf> [--gpu]
//       Invariants of both features (ctest on CPU; --gpu runs them on the F16-KV / FlashAttention profile).
#include "zonos3.h"
#include "zonos3-accent.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

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

int selftest(const char * model_path, const char * accents_path, bool gpu) {
    zonos3_model model;
    CHECK(zonos3_model_load(model, model_path, gpu), "load backbone");
    zonos3_accents bank;
    std::string err;
    CHECK(zonos3_accents_load(bank, accents_path, model, err), err.c_str());
    zonos3_accents wrong;
    CHECK(!zonos3_accents_load(wrong, model_path, model, err), "accents: a backbone file is refused");
    model.accents = &bank;
    zonos3_prompt_options opt;
    zonos3_prompt prompt;
    CHECK(zonos3_build_prompt(model, "The quick brown fox jumps.", opt, prompt), "prompt");
    const auto rows = random_rows(model, 5);
    test_accents(model, prompt, rows);
    test_probe(model, prompt, rows);
    model.accents = nullptr;
    zonos3_accents_free(bank);
    zonos3_model_free(model);
    printf("PASS\n");
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc == 4 && !strcmp(argv[1], "--selftest")) return selftest(argv[2], argv[3], false);
    if (argc == 5 && !strcmp(argv[1], "--selftest") && !strcmp(argv[4], "--gpu")) return selftest(argv[2], argv[3], true);
    fprintf(stderr, "usage: %s --selftest <model.gguf> <accents.gguf> [--gpu]\n", argv[0]);
    return 2;
}
