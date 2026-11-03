#include "zonos3.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <utility>
#include <vector>

// Keep the existing continuation gate. These are FP32 accumulation/reduction
// comparisons, not a license to hide half-precision or autoregressive drift.
static bool close_logits(const std::vector<float> & a, const std::vector<float> & b, const char * label) {
    if (a.size() != b.size()) return false;
    size_t failures = 0;
    double squared = 0;
    float maximum = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const float difference = std::abs(a[i] - b[i]);
        const float tolerance = 2e-3f + 2e-3f * std::max(std::abs(a[i]), std::abs(b[i]));
        if (!std::isfinite(a[i]) || !std::isfinite(b[i]) || difference > tolerance) {
            if (!failures) fprintf(stderr, "%s: first mismatch %zu: %.9g != %.9g\n", label, i, a[i], b[i]);
            ++failures;
        }
        maximum = std::max(maximum, difference);
        squared += (double) difference * difference;
    }
    printf("%s: max_abs=%.9g rmse=%.9g outside_gate=%zu/%zu\n", label, maximum,
           std::sqrt(squared / std::max((size_t) 1, a.size())), failures, a.size());
    return failures == 0;
}

struct state_lane {
    std::string name;
    ggml_type type;
    std::vector<uint8_t> bytes;
};

// Read actual persistent tensors, not the cached host logits. Global attention's
// extra idle-write row is deliberately excluded; only the live causal prefix
// can influence a resumed consumer.
static std::vector<state_lane> snapshot(const zonos3_batch_ctx & batch, int slot, int rows) {
    std::vector<state_lane> result;
    auto read_lane = [&](ggml_tensor * tensor, size_t offset, size_t bytes, const std::string & name) {
        state_lane lane{name, tensor->type, std::vector<uint8_t>(bytes)};
        ggml_backend_tensor_get(tensor, lane.bytes.data(), offset, bytes);
        result.push_back(std::move(lane));
    };
    for (int L = 0; L < (int) batch.model->hp.n_layer; ++L) {
        const std::string suffix = "_layer" + std::to_string(L);
        if (batch.gdn_state[L]) {
            auto * state = batch.gdn_state[L], * history = batch.gdn_history[L];
            read_lane(state, (size_t) slot * state->nb[3], state->nb[3], "GDN" + suffix);
            read_lane(history, (size_t) slot * history->nb[2], history->nb[2], "convolution" + suffix);
        } else {
            for (int h = 0; h < (int) batch.model->hp.n_head_kv; ++h) {
                for (int kv = 0; kv < 2; ++kv) {
                    auto * cache = kv ? batch.v_cache[L] : batch.k_cache[L];
                    const size_t offset = (size_t) h * cache->nb[2] +
                                          (size_t) slot * (batch.slot_cap + 1) * cache->nb[1];
                    read_lane(cache, offset, (size_t) rows * cache->nb[1],
                              (kv ? "V" : "K") + suffix + "_head" + std::to_string(h));
                }
            }
        }
    }
    return result;
}

static bool same_state(const std::vector<state_lane> & a, const std::vector<state_lane> & b,
                       const char * label, bool exact) {
    if (a.size() != b.size()) return false;
    bool ok = true;
    for (size_t lane = 0; lane < a.size(); ++lane) {
        if (a[lane].type != b[lane].type || a[lane].bytes.size() != b[lane].bytes.size()) return false;
        const std::string name = std::string(label) + "/" + a[lane].name;
        if (exact) {
            if (a[lane].bytes != b[lane].bytes) {
                fprintf(stderr, "%s: inactive state changed\n", name.c_str());
                ok = false;
            }
            continue;
        }
        const auto type = a[lane].type;
        if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16) return false;
        const size_t count = a[lane].bytes.size() / ggml_type_size(type);
        std::vector<float> af(count), bf(count);
        for (size_t i = 0; i < count; ++i) {
            if (type == GGML_TYPE_F32) {
                std::memcpy(&af[i], a[lane].bytes.data() + i * sizeof(float), sizeof(float));
                std::memcpy(&bf[i], b[lane].bytes.data() + i * sizeof(float), sizeof(float));
            } else {
                ggml_fp16_t ah, bh;
                std::memcpy(&ah, a[lane].bytes.data() + i * sizeof(ah), sizeof(ah));
                std::memcpy(&bh, b[lane].bytes.data() + i * sizeof(bh), sizeof(bh));
                af[i] = ggml_fp16_to_fp32(ah); bf[i] = ggml_fp16_to_fp32(bh);
            }
        }
        ok = close_logits(af, bf, name.c_str()) && ok;
    }
    return ok;
}

static bool teacher_forced(zonos3_model & model) {
    const int ncb = (int) model.hp.n_codebooks, W = ncb + 1;
    const size_t stride = (size_t) ncb * model.hp.audio_vocab;
    bool has_gdn = false, has_global = false;
    for (const auto & layer : model.layers) {
        has_gdn |= layer.is_gdn;
        has_global |= !layer.is_gdn;
    }
    if (!has_gdn || !has_global) {
        fprintf(stderr, "state regression requires both GDN and global layers\n"); return false;
    }
    std::vector<std::vector<float>> prompts(2);
    zonos3_prompt prompt_meta[2];
    int lengths[2] = {};
    for (int slot = 0; slot < 2; ++slot) {
        const int frames = slot ? 11 : 300; // Unequal prefill, crossing the 256-row chunk boundary.
        std::vector<int32_t> codes((size_t) frames * ncb);
        for (size_t i = 0; i < codes.size(); ++i)
            codes[i] = (int32_t) ((i * 17 + slot * 31) % model.hp.codebook_size);
        zonos3_prompt_options options;
        options.reference_codes = codes.data();
        options.reference_frames = frames;
        options.condition_buckets = {(int32_t) (2 + slot), 5, 10};
        if (!zonos3_build_prompt(model, slot ? "short" : "long unequal reference", options, prompt_meta[slot])) return false;
        lengths[slot] = prompt_meta[slot].n_rows;
        prompts[slot].assign(prompt_meta[slot].ids.begin(), prompt_meta[slot].ids.end());
    }
    zonos3_batch_ctx batch, isolated[2];
    const int capacity = std::max(lengths[0], lengths[1]) + 40;
    bool ok = zonos3_batch_init(batch, model, 2, capacity);
    for (int i = 0; ok && i < 2; ++i) ok = zonos3_batch_init(isolated[i], model, 1, capacity);
    zonos3_slot slots[2], singles[2];
    std::vector<float> got(stride), expected(stride);
    for (int i = 0; ok && i < 2; ++i) {
        slots[i].index = i; singles[i].index = 0;
        slots[i].active = singles[i].active = true;
        slots[i].n_past = singles[i].n_past = lengths[i];
        ok = zonos3_batch_slot_prefill(batch, i, prompts[i].data(), lengths[i], prompt_meta[i].cond, got.data()) &&
             zonos3_batch_slot_prefill(isolated[i], 0, prompts[i].data(), lengths[i], prompt_meta[i].cond, expected.data()) &&
             close_logits(got, expected, "unequal prefill logits") &&
             same_state(snapshot(batch, i, lengths[i]), snapshot(isolated[i], 0, lengths[i]), "unequal prefill state", false);
    }
    // Fixed ordinary rows are shared with the isolated oracle regardless of its
    // argmax. Pause each slot in turn; slot 1 alone still executes the idle slot 0
    // column, while slot 0 alone shrinks the graph ladder to width 1.
    for (int step = 0; ok && step < 32; ++step) {
        const int paused = step >= 8 && step < 16 ? 0 : step >= 20 && step < 24 ? 1 : -1;
        const auto before = paused >= 0 ? snapshot(batch, paused, slots[paused].n_past) : std::vector<state_lane>{};
        std::vector<float> paused_logits;
        int paused_position = -1;
        if (paused >= 0) {
            const float * logits = zonos3_batch_slot_logits(batch, paused);
            paused_logits.assign(logits, logits + stride);
            paused_position = slots[paused].n_past;
        }
        std::vector<zonos3_slot *> active;
        for (int i = 0; i < 2; ++i) if (i != paused) {
            slots[i].next_ids.resize(W);
            for (int cb = 0; cb < ncb; ++cb)
                slots[i].next_ids[cb] = (int32_t) ((step * 113 + cb * 7 + i * 19) % model.hp.codebook_size);
            slots[i].next_ids[ncb] = (int32_t) model.hp.text_vocab;
            singles[i].next_ids = slots[i].next_ids;
            active.push_back(&slots[i]);
        }
        ok = zonos3_batch_step(batch, active);
        if (ok && paused >= 0) {
            ok = same_state(before, snapshot(batch, paused, slots[paused].n_past), "paused", true) &&
                 paused_position == slots[paused].n_past &&
                 std::memcmp(paused_logits.data(), zonos3_batch_slot_logits(batch, paused), stride * sizeof(float)) == 0;
            if (!ok) fprintf(stderr, "paused slot position, logits, or persistent state changed\n");
        }
        for (int i = 0; ok && i < 2; ++i) if (i != paused) {
            ok = zonos3_batch_step(isolated[i], {&singles[i]});
            if (!ok) break;
            std::copy_n(zonos3_batch_slot_logits(batch, i), stride, got.begin());
            std::copy_n(zonos3_batch_slot_logits(isolated[i], 0), stride, expected.begin());
            const std::string label = "teacher row " + std::to_string(step) + " slot " + std::to_string(i);
            // Capture/report every layer's live state before applying the logits
            // gate, so a failure identifies the earliest recurrent/cache drift.
            const bool state_ok = same_state(snapshot(batch, i, slots[i].n_past),
                                             snapshot(isolated[i], 0, singles[i].n_past), label.c_str(), false);
            const bool logits_ok = close_logits(got, expected, label.c_str());
            ok = state_ok && logits_ok;
        }
    }
    // Reuse the long-reference slot for the other request; compare with a newly
    // reset isolated context, including persistent state and subsequent decode.
    if (ok) {
        const auto preserved = snapshot(batch, 1, slots[1].n_past);
        ok = zonos3_batch_slot_prefill(batch, 0, prompts[1].data(), lengths[1], prompt_meta[1].cond, got.data()) &&
             zonos3_batch_slot_prefill(isolated[0], 0, prompts[1].data(), lengths[1], prompt_meta[1].cond, expected.data()) &&
             close_logits(got, expected, "reuse logits") &&
             same_state(snapshot(batch, 0, lengths[1]), snapshot(isolated[0], 0, lengths[1]), "reuse state", false) &&
             same_state(preserved, snapshot(batch, 1, slots[1].n_past), "reuse neighbor", true);
        slots[0].n_past = singles[0].n_past = lengths[1];
        slots[0].next_ids = singles[0].next_ids = std::vector<int32_t>(W, 1);
        slots[0].next_ids[ncb] = singles[0].next_ids[ncb] = (int32_t) model.hp.text_vocab;
        if (ok) ok = zonos3_batch_step(batch, {&slots[0]}) && zonos3_batch_step(isolated[0], {&singles[0]});
        if (ok) {
            std::copy_n(zonos3_batch_slot_logits(batch, 0), stride, got.begin());
            std::copy_n(zonos3_batch_slot_logits(isolated[0], 0), stride, expected.begin());
            ok = close_logits(got, expected, "reuse continuation") &&
                 same_state(snapshot(batch, 0, slots[0].n_past), snapshot(isolated[0], 0, singles[0].n_past), "reuse continuation", false) &&
                 same_state(preserved, snapshot(batch, 1, slots[1].n_past), "reuse continuation neighbor", true);
        }
    }
    zonos3_batch_free(batch);
    for (auto & context : isolated) zonos3_batch_free(context);
    return ok;
}

static bool all_pad_row(const zonos3_model & model, const zonos3_prompt & p, int row) {
    const int ncb = (int) model.hp.n_codebooks, W = ncb + 1;
    for (int cb = 0; cb < ncb; ++cb) if (p.ids[(size_t) row * W + cb] != (int32_t) model.hp.audio_pad_id) return false;
    return p.ids[(size_t) row * W + ncb] == (int32_t) model.hp.text_vocab;
}

static bool rows_equal(const zonos3_prompt & a, int a0, const zonos3_prompt & b, int b0, int count, int W) {
    return std::equal(a.ids.begin() + (size_t) a0 * W, a.ids.begin() + (size_t) (a0 + count) * W, b.ids.begin() + (size_t) b0 * W);
}

#define REQUIRE(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); return false; } } while (0)

// Layout, bucket mapping, option validation, coverage check and CFG twin (no numerics).
static bool prompt_contract(zonos3_model & model, const std::vector<float> & speaker) {
    const auto & hp = model.hp;
    const int ncb = (int) hp.n_codebooks, W = ncb + 1, frames = 3, ncond = (int) hp.n_cond;
    const int unknown = (int) hp.n_cond_buckets;
    REQUIRE(ncb == 10 && hp.codebook_size >= 32 && ncond == 3 && speaker.size() == hp.speaker_dim, "model geometry");
    const std::string text = "raw reference test";
    const auto tokens = zonos3_tokenize(model.tok, text);
    std::vector<int32_t> codes(frames * ncb);
    for (int f = 0; f < frames; ++f)
        for (int cb = 0; cb < ncb; ++cb) codes[f * ncb + cb] = f * ncb + cb;

    // (1a) unconditioned prompt: [3 condition rows] text [3 silence rows]
    zonos3_prompt plain, plain0;
    zonos3_prompt_options o;
    REQUIRE(zonos3_build_prompt(model, text, o, plain), "plain build");
    o.silence_prefill_frames = 0;
    REQUIRE(zonos3_build_prompt(model, text, o, plain0), "plain build without silence");
    const int text_rows = (int) tokens.size();
    const zonos3_generation_registry & limits = zonos3_generation();
    REQUIRE(plain0.n_rows == ncond + text_rows && plain.n_rows == plain0.n_rows + limits.silence_prefill_default_frames, "plain row counts");
    REQUIRE(plain.ids.size() == (size_t) plain.n_rows * W && plain.speaker_rows == 0 &&
            plain.prefill_frames == limits.silence_prefill_default_frames, "plain metadata");
    REQUIRE(plain0.prefill_frames == 0, "silence 0 disables prefill");
    REQUIRE(plain.cond.speaker_row == -1 && plain.cond.condition_row == 0 && plain.cond.buckets.size() == 3, "plain cond");
    for (int b : plain.cond.buckets) REQUIRE(b == unknown, "plain buckets are unknown");
    for (int r = 0; r < ncond; ++r) REQUIRE(all_pad_row(model, plain, r), "condition rows are all-pad");
    for (int t = 0; t < text_rows; ++t) {
        for (int cb = 0; cb < ncb; ++cb) REQUIRE(plain.ids[(size_t) (ncond + t) * W + cb] == (int32_t) hp.audio_pad_id, "text rows have pad audio");
        REQUIRE(plain.ids[(size_t) (ncond + t) * W + ncb] == tokens[t], "text token");
    }
    // silence rows: sheared copies of the silence frame, no flush tail, text column text_vocab
    for (int k = 0; k < plain.prefill_frames; ++k) {
        const int row = ncond + text_rows + k;
        for (int cb = 0; cb < ncb; ++cb) {
            const int32_t expected = k - cb >= 0 ? hp.silence_frame[cb] : (int32_t) hp.audio_pad_id;
            REQUIRE(plain.ids[(size_t) row * W + cb] == expected, "silence row shear");
        }
        REQUIRE(plain.ids[(size_t) row * W + ncb] == (int32_t) hp.text_vocab, "silence text column");
    }
    REQUIRE(rows_equal(plain, 0, plain0, 0, plain0.n_rows, W), "silence rows are appended after EOS only");
    const auto pcodes = zonos3_prefill_codes(model, plain);
    REQUIRE(pcodes.size() == (size_t) 3 * ncb, "prefill code count");
    for (int k = 0; k < 3; ++k) for (int cb = 0; cb < ncb; ++cb)
        REQUIRE(pcodes[k * ncb + cb] == plain.ids[(size_t) (plain.n_rows - 3 + k) * W + cb], "prefill codes");
    REQUIRE(zonos3_prefill_codes(model, plain0).empty(), "no prefill codes");

    // (1b) embedding prompt: [speaker row][conditions] text [silence]
    zonos3_prompt emb;
    o = zonos3_prompt_options();
    o.speaker_embedding = speaker.data();
    o.condition_buckets = {3, 5, 10};
    REQUIRE(zonos3_build_prompt(model, text, o, emb), "embedding build");
    REQUIRE(emb.n_rows == plain.n_rows + 1 && emb.speaker_rows == 1 && emb.prefill_frames == 3, "embedding row counts");
    REQUIRE(emb.cond.speaker_row == 0 && emb.cond.condition_row == 1 && emb.cond.speaker == speaker, "embedding cond");
    REQUIRE((emb.cond.buckets == std::vector<int32_t>{3, 5, 10}), "embedding buckets");
    REQUIRE(all_pad_row(model, emb, 0), "speaker row is all-pad");
    REQUIRE(rows_equal(emb, 1, plain, 0, plain.n_rows, W), "rows after the speaker row equal the plain prompt");

    // (1c) reference prompt: delayed prefix unchanged, then conditions, text, silence
    zonos3_prompt ref;
    o = zonos3_prompt_options();
    o.reference_codes = codes.data(); o.reference_frames = frames;
    REQUIRE(zonos3_build_prompt(model, text, o, ref), "reference build");
    const int prefix_rows = frames + ncb - 1;
    REQUIRE(ref.n_rows == plain.n_rows + prefix_rows && ref.speaker_rows == prefix_rows, "reference row counts");
    REQUIRE(ref.cond.speaker_row == -1 && ref.cond.condition_row == prefix_rows && ref.cond.speaker.empty(), "reference cond");
    for (int t = 0; t < prefix_rows; ++t) {
        for (int cb = 0; cb < ncb; ++cb) {
            const int f = t - cb;
            const int32_t expected = f >= 0 && f < frames ? codes[f * ncb + cb] : (int32_t) hp.audio_pad_id;
            REQUIRE(ref.ids[(size_t) t * W + cb] == expected, "raw prefix mismatch");
        }
        REQUIRE(ref.ids[(size_t) t * W + ncb] == (int32_t) hp.text_vocab, "raw prefix text column");
    }
    REQUIRE(rows_equal(ref, prefix_rows, plain, 0, plain.n_rows, W), "rows after the reference prefix equal the plain prompt");

    // (2) bucket mapping is bisect_right
    const int fmax = zonos3_condition_index(model, "fmax"), loud = zonos3_condition_index(model, "loudness");
    REQUIRE(fmax == 1 && loud == 0 && zonos3_condition_index(model, "speaking_rate") == 2 &&
            zonos3_condition_index(model, "pitch") == -1, "condition index");
    REQUIRE(zonos3_condition_bucket(model, fmax, 8000.0) == 2, "fmax 8000.0 -> 2");
    REQUIRE(zonos3_condition_bucket(model, fmax, 7999.9) == 1, "fmax 7999.9 -> 1");
    REQUIRE(zonos3_condition_bucket(model, fmax, 7000.0) == 0, "below first edge -> 0");
    // Edges are stored as float32 in the GGUF; the last stored edge itself maps to the last bucket.
    REQUIRE(zonos3_condition_bucket(model, fmax, (double) hp.cond_edges[fmax].back()) == 9 &&
            zonos3_condition_bucket(model, fmax, 1e9) == 9, "last edge -> 9");
    REQUIRE(zonos3_condition_bucket(model, loud, -23.0) == 4, "loudness -23 -> 4");
    REQUIRE(zonos3_condition_bucket(model, fmax, std::nan("")) == -1 &&
            zonos3_condition_bucket(model, fmax, INFINITY) == -1 &&
            zonos3_condition_bucket(model, -1, 1.0) == -1, "non-finite / bad index -> -1");

    // (3) embedding and reference are mutually exclusive; option validation
    zonos3_prompt bad;
    o = zonos3_prompt_options();
    o.speaker_embedding = speaker.data(); o.reference_codes = codes.data(); o.reference_frames = frames;
    REQUIRE(!zonos3_build_prompt(model, text, o, bad), "embedding + reference must be rejected");
    o = zonos3_prompt_options(); o.condition_buckets = {1, 2};
    REQUIRE(!zonos3_build_prompt(model, text, o, bad), "bucket count");
    o.condition_buckets = {1, 2, unknown + 1};
    REQUIRE(!zonos3_build_prompt(model, text, o, bad), "bucket range");
    o.condition_buckets = {1, -1, 2};
    REQUIRE(!zonos3_build_prompt(model, text, o, bad), "negative bucket");
    o = zonos3_prompt_options(); o.silence_prefill_frames = limits.silence_prefill_max_frames + 1;
    REQUIRE(!zonos3_build_prompt(model, text, o, bad), "silence prefill upper bound");
    o.silence_prefill_frames = -1;
    REQUIRE(!zonos3_build_prompt(model, text, o, bad), "silence prefill lower bound");
    o.silence_prefill_frames = limits.silence_prefill_max_frames;
    REQUIRE(zonos3_build_prompt(model, text, o, bad), "silence prefill max accepted");
    std::vector<float> nonfinite = speaker;
    nonfinite[7] = INFINITY;
    o = zonos3_prompt_options(); o.speaker_embedding = nonfinite.data();
    REQUIRE(!zonos3_build_prompt(model, text, o, bad), "non-finite embedding");

    // reference-code validation (non-ordinary codes, inconsistent pointer/length, the builder's reference_max_frames
    // bound: request resolvers truncate before it)
    o = zonos3_prompt_options();
    o.reference_codes = codes.data(); o.reference_frames = frames;
    codes[0] = (int32_t) hp.codebook_size;
    REQUIRE(!zonos3_build_prompt(model, "test", o, bad), "non-ordinary code");
    codes[0] = 0;
    o.reference_codes = nullptr;
    REQUIRE(!zonos3_build_prompt(model, "test", o, bad), "null reference with frames");
    o.reference_codes = codes.data(); o.reference_frames = -1;
    REQUIRE(!zonos3_build_prompt(model, "test", o, bad), "negative frames");
    const int max_ref = limits.reference_max_frames;
    std::vector<int32_t> boundary((size_t) (max_ref + 1) * ncb, 0);
    o.reference_codes = boundary.data(); o.reference_frames = max_ref;
    const auto saved_context = model.hp.n_ctx_train;
    model.hp.n_ctx_train = std::max(saved_context, (uint32_t) 4096);
    const bool accepts_max = zonos3_build_prompt(model, "boundary", o, bad);
    o.reference_frames = max_ref + 1;
    const bool rejects_above = !zonos3_build_prompt(model, "boundary", o, bad) && bad.n_rows == 0;
    model.hp.n_ctx_train = saved_context;
    REQUIRE(accepts_max && rejects_above, "reference_max_frames prefix bound");

    // (4) coverage contract
    std::vector<float> ids(emb.ids.begin(), emb.ids.end());
    REQUIRE(zonos3_check_conditioning(model, ids.data(), emb.n_rows, emb.cond), "valid embedding conditioning accepted");
    zonos3_conditioning uncovered = emb.cond;
    uncovered.speaker_row = -1;
    REQUIRE(!zonos3_check_conditioning(model, ids.data(), emb.n_rows, uncovered), "uncovered all-pad speaker row must be rejected");
    zonos3_conditioning no_cond = emb.cond;
    no_cond.condition_row = -1;
    REQUIRE(!zonos3_check_conditioning(model, ids.data(), emb.n_rows, no_cond), "uncovered condition rows must be rejected");
    zonos3_conditioning wrong_row = emb.cond;
    wrong_row.condition_row = emb.n_rows - 4; // rows are text/silence, not all-pad
    REQUIRE(!zonos3_check_conditioning(model, ids.data(), emb.n_rows, wrong_row), "non-pad conditioning row must be rejected");
    zonos3_conditioning short_speaker = emb.cond;
    short_speaker.speaker.pop_back();
    REQUIRE(!zonos3_check_conditioning(model, ids.data(), emb.n_rows, short_speaker), "speaker size");
    REQUIRE(!zonos3_check_conditioning(model, ids.data(), emb.n_rows, zonos3_conditioning{}), "empty conditioning on a Mini prompt");

    // (7) CFG twin
    zonos3_prompt twin;
    REQUIRE(zonos3_build_cfg_twin(model, emb, twin), "embedding twin");
    REQUIRE(twin.n_rows == emb.n_rows - 1 && twin.speaker_rows == 0 && twin.prefill_frames == emb.prefill_frames, "twin geometry");
    REQUIRE(twin.cond.speaker_row == -1 && twin.cond.condition_row == 0 && twin.cond.speaker.empty(), "twin cond rows");
    for (int b : twin.cond.buckets) REQUIRE(b == unknown, "twin buckets unknown");
    REQUIRE(twin.cond.buckets.size() == 3 && rows_equal(twin, 0, emb, 1, twin.n_rows, W), "twin keeps conditions, text and silence rows");
    std::vector<float> tids(twin.ids.begin(), twin.ids.end());
    REQUIRE(zonos3_check_conditioning(model, tids.data(), twin.n_rows, twin.cond), "twin conditioning is coverable");
    zonos3_prompt rtwin;
    REQUIRE(zonos3_build_cfg_twin(model, ref, rtwin), "reference twin");
    REQUIRE(rtwin.n_rows == ref.n_rows - prefix_rows && rows_equal(rtwin, 0, plain, 0, plain.n_rows, W), "reference twin drops exactly the prefix");
    REQUIRE(!zonos3_build_cfg_twin(model, plain, twin), "CFG without a speaker prompt must be rejected");

    // (8) tokenizer spec: pronunciation groups (superbpe.payload_ids) and edit marker rows (edit.edit_marker_rows)
    const std::string group_text = "Say \xEE\x80\x80t\xC9\x99\xEE\x80\x81 again.";   // Say U+E000 tə U+E001 again.
    std::string err;
    REQUIRE(!model.tok.has_phonemes && !model.tok.spec_json.empty(), "the dummy spec has no phonemes");
    REQUIRE(zonos3_check_phoneme_groups(model.tok, group_text) ==
            "This checkpoint has no phoneme tokens: remove the U+E000 / U+E001 pronunciation groups", "groups need phonemes");
    REQUIRE(!zonos3_build_prompt(model, group_text, zonos3_prompt_options{}, bad, &err) &&
            err.rfind("This checkpoint has no phoneme", 0) == 0,
            "build_prompt reports the group error");
    const std::string spec_base = R"({"kind": "superbpe", "vocab_size": 8192, "offset": 4, "bos": 2, "eos": 3)";
    // two edit marker rows, then open/close and the phones t, ə, ˈ
    const std::string spec = spec_base + R"(, "phonemes": {"open_marker": "\ue000", "close_marker": "\ue001",
        "open_id": 8199, "close_id": 8200, "symbols": ["t", "\u0259", "\u02c8"]}})";
    zonos3_tokenizer tok = model.tok;
    REQUIRE(!zonos3_tokenizer_set_spec(tok, spec, 8196, 6, err) && err.find("text_extra_tokens") != std::string::npos,
            "phoneme block must end the extra rows");
    REQUIRE(zonos3_tokenizer_set_spec(tok, spec, 8196, 7, err) && tok.has_phonemes, "phoneme spec accepted");
    std::vector<int32_t> got, say, again;
    REQUIRE(zonos3_text_ids(tok, group_text, got, err), "valid group");
    REQUIRE(zonos3_payload_ids(tok, "Say ", say, err) && zonos3_payload_ids(tok, " again.", again, err), "plain stretches");
    std::vector<int32_t> want{2};
    want.insert(want.end(), say.begin(), say.end());
    for (int32_t id : {8199, 8201, 8202, 8200}) want.push_back(id);
    want.insert(want.end(), again.begin(), again.end());
    want.push_back(3);
    REQUIRE(got == want, "group ids: open, open+2+index per phone, close; BPE between groups");
    REQUIRE(zonos3_check_phoneme_groups(tok, "a \xEE\x80\x80\xEE\x80\x81 b") ==
            "Pronunciation groups: Unbalanced or empty phoneme group", "empty group");
    REQUIRE(zonos3_check_phoneme_groups(tok, "a \xEE\x80\x81 b") == "Pronunciation groups: Unbalanced phoneme group markers",
            "stray close marker");
    REQUIRE(zonos3_check_phoneme_groups(tok, "\xEE\x80\x80xtx'\xEE\x80\x81") ==
            "Pronunciation groups: Phones outside the inventory: [\"'\", 'x']", "unknown phones, sorted, Python repr");
    REQUIRE(zonos3_check_phoneme_groups(tok, "no groups") == "", "no groups");
    REQUIRE(!zonos3_tokenizer_set_spec(tok, spec_base + R"(, "phonemes": {"open_marker": "\ue000", "close_marker": "\ue001",
        "open_id": 8196, "close_id": 8197, "symbols": ["t"]}})", 8196, 3, err), "phoneme ids must follow the padding row");
    REQUIRE(!zonos3_tokenizer_set_spec(tok, R"({"kind": "superbpe", "vocab_size": 8000, "offset": 4, "bos": 2, "eos": 3})",
                                       8196, 0, err) && err == "SuperBPE ID convention mismatch", "vocab convention");
    const zonos3_tokenizer saved_tok = model.tok;
    const uint32_t saved_extra = model.hp.text_extra_tokens;
    REQUIRE(zonos3_tokenizer_set_spec(model.tok, spec, 8196, 7, err), "phoneme spec on the model");
    model.hp.text_extra_tokens = 7;
    const int markers = zonos3_edit_marker_rows(model);
    model.hp.text_extra_tokens = 4;
    const int markers_plain = (model.tok = saved_tok, zonos3_edit_marker_rows(model));
    model.hp.text_extra_tokens = saved_extra;
    REQUIRE(markers == 2 && markers_plain == 4 && zonos3_edit_marker_id(model, 1) == 8198, "edit marker rows / ids");
    REQUIRE(!zonos3_supports_editing(model), "the dummy has no span_end_id");
    return true;
}

// Teacher-forced numerical continuation for one prompt: cached single-shot, chunked prefills split at
// the given absolute rows (with conditioning rows split across chunks), slot isolation and reset.
static bool continuation(const zonos3_model & model, const zonos3_prompt & p, const zonos3_prompt & plain,
                         const char * label, std::vector<int> splits) {
    const int ncb = (int) model.hp.n_codebooks, W = ncb + 1, rows = p.n_rows;
    const size_t stride = (size_t) ncb * model.hp.audio_vocab;
    const std::vector<float> ids(p.ids.begin(), p.ids.end()), plain_ids(plain.ids.begin(), plain.ids.end());
    std::vector<float> full;
    if (!zonos3_logits(model, ids.data(), rows, p.cond, full) || full.size() != (size_t) rows * stride) return false;
    const std::vector<float> uncached(full.end() - stride, full.end());
    zonos3_batch_ctx batch;
    if (!zonos3_batch_init(batch, model, 2, rows + 16)) return false;
    const bool full_f32 = !model.tensors.empty() && std::all_of(model.tensors.begin(), model.tensors.end(),
        [](const std::pair<const std::string, ggml_tensor *> & weight) { return weight.second->type == GGML_TYPE_F32; });
    const ggml_type expected_kv_type = model.is_gpu && !full_f32 ? GGML_TYPE_F16 : GGML_TYPE_F32;
    bool cache_type_ok = true;
    for (size_t L = 0; L < batch.k_cache.size(); ++L) if (batch.k_cache[L]) {
        cache_type_ok = cache_type_ok && batch.k_cache[L]->type == expected_kv_type &&
                        batch.v_cache[L]->type == expected_kv_type;
    }
    if (!cache_type_ok) fprintf(stderr, "KV cache dtype does not match the complete weight precision profile\n");
    std::vector<float> expected(stride), got(stride), other(stride), preserved(stride);
    // A cold single-chunk graph is the canonical continuation oracle. GPU F16
    // and quantized profiles store KV as F16; the entirely-F32 diagnostic model
    // and CPU retain F32 KV. Only the F16 storage boundary differs from uncached
    // zonos3_logits, which always retains F32 K/V.
    bool ok = cache_type_ok && zonos3_batch_slot_extend(batch, 1, ids.data(), rows, 0, p.cond, expected.data());
    const auto expected_state = ok ? snapshot(batch, 1, rows) : std::vector<state_lane>{};
    const std::string name(label);
    if (ok && expected_kv_type == GGML_TYPE_F32)
        ok = close_logits(uncached, expected, (name + " uncached vs F32 cached attention").c_str());
    if (ok && expected_kv_type == GGML_TYPE_F16) {
        float maximum = 0;
        double squared = 0;
        for (size_t i = 0; i < stride; ++i) {
            if (!std::isfinite(uncached[i]) || !std::isfinite(expected[i])) ok = false;
            const float difference = std::abs(uncached[i] - expected[i]);
            maximum = std::max(maximum, difference);
            squared += (double) difference * difference;
        }
        printf("%s F32 uncached vs F16 KV storage diagnostic: max_abs=%.9g rmse=%.9g "
               "(different arithmetic; not a continuation equivalence claim)\n",
               label, maximum, std::sqrt(squared / stride));
    }
    std::sort(splits.begin(), splits.end());
    splits.erase(std::unique(splits.begin(), splits.end()), splits.end());
    for (const int split : splits) {
        if (!ok) break;
        if (split <= 0 || split >= rows) continue;
        // Fresh context per split: extend continues state in place and never resets a slot.
        zonos3_batch_ctx chunked;
        ok = zonos3_batch_init(chunked, model, 1, rows + 16) &&
             zonos3_batch_slot_extend(chunked, 0, ids.data(), split, 0, p.cond, got.data());
        for (int position = split; ok && position < rows;) {
            const int count = std::min(3, rows - position);
            ok = zonos3_batch_slot_extend(chunked, 0, ids.data() + (size_t) position * W, count, position, p.cond, got.data());
            position += count;
        }
        if (ok) {
            const std::string tag = name + " split " + std::to_string(split);
            const bool state_ok = same_state(expected_state, snapshot(chunked, 0, rows), (tag + " state").c_str(), false);
            const bool logits_ok = close_logits(expected, got, (tag + " logits").c_str());
            ok = state_ok && logits_ok;
        }
        zonos3_batch_free(chunked);
    }
    // Filling another slot must not mutate this slot's GDN, convolution or global KV state.
    if (ok) {
        ok = zonos3_batch_slot_prefill(batch, 0, ids.data(), rows, p.cond, got.data());
        std::copy_n(zonos3_batch_slot_logits(batch, 0), stride, preserved.begin());
        ok = ok && close_logits(expected, got, (name + " single-shot prefill vs extend").c_str());
        ok = ok && zonos3_batch_slot_prefill(batch, 1, plain_ids.data(), plain.n_rows, plain.cond, other.data());
        std::copy_n(zonos3_batch_slot_logits(batch, 0), stride, got.begin());
        ok = ok && close_logits(preserved, got, (name + " slot isolation").c_str());
    }
    // Prefill on a reused slot resets every state lane, including prior longer prompts.
    if (ok) {
        ok = zonos3_batch_slot_prefill(batch, 0, plain_ids.data(), plain.n_rows, plain.cond, got.data()) &&
             close_logits(other, got, (name + " slot reset").c_str());
    }
    zonos3_batch_free(batch);
    return ok;
}

static bool max_abs_diff_exceeds(const std::vector<float> & a, const std::vector<float> & b, float floor, const char * label) {
    float maximum = 0;
    for (size_t i = 0; i < a.size(); ++i) maximum = std::max(maximum, std::abs(a[i] - b[i]));
    printf("%s: last-row logits differ by max_abs=%.9g\n", label, maximum);
    return maximum > floor;
}

static bool exercise(zonos3_model & model, const std::vector<float> & speaker) {
    if (!prompt_contract(model, speaker)) return false;
    const int ncb = (int) model.hp.n_codebooks, frames = 3;
    std::vector<int32_t> codes(frames * ncb);
    for (int f = 0; f < frames; ++f)
        for (int cb = 0; cb < ncb; ++cb) codes[f * ncb + cb] = f * ncb + cb;
    zonos3_prompt plain, emb, ref;
    zonos3_prompt_options o;
    if (!zonos3_build_prompt(model, "raw reference test", o, plain)) return false;
    o.speaker_embedding = speaker.data();
    o.condition_buckets = {3, 5, 10};
    if (!zonos3_build_prompt(model, "raw reference test", o, emb)) return false;
    o = zonos3_prompt_options();
    o.reference_codes = codes.data(); o.reference_frames = frames;
    if (!zonos3_build_prompt(model, "raw reference test", o, ref)) return false;

    // (5) chunked == single-shot, with splits around the speaker/condition rows (absolute vs local rows).
    const int c = emb.cond.condition_row;
    if (!continuation(model, emb, plain, "embedding", {1, c, c + 1, c + 3, emb.n_rows - 1})) return false;
    const int rc = ref.cond.condition_row;
    if (!continuation(model, ref, plain, "reference", {1, 3, rc - 1, rc, rc + 1, rc + 2, rc + 3, ref.n_rows - 1})) return false;

    // (6) speaker vector and condition buckets steer the last-row logits.
    const size_t stride = (size_t) ncb * model.hp.audio_vocab;
    const auto last_row = [&](const zonos3_prompt & p, const zonos3_conditioning & cond) {
        std::vector<float> ids(p.ids.begin(), p.ids.end()), logits;
        if (!zonos3_logits(model, ids.data(), p.n_rows, cond, logits)) logits.clear();
        return logits.size() >= stride ? std::vector<float>(logits.end() - stride, logits.end()) : std::vector<float>{};
    };
    const auto base = last_row(emb, emb.cond);
    if (base.empty()) return false;
    zonos3_conditioning other_speaker = emb.cond;
    for (size_t i = 0; i < 16; ++i) other_speaker.speaker[i] = -other_speaker.speaker[i];
    const auto spk = last_row(emb, other_speaker);
    zonos3_conditioning other_bucket = emb.cond;
    other_bucket.buckets[0] = 4;
    const auto bkt = last_row(emb, other_bucket);
    if (spk.empty() || bkt.empty() || !max_abs_diff_exceeds(base, spk, 1e-7f, "speaker vector sensitivity") ||
        !max_abs_diff_exceeds(base, bkt, 1e-7f, "condition bucket sensitivity")) {
        fprintf(stderr, "FAIL: conditioning does not reach the logits\n");
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 3 || (argc == 3 && std::strcmp(argv[2], "--gpu") != 0)) {
        fprintf(stderr, "usage: raw-reference-state MODEL.gguf [--gpu]\n"); return 2;
    }
    zonos3_model model;
    if (!zonos3_model_load(model, argv[1], argc == 3)) return 1;
    if (argc == 3 && !model.is_gpu) {
        fprintf(stderr, "raw-reference-state: --gpu requested but no GPU backend loaded\n");
        zonos3_model_free(model); return 1;
    }
    std::vector<float> speaker(model.hp.speaker_dim);
    {
        std::mt19937 rng(0);
        std::normal_distribution<float> normal(0.0f, 0.35f);
        for (float & v : speaker) v = normal(rng);
    }
    const bool prompt_ok = exercise(model, speaker);
    const bool state_ok = teacher_forced(model);
    zonos3_model_free(model);
    const bool ok = prompt_ok && state_ok;
    if (!ok) fprintf(stderr, "raw-reference/state contract failed\n");
    return ok ? 0 : 1;
}
