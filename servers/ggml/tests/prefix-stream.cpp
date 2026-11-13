// Prefix streaming with early partial-frame decode (stream_codebooks): the ready rule of prefix_submit_ready
// (zonos3.tts.stream.PrefixDecoder.ready), the codec leaving missing residual codebooks out, the request fields, and
// whole prefix streams through the scheduler on the dummy backbone and codec.
//
// usage: prefix-stream <dummy-zonos3.gguf> <dummy-zac.gguf> [--gpu]   (fixtures: scripts/make-dummy-zac.py; --gpu
// also runs the codec checks on the GPU backend)
#include "server/generate.h"
#include "server/http.h"
#include "server/projects.h"   // ServerState's ProjectStore
#include "server/scheduler.h"
#include "server/state.h"
#include "zac.h"
#include "zonos3-generation.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CHECK(cond, label) do { if (!(cond)) { fprintf(stderr, "FAIL: %s (%s:%d)\n", label, __FILE__, __LINE__); exit(1); } } while (0)

using json = nlohmann::json;

// Not NCB: <windows.h> (through httplib.h) declares a typedef NCB (nb30.h), MSVC error C2377.
static constexpr int N_CB = 10, PAD = 4097, P = 3;   // the release geometry: 10 codebooks, audio pad 4097, prefill 3

// ------------------------------------------------------------------------------------------------- the ready rule

// Rows appended one at a time (semantic code `code0[r]`, default 5), prefix_submit_ready's arithmetic after each:
// the (rows, end) of every chunk handed to the codec.
struct Chunk { int rows, end; };
static std::vector<Chunk> run_rule(int k, bool first_only, int n_rows, int eos_frame = -1,
                                   const std::vector<std::pair<int, int>> & code0 = {},
                                   const zs::PrefixStreamSettings & ps = {}) {
    std::vector<int32_t> rows;
    std::vector<Chunk> out;
    int emitted = P;
    for (int H = 1; H <= n_rows; ++H) {
        std::vector<int32_t> row(N_CB, 7);
        row[0] = 5;
        for (const auto & [r, v] : code0) if (r == H - 1) row[0] = v;
        rows.insert(rows.end(), row.begin(), row.end());
        const int eos = eos_frame >= 0 && H > eos_frame ? eos_frame : -1;   // known once its row was sampled
        int complete = H - (N_CB - 1);
        if (eos >= 0) complete = std::min(complete, eos);
        const int ready = zs::prefix_ready(rows.data(), H, N_CB, PAD, complete, eos, emitted, P, k, first_only);
        const int end = zs::prefix_chunk_end(ps, ready, emitted, P);
        if (end < 0) continue;
        CHECK(end > emitted, "a chunk moves forward");
        out.push_back({H, end});
        emitted = end;
    }
    return out;
}

// The complete-frame rule before early decode: decodable = complete - hold_back, need ramps.
static std::vector<Chunk> run_complete(int n_rows, int eos_frame, const zs::PrefixStreamSettings & ps) {
    std::vector<Chunk> out;
    int emitted = P;
    for (int H = 1; H <= n_rows; ++H) {
        int complete = H - (N_CB - 1);
        if (eos_frame >= 0 && H > eos_frame) complete = std::min(complete, eos_frame);
        const int decodable = complete - ps.hold_back_frames;
        const int need = std::min(ps.chunk_frames, ps.first_chunk_frames + emitted - P);
        if (decodable - emitted < need) continue;
        out.push_back({H, decodable});
        emitted = decodable;
    }
    return out;
}

static bool same(const std::vector<Chunk> & a, const std::vector<Chunk> & b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) if (a[i].rows != b[i].rows || a[i].end != b[i].end) return false;
    return true;
}

static void test_ready_rule() {
    // First chunk: one generated row (k=1) instead of ncb (complete frames).
    auto c1 = run_rule(1, true, 60);
    CHECK(!c1.empty() && c1[0].rows == P + 1 && c1[0].end == P + 1, "k=1: the first chunk after the first generated row");
    auto c3 = run_rule(3, true, 60);
    CHECK(!c3.empty() && c3[0].rows == P + 3 && c3[0].end == P + 1, "k=3: the first chunk after three generated rows");
    auto c10 = run_rule(N_CB, true, 60);
    CHECK(!c10.empty() && c10[0].rows == P + N_CB && c10[0].end == P + 1, "k=ncb: the first chunk once the frame is complete");

    // Scope first: the chunks after the first wait for complete frames (frame P + 1 complete at row P + 1 + N_CB, need 2).
    CHECK(c1.size() >= 2 && c1[1].rows == P + 2 + N_CB - 1 + 1 && c1[1].end == P + 3, "scope first: the second chunk is complete");
    for (size_t i = 1; i < c1.size(); ++i) CHECK(c1[i].end <= c1[i].rows - (N_CB - 1), "scope first: later chunks complete");
    // Scope all: every chunk early (need 2 -> two more generated rows).
    auto all = run_rule(1, false, 60);
    CHECK(all.size() >= 2 && all[1].rows == P + 3 && all[1].end == P + 3, "scope all: the second chunk early too");

    // k = ncb is the complete-frame rule, chunk for chunk (also with a hold-back and an EOS).
    zs::PrefixStreamSettings hold;
    hold.hold_back_frames = 2;
    for (bool first : {true, false}) {
        CHECK(same(run_rule(N_CB, first, 80), run_complete(80, -1, {})), "k=ncb: unchanged ramp");
        CHECK(same(run_rule(N_CB, first, 80, 30, {}, hold), run_complete(80, 30, hold)), "k=ncb: unchanged with hold-back/EOS");
    }

    // EOS caps the early frames: EOS at frame 4 (its row sampled), k=1 never decodes past it.
    auto eos = run_rule(1, false, 40, 4, {{4, PAD - 1}});
    CHECK(!eos.empty() && eos.back().end <= 4, "k=1: capped at the EOS frame");
    // A semantic EOA / padding row is not a frame: the first generated row is EOA -> nothing early.
    auto eoa = run_rule(1, true, P + 1, -1, {{P, PAD - 1}});
    CHECK(eoa.empty(), "k=1: no chunk ending on an EOA row");
    auto pad = run_rule(1, true, P + 2, -1, {{P + 1, PAD}});
    CHECK(pad.size() == 1 && pad[0].end == P + 1, "k=1: the walk-back stops before a padding row");
}

// --------------------------------------------------------------------------------------------------- the codec

static std::vector<int32_t> lcg_codes(int rows) {
    std::vector<int32_t> c((size_t) rows * N_CB);
    uint32_t x = 12345;
    for (auto & v : c) { x = x * 1664525u + 1013904223u; v = (int32_t) ((x >> 8) % 4096); }
    return c;
}

static std::vector<float> stream(const zac_model & m, const std::vector<int32_t> & codes, int H, std::vector<int> ends,
                                 bool * ok = nullptr) {
    std::vector<float> pcm;
    zac_stream st;
    bool good = true;
    for (int e : ends) good = good && zac_stream_decode(m, st, codes.data(), H, e, pcm);
    good = good && zac_stream_flush(m, st, pcm);
    if (ok) *ok = good;
    else CHECK(good, "stream decode");
    return pcm;
}

static double max_diff(const std::vector<float> & a, const std::vector<float> & b, size_t n) {
    double d = 0;
    for (size_t i = 0; i < n; ++i) d = std::max(d, std::fabs((double) a[i] - (double) b[i]));
    return d;
}

static void test_codec(const char * zac_path, bool gpu) {
    zac_model m;
    CHECK(zac_load(m, zac_path, gpu), "load the dummy codec");
    CHECK(zac_partial_frames(m), "the dummy codec has res.{i}.bias");
    const int n = 12, L = m.latent_dim;
    const std::vector<int32_t> codes = lcg_codes(40);

    // Complete frames are bit-identical through the masked graph: decoding [0, n) with frame n - 1's last codebook
    // missing emits, ahead of what that frame touches, exactly what decoding [0, n - 1) alone emits.
    {
        std::vector<float> part, full;
        zac_stream a, b;
        CHECK(zac_stream_decode(m, a, codes.data(), n + N_CB - 2, n, part), "partial chunk");
        CHECK(zac_stream_decode(m, b, codes.data(), 40, n - 1, full), "complete chunk");
        CHECK(!full.empty() && part.size() >= full.size(), "the partial chunk emits as far");
        CHECK(max_diff(part, full, full.size()) == 0.0, "complete frames bit-identical next to a partial one");
    }

    // A missing residual code drops its codebook's whole term: with table row X of every residual codebook set to
    // -res.{c}.bias (code X adds nothing), a frame missing its codes decodes like one coded X.
    const int X = 123;
    for (int c = 0; c + 1 < N_CB; ++c) {
        std::vector<float> row(m.res_cb_bias.begin() + (size_t) c * L, m.res_cb_bias.begin() + (size_t) (c + 1) * L);
        for (float & v : row) v = -v;
        ggml_tensor * t = m.get("res." + std::to_string(c) + ".table");
        CHECK(t && t->type == GGML_TYPE_F32, "F32 residual table");
        ggml_backend_tensor_set(t, row.data(), (size_t) X * L * sizeof(float), (size_t) L * sizeof(float));
    }
    std::vector<int32_t> coded(codes.begin(), codes.begin() + (size_t) (n + N_CB - 1) * N_CB), minus = coded, zero = coded;
    for (int r = n; r < n + N_CB - 1; ++r)
        for (int c = 1; c < N_CB; ++c) {
            coded[(size_t) r * N_CB + c] = X;
            minus[(size_t) r * N_CB + c] = -1;
            zero[(size_t) r * N_CB + c] = 0;
        }
    const std::vector<float> missing = stream(m, codes, n, {n});   // rows past H: frames n - 9 .. n - 1 partial
    const std::vector<float> neg = stream(m, minus, n + N_CB - 1, {n}); // the same cells as -1
    const std::vector<float> as_x = stream(m, coded, n + N_CB - 1, {n});
    const std::vector<float> as_0 = stream(m, zero, n + N_CB - 1, {n});
    CHECK(missing.size() == as_x.size() && neg.size() == as_x.size() && as_0.size() == as_x.size(), "lengths");
    double peak = 0;
    for (float v : as_x) peak = std::max(peak, (double) std::fabs(v));
    CHECK(peak > 0, "nonzero audio");
    CHECK(max_diff(missing, neg, missing.size()) == 0.0, "rows past H and -1 codes are the same missing cells");
    const double d_x = max_diff(missing, as_x, missing.size()), d_0 = max_diff(missing, as_0, missing.size());
    fprintf(stderr, "codec: peak %.3e, missing vs code-X %.3e, missing vs code-0 %.3e\n", peak, d_x, d_0);
    CHECK(d_x <= 1e-4 * peak, "a missing code leaves its table row and res.{c}.bias out");
    CHECK(d_0 > 1e-2 * peak, "a missing code is not a code");

    // Without res.{i}.bias the codec cannot leave codebooks out: missing cells fail, complete frames still decode.
    zac_model old;
    CHECK(zac_clone(old, m), "clone");
    old.res_cb_bias.clear();
    CHECK(!zac_partial_frames(old), "no res.{i}.bias: no partial frames");
    bool ok = true;
    stream(old, codes, n, {n}, &ok);
    CHECK(!ok, "missing cells need res.{i}.bias");
    stream(old, codes, 40, {n}, &ok);
    CHECK(ok, "complete frames decode without res.{i}.bias");
    zac_free(old);
    zac_free(m);
}

// ------------------------------------------------------------------------------------- requests and the scheduler

static zs::GenReq plan(zs::ServerState & s, const json & body) {
    return zs::plan_generate(s, body, std::nullopt).gr;
}

static int plan_status(zs::ServerState & s, const json & body, std::string & detail) {
    try {
        zs::plan_generate(s, body, std::nullopt);
    } catch (const zs::HttpError & e) {
        detail = e.body.value("detail", json()).is_string() ? e.body["detail"].get<std::string>() : e.body.dump();
        return e.status;
    }
    return 200;
}

static std::vector<float> generate(zs::ServerState & s, json body, size_t * blocks) {
    auto job = zs::make_job(s, zs::plan_generate(s, body, std::nullopt));
    zs::submit_job(s, job);
    std::vector<float> pcm;
    std::unique_lock<std::mutex> lk(job->m);
    job->cv.wait(lk, [&] { return job->finished; });
    CHECK(!job->failed, "generation");
    *blocks = job->pcm.size();
    for (auto & b : job->pcm) pcm.insert(pcm.end(), b.begin(), b.end());
    return pcm;
}

static void test_server(const char * model_path, const char * zac_path) {
    zs::ServerState s;
    s.default_voice_dir.clear();
    s.batch_slots = 2;
    s.zac_threads = 1;
    std::string err;
    CHECK(zs::configure_silence_prefill(s, err), "silence prefill");
    CHECK(zonos3_model_load(s.model, model_path, /*use_gpu=*/false), "load the dummy backbone");
    s.max_frames = 40;
    CHECK(zs::configure_frame_budget(s, err), "frame budget");
    CHECK(zs::load_decoder(s, zac_path, /*use_gpu=*/false, err), "load the dummy codec");
    zac_model & zac = zs::default_decoder(s).zac;
    const int ncb = (int) s.model.hp.n_codebooks;

    // Request fields: registry defaults, both fields, 400 above n_codebooks.
    const zonos3_generation_registry & reg = zonos3_generation();
    const json base = {{"text", "hello there"}, {"seed", 1}, {"max_tokens", 24}, {"temperature", 0.0}};
    zs::GenReq gr = plan(s, base);
    CHECK(gr.stream_codebooks == reg.stream_codebooks_default && reg.stream_codebooks_default == 1, "default k = registry 1");
    CHECK(gr.stream_codebooks_first_only == reg.stream_codebooks_first_only && gr.stream_codebooks_first_only, "default scope first");
    json b = base;
    b["stream_codebooks"] = ncb;
    b["stream_codebooks_scope"] = "all";
    gr = plan(s, b);
    CHECK(gr.stream_codebooks == ncb && !gr.stream_codebooks_first_only, "k = ncb, scope all");
    std::string detail;
    b["stream_codebooks"] = ncb + 1;
    CHECK(plan_status(s, b, detail) == 400 && detail == "stream_codebooks must be 1.." + std::to_string(ncb), "k > ncb: 400");
    b["stream_codebooks"] = 0;
    CHECK(plan_status(s, b, detail) == 400, "k = 0: 400");
    // decoder: the default when unset (the dummy codec's head variant), an id the server does not serve a 400.
    CHECK(zs::plan_generate(s, base, std::nullopt).gr.decoder == &zs::default_decoder(s), "default decoder");
    b = base;
    b["decoder"] = "nope";
    CHECK(plan_status(s, b, detail) == 400 &&
              detail == "Unknown decoder 'nope'; this server serves: " + zs::default_decoder(s).id + ".",
          "unknown decoder: 400");

    zs::start_scheduler(s);
    CHECK(!s.worker_failed, "scheduler");

    // Whole prefix streams on the dummy backbone (greedy): k = ncb is the whole decode; k = 1 hands the codec
    // partial frames (different first audio, same length); scope all differs again.
    size_t n_whole, n_complete, n_first, n_all;
    json w = base;
    w["stream_decode"] = "whole";
    const std::vector<float> whole = generate(s, w, &n_whole);
    json k10 = base;
    k10["stream_decode"] = "prefix";
    k10["stream_codebooks"] = ncb;
    const std::vector<float> complete = generate(s, k10, &n_complete);
    json k1 = k10;
    k1["stream_codebooks"] = 1;
    const std::vector<float> first = generate(s, k1, &n_first);
    k1["stream_codebooks_scope"] = "all";
    const std::vector<float> all = generate(s, k1, &n_all);
    fprintf(stderr, "server: %zu samples; blocks whole %zu, k=ncb %zu, k=1 first %zu, k=1 all %zu\n", whole.size(), n_whole,
            n_complete, n_first, n_all);
    CHECK(!whole.empty() && complete.size() == whole.size(), "k=ncb: whole length");
    CHECK(max_diff(complete, whole, whole.size()) < 1e-6, "k=ncb: the whole decode");
    CHECK(n_complete > 1 && n_first > 1, "prefix streams in chunks");
    CHECK(first.size() == whole.size(), "k=1 scope first: same length");
    // Scope all may have emitted frames past a max-token stop's complete frames (finish keeps max(target, emitted)).
    CHECK(all.size() >= whole.size(), "k=1 scope all: at least as long");
    CHECK(max_diff(first, complete, first.size()) > 0, "k=1: the first chunk decodes partial frames");
    CHECK(max_diff(all, first, first.size()) > 0, "scope all: later chunks partial too");

    // A codec without res.{i}.bias: every request streams complete frames.
    zac.res_cb_bias.clear();
    gr = plan(s, base);
    CHECK(gr.stream_codebooks == ncb, "no res.{i}.bias: k = ncb");
    zs::stop_scheduler(s);
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <dummy-zonos3.gguf> <dummy-zac.gguf> [--gpu]\n", argv[0]);
        return 1;
    }
    test_ready_rule();
    test_codec(argv[2], /*gpu=*/false);
    if (argc > 3 && std::string(argv[3]) == "--gpu") test_codec(argv[2], /*gpu=*/true);
    test_server(argv[1], argv[2]);
    printf("PASS\n");
    return 0;
}
