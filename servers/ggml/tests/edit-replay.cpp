// edit-replay: run one src/zonos3-edit.h function on a JSON request {"fn", "args"} read from stdin and print the
// result as JSON (matrices as nested lists, tuples as lists). A ValueError prints {"error": "ValueError", "message"}.
// Parity against model/src/zonos3/tts/{edit,edit_align}.py: servers/ggml/tests/parity/test_ggml_edit.py, which also
// writes tests/fixtures/edit-replay/cases.json; `edit-replay --cases <file>` (ctest) replays those cases and prints
// PASS when every answer equals the recorded Python one.
#include "zonos3-edit.h"

#include "json.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

using json = nlohmann::json;
using namespace zonos3_edit;

namespace {

std::vector<int32_t> flat(const json & m, int & cols) {
    std::vector<int32_t> out;
    for (const auto & row : m) {
        cols = (int) row.size();
        for (const auto & v : row) out.push_back(v.get<int32_t>());
    }
    return out;
}

std::vector<int32_t> flat(const json & m) { int cols = 0; return flat(m, cols); }

json rows(const std::vector<int32_t> & v, int cols) {
    json out = json::array();
    for (size_t r = 0; cols > 0 && r < v.size() / cols; ++r)
        out.push_back(std::vector<int32_t>(v.begin() + r * cols, v.begin() + (r + 1) * cols));
    return out;
}

std::vector<Span> spans_of(const json & j) {
    std::vector<Span> out;
    for (const auto & s : j) out.push_back({s.at(0).get<int>(), s.at(1).get<int>()});
    return out;
}

json spans_json(const std::vector<Span> & spans) {
    json out = json::array();
    for (const auto & s : spans) out.push_back({s.start, s.length});
    return out;
}

std::vector<AlignedWord> words_of(const json & j) {
    std::vector<AlignedWord> out;
    for (const auto & w : j) {
        AlignedWord a;
        a.word = w.at("word").get<std::string>();
        a.start = w.at("start").get<double>();
        a.end = w.at("end").get<double>();
        if (w.contains("confidence") && !w["confidence"].is_null()) a.confidence = w["confidence"].get<double>();
        a.tokens = w.at("tokens").get<std::vector<std::string>>();
        out.push_back(std::move(a));
    }
    return out;
}

json words_json(const std::vector<AlignedWord> & words) {
    json out = json::array();
    for (const auto & w : words)
        out.push_back({{"word", w.word}, {"start", w.start}, {"end", w.end},
                       {"confidence", w.confidence ? json(*w.confidence) : json()}, {"tokens", w.tokens}});
    return out;
}

using Lists = std::vector<std::vector<std::string>>;

json call(const std::string & fn, const json & a) {
    if (fn == "edit_prompt_tokens") {
        Layout l;
        l.n_codebooks = a.at("n_codebooks").get<int>();
        l.audio_pad_id = a.at("audio_pad_id").get<int>();
        l.text_vocab = a.at("text_vocab").get<int>();
        l.span_end_id = a.at("span_end_id").get<int>();
        std::vector<std::vector<int32_t>> fills;
        for (const auto & f : a.at("fills")) fills.push_back(flat(f));
        return rows(edit_prompt_tokens(flat(a.at("text_rows")), flat(a.at("codes")), spans_of(a.at("spans")), fills, l),
                    l.n_codebooks + 1);
    }
    if (fn == "extract_fill") {
        const auto sizes = a.at("codebook_sizes").get<std::vector<int>>();
        const Fill f = extract_fill(flat(a.at("delayed_rows")), a.at("eos_frame").is_null() ? -1 : a["eos_frame"].get<int>(),
                                    a.at("audio_pad_id").get<int>(), sizes);
        return {rows(f.frames, (int) sizes.size()), f.ended};
    }
    if (fn == "splice_edit") {
        int k = 0;
        const auto codes = flat(a.at("codes"), k);
        std::vector<std::vector<int32_t>> fills;
        for (const auto & f : a.at("fills")) fills.push_back(flat(f));
        return rows(splice_edit(codes, k, spans_of(a.at("spans")), fills), k);
    }
    if (fn == "validate_edit_spans") {
        validate_edit_spans(spans_of(a.at("spans")), a.at("total_frames").get<int>(), a.at("max_spans").get<int>());
        return nullptr;
    }
    if (fn == "seconds_to_spans") {
        std::vector<std::pair<double, double>> pairs;
        for (const auto & p : a.at("pairs")) pairs.emplace_back(p.at(0).get<double>(), p.at(1).get<double>());
        return spans_json(seconds_to_spans(pairs, a.at("total_frames").get<int>(), a.at("frame_rate").get<double>()));
    }
    if (fn == "sequence_opcodes") {
        json out = json::array();
        for (const auto & o : sequence_opcodes(a.at("a").get<std::vector<std::string>>(), a.at("b").get<std::vector<std::string>>())) {
            const char * tag = o.tag == 'e' ? "equal" : o.tag == 'r' ? "replace" : o.tag == 'd' ? "delete" : "insert";
            out.push_back({tag, o.i1, o.i2, o.j1, o.j2});
        }
        return out;
    }
    if (fn == "diff_spans") {
        const EditDiff d = diff_spans(words_of(a.at("words")), a.at("target_tokens").get<std::vector<std::string>>(),
                                      a.at("total_frames").get<int>(), a.at("frame_rate").get<double>(),
                                      a.at("margin_seconds").get<double>(), a.at("min_gap_frames").get<int>(),
                                      a.at("max_spans").get<int>());
        return {{"spans", spans_json(d.spans)}, {"new_target_tokens", d.new_target_tokens}};
    }
    if (fn == "plan_edit_windows") {
        const auto spans = spans_of(a.at("spans"));
        const auto ws = plan_edit_windows(
            spans, words_of(a.at("words")), a.at("target_words").get<std::vector<std::string>>(),
            a.at("target_word_tokens").get<Lists>(), a.at("total_frames").get<int>(), a.at("frame_rate").get<double>(),
            a.at("context_seconds").get<double>(),
            a.at("target_text").is_null() ? std::nullopt : std::optional<std::string>(a["target_text"].get<std::string>()));
        json out = json::array();
        for (const auto & w : ws)
            out.push_back({{"start", w.start}, {"end", w.end}, {"regions", w.regions}, {"text", w.text},
                           {"spans", spans_json(w.spans(spans))}});
        return out;
    }
    if (fn == "expected_fill_seconds")
        return expected_fill_seconds(spans_of(a.at("spans")), words_of(a.at("words")),
                                     a.at("target_words").get<std::vector<std::string>>(),
                                     a.at("target_word_tokens").get<Lists>(), a.at("frame_rate").get<double>());
    if (fn == "wildcard_verify") {
        const Verify v = wildcard_verify(a.at("reference_tokens").get<std::vector<std::string>>(),
                                         a.at("hypothesis_tokens").get<std::vector<std::string>>());
        json groups = json::array();
        for (const auto & g : v.groups) groups.push_back({{"ipa", g.ipa}, {"heard", g.heard ? json(*g.heard) : json()}});
        return {{"word_errors", v.word_errors}, {"ref_words", v.ref_words}, {"cer", v.cer}, {"groups", groups}};
    }
    if (fn == "fill_groups") {
        std::vector<std::optional<std::string>> heard;
        for (const auto & h : a.at("heard")) heard.push_back(h.is_null() ? std::nullopt : std::optional<std::string>(h.get<std::string>()));
        return fill_groups(a.at("text").get<std::string>(), heard);
    }
    if (fn == "canonical_markup") return canonical_markup(a.at("text").get<std::string>());
    if (fn == "target_normalize_request") {
        const TargetNormalize t = target_normalize_request(a.at("target_text").get<std::string>());
        return {{"texts", t.texts}, {"assembled", t.assemble(a.at("normalized").get<Lists>())}};
    }
    if (fn == "edit_words") {
        std::vector<WordIn> ws;
        for (const auto & w : a.at("words")) {
            WordIn in;
            in.word = w.at("word").get<std::string>();
            in.start = w.at("start").get<double>();
            in.end = w.at("end").get<double>();
            if (w.contains("confidence") && !w["confidence"].is_null()) in.confidence = w["confidence"].get<double>();
            ws.push_back(std::move(in));
        }
        return words_json(edit_words(ws, a.at("word_tokens").get<Lists>(),
                                     a.at("original_tokens").get<std::vector<std::string>>(), a.at("duration").get<double>()));
    }
    if (fn == "mms_language") {
        const std::string lang = a.at("language").get<std::string>();
        const auto code = mms_language(lang);
        if (!code) throw std::invalid_argument(mms_language_error(lang));
        return *code;
    }
    if (fn == "split_words") return split_words(a.at("text").get<std::string>());
    if (fn == "token_chars") return token_chars(a.at("token").get<std::string>());
    throw std::runtime_error("unknown fn " + fn);
}

json run(const json & req) {
    try {
        return call(req.at("fn").get<std::string>(), req.at("args"));
    } catch (const std::invalid_argument & e) {
        return {{"error", "ValueError"}, {"message", e.what()}};
    }
}

// Floats equal within 1e-9 (relative), everything else exactly.
bool same(const json & x, const json & y) {
    if (x.is_number() && y.is_number()) {
        const double a = x.get<double>(), b = y.get<double>();
        return a == b || std::fabs(a - b) <= 1e-9 * std::max(std::fabs(a), std::fabs(b));
    }
    if (x.type() != y.type()) return false;
    if (x.is_array()) {
        if (x.size() != y.size()) return false;
        for (size_t i = 0; i < x.size(); ++i) if (!same(x[i], y[i])) return false;
        return true;
    }
    if (x.is_object()) {
        if (x.size() != y.size()) return false;
        for (auto it = x.begin(); it != x.end(); ++it) if (!y.contains(it.key()) || !same(it.value(), y[it.key()])) return false;
        return true;
    }
    return x == y;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc > 2 && std::string(argv[1]) == "--cases") {
        std::ifstream f(argv[2]);
        if (!f) { std::cerr << "cannot read " << argv[2] << "\n"; return 1; }
        const json cases = json::parse(f);
        int failed = 0;
        for (const auto & c : cases) {
            const json got = run(c);
            if (!same(got, c.at("expected"))) {
                ++failed;
                std::cerr << "MISMATCH " << c.value("name", c.at("fn").get<std::string>()) << "\n  expected "
                          << c["expected"].dump() << "\n  got      " << got.dump() << "\n";
            }
        }
        std::cout << (failed ? "FAIL " : "PASS ") << cases.size() - failed << "/" << cases.size() << " cases\n";
        return failed ? 1 : 0;
    }
    const std::string in((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
    std::cout << run(json::parse(in)).dump() << "\n";
    return 0;
}
