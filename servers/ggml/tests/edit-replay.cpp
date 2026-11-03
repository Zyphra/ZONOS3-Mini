// edit-replay: run one src/zonos3-edit.h function on a JSON request {"fn", "args"} read from stdin and print the result
// as JSON (matrices as nested lists). A ValueError prints {"error": "ValueError", "message"}. Parity against
// model/src/zonos3/tts/edit.py: servers/ggml/tests/parity/test_ggml_edit.py, which also writes
// tests/fixtures/edit-replay/cases.json; `edit-replay --cases <file>` (ctest) replays those cases and prints PASS when
// every answer equals the recorded Python one.
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

std::vector<int32_t> flat(const json & m) {
    std::vector<int32_t> out;
    for (const auto & row : m)
        for (const auto & v : row) out.push_back(v.get<int32_t>());
    return out;
}

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
    if (fn == "validate_edit_spans") {
        validate_edit_spans(spans_of(a.at("spans")), a.at("total_frames").get<int>(), a.at("max_spans").get<int>());
        return nullptr;
    }
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
