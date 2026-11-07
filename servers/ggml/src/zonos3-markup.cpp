#include "zonos3-markup.h"
#include "zonos3-text.h"

#include <cstdint>

namespace {

// Length in bytes of a marker starting at text[pos], or 0.
size_t marker_at(const std::string & text, size_t pos) {
    if (text[pos] != '<') return 0;
    size_t i = pos + 1;
    if (i < text.size() && text[i] == '/') ++i;
    bool named = false;
    for (const std::string & tag : zonos3_nonverbal_markup_tags())
        if (text.compare(i, tag.size(), tag) == 0) {
            i += tag.size();
            named = true;
            break;
        }
    if (!named) return 0;
    while (i < text.size()) {
        size_t j = i;
        if (!is_py_space(utf8_next(text, j))) break;
        i = j;
    }
    if (i < text.size() && text[i] == '/') ++i;
    if (i < text.size() && text[i] == '>') return i + 1 - pos;
    return 0;
}

// Byte ranges of the markers, left to right, non-overlapping.
std::vector<std::pair<size_t, size_t>> marker_bytes(const std::string & text) {
    std::vector<std::pair<size_t, size_t>> out;
    for (size_t pos = 0; pos < text.size();) {
        const size_t n = marker_at(text, pos);
        if (n) {
            out.emplace_back(pos, pos + n);
            pos += n;
        } else ++pos;
    }
    return out;
}

} // namespace

const std::vector<std::string> & zonos3_nonverbal_markup_tags() {
    static const std::vector<std::string> tags = {"laugh", "breath", "sigh", "cough", "panting", "shouting", "crying"};
    return tags;
}

bool zonos3_has_nonverbal_markup(const std::string & text) {
    for (size_t pos = 0; pos < text.size(); ++pos)
        if (marker_at(text, pos)) return true;
    return false;
}

std::vector<std::string> zonos3_split_nonverbal_markup(const std::string & text) {
    std::vector<std::string> pieces;
    size_t prev = 0;
    for (const auto & [a, b] : marker_bytes(text)) {
        pieces.push_back(text.substr(prev, a - prev));
        pieces.push_back(text.substr(a, b - a));
        prev = b;
    }
    pieces.push_back(text.substr(prev));
    return pieces;
}

std::vector<std::pair<int, int>> zonos3_nonverbal_marker_spans(const std::string & text) {
    std::vector<std::pair<int, int>> out;
    const auto bytes = marker_bytes(text);
    size_t k = 0, i = 0;
    int cp = 0, start = 0;
    while (k < bytes.size()) {
        if (i == bytes[k].first) start = cp;
        if (i == bytes[k].second) {
            out.emplace_back(start, cp);
            ++k;
            continue;
        }
        utf8_next(text, i);
        ++cp;
    }
    return out;
}

std::string zonos3_strip_nonverbal_markup(const std::string & text) {
    const auto bytes = marker_bytes(text);
    if (bytes.empty()) return text;
    std::string replaced;
    size_t prev = 0;
    for (const auto & [a, b] : bytes) {
        replaced.append(text, prev, a - prev);
        replaced += ' ';
        prev = b;
    }
    replaced.append(text, prev, std::string::npos);
    // " ".join(replaced.split())
    std::string out;
    bool pending_space = false;
    for (size_t i = 0; i < replaced.size();) {
        const size_t at = i;
        const uint32_t cp = utf8_next(replaced, i);
        if (is_py_space(cp)) {
            pending_space = !out.empty();
            continue;
        }
        if (pending_space) {
            out += ' ';
            pending_space = false;
        }
        out.append(replaced, at, i - at);
    }
    return out;
}
