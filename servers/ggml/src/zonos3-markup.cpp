#include "zonos3-markup.h"

#include <cstdint>

namespace {

// Decode one UTF-8 code point at s[i] (advances i); a malformed byte decodes alone as U+FFFD.
uint32_t next_cp(const std::string & s, size_t & i) {
    const unsigned char c = (unsigned char) s[i];
    const int len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 0;
    if (len == 0 || i + len > s.size()) { ++i; return 0xFFFD; }
    uint32_t cp = len == 1 ? c : len == 2 ? (c & 0x1F) : len == 3 ? (c & 0x0F) : (c & 0x07);
    for (int k = 1; k < len; ++k) {
        const unsigned char cc = (unsigned char) s[i + k];
        if ((cc >> 6) != 0x2) { ++i; return 0xFFFD; }
        cp = (cp << 6) | (cc & 0x3F);
    }
    i += len;
    return cp;
}

// Python str.isspace() (also what re's \s matches in a str pattern).
bool is_space(uint32_t cp) {
    return (cp >= 0x09 && cp <= 0x0D) || (cp >= 0x1C && cp <= 0x20) || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F ||
           cp == 0x3000;
}

// Length in bytes of a marker starting at text[pos], or 0.
size_t marker_at(const std::string & text, size_t pos) {
    if (text[pos] != '<') return 0;
    size_t i = pos + 1;
    if (i < text.size() && text[i] == '/') ++i;
    bool named = false;
    for (const std::string & tag : zonos3_nonverbal_markup_tags())
        if (text.compare(i, tag.size(), tag) == 0) { i += tag.size(); named = true; break; }
    if (!named) return 0;
    while (i < text.size()) {
        size_t j = i;
        if (!is_space(next_cp(text, j))) break;
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
        if (n) { out.emplace_back(pos, pos + n); pos += n; }
        else ++pos;
    }
    return out;
}

}  // namespace

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
        if (i == bytes[k].second) { out.emplace_back(start, cp); ++k; continue; }
        next_cp(text, i);
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
        const uint32_t cp = next_cp(replaced, i);
        if (is_space(cp)) { pending_space = !out.empty(); continue; }
        if (pending_space) { out += ' '; pending_space = false; }
        out.append(replaced, at, i - at);
    }
    return out;
}
