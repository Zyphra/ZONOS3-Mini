#include "zonos3-textmatch.h"

#include "unicode-data.h"

#include <algorithm>
#include <cstdint>
#include <iterator>

namespace zonos3_textmatch {

namespace {

uint32_t next_cp(const std::string & s, size_t & i) {
    const unsigned char c = (unsigned char) s[i];
    int n = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 0;
    if (n == 0 || i + n > s.size()) { ++i; return 0xFFFD; }
    uint32_t cp = n == 1 ? c : n == 2 ? (c & 0x1F) : n == 3 ? (c & 0x0F) : (c & 0x07);
    for (int k = 1; k < n; ++k) {
        const unsigned char d = (unsigned char) s[i + k];
        if ((d & 0xC0) != 0x80) { ++i; return 0xFFFD; }
        cp = (cp << 6) | (d & 0x3F);
    }
    i += n;
    return cp;
}

std::u32string cps(const std::string & s) {
    std::u32string out;
    for (size_t i = 0; i < s.size();) out += next_cp(s, i);
    return out;
}

void put_cp(std::string & out, uint32_t cp) {
    if (cp < 0x80) out += (char) cp;
    else if (cp < 0x800) { out += (char) (0xC0 | (cp >> 6)); out += (char) (0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) {
        out += (char) (0xE0 | (cp >> 12)); out += (char) (0x80 | ((cp >> 6) & 0x3F)); out += (char) (0x80 | (cp & 0x3F));
    } else {
        out += (char) (0xF0 | (cp >> 18)); out += (char) (0x80 | ((cp >> 12) & 0x3F));
        out += (char) (0x80 | ((cp >> 6) & 0x3F)); out += (char) (0x80 | (cp & 0x3F));
    }
}

std::string utf8(const std::u32string & s) {
    std::string out;
    for (char32_t c : s) put_cp(out, c);
    return out;
}

template <size_t N>
bool in_ranges(const uint32_t (&table)[N][2], uint32_t c) {
    auto it = std::upper_bound(std::begin(table), std::end(table), c,
                               [](uint32_t v, const uint32_t (&r)[2]) { return v < r[0]; });
    return it != std::begin(table) && c <= (*(it - 1))[1];
}

template <size_t N, size_t W>
const uint32_t * find_row(const uint32_t (&table)[N][W], uint32_t c) {
    auto it = std::lower_bound(std::begin(table), std::end(table), c,
                               [](const uint32_t (&r)[W], uint32_t v) { return r[0] < v; });
    return it != std::end(table) && (*it)[0] == c ? *it : nullptr;
}

// One character's full case mapping ({cp, length, mapped[3]} rows) appended to `out`.
template <size_t N>
void map_case(const uint32_t (&table)[N][5], uint32_t c, std::u32string & out) {
    if (const uint32_t * row = find_row(table, c)) out.append(row + 2, row + 2 + row[1]);
    else out += (char32_t) c;
}

// Python's \w on str: str.isalnum() (letters and numbers, General_Category L* and N*) or _.
bool is_word(uint32_t c) { return c == '_' || in_ranges(ZONOS3_UNI_L, c) || in_ranges(ZONOS3_UNI_N, c); }

// str.lower() Final_Sigma classes: 1 case-ignorable, 2 cased (not case-ignorable), 0 neither.
int sigma_class(uint32_t c) {
    auto it = std::upper_bound(std::begin(ZONOS3_UNI_SIGMA_CONTEXT), std::end(ZONOS3_UNI_SIGMA_CONTEXT), c,
                               [](uint32_t v, const uint32_t (&r)[3]) { return v < r[0]; });
    if (it == std::begin(ZONOS3_UNI_SIGMA_CONTEXT) || c > (*(it - 1))[1]) return 0;
    return (int) (*(it - 1))[2];
}

// CPython handle_capital_sigma: U+03A3 lowers to final sigma after a cased letter not followed by one (case-ignorable
// characters skipped on both sides).
uint32_t lower_sigma(const std::u32string & s, size_t i) {
    size_t j = i;
    int before = 0;
    while (j > 0) {
        before = sigma_class(s[--j]);
        if (before != 1) break;
    }
    bool final_sigma = before == 2;
    if (final_sigma) {
        for (size_t k = i + 1; k < s.size(); ++k) {
            const int after = sigma_class(s[k]);
            if (after == 1) continue;
            final_sigma = after != 2;
            break;
        }
    }
    return final_sigma ? 0x3C2 : 0x3C3;
}

int levenshtein(const std::u32string & a, const std::u32string & b) {
    std::vector<int> prev(b.size() + 1), cur(b.size() + 1);
    for (size_t j = 0; j <= b.size(); ++j) prev[j] = (int) j;
    for (size_t i = 1; i <= a.size(); ++i) {
        cur[0] = (int) i;
        for (size_t j = 1; j <= b.size(); ++j)
            cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1] ? 1 : 0)});
        std::swap(prev, cur);
    }
    return prev[b.size()];
}

std::u32string normalized_cps(const std::string & text) {
    std::u32string folded;
    for (char32_t c : cps(text)) {
        if (c == 0x2018 || c == 0x2019 || c == 0x201B || c == 0x2BC) c = '\'';   // _QUOTES
        else if (c == '_') c = ' ';
        map_case(ZONOS3_UNI_UPPER, c, folded);
    }
    std::u32string out;   // [^\w']+ -> " ", then strip()
    bool gap = false;
    for (char32_t c : folded) {
        if (is_word(c) || c == '\'') {
            if (gap && !out.empty()) out += U' ';
            gap = false;
            out += c;
        } else {
            gap = true;
        }
    }
    return out;
}

}  // namespace

std::string normalize(const std::string & text) { return utf8(normalized_cps(text)); }

std::vector<std::string> tokens(const std::string & text) {
    std::vector<std::string> out;
    std::string cur;
    for (char32_t c : normalized_cps(text)) {
        if (c == U' ') { out.push_back(std::move(cur)); cur.clear(); }
        else put_cp(cur, c);
    }
    if (!cur.empty()) out.push_back(std::move(cur));
    return out;
}

double char_error_rate(const std::string & reference, const std::string & hypothesis) {
    std::u32string ref = normalized_cps(reference), hyp = normalized_cps(hypothesis);
    ref.erase(std::remove(ref.begin(), ref.end(), U' '), ref.end());
    hyp.erase(std::remove(hyp.begin(), hyp.end(), U' '), hyp.end());
    if (ref.empty()) return 0.0;
    if (hyp.empty()) return 1.0;
    return (double) levenshtein(ref, hyp) / (double) ref.size();
}

int word_errors(const std::vector<std::string> & reference, const std::vector<std::string> & hypothesis) {
    std::vector<int> prev(hypothesis.size() + 1), row;
    for (size_t j = 0; j <= hypothesis.size(); ++j) prev[j] = (int) j;
    for (size_t i = 1; i <= reference.size(); ++i) {
        row.assign(1, (int) i);
        for (size_t j = 1; j <= hypothesis.size(); ++j)
            row.push_back(std::min({row.back() + 1, prev[j] + 1, prev[j - 1] + (reference[i - 1] != hypothesis[j - 1] ? 1 : 0)}));
        prev.swap(row);
    }
    return prev.back();
}

std::string py_upper(const std::string & text) {
    std::u32string out;
    for (char32_t c : cps(text)) map_case(ZONOS3_UNI_UPPER, c, out);
    return utf8(out);
}

std::string py_lower(const std::string & text) {
    const std::u32string s = cps(text);
    std::u32string out;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == 0x3A3) out += (char32_t) lower_sigma(s, i);
        else map_case(ZONOS3_UNI_LOWER, s[i], out);
    }
    return utf8(out);
}

}  // namespace zonos3_textmatch
