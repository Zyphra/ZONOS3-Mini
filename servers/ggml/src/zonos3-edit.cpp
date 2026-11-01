#include "zonos3-edit.h"

#include "zonos3-markup.h"

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <unordered_map>

namespace zonos3_edit {

namespace {

[[noreturn]] void value_error(const std::string & msg) { throw std::invalid_argument(msg); }

// ------------------------------------------------------------------------------------------------------- UTF-8 text

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

std::u32string cps(const std::string & s) {
    std::u32string out;
    for (size_t i = 0; i < s.size();) out += next_cp(s, i);
    return out;
}

// Python str.isspace() (the whitespace str.split() splits on).
bool py_space(uint32_t c) {
    return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

// str.lower() for the scripts MMS transcripts use (Latin, Greek, Cyrillic).
uint32_t py_lower(uint32_t c) {
    if (c >= 'A' && c <= 'Z') return c + 32;
    if (c >= 0xC0 && c <= 0xDE && c != 0xD7) return c + 32;
    if (c >= 0x100 && c <= 0x137 && !(c & 1)) return c + 1;
    if (c >= 0x139 && c <= 0x148 && (c & 1)) return c + 1;
    if (c >= 0x14A && c <= 0x177 && !(c & 1)) return c + 1;
    if (c == 0x178) return 0xFF;
    if (c >= 0x179 && c <= 0x17E && (c & 1)) return c + 1;
    if (c >= 0x391 && c <= 0x3AB && c != 0x3A2) return c + 32;
    if (c >= 0x410 && c <= 0x42F) return c + 32;
    if (c >= 0x400 && c <= 0x40F) return c + 80;
    return c;
}

std::string lower(const std::string & s) {
    std::string out;
    for (size_t i = 0; i < s.size();) put_cp(out, py_lower(next_cp(s, i)));
    return out;
}

size_t cp_len(const std::string & s) {
    size_t n = 0;
    for (unsigned char c : s) n += (c & 0xC0) != 0x80;
    return n;
}

std::string join(const std::vector<std::string> & items, const std::string & sep) {
    std::string out;
    for (size_t i = 0; i < items.size(); ++i) out += (i ? sep : "") + items[i];
    return out;
}

// Python repr() of a str.
std::string repr(const std::string & s) {
    const bool dq = s.find('\'') != std::string::npos && s.find('"') == std::string::npos;
    const char q = dq ? '"' : '\'';
    std::string out(1, q);
    for (size_t i = 0; i < s.size();) {
        const uint32_t c = next_cp(s, i);
        if (c == (uint32_t) q || c == '\\') { out += '\\'; out += (char) c; }
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c < 0x20 || c == 0x7F) { char b[8]; snprintf(b, sizeof b, "\\x%02x", c); out += b; }
        else put_cp(out, c);
    }
    return out + q;
}

std::string fmt3(double v) {
    char b[64];
    snprintf(b, sizeof b, "%.3f", v);
    return b;
}

const std::string GROUP_OPEN = "\xEE\x80\x80", GROUP_CLOSE = "\xEE\x80\x81";   // U+E000, U+E001
constexpr double GROUP_LETTERS_PER_SYMBOL = 1.4;

// _GROUP_RE ("\ue000[^\ue000\ue001]*\ue001") matches of `text` as byte ranges [start, end).
std::vector<std::pair<size_t, size_t>> group_matches(const std::string & text) {
    std::vector<std::pair<size_t, size_t>> out;
    size_t i = 0;
    while ((i = text.find(GROUP_OPEN, i)) != std::string::npos) {
        size_t j = i + GROUP_OPEN.size();
        bool closed = false;
        while (j < text.size()) {
            if (text.compare(j, 3, GROUP_CLOSE) == 0) { closed = true; break; }
            if (text.compare(j, 3, GROUP_OPEN) == 0) break;
            size_t k = j;
            next_cp(text, k);
            j = k;
        }
        if (closed) { out.push_back({i, j + 3}); i = j + 3; }
        else i = j;   // an unclosed group: the scan resumes at the next opener
    }
    return out;
}

template <class T>
std::vector<T> py_slice(const std::vector<T> & v, std::optional<int> a, std::optional<int> b) {
    const int n = (int) v.size();
    auto clamp = [&](int x) { return x < 0 ? std::max(0, x + n) : std::min(x, n); };
    const int lo = a ? clamp(*a) : 0, hi = b ? clamp(*b) : n;
    return lo < hi ? std::vector<T>(v.begin() + lo, v.begin() + hi) : std::vector<T>{};
}

int py_floor(double x) { return (int) std::floor(x); }
int py_ceil(double x) { return (int) std::ceil(x); }

}  // namespace

// ---------------------------------------------------------------------------------------------------------- edit.py

void validate_edit_spans(const std::vector<Span> & spans, int total_frames, int max_spans) {
    if (spans.empty() || (int) spans.size() > max_spans) value_error("Edit requires between one and max_spans regions");
    int previous_end = -1;
    for (const Span & s : spans) {
        if (s.start < 0 || s.length < 1 || s.start + s.length > total_frames || s.start <= previous_end)
            value_error("Edit spans must be ascending, separated and within the audio");
        previous_end = s.start + s.length;
    }
}

std::vector<int32_t> sheared_block(const int32_t * frames, int n_frames, const Layout & l, int end_id) {
    const int k = l.n_codebooks, W = k + 1;
    const bool has_end = end_id >= 0;
    const int rows = n_frames + (has_end ? 1 : 0) + k - 1;
    std::vector<int32_t> audio((size_t) rows * k, l.audio_pad_id);
    std::copy(frames, frames + (size_t) n_frames * k, audio.begin());
    if (has_end) std::fill(audio.begin() + (size_t) n_frames * k, audio.begin() + (size_t) (n_frames + 1) * k, end_id);
    std::vector<int32_t> out((size_t) rows * W);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < k; ++c) out[(size_t) r * W + c] = r - c >= 0 ? audio[(size_t) (r - c) * k + c] : l.audio_pad_id;
        out[(size_t) r * W + k] = l.text_vocab;
    }
    return out;
}

std::vector<int32_t> edit_prompt_tokens(const std::vector<int32_t> & text_rows, const std::vector<int32_t> & codes,
                                        const std::vector<Span> & spans,
                                        const std::vector<std::vector<int32_t>> & fills, const Layout & l) {
    if (l.span_end_id != l.audio_pad_id + 1) value_error("span_end_id must equal audio_pad_id + 1");
    const int k = l.n_codebooks, W = k + 1;
    const int n = (int) (codes.size() / k);
    validate_edit_spans(spans, n, (int) spans.size());
    if (fills.size() >= spans.size()) value_error("Prompt needs an ungenerated span");
    std::vector<int32_t> out(text_rows);
    auto marker = [&](int j) {
        for (int c = 0; c < k; ++c) out.push_back(l.audio_pad_id);
        out.push_back(l.text_vocab + 1 + j);
    };
    auto append = [&](const std::vector<int32_t> & rows) { out.insert(out.end(), rows.begin(), rows.end()); };
    int cursor = 0;
    for (size_t j = 0; j < spans.size(); ++j) {
        if (spans[j].start > cursor)
            append(sheared_block(codes.data() + (size_t) cursor * k, spans[j].start - cursor, l, -1));
        marker((int) j);
        cursor = spans[j].start + spans[j].length;
    }
    append(sheared_block(codes.data() + (size_t) cursor * k, n - cursor, l, l.audio_pad_id - 1));
    for (size_t j = 0; j < fills.size(); ++j) {
        marker((int) j);
        append(sheared_block(fills[j].data(), (int) (fills[j].size() / k), l, l.span_end_id));
    }
    marker((int) fills.size());
    (void) W;
    return out;
}

Fill extract_fill(const std::vector<int32_t> & delayed_rows, int eos_frame, int audio_pad_id,
                  const std::vector<int> & codebook_sizes) {
    const int k = (int) codebook_sizes.size();
    const int H = (int) (delayed_rows.size() / k);
    std::vector<int32_t> frames((size_t) H * k, audio_pad_id);   // shear_up
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < k && r + c < H; ++c) frames[(size_t) r * k + c] = delayed_rows[(size_t) (r + c) * k + c];
    int n = std::max(0, eos_frame >= 0 ? eos_frame : H - (k - 1));
    n = std::min(n, H);
    bool truncated = false;
    for (int r = 0; r < n && !truncated; ++r)
        for (int c = 0; c < k; ++c) {
            const int32_t v = frames[(size_t) r * k + c];
            if (v < 0 || v >= codebook_sizes[c]) { truncated = true; n = r; break; }
        }
    Fill out;
    out.frames.assign(frames.begin(), frames.begin() + (size_t) n * k);
    out.ended = eos_frame >= 0 && !truncated;
    return out;
}

std::vector<int32_t> splice_edit(const std::vector<int32_t> & codes, int n_codebooks, const std::vector<Span> & spans,
                                 const std::vector<std::vector<int32_t>> & fills) {
    const int k = n_codebooks;
    validate_edit_spans(spans, (int) (codes.size() / k), (int) spans.size());
    if (fills.size() != spans.size()) value_error("One fill is required per span");
    std::vector<int32_t> out;
    int cursor = 0;
    for (size_t j = 0; j < spans.size(); ++j) {
        if (fills[j].size() % k) value_error("Invalid fill shape");
        out.insert(out.end(), codes.begin() + (size_t) cursor * k, codes.begin() + (size_t) spans[j].start * k);
        out.insert(out.end(), fills[j].begin(), fills[j].end());
        cursor = spans[j].start + spans[j].length;
    }
    out.insert(out.end(), codes.begin() + (size_t) cursor * k, codes.end());
    return out;
}

// ---------------------------------------------------------------------------------------------------- edit_align.py

static const std::vector<std::pair<std::string, std::string>> & mms_languages() {
    static const std::vector<std::pair<std::string, std::string>> m = {
        {"en_us", "en"}, {"fr_fr", "fr"}, {"de", "de"}, {"es", "es"}, {"it", "it"},
        {"pt_br", "pt"}, {"ja", "ja"}, {"cmn", "zh"}, {"ko", "ko"}};
    return m;
}

std::optional<std::string> mms_language(const std::string & language) {
    for (const auto & [k, v] : mms_languages()) if (k == language) return v;
    return std::nullopt;
}

std::string mms_language_error(const std::string & language) {
    std::vector<std::string> keys;
    for (const auto & kv : mms_languages()) keys.push_back(kv.first);
    return "Speech editing has no MMS alignment for language " + repr(language) + " (supported: " + join(keys, ", ") + ").";
}

std::vector<std::string> split_words(const std::string & text) {
    std::vector<std::string> out;
    std::string cur;
    bool in = false;
    for (size_t i = 0; i < text.size();) {
        const size_t at = i;
        const uint32_t c = next_cp(text, i);
        if (py_space(c)) { if (in) { out.push_back(cur); cur.clear(); in = false; } }
        else { cur.append(text, at, i - at); in = true; }
    }
    if (in) out.push_back(cur);
    return out;
}

bool is_group_token(const std::string & token) {
    return token.size() >= 3 && token.compare(0, 3, GROUP_OPEN) == 0 &&
           token.compare(token.size() - 3, 3, GROUP_CLOSE) == 0;
}

double token_chars(const std::string & token) {
    return is_group_token(token) ? GROUP_LETTERS_PER_SYMBOL * ((double) cp_len(token) - 2) : (double) cp_len(token);
}

bool is_marker_token(const std::string & token) {
    const auto pieces = zonos3_split_nonverbal_markup(token);
    return pieces.size() == 3 && pieces[0].empty() && pieces[2].empty();
}

std::string canonical_markup(const std::string & text) {
    if (!zonos3_has_nonverbal_markup(text)) return text;
    const auto pieces = zonos3_split_nonverbal_markup(text);
    std::string out;
    for (size_t p = 0; p < pieces.size(); ++p) {
        if (p % 2 == 0) { out += pieces[p]; continue; }
        for (size_t i = 0; i < pieces[p].size();) {
            const uint32_t c = next_cp(pieces[p], i);
            if (!py_space(c)) put_cp(out, c);
        }
    }
    return out;
}

// _PIECE_RE.split(word) without the empty pieces: pronunciation groups, nonverbal markers and the text around them.
static std::vector<std::string> word_pieces(const std::string & word) {
    std::vector<std::string> out;
    auto text_part = [&](const std::string & t) {
        for (const auto & piece : zonos3_split_nonverbal_markup(t)) if (!piece.empty()) out.push_back(piece);
    };
    size_t cursor = 0;
    for (const auto & [a, b] : group_matches(word)) {
        text_part(word.substr(cursor, a - cursor));
        out.push_back(word.substr(a, b - a));
        cursor = b;
    }
    text_part(word.substr(cursor));
    return out;
}

TargetNormalize target_normalize_request(const std::string & target_text) {
    TargetNormalize out;
    const auto words = split_words(target_text);
    if (target_text.find(GROUP_OPEN) == std::string::npos && !zonos3_has_nonverbal_markup(target_text)) {
        out.texts.push_back(target_text);
        out.texts.insert(out.texts.end(), words.begin(), words.end());
        return out;
    }
    out.plain = false;
    for (const auto & w : words) out.pieces.push_back(word_pieces(w));
    for (const auto & word : out.pieces)
        for (const auto & piece : word)
            if (!is_group_token(piece) && !is_marker_token(piece)) out.texts.push_back(piece);
    return out;
}

std::vector<std::vector<std::string>> TargetNormalize::assemble(
        const std::vector<std::vector<std::string>> & normalized) const {
    if (plain) return normalized;
    size_t next = 0;
    std::vector<std::vector<std::string>> word_tokens;
    for (const auto & word : pieces) {
        std::vector<std::string> tokens;
        for (const auto & piece : word) {
            if (is_group_token(piece)) tokens.push_back(piece);
            else if (is_marker_token(piece)) {}
            else {
                if (next >= normalized.size()) value_error("the MMS normalize answered too few texts");
                tokens.insert(tokens.end(), normalized[next].begin(), normalized[next].end());
                ++next;
            }
        }
        word_tokens.push_back(std::move(tokens));
    }
    std::vector<std::vector<std::string>> out(1);
    for (const auto & t : word_tokens) out[0].insert(out[0].end(), t.begin(), t.end());
    out.insert(out.end(), word_tokens.begin(), word_tokens.end());
    return out;
}

std::vector<Span> seconds_to_spans(const std::vector<std::pair<double, double>> & pairs, int total_frames,
                                   double frame_rate) {
    std::vector<Span> spans;
    for (const auto & [start, end] : pairs) {
        const int first = std::max(0, py_floor(start * frame_rate + 1e-6));
        spans.push_back({first, std::min(total_frames, py_ceil(end * frame_rate - 1e-6)) - first});
    }
    return spans;
}

// ------------------------------------------------------------------------------------- difflib.SequenceMatcher port

namespace {

struct Matcher {
    std::vector<int> a, b;
    std::unordered_map<int, std::vector<int>> b2j;

    Matcher(const std::vector<std::string> & sa, const std::vector<std::string> & sb) {
        std::map<std::string, int> ids;
        auto id = [&](const std::string & s) { return ids.emplace(s, (int) ids.size()).first->second; };
        for (const auto & s : sa) a.push_back(id(s));
        for (const auto & s : sb) b.push_back(id(s));
        for (int j = 0; j < (int) b.size(); ++j) b2j[b[j]].push_back(j);
    }

    // find_longest_match(alo, ahi, blo, bhi) without junk (autojunk=False, isjunk=None).
    void longest(int alo, int ahi, int blo, int bhi, int & besti, int & bestj, int & bestsize) const {
        besti = alo; bestj = blo; bestsize = 0;
        std::unordered_map<int, int> j2len, newj2len;
        for (int i = alo; i < ahi; ++i) {
            newj2len.clear();
            auto it = b2j.find(a[i]);
            if (it != b2j.end()) {
                for (int j : it->second) {
                    if (j < blo) continue;
                    if (j >= bhi) break;
                    auto prev = j2len.find(j - 1);
                    const int k = newj2len[j] = (prev == j2len.end() ? 0 : prev->second) + 1;
                    if (k > bestsize) { besti = i - k + 1; bestj = j - k + 1; bestsize = k; }
                }
            }
            std::swap(j2len, newj2len);
        }
        while (besti > alo && bestj > blo && a[besti - 1] == b[bestj - 1]) { --besti; --bestj; ++bestsize; }
        while (besti + bestsize < ahi && bestj + bestsize < bhi && a[besti + bestsize] == b[bestj + bestsize]) ++bestsize;
    }

    std::vector<std::array<int, 3>> matching_blocks() const {
        const int la = (int) a.size(), lb = (int) b.size();
        std::vector<std::array<int, 4>> queue{{0, la, 0, lb}};
        std::vector<std::array<int, 3>> blocks;
        while (!queue.empty()) {
            const auto [alo, ahi, blo, bhi] = queue.back();
            queue.pop_back();
            int i, j, k;
            longest(alo, ahi, blo, bhi, i, j, k);
            if (k) {
                blocks.push_back({i, j, k});
                if (alo < i && blo < j) queue.push_back({alo, i, blo, j});
                if (i + k < ahi && j + k < bhi) queue.push_back({i + k, ahi, j + k, bhi});
            }
        }
        std::sort(blocks.begin(), blocks.end());
        std::vector<std::array<int, 3>> out;
        int i1 = 0, j1 = 0, k1 = 0;
        for (const auto & [i2, j2, k2] : blocks) {
            if (i1 + k1 == i2 && j1 + k1 == j2) k1 += k2;
            else {
                if (k1) out.push_back({i1, j1, k1});
                i1 = i2; j1 = j2; k1 = k2;
            }
        }
        if (k1) out.push_back({i1, j1, k1});
        out.push_back({la, lb, 0});
        return out;
    }
};

}  // namespace

std::vector<Opcode> sequence_opcodes(const std::vector<std::string> & a, const std::vector<std::string> & b) {
    const Matcher m(a, b);
    std::vector<Opcode> out;
    int i = 0, j = 0;
    for (const auto & [ai, bj, size] : m.matching_blocks()) {
        char tag = 0;
        if (i < ai && j < bj) tag = 'r';
        else if (i < ai) tag = 'd';
        else if (j < bj) tag = 'i';
        if (tag) out.push_back({tag, i, ai, j, bj});
        i = ai + size;
        j = bj + size;
        if (size) out.push_back({'e', ai, i, bj, j});
    }
    return out;
}

EditDiff diff_spans(const std::vector<AlignedWord> & words, const std::vector<std::string> & target_tokens,
                    int total_frames, double frame_rate, double margin_seconds, int min_gap_frames, int max_spans) {
    std::vector<std::pair<std::string, const AlignedWord *>> expanded;
    for (const auto & w : words) for (const auto & t : w.tokens) expanded.push_back({t, &w});
    std::vector<std::string> original;
    for (const auto & e : expanded) original.push_back(e.first);
    EditDiff out;
    const int ne = (int) expanded.size();
    for (const Opcode & op : sequence_opcodes(original, target_tokens)) {
        if (op.tag == 'e') continue;
        double t0, t1;
        if (op.i1 != op.i2) {
            t0 = expanded[op.i1].second->start;
            t1 = expanded[op.i2 - 1].second->end;
        } else {
            t0 = op.i1 ? expanded[op.i1 - 1].second->end : 0.0;
            t1 = op.i1 < ne ? expanded[op.i1].second->start : total_frames / frame_rate;
        }
        const int start = std::min(total_frames - 1, std::max(0, py_floor((t0 - margin_seconds) * frame_rate)));
        const int end = std::max(start + 1, std::min(total_frames, py_ceil((t1 + margin_seconds) * frame_rate)));
        if (!out.spans.empty() && start - (out.spans.back().start + out.spans.back().length) < min_gap_frames) {
            Span & prev = out.spans.back();
            prev.length = std::max(end, prev.start + prev.length) - prev.start;
        } else {
            out.spans.push_back({start, end - start});
        }
        for (int j = op.j1; j < op.j2; ++j) out.new_target_tokens.push_back(j);
    }
    if (out.spans.empty()) value_error("target text equals the original transcript");
    if ((int) out.spans.size() > max_spans)
        value_error("edit touches " + std::to_string(out.spans.size()) + " regions; at most " + std::to_string(max_spans) +
                    " are supported");
    return out;
}

std::vector<AlignedWord> edit_words(const std::vector<WordIn> & words,
                                    const std::vector<std::vector<std::string>> & word_tokens,
                                    const std::vector<std::string> & original_tokens, double duration) {
    if (words.empty()) value_error("original_words is empty.");
    double previous_end = 0.0;
    if (words[0].start < 0) value_error("original_words[0] (" + repr(words[0].word) + ") starts before the clip.");
    for (size_t k = 0; k < words.size(); ++k) {
        const double start = words[k].start, end = words[k].end;
        const std::string label = "original_words[" + std::to_string(k) + "] (" + repr(words[k].word) + ")";
        if (!(std::isfinite(start) && std::isfinite(end))) value_error(label + " has a non-finite time.");
        if (!(start < end)) value_error(label + " must start before it ends (" + fmt3(start) + " >= " + fmt3(end) + ").");
        if (start < previous_end - 1e-6)
            value_error(label + " starts at " + fmt3(start) + " s, before the previous word ends (" + fmt3(previous_end) + " s).");
        if (end > duration + 1e-3)
            value_error(label + " ends at " + fmt3(end) + " s, after the clip (" + fmt3(duration) + " s).");
        previous_end = end;
    }
    std::vector<std::string> spelled;
    for (const auto & tokens : word_tokens) spelled.insert(spelled.end(), tokens.begin(), tokens.end());
    if (spelled != original_tokens) {
        size_t k = std::min(spelled.size(), original_tokens.size());
        for (size_t i = 0; i < std::min(spelled.size(), original_tokens.size()); ++i)
            if (spelled[i] != original_tokens[i]) { k = i; break; }
        auto around = [&](const std::vector<std::string> & v) {
            const auto s = py_slice(v, std::max<int>(0, (int) k - 2), (int) k + 3);
            const std::string j = join(s, " ");
            return repr(j.empty() ? "(nothing)" : j);
        };
        value_error("original_words do not spell original_text: they read " + around(spelled) + " where the text reads " +
                    around(original_tokens) + " (normalized).");
    }
    std::vector<AlignedWord> out;
    for (size_t k = 0; k < words.size() && k < word_tokens.size(); ++k)
        out.push_back({words[k].word, words[k].start, words[k].end, words[k].confidence, word_tokens[k]});
    return out;
}

std::vector<Span> EditWindow::spans(const std::vector<Span> & all) const {
    std::vector<Span> out;
    for (int k : regions) out.push_back({all[k].start - start, all[k].length});
    return out;
}

EditWindow whole_clip_window(const std::vector<Span> & spans, int total_frames, const std::string & target_text) {
    EditWindow w;
    w.start = 0;
    w.end = total_frames;
    for (int k = 0; k < (int) spans.size(); ++k) w.regions.push_back(k);
    w.text = target_text;
    return w;
}

namespace {

// WordDiff: the original's MMS words against the target's whitespace words, diffed over their zsa tokens.
struct WordDiff {
    const std::vector<AlignedWord> & words;
    std::vector<int> orig_start{0}, tgt_start{0};
    std::vector<Opcode> opcodes;

    WordDiff(const std::vector<AlignedWord> & w, const std::vector<std::string> & target_words,
             const std::vector<std::vector<std::string>> & target_word_tokens) : words(w) {
        if (target_word_tokens.size() != target_words.size()) value_error("One token list is needed per target word.");
        std::vector<std::string> a, b;
        for (const auto & x : w) { orig_start.push_back(orig_start.back() + (int) x.tokens.size()); a.insert(a.end(), x.tokens.begin(), x.tokens.end()); }
        for (const auto & t : target_word_tokens) { tgt_start.push_back(tgt_start.back() + (int) t.size()); b.insert(b.end(), t.begin(), t.end()); }
        opcodes = sequence_opcodes(a, b);
    }

    std::optional<int> target_cut(int i, bool left) const {
        const int n = (int) opcodes.size();
        for (int q = 0; q < n; ++q) {
            const Opcode & o = opcodes[left ? q : n - 1 - q];
            if (o.i1 < i && i < o.i2) return o.tag == 'e' ? std::optional<int>(o.j1 + i - o.i1) : std::nullopt;
            if ((left ? o.i1 : o.i2) == i) return left ? o.j1 : o.j2;
        }
        return i == 0 ? 0 : tgt_start.back();
    }

    std::optional<int> target_word(int k, bool left) const {
        const auto j = target_cut(orig_start[k], left);
        if (!j) return std::nullopt;
        std::optional<int> found;
        for (int t = 0; t < (int) tgt_start.size(); ++t)
            if (tgt_start[t] == *j) { found = t; if (left) break; }
        return found;
    }

    std::pair<int, int> clean(int a, int b) const {
        while (a > 0 && !target_word(a, true)) --a;
        while (b < (int) words.size() && !target_word(b, false)) ++b;
        return {a, b};
    }

    std::pair<std::optional<int>, std::optional<int>> target_range(int a, int b) const {
        return {target_word(a, true), target_word(b, false)};
    }
};

}  // namespace

std::vector<EditWindow> plan_edit_windows(const std::vector<Span> & spans, const std::vector<AlignedWord> & words,
                                          const std::vector<std::string> & target_words,
                                          const std::vector<std::vector<std::string>> & target_word_tokens,
                                          int total_frames, double frame_rate, double context_seconds,
                                          const std::optional<std::string> & target_text) {
    const int n = (int) words.size();
    if (!n) value_error("MMS found no words in the audio, so the edit cannot be windowed.");
    const WordDiff diff(words, target_words, target_word_tokens);
    auto boundary = [&](int k) {
        if (k == 0) return 0;
        if (k == n) return total_frames;
        return std::min(total_frames, (int) std::nearbyint((words[k - 1].end + words[k].start) / 2 * frame_rate));
    };
    struct Win { int a, b; std::vector<int> regions; };
    std::vector<Win> windows;
    for (int index = 0; index < (int) spans.size(); ++index) {
        const double t0 = spans[index].start / frame_rate - context_seconds;
        const double t1 = (spans[index].start + spans[index].length) / frame_rate + context_seconds;
        int a = n;
        for (int k = 0; k < n; ++k) if (words[k].end > t0) { a = k; break; }
        int last = 0;
        for (int k = n - 1; k >= 0; --k) if (words[k].start < t1) { last = k + 1; break; }
        const auto [ca, cb] = diff.clean(a, std::max(a, last));
        windows.push_back({ca, cb, {index}});
    }
    auto frames = [&](const Win & w) {
        int first = spans[w.regions[0]].start, last = 0;
        for (int k : w.regions) {
            first = std::min(first, spans[k].start);
            last = std::max(last, spans[k].start + spans[k].length);
        }
        return std::make_pair(std::min(boundary(w.a), first), std::max(boundary(w.b), last));
    };
    std::stable_sort(windows.begin(), windows.end(), [](const Win & x, const Win & y) {
        return std::make_pair(x.a, x.b) < std::make_pair(y.a, y.b);
    });
    std::vector<Win> merged{windows[0]};
    for (size_t q = 1; q < windows.size(); ++q) {
        Win & current = merged.back();
        const Win & window = windows[q];
        if (window.a < current.b || frames(window).first < frames(current).second) {
            current.a = std::min(current.a, window.a);
            current.b = std::max(current.b, window.b);
            current.regions.insert(current.regions.end(), window.regions.begin(), window.regions.end());
            std::sort(current.regions.begin(), current.regions.end());
        } else {
            merged.push_back(window);
        }
    }
    std::vector<EditWindow> out;
    for (const Win & w : merged) {
        const auto [start, end] = frames(w);
        EditWindow e;
        e.start = start;
        e.end = end;
        e.regions = w.regions;
        if (w.a == 0 && w.b == n && target_text) e.text = *target_text;
        else {
            const auto [ja, jb] = diff.target_range(w.a, w.b);
            e.text = join(py_slice(target_words, ja, jb), " ");
        }
        out.push_back(std::move(e));
    }
    return out;
}

std::vector<double> expected_fill_seconds(const std::vector<Span> & spans, const std::vector<AlignedWord> & words,
                                          const std::vector<std::string> & target_words,
                                          const std::vector<std::vector<std::string>> & target_word_tokens,
                                          double frame_rate) {
    const WordDiff diff(words, target_words, target_word_tokens);
    auto chars = [](const std::vector<std::vector<std::string>> & lists) {
        double s = 0;
        for (const auto & tokens : lists) for (const auto & t : tokens) s += token_chars(t);
        return s;
    };
    auto word_tokens = [&](int a, int b) {
        std::vector<std::vector<std::string>> out;
        for (int k = std::max(0, a); k < std::min(b, (int) words.size()); ++k) out.push_back(words[k].tokens);
        return out;
    };
    double spoken = 0;
    for (const auto & w : words) spoken += w.end - w.start;
    const double rate = spoken > 0 ? chars(word_tokens(0, (int) words.size())) / spoken : 0.0;
    std::vector<double> mids;
    for (const auto & w : words) mids.push_back((w.start + w.end) / 2);
    std::vector<double> out;
    for (const Span & s : spans) {
        const double r0 = s.start / frame_rate, r1 = (s.start + s.length) / frame_rate;
        std::vector<int> inside;
        for (int k = 0; k < (int) mids.size(); ++k) if (r0 <= mids[k] && mids[k] < r1) inside.push_back(k);
        int a;
        if (!inside.empty()) a = inside[0];
        else {
            a = (int) words.size();
            for (int k = 0; k < (int) mids.size(); ++k) if (mids[k] >= r0) { a = k; break; }
        }
        const auto [ca, cb] = diff.clean(a, inside.empty() ? a : inside.back() + 1);
        const auto [ja, jb] = diff.target_range(ca, cb);
        const double original = chars(word_tokens(ca, cb)), target = chars(py_slice(target_word_tokens, ja, jb));
        if (original) out.push_back((r1 - r0) * target / original);
        else out.push_back(rate > 0 ? target / rate : r1 - r0);
    }
    return out;
}

static int levenshtein(const std::u32string & a, const std::u32string & b) {
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

Verify wildcard_verify(const std::vector<std::string> & reference_tokens,
                       const std::vector<std::string> & hypothesis_tokens) {
    const auto & ref = reference_tokens;
    const auto & hyp = hypothesis_tokens;
    const int n = (int) ref.size(), m = (int) hyp.size();
    std::vector<char> wild(n);
    std::vector<std::u32string> rc(n), hc(m);
    for (int i = 0; i < n; ++i) { wild[i] = is_group_token(ref[i]); rc[i] = cps(ref[i]); }
    for (int j = 0; j < m; ++j) hc[j] = cps(hyp[j]);
    using Cost = std::pair<int, int>;
    auto add = [](Cost x, Cost y) { return Cost{x.first + y.first, x.second + y.second}; };
    auto sub = [&](int i, int j) {
        if (wild[i]) return Cost{0, 0};
        return Cost{ref[i] != hyp[j] ? 1 : 0, levenshtein(rc[i], hc[j])};
    };
    auto del = [&](int i) { return Cost{1, wild[i] ? 0 : (int) rc[i].size()}; };
    auto ins = [&](int j) { return Cost{1, (int) hc[j].size()}; };
    std::vector<std::vector<Cost>> cost(n + 1, std::vector<Cost>(m + 1, Cost{0, 0}));
    for (int i = 1; i <= n; ++i) cost[i][0] = add(cost[i - 1][0], del(i - 1));
    for (int j = 1; j <= m; ++j) cost[0][j] = add(cost[0][j - 1], ins(j - 1));
    for (int i = 1; i <= n; ++i)
        for (int j = 1; j <= m; ++j)
            cost[i][j] = std::min({add(cost[i - 1][j - 1], sub(i - 1, j - 1)), add(cost[i - 1][j], del(i - 1)),
                                   add(cost[i][j - 1], ins(j - 1))});
    std::map<int, std::string> heard;
    int i = n, j = m;
    while (i > 0) {
        if (j > 0 && cost[i][j] == add(cost[i - 1][j - 1], sub(i - 1, j - 1))) {
            if (wild[i - 1]) heard[i - 1] = hyp[j - 1];
            --i; --j;
        } else if (j > 0 && cost[i][j] == add(cost[i][j - 1], ins(j - 1))) {
            --j;
        } else {
            --i;
        }
    }
    std::string filled, said;
    for (int k = 0; k < n; ++k) {
        if (!wild[k]) filled += ref[k];
        else if (heard.count(k)) filled += heard[k];
    }
    for (const auto & h : hyp) said += h;
    const std::u32string fc = cps(filled), sc = cps(said);
    Verify out;
    out.word_errors = cost[n][m].first;
    out.ref_words = n;
    out.cer = fc.empty() ? 0.0 : sc.empty() ? 1.0 : (double) levenshtein(fc, sc) / (double) fc.size();
    for (int k = 0; k < n; ++k) {
        if (!wild[k]) continue;
        VerifyGroup g;
        g.ipa = ref[k].substr(3, ref[k].size() - 6);
        if (heard.count(k)) g.heard = heard[k];
        out.groups.push_back(std::move(g));
    }
    return out;
}

std::string fill_groups(const std::string & text, const std::vector<std::optional<std::string>> & heard) {
    std::string out;
    size_t cursor = 0, next = 0;
    for (const auto & [a, b] : group_matches(text)) {
        out += text.substr(cursor, a - cursor);
        if (next >= heard.size()) value_error("fill_groups: fewer heard words than groups");
        const auto & h = heard[next++];
        if (h) out += lower(*h);
        cursor = b;
    }
    out += text.substr(cursor);
    return join(split_words(out), " ");
}

}  // namespace zonos3_edit
