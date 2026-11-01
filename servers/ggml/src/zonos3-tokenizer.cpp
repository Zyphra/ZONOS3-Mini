// zonos3-tokenizer.cpp — SuperBPE: added-token split, the Split regex pre-tokenizer
//   \p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S)   (behavior Isolated)
// as a hand-written matcher, GPT-2 byte-level mapping, then rank-ordered BPE merges.
#include "zonos3-tokenizer.h"
#include "unicode-data.h"

#include "json.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace {

bool in_ranges(const uint32_t (*r)[2], size_t n, uint32_t cp) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (cp < r[mid][0]) hi = mid;
        else if (cp > r[mid][1]) lo = mid + 1;
        else return true;
    }
    return false;
}

constexpr uint32_t INVALID = 0xFFFFFFFFu; // undecodable byte: neither L, N nor space

bool is_L(uint32_t cp) { return cp != INVALID && in_ranges(ZONOS3_UNI_L, sizeof(ZONOS3_UNI_L) / sizeof(ZONOS3_UNI_L[0]), cp); }
bool is_N(uint32_t cp) { return cp != INVALID && in_ranges(ZONOS3_UNI_N, sizeof(ZONOS3_UNI_N) / sizeof(ZONOS3_UNI_N[0]), cp); }
bool is_space(uint32_t cp) { // Unicode White_Space (Oniguruma \s)
    return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F ||
           cp == 0x205F || cp == 0x3000;
}
bool is_P(uint32_t cp) { return !is_space(cp) && !is_L(cp) && !is_N(cp); }

struct cp_span { uint32_t cp; size_t b0, b1; }; // code point and its byte range in the segment

std::vector<cp_span> decode_utf8(const std::string & s, size_t b, size_t e) {
    std::vector<cp_span> out;
    const auto * p = (const unsigned char *) s.data();
    size_t i = b;
    while (i < e) {
        const unsigned char c = p[i];
        int len = 0;
        uint32_t cp = 0;
        if (c < 0x80)              { len = 1; cp = c; }
        else if ((c & 0xE0) == 0xC0) { len = 2; cp = c & 0x1F; }
        else if ((c & 0xF0) == 0xE0) { len = 3; cp = c & 0x0F; }
        else if ((c & 0xF8) == 0xF0) { len = 4; cp = c & 0x07; }
        bool ok = len > 0 && i + len <= e;
        for (int k = 1; ok && k < len; ++k) {
            if ((p[i + k] & 0xC0) != 0x80) ok = false;
            else cp = (cp << 6) | (p[i + k] & 0x3F);
        }
        if (!ok) { out.push_back({INVALID, i, i + 1}); ++i; continue; }
        out.push_back({cp, i, i + len});
        i += len;
    }
    return out;
}

// Split [0, n) into pieces: each regex match is a piece, each maximal gap between matches is one.
std::vector<std::pair<size_t, size_t>> pretokenize(const std::vector<cp_span> & cps) {
    const size_t n = cps.size();
    std::vector<std::pair<size_t, size_t>> pieces;
    size_t gap = 0, i = 0;
    auto match_at = [&](size_t i) -> size_t { // end of the match starting at i, or i if none
        // A1: \p{N}{1,3}
        if (is_N(cps[i].cp)) {
            size_t j = i;
            while (j < n && j - i < 3 && is_N(cps[j].cp)) ++j;
            return j;
        }
        // A2: ' '?[^\s\p{L}\p{N}]{2,}[\r\n/]*
        {
            const size_t j = i + (cps[i].cp == 0x20 ? 1 : 0);
            size_t k = j;
            while (k < n && is_P(cps[k].cp)) ++k;
            if (k - j >= 2) {
                while (k < n && (cps[k].cp == '\r' || cps[k].cp == '\n' || cps[k].cp == '/')) ++k;
                return k;
            }
        }
        // A3: ' +(?!\S)'
        if (cps[i].cp == 0x20) {
            size_t e = i;
            while (e < n && cps[e].cp == 0x20) ++e;
            if (e == n || is_space(cps[e].cp)) return e;
            if (e - i >= 2) return e - 1;
        }
        return i;
    };
    while (i < n) {
        const size_t e = match_at(i);
        if (e == i) { ++i; continue; }
        if (gap < i) pieces.push_back({gap, i});
        pieces.push_back({i, e});
        i = gap = e;
    }
    if (gap < n) pieces.push_back({gap, n});
    return pieces;
}

// GPT-2 bytes_to_unicode: printable bytes map to themselves, the rest to U+0100 + k.
const std::vector<std::string> & byte_symbols() {
    static const std::vector<std::string> tbl = [] {
        std::vector<std::string> t(256);
        int extra = 0;
        for (int b = 0; b < 256; ++b) {
            const bool keep = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            const uint32_t cp = keep ? (uint32_t) b : 256u + (uint32_t) extra++;
            std::string s;
            if (cp < 0x80) s += (char) cp;
            else { s += (char) (0xC0 | (cp >> 6)); s += (char) (0x80 | (cp & 0x3F)); }
            t[b] = s;
        }
        return t;
    }();
    return tbl;
}

void bpe_piece(const zonos3_tokenizer & tok, const std::string & s, size_t b0, size_t b1,
               std::vector<int32_t> & out) {
    const auto & bs = byte_symbols();
    std::vector<std::string> sym;
    sym.reserve(b1 - b0);
    for (size_t i = b0; i < b1; ++i) sym.push_back(bs[(unsigned char) s[i]]);
    std::string key;
    while (sym.size() > 1) {
        int32_t best = INT32_MAX;
        size_t at = 0;
        for (size_t i = 0; i + 1 < sym.size(); ++i) {
            key.assign(sym[i]); key += '\x01'; key += sym[i + 1];
            const auto it = tok.merge_rank.find(key);
            if (it != tok.merge_rank.end() && it->second < best) { best = it->second; at = i; }
        }
        if (best == INT32_MAX) break;
        sym[at] += sym[at + 1];
        sym.erase(sym.begin() + (ptrdiff_t) at + 1);
    }
    for (const auto & x : sym) {
        const auto it = tok.vocab.find(x);
        if (it == tok.vocab.end()) { fprintf(stderr, "zonos3: tokenizer: symbol not in vocab (skipped)\n"); continue; }
        out.push_back(it->second + tok.offset);
    }
}

} // namespace

bool zonos3_tokenizer_load(zonos3_tokenizer & tok, const std::string & json, int32_t offset,
                           int32_t bos, int32_t eos, std::string & err) {
    tok = zonos3_tokenizer();
    tok.offset = offset; tok.bos = bos; tok.eos = eos;
    nlohmann::json j = nlohmann::json::parse(json, nullptr, /*allow_exceptions=*/false);
    if (j.is_discarded() || !j.is_object()) { err = "tokenizer: invalid JSON"; return false; }
    const auto & model = j["model"];
    if (!model.is_object() || model.value("type", "") != "BPE") { err = "tokenizer: model.type is not BPE"; return false; }
    if (!model["vocab"].is_object() || !model["merges"].is_array()) { err = "tokenizer: missing vocab/merges"; return false; }
    for (auto it = model["vocab"].begin(); it != model["vocab"].end(); ++it) {
        if (!it.value().is_number_integer()) { err = "tokenizer: non-integer vocab id"; return false; }
        tok.vocab[it.key()] = it.value().get<int32_t>();
    }
    int32_t rank = 0;
    for (const auto & mg : model["merges"]) {
        if (!mg.is_array() || mg.size() != 2 || !mg[0].is_string() || !mg[1].is_string()) {
            err = "tokenizer: merges must be [a,b] arrays"; return false;
        }
        tok.merge_rank.emplace(mg[0].get<std::string>() + '\x01' + mg[1].get<std::string>(), rank++);
    }
    if (j.contains("added_tokens") && j["added_tokens"].is_array()) {
        for (const auto & a : j["added_tokens"]) {
            if (!a.is_object() || !a["content"].is_string() || !a["id"].is_number_integer()) {
                err = "tokenizer: malformed added_tokens"; return false;
            }
            const std::string c = a["content"].get<std::string>();
            if (!c.empty()) tok.added.push_back({c, a["id"].get<int32_t>()});
        }
    }
    if (tok.vocab.empty()) { err = "tokenizer: empty vocab"; return false; }
    return true;
}

namespace {

// SuperBPE of `utf8` (added tokens split first), no BOS/EOS.
void bpe_text(const zonos3_tokenizer & tok, const std::string & utf8, std::vector<int32_t> & out) {
    auto encode_segment = [&](size_t b, size_t e) {
        if (b >= e) return;
        const std::vector<cp_span> cps = decode_utf8(utf8, b, e);
        for (const auto & pc : pretokenize(cps))
            bpe_piece(tok, utf8, cps[pc.first].b0, cps[pc.second - 1].b1, out);
    };
    size_t pos = 0;
    while (pos < utf8.size()) {
        size_t best = std::string::npos, best_len = 0;
        int32_t best_id = 0;
        for (const auto & a : tok.added) {
            const size_t f = utf8.find(a.first, pos);
            if (f != std::string::npos && (f < best || (f == best && a.first.size() > best_len))) {
                best = f; best_len = a.first.size(); best_id = a.second;
            }
        }
        if (best == std::string::npos) { encode_segment(pos, utf8.size()); break; }
        encode_segment(pos, best);
        out.push_back(best_id + tok.offset);
        pos = best + best_len;
    }
}

// Python repr() of a one-code-point str (printable code points verbatim; quotes as Python picks them).
std::string py_char_repr(const std::string & ch) {
    const std::vector<cp_span> cps = decode_utf8(ch, 0, ch.size());
    const uint32_t cp = cps.size() == 1 ? cps[0].cp : INVALID;
    char buf[16];
    std::string body;
    if (cp == '\\') body = "\\\\";
    else if (cp == '\n') body = "\\n";
    else if (cp == '\r') body = "\\r";
    else if (cp == '\t') body = "\\t";
    else if (cp < 0x20 || (cp >= 0x7F && cp <= 0xA0) || cp == 0xAD) { snprintf(buf, sizeof buf, "\\x%02x", cp); body = buf; }
    else if (cp != INVALID && ((cp >= 0xE000 && cp <= 0xF8FF) || (cp >= 0x2000 && cp <= 0x200F) ||
                               (cp >= 0x2028 && cp <= 0x202F) || (cp >= 0x205F && cp <= 0x206F) || cp == 0x1680 ||
                               cp == 0x3000 || cp == 0xFEFF)) {
        snprintf(buf, sizeof buf, "\\u%04x", cp); body = buf;
    } else body = ch;
    if (cp == '\'') return "\"'\"";
    return "'" + body + "'";
}

} // namespace

std::vector<int32_t> zonos3_tokenize(const zonos3_tokenizer & tok, const std::string & utf8) {
    std::vector<int32_t> out{tok.bos};
    bpe_text(tok, utf8, out);
    out.push_back(tok.eos);
    return out;
}

bool zonos3_tokenizer_set_spec(zonos3_tokenizer & tok, const std::string & spec_json, uint32_t text_vocab,
                               uint32_t text_extra_tokens, std::string & err) {
    const nlohmann::json spec = nlohmann::json::parse(spec_json, nullptr, /*allow_exceptions=*/false);
    if (spec.is_discarded() || !spec.is_object()) { err = "text_tokenizer_spec: invalid JSON"; return false; }
    try {
        if (spec.at("kind").get<std::string>() != "superbpe" || spec.at("offset").get<int64_t>() != 4 ||
            spec.at("bos").get<int64_t>() != 2 || spec.at("eos").get<int64_t>() != 3 ||
            (int64_t) text_vocab != spec.at("vocab_size").get<int64_t>() + 4 || tok.offset != 4 || tok.bos != 2 ||
            tok.eos != 3) {
            err = "SuperBPE ID convention mismatch"; return false;
        }
        tok.has_phonemes = false;
        tok.phonemes = zonos3_phoneme_spec();
        if (spec.contains("phonemes") && !spec["phonemes"].is_null()) {
            const auto & p = spec["phonemes"];
            zonos3_phoneme_spec ph;
            ph.open_marker = p.at("open_marker").get<std::string>();
            ph.close_marker = p.at("close_marker").get<std::string>();
            ph.open_id = p.at("open_id").get<int32_t>();
            ph.close_id = p.at("close_id").get<int32_t>();
            ph.symbols = p.at("symbols").get<std::vector<std::string>>();
            if (ph.open_id < (int64_t) text_vocab + 1 || ph.close_id != ph.open_id + 1) {
                err = "Phoneme token IDs must follow the text padding row (and any edit marker rows)"; return false;
            }
            bool valid = decode_utf8(ph.open_marker, 0, ph.open_marker.size()).size() == 1 &&
                         decode_utf8(ph.close_marker, 0, ph.close_marker.size()).size() == 1;
            for (size_t i = 0; valid && i < ph.symbols.size(); ++i) {
                const auto & s = ph.symbols[i];
                valid = decode_utf8(s, 0, s.size()).size() == 1 && s != ph.open_marker && s != ph.close_marker &&
                        std::find(ph.symbols.begin(), ph.symbols.begin() + (ptrdiff_t) i, s) == ph.symbols.begin() + (ptrdiff_t) i;
            }
            if (!valid) { err = "Invalid phoneme symbols"; return false; }
            const int64_t before = (int64_t) ph.open_id - text_vocab - 1;
            if (before + 2 + (int64_t) ph.symbols.size() != (int64_t) text_extra_tokens) {
                err = "Phoneme tokens do not match text_extra_tokens (they must be the last extra rows)"; return false;
            }
            tok.phonemes = std::move(ph);
            tok.has_phonemes = true;
        }
    } catch (const std::exception & e) {
        err = std::string("text_tokenizer_spec: ") + e.what();
        return false;
    }
    tok.spec_json = spec_json;
    return true;
}

bool zonos3_payload_ids(const zonos3_tokenizer & tok, const std::string & text, std::vector<int32_t> & ids,
                        std::string & err) {
    if (!tok.has_phonemes) { bpe_text(tok, text, ids); return true; }
    const auto & ph = tok.phonemes;
    const std::string & open = ph.open_marker, & close = ph.close_marker;
    if (text.find(open) == std::string::npos && text.find(close) == std::string::npos) { bpe_text(tok, text, ids); return true; }
    size_t pos = 0;
    while (pos < text.size()) {
        const size_t start = text.find(open, pos);
        const std::string stretch = start == std::string::npos ? text.substr(pos) : text.substr(pos, start - pos);
        if (stretch.find(close) != std::string::npos) { err = "Unbalanced phoneme group markers"; return false; }
        if (!stretch.empty()) bpe_text(tok, stretch, ids);
        if (start == std::string::npos) break;
        const size_t gstart = start + open.size();
        const size_t end = text.find(close, gstart);
        const std::string group = end != std::string::npos ? text.substr(gstart, end - gstart) : std::string();
        if (end == std::string::npos || group.empty() || group.find(open) != std::string::npos) {
            err = "Unbalanced or empty phoneme group"; return false;
        }
        std::vector<std::string> unknown;
        std::vector<int32_t> phones;
        for (const auto & c : decode_utf8(group, 0, group.size())) {
            const std::string s = group.substr(c.b0, c.b1 - c.b0);
            const auto it = std::find(ph.symbols.begin(), ph.symbols.end(), s);
            if (it == ph.symbols.end()) unknown.push_back(s);
            else phones.push_back(ph.open_id + 2 + (int32_t) (it - ph.symbols.begin()));
        }
        if (!unknown.empty()) {
            std::sort(unknown.begin(), unknown.end());
            unknown.erase(std::unique(unknown.begin(), unknown.end()), unknown.end());
            err = "Phones outside the inventory: [";
            for (size_t i = 0; i < unknown.size(); ++i) err += (i ? ", " : "") + py_char_repr(unknown[i]);
            err += "]";
            return false;
        }
        ids.push_back(ph.open_id);
        ids.insert(ids.end(), phones.begin(), phones.end());
        ids.push_back(ph.close_id);
        pos = end + close.size();
    }
    return true;
}

bool zonos3_text_ids(const zonos3_tokenizer & tok, const std::string & utf8, std::vector<int32_t> & out,
                     std::string & err) {
    out.assign(1, tok.bos);
    if (!zonos3_payload_ids(tok, utf8, out, err)) return false;
    out.push_back(tok.eos);
    return true;
}

std::string zonos3_check_phoneme_groups(const zonos3_tokenizer & tok, const std::string & utf8) {
    const zonos3_phoneme_spec defaults;
    const auto & ph = tok.has_phonemes ? tok.phonemes : defaults;
    if (utf8.find(ph.open_marker) == std::string::npos && utf8.find(ph.close_marker) == std::string::npos) return "";
    if (!tok.has_phonemes) return "This checkpoint has no phoneme tokens: remove the U+E000 / U+E001 pronunciation groups";
    std::vector<int32_t> ids;
    std::string err;
    if (!zonos3_payload_ids(tok, utf8, ids, err)) return "Pronunciation groups: " + err;
    return "";
}
