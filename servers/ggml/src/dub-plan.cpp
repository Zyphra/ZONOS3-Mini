// dub-plan.cpp — port of model/src/zonos3/dub/plan.py. Floating-point steps follow the numpy evaluation order
// (sequential cumsum, pairwise sums for mean, numpy.interp, numpy.linspace) so decisions match bit for bit.
#include "dub-plan.h"
#include "unicode-data.h"

#define POCKETFFT_NO_MULTITHREADING
#include "pocketfft_hdronly.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <set>
#include <stdexcept>

namespace dubplan {

namespace {

constexpr double INF = std::numeric_limits<double>::infinity();

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

// unicodedata.category(ch)[0] in "LN" or ch == "'"
bool wordy(uint32_t cp) {
    return cp == '\'' || in_ranges(ZONOS3_UNI_L, sizeof(ZONOS3_UNI_L) / sizeof(ZONOS3_UNI_L[0]), cp) ||
           in_ranges(ZONOS3_UNI_N, sizeof(ZONOS3_UNI_N) / sizeof(ZONOS3_UNI_N[0]), cp);
}

// Python str.isspace (also what re's \s / \S use on str patterns)
bool py_space(uint32_t cp) {
    return (cp >= 0x09 && cp <= 0x0D) || (cp >= 0x1C && cp <= 0x20) || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F ||
           cp == 0x3000;
}

constexpr uint32_t CJK_BLOCKS[][2] = {
    {0x3040, 0x30FF}, {0x31F0, 0x31FF}, {0x3400, 0x4DBF}, {0x4E00, 0x9FFF}, {0xF900, 0xFAFF}, {0xFF66, 0xFF9F},
};

bool cjk(uint32_t cp) {
    for (const auto &b : CJK_BLOCKS)
        if (b[0] <= cp && cp <= b[1]) return true;
    return false;
}

// Code points of a UTF-8 string with the byte offset of each (offs has one extra entry: the total byte length).
struct Decoded {
    std::vector<uint32_t> cps;
    std::vector<size_t> offs;
};

Decoded decode(const std::string &s) {
    Decoded d;
    const auto *p = (const unsigned char *) s.data();
    size_t i = 0, n = s.size();
    while (i < n) {
        d.offs.push_back(i);
        uint32_t c = p[i];
        int len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 1;
        if (i + len > n) len = 1;
        if (len == 2) c &= 0x1F;
        else if (len == 3) c &= 0x0F;
        else if (len == 4) c &= 0x07;
        for (int k = 1; k < len; ++k) c = (c << 6) | (p[i + k] & 0x3F);
        d.cps.push_back(c);
        i += len;
    }
    d.offs.push_back(n);
    return d;
}

std::string slice(const std::string &s, const Decoded &d, int a, int b) {
    const int n = int(d.cps.size());
    a = std::clamp(a, 0, n);
    b = std::clamp(b, 0, n);
    if (b <= a) return "";
    return s.substr(d.offs[a], d.offs[b] - d.offs[a]);
}

// numpy's pairwise summation (the inner-axis add.reduce of np.sum / np.mean)
double pairwise_sum(const double *a, size_t n) {
    if (n < 8) {
        double r = 0.0;
        for (size_t i = 0; i < n; ++i) r += a[i];
        return r;
    }
    if (n <= 128) {
        double r[8];
        for (int j = 0; j < 8; ++j) r[j] = a[j];
        size_t i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; ++j) r[j] += a[i + j];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    size_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
}

// LipMap._map: numpy.interp inside, the nearer end's offset outside
double map_time(double t, const std::vector<double> &xp, const std::vector<double> &fp) {
    if (xp.empty()) throw std::invalid_argument("array of sample points is empty");
    if (t < xp.front()) return t + (fp.front() - xp.front());
    if (t > xp.back()) return t + (fp.back() - xp.back());
    return interp(t, xp, fp);
}

}  // namespace

bool unspaced(const std::string &lang) {
    for (const char *u : UNSPACED)
        if (lang == u) return true;
    return false;
}

std::vector<TextUnit> text_units(const std::string &text, const std::string &lang) {
    const Decoded d = decode(text);
    const int n = int(d.cps.size());
    std::vector<std::array<int, 2>> spans;  // [start, end)
    int lead = -1;                          // start of non-wordy chars seen before the first unit
    auto attach = [&](int start, int end) {
        if (!spans.empty()) spans.back()[1] = end;
        else if (lead < 0) lead = start;
    };
    if (unspaced(lang)) {
        int run_end = -1;  // end of the open run of non-CJK wordy chars
        for (int i = 0; i < n; ++i) {
            const uint32_t ch = d.cps[i];
            if (py_space(ch)) continue;
            if (wordy(ch)) {
                if (cjk(ch)) {
                    spans.push_back({i, i + 1});
                    run_end = -1;
                } else if (run_end == i) {
                    spans.back()[1] = i + 1;
                    run_end = i + 1;
                } else {
                    spans.push_back({i, i + 1});
                    run_end = i + 1;
                }
            } else {
                attach(i, i + 1);
            }
        }
    } else {
        int i = 0;
        while (i < n) {  // _TOKEN = \S+
            if (py_space(d.cps[i])) {
                ++i;
                continue;
            }
            int j = i;
            bool any = false;
            for (; j < n && !py_space(d.cps[j]); ++j) any = any || wordy(d.cps[j]);
            if (any) spans.push_back({i, j});
            else attach(i, j);
            i = j;
        }
    }
    if (!spans.empty() && lead >= 0) spans[0][0] = lead;
    std::vector<TextUnit> out;
    out.reserve(spans.size());
    for (const auto &s : spans) out.push_back({slice(text, d, s[0], s[1]), s[0], s[1]});
    return out;
}

std::vector<int> unit_bytes(const std::string &text, const std::vector<TextUnit> &units) {
    const Decoded d = decode(text);
    std::vector<int> out;
    out.reserve(units.size());
    for (size_t k = 0; k < units.size(); ++k) {
        const int stop = k + 1 < units.size() ? units[k + 1].start : units[k].end;
        out.push_back(int(slice(text, d, units[k].start, stop).size()));
    }
    return out;
}

std::vector<bool> break_after(const std::string &text, const std::vector<TextUnit> &units) {
    const Decoded d = decode(text);
    const Decoded punct = decode(BREAK_PUNCT);
    std::vector<bool> out;
    out.reserve(units.size());
    for (const auto &u : units) {
        int i = u.end - 1;
        if (i < 0) i += int(d.cps.size());  // Python negative index
        const uint32_t ch = d.cps.at(size_t(i));
        out.push_back(std::find(punct.cps.begin(), punct.cps.end(), ch) != punct.cps.end());
    }
    return out;
}

std::string span_text(const std::string &text, const std::vector<TextUnit> &units, int a, int b) {
    if (a == b) return "";
    const Decoded d = decode(text);
    return slice(text, d, units.at(size_t(a)).start, units.at(size_t(b - 1)).end);
}

std::string join_words(const std::vector<std::string> &words, const std::string &lang) {
    const std::string sep = unspaced(lang) ? "" : " ";
    std::string out;
    for (size_t i = 0; i < words.size(); ++i) {
        if (i) out += sep;
        out += words[i];
    }
    return out;
}

std::vector<Range> ref_segments(const std::vector<TimedWord> &words, double min_pause) {
    if (words.empty()) return {};
    std::vector<int> starts = {0};
    bool have_prev = false;
    double prev_end = 0.0;
    for (size_t j = 0; j < words.size(); ++j) {
        const auto &w = words[j];
        if (w.end - w.start < 0.001) continue;  // zero-width words never start a segment
        if (have_prev && j > 0 && w.start - prev_end >= min_pause) starts.push_back(int(j));
        prev_end = w.end;
        have_prev = true;
    }
    std::vector<Range> out;
    for (size_t k = 0; k < starts.size(); ++k)
        out.emplace_back(starts[k], k + 1 < starts.size() ? starts[k + 1] : int(words.size()));
    return out;
}

std::vector<Range> project_segments(const std::vector<Range> &ref_ranges,
                                    const std::vector<std::pair<double, double>> &ref_spans,
                                    const std::vector<std::pair<int, int>> &links, const std::vector<int> &ubytes,
                                    const std::vector<bool> &breaks) {
    const size_t n_seg = ref_ranges.size(), m = ubytes.size();
    if (n_seg == 0) return {};
    if (n_seg == 1) return {{0, int(m)}};
    std::map<int, size_t> seg_of;
    for (size_t k = 0; k < n_seg; ++k)
        for (int i = ref_ranges[k].first; i < ref_ranges[k].second; ++i) seg_of[i] = k;
    // per_unit[k][j]: links into target unit j whose reference word lies outside segment k
    std::vector<std::vector<double>> per_unit(n_seg, std::vector<double>(m, 0.0));
    for (const auto &[i, j] : links) {
        auto it = seg_of.find(i);
        if (j < 0 || size_t(j) >= m || it == seg_of.end()) continue;
        for (size_t k = 0; k < n_seg; ++k) per_unit[k][j] += 1.0;
        per_unit[it->second][j] -= 1.0;
    }
    std::vector<std::vector<double>> prefix(n_seg, std::vector<double>(m + 1, 0.0));
    for (size_t k = 0; k < n_seg; ++k)
        for (size_t j = 0; j < m; ++j) prefix[k][j + 1] = prefix[k][j] + per_unit[k][j];
    std::vector<double> cum_bytes(m + 1, 0.0);
    for (size_t j = 0; j < m; ++j) cum_bytes[j + 1] = cum_bytes[j] + double(ubytes[j]);
    const double total_bytes = cum_bytes[m];
    std::vector<double> durations;
    double total_dur = 0.0;
    for (const auto &[s, e] : ref_spans) {
        durations.push_back(std::max(0.0, e - s));
        total_dur += durations.back();
    }
    std::vector<double> pen(m + 1, 0.0);
    for (size_t b = 1; b < m; ++b)
        if (!breaks.at(b - 1)) pen[b] = 0.5;

    std::vector<double> f(m + 1, INF), g(m + 1);
    f[0] = 0.0;
    std::vector<std::vector<size_t>> back(n_seg, std::vector<size_t>(m + 1, 0));
    const bool shares = total_bytes > 0 && total_dur > 0;
    for (size_t k = 0; k < n_seg; ++k) {
        const double want = shares ? durations.at(k) / total_dur : 0.0;
        for (size_t b = 0; b <= m; ++b) {
            size_t arg = 0;  // np.argmin: first minimum, over a (a > b is inf)
            double best = 0.0;
            for (size_t a = 0; a <= m; ++a) {
                double total = INF;
                if (a <= b) {
                    double cost = prefix[k][b] - prefix[k][a];
                    if (shares) cost = cost + std::abs((cum_bytes[b] - cum_bytes[a]) / total_bytes - want);
                    total = f[a] + cost;
                }
                if (a == 0 || total < best) {
                    arg = a;
                    best = total;
                }
            }
            back[k][b] = arg;
            g[b] = k + 1 < n_seg ? best + pen[b] : best;
        }
        f = g;
    }
    std::vector<int> bounds = {int(m)};
    size_t b = m;
    for (size_t k = n_seg; k-- > 0;) {
        b = back[k][b];
        bounds.push_back(int(b));
    }
    std::reverse(bounds.begin(), bounds.end());  // b_0 .. b_K
    std::vector<Range> out;
    for (size_t k = 0; k < n_seg; ++k) out.emplace_back(bounds[k], bounds[k + 1]);
    return out;
}

RateScale RateScale::extended(const std::vector<double> &row_edges, const std::vector<double> &below,
                              const std::vector<double> &above, const std::map<int, double> &doses) {
    const int n = int(row_edges.size()) + 1;
    int lo = 0, hi = n - 1;
    for (const auto &[b, _] : doses) {
        if (b < 0) lo = std::min(lo, b);
        if (b >= n) hi = std::max(hi, b);
    }
    std::set<int> want, have;
    for (int b = lo; b < 0; ++b) want.insert(b);
    for (int b = n; b <= hi; ++b) want.insert(b);
    for (const auto &[b, _] : doses) have.insert(b);
    if (have != want) {
        std::string list = "[";
        for (int b : have) list += (list.size() > 1 ? ", " : "") + std::to_string(b);
        throw std::invalid_argument("steered buckets " + list + "] are not contiguous from the row's 0.." +
                                    std::to_string(n - 1));
    }
    if (-lo > int(below.size()) || hi - n + 1 > int(above.size()))
        throw std::invalid_argument("steered buckets " + std::to_string(lo) + ".." + std::to_string(hi) +
                                    " have no nominal edges");
    RateScale s;
    for (int k = -lo - 1; k >= 0; --k) s.edges.push_back(below[size_t(k)]);
    s.edges.insert(s.edges.end(), row_edges.begin(), row_edges.end());
    for (int k = 0; k < hi - n + 1; ++k) s.edges.push_back(above[size_t(k)]);
    s.first = lo;
    s.doses = doses;
    return s;
}

int RateScale::bucket(double r) const {
    return first + int(std::upper_bound(edges.begin(), edges.end(), r) - edges.begin());  // bisect_right
}

double RateScale::rate(int b) const {
    // midpoint; the open end buckets extend by half a neighbour (laci.RateBuckets)
    const auto &e = edges;
    const int i = b - first, n = int(e.size());
    if (i <= 0) return e.at(0) - (e.at(1) - e.at(0)) / 2;
    if (i >= n) {
        if (n < 2) throw std::out_of_range("tuple index out of range");
        return e[n - 1] + (e[n - 1] - e[n - 2]) / 2;
    }
    return (e[i - 1] + e[i]) / 2;
}

std::vector<Candidate> candidate_plan(int bucket, const RateScale &scale, int64_t seed, int segment_index) {
    std::vector<Candidate> out;
    std::set<int> used;
    for (size_t oi = 0; oi < CANDIDATE_BUCKET_OFFSETS.size(); ++oi) {
        const int offset = CANDIDATE_BUCKET_OFFSETS[oi];
        const int b = bucket + offset;
        if (!(scale.first <= b && b <= scale.last()) || used.count(b)) continue;
        used.insert(b);
        for (int r = 0; r < CANDIDATE_SEEDS; ++r)
            out.push_back({b, offset, seed + 100 * int64_t(segment_index) + 10 * int64_t(oi) + r});
    }
    return out;
}

std::vector<Candidate> pinned_candidate_plan(int bucket, int64_t seed, int segment_index) {
    // the full count of an automatic plan, all in bucket, told apart by seed only
    const int n = int(CANDIDATE_BUCKET_OFFSETS.size()) * CANDIDATE_SEEDS;
    std::vector<Candidate> out;
    for (int r = 0; r < n; ++r) out.push_back({bucket, 0, seed + 100 * int64_t(segment_index) + r});
    return out;
}

int max_tokens_for(double duration_s) {
    return int(std::min(9000.0, std::ceil(FRAME_RATE * (2.0 * duration_s + 3.0))));
}

std::pair<double, double> voice_prompt_span(const std::vector<RefSegment> &segments, double duration) {
    if (segments.empty()) throw std::out_of_range("list index out of range");
    const double start = std::max(0.0, segments[0].ref_start - 0.1);
    std::optional<double> end;
    for (const auto &seg : segments) {
        const double cand = seg.ref_end + 0.1;
        if (cand - start <= VOICE_PROMPT_MAX_S && (!end || cand > *end)) end = cand;
    }
    if (!end) end = start + VOICE_PROMPT_MAX_S;
    return {std::min(start, duration), std::min(*end, duration)};
}

std::optional<std::pair<double, double>> speech_span(const std::vector<TimedWord> &items, const float *audio,
                                                     size_t n_audio, int sr) {
    bool any = false;
    double lo = 0.0, hi = 0.0;
    for (const auto &it : items) {
        if (!(it.placed && it.end > it.start)) continue;
        if (!any || it.start < lo) lo = it.start;
        if (!any || it.end > hi) hi = it.end;
        any = true;
    }
    if (any) return std::make_pair(lo, hi);
    const size_t frame = size_t(std::max(1, int(0.02 * sr)));
    const size_t n = n_audio / frame;
    if (n == 0) return std::nullopt;
    std::vector<double> rms(n), sq(frame);
    double peak = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < frame; ++j) {
            const double v = double(audio[i * frame + j]);
            sq[j] = v * v;
        }
        rms[i] = std::sqrt(pairwise_sum(sq.data(), frame) / double(frame));
        if (i == 0 || rms[i] > peak) peak = rms[i];
    }
    if (peak <= 0) return std::nullopt;
    size_t first = n, last = 0;
    for (size_t i = 0; i < n; ++i)
        if (rms[i] > 0.1 * peak) {
            if (first == n) first = i;
            last = i;
        }
    return std::make_pair(double(first * frame) / sr, double((last + 1) * frame) / sr);
}

std::optional<double> candidate_score(const std::optional<TakeResult> &c, double ref_dur, double room) {
    if (!c || c->speech_s <= 0) return std::nullopt;
    double score = std::abs(std::log(c->speech_s / ref_dur));
    if (c->speech_s > room) score += 0.5;
    if (c->capped) score += 1.0;
    if (c->end == "max_tokens") score += 1.0;
    return score;
}

std::optional<int> choose_candidate(const std::vector<std::optional<TakeResult>> &cands, double ref_dur, double room) {
    std::optional<int> best;
    std::optional<double> best_score;
    for (size_t i = 0; i < cands.size(); ++i) {
        const auto s = candidate_score(cands[i], ref_dur, room);
        if (s && (!best_score || *s < *best_score)) {
            best = int(i);
            best_score = s;
        }
    }
    return best;
}

std::pair<std::vector<Placed>, double> place_segments(const std::vector<PlaceItem> &items, double ref_duration) {
    std::vector<Placed> placed;
    double prev_end = -INF, timeline = ref_duration;
    for (const auto &it : items) {
        const double s0 = it.s0, s1 = it.s1;
        double crop_start = std::max(0.0, s0 - PRE_PAD_S);
        const double crop_end = std::min(it.take_len, s1 + POST_PAD_S);
        const double speech_start = std::max(it.ref_start, prev_end + MIN_GAP_S);
        const double speech_end = speech_start + (s1 - s0);
        double place_start = speech_start - (s0 - crop_start);
        if (place_start < 0) {
            crop_start += -place_start;
            place_start = 0.0;
        }
        prev_end = speech_end;
        timeline = std::max(timeline, place_start + crop_end - crop_start);
        placed.push_back({crop_start, crop_end, place_start, speech_start, speech_end, speech_start - it.ref_start,
                          speech_start - it.ref_start, speech_end - it.ref_end,
                          (s1 - s0) - (it.ref_end - it.ref_start)});
    }
    return {placed, timeline};
}

std::vector<Range> word_windows(const std::vector<int> &counts) {
    const int n = int(counts.size());
    for (int c : counts)
        if (c > WINDOW_MAX_WP)
            throw std::invalid_argument("a word exceeds " + std::to_string(WINDOW_MAX_WP) + " wordpieces");
    std::vector<Range> windows;
    int s = 0;
    while (n) {
        int e = s, total = 0;
        while (e < n && total + counts[e] <= WINDOW_MAX_WP) total += counts[e++];
        windows.emplace_back(s, e);
        if (e == n) break;
        s += std::max(1, int((e - s) * WINDOW_STRIDE_FRACTION));
    }
    return windows;
}

std::vector<int> window_of_words(const std::vector<Range> &windows, int n) {
    std::vector<int> owner(size_t(std::max(n, 0)), 0), best(size_t(std::max(n, 0)), -1);
    for (size_t wi = 0; wi < windows.size(); ++wi) {
        const auto [s, e] = windows[wi];
        for (int w = s; w < e; ++w) {
            const int margin = std::min(w - s, e - 1 - w);
            if (margin > best.at(size_t(w))) {
                best[w] = margin;
                owner[w] = int(wi);
            }
        }
    }
    return owner;
}

std::vector<uint8_t> band_mask(int w1, int w2) {
    std::vector<uint8_t> out(size_t(std::max(w1, 0)) * size_t(std::max(w2, 0)));
    const double band = std::max(double(BAND_MIN_WP), BAND_FRACTION * w2);
    for (int p = 0; p < w1; ++p)
        for (int q = 0; q < w2; ++q) out[size_t(p) * w2 + q] = std::abs(double(p) * w2 / w1 - double(q)) <= band;
    return out;
}

double interp(double x, const std::vector<double> &xp, const std::vector<double> &fp) {
    // numpy/_core/src/multiarray/compiled_base.c arr_interp
    const size_t n = xp.size();
    if (n == 0) throw std::invalid_argument("array of sample points is empty");
    if (std::isnan(x)) return x;
    if (x < xp[0]) return fp[0];
    if (x > xp[n - 1]) return fp[n - 1];
    if (x == xp[n - 1]) return fp[n - 1];
    const size_t j = size_t(std::upper_bound(xp.begin(), xp.end(), x) - xp.begin()) - 1;  // xp[j] <= x < xp[j + 1]
    if (xp[j] == x) return fp[j];
    const double slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    double r = slope * (x - xp[j]) + fp[j];
    if (std::isnan(r)) {
        r = slope * (x - xp[j + 1]) + fp[j + 1];
        if (std::isnan(r) && fp[j] == fp[j + 1]) r = fp[j];
    }
    return r;
}

LipMap LipMap::from_frames(double start, double rate, const std::vector<double> &src_frames) {
    // source times made strictly increasing (the inverse needs it)
    LipMap m;
    double run = -INF;
    for (size_t i = 0; i < src_frames.size(); ++i) {
        run = i == 0 ? src_frames[0] : std::max(run, src_frames[i]);  // np.maximum.accumulate
        m.out_t.push_back(start + double(i) / rate);
        m.src_t.push_back(start + (run / rate + double(i) * 1e-9));
    }
    return m;
}

double LipMap::src_time(double t) const { return map_time(t, out_t, src_t); }
double LipMap::out_time(double t) const { return map_time(t, src_t, out_t); }

std::vector<double> LipMap::shifts() const {
    std::vector<double> out(out_t.size());
    for (size_t i = 0; i < out.size(); ++i) out[i] = out_t[i] - src_t[i];
    return out;
}

std::vector<double> guard_lip_frames(const std::vector<double> &src_frames, double rate, double max_rate,
                                     double max_shift_s) {
    const size_t n = src_frames.size();
    if (n == 0) throw std::out_of_range("index 0 is out of bounds for axis 0 with size 0");
    std::vector<double> src(n), lo(n), hi(n);
    const double s = max_shift_s * rate;
    for (size_t i = 0; i < n; ++i) {
        src[i] = i == 0 ? src_frames[0] : std::max(src[i - 1], src_frames[i]);
        lo[i] = double(i) - s;
        hi[i] = double(i) + s;
    }
    hi[0] = std::min(hi[0], 0.0);  // no advance at the track's first frame
    const double lo_step = 1.0 / max_rate, hi_step = max_rate;
    // shift bounds and rate bounds alternate; the last pass enforces the rate
    for (int pass = 0; pass < 4; ++pass) {
        for (size_t i = 0; i < n; ++i) src[i] = std::min(std::max(src[i], lo[i]), hi[i]);
        for (size_t i = 1; i < n; ++i) src[i] = std::min(std::max(src[i], src[i - 1] + lo_step), src[i - 1] + hi_step);
    }
    return src;
}

LipWarp lip_warp(const LipMap &m, double src_start, double src_end) {
    const double a = m.out_time(src_start), b = m.out_time(src_end);
    if (src_end - src_start < 1e-6 || b - a < 1e-6) return {1.0, 1.0, 1.0, 1.0};
    const size_t num = size_t(std::max(2, int((b - a) / 0.01) + 1));
    const double step = (b - a) / double(num - 1);  // np.linspace
    const double h = LIP_SCALE_WINDOW_S / 2;
    std::vector<double> abs_log(num);
    double mx = -INF, mn = INF;
    for (size_t i = 0; i < num; ++i) {
        const double t = i + 1 == num ? b : double(i) * step + a;
        const double scale = 2 * h / std::max(m.src_time(t + h) - m.src_time(t - h), 1e-9);
        const double l = std::log(std::min(std::max(scale, 1 / LIP_SCALE_MAX), LIP_SCALE_MAX));
        abs_log[i] = std::abs(l);
        mx = std::max(mx, l);
        mn = std::min(mn, l);
    }
    return {(b - a) / (src_end - src_start), std::exp(std::max(0.0, mx)), std::exp(std::max(0.0, -mn)),
            std::exp(pairwise_sum(abs_log.data(), num) / double(num))};
}

std::vector<double> wsola(const std::vector<double> &x, const std::function<double(double)> &src_of, int64_t n_out,
                          int window, int tolerance) {
    // Output frame k is centred at k * hop; its source frame is the one within tolerance of src_of(k * hop) that best
    // continues the previous source frame (normalized cross-correlation less WSOLA_DRIFT_PENALTY per tolerance of
    // distance), so the overlap-add stays phase-coherent.
    const int64_t hop = window / 2, half = window / 2;
    std::vector<double> win((size_t) window);  // np.hanning(window + 1)[:window]: periodic Hann
    {
        const int M = window + 1;
        constexpr double PI = 3.141592653589793;  // numpy.pi
        for (int i = 0; i < window; ++i) win[i] = 0.5 + 0.5 * std::cos(PI * double(1 - M + 2 * i) / double(M - 1));
    }
    const int64_t pad = half + tolerance + hop;
    const int64_t nx = int64_t(x.size());
    std::vector<double> xp(size_t(nx + 2 * pad), 0.0);
    std::copy(x.begin(), x.end(), xp.begin() + pad);
    const int64_t n_frames = n_out / hop + 2;
    std::vector<int64_t> nominal((size_t) n_frames);
    for (int64_t k = 0; k < n_frames; ++k) {
        const double v = std::min(std::max(src_of(double(k * hop)), 0.0), double(nx));
        nominal[k] = int64_t(std::nearbyint(v)) + pad;
    }
    const size_t size = size_t(1) << int(std::ceil(std::log2(double(2 * window + 2 * tolerance))));
    const size_t n_lag = size_t(2 * tolerance + 1), n_bins = size / 2 + 1, n_region = size_t(2 * tolerance + window);
    std::vector<double> y(size_t(n_frames * hop + window), 0.0);
    std::vector<double> penalty(n_lag);
    for (size_t l = 0; l < n_lag; ++l)
        penalty[l] = WSOLA_DRIFT_PENALTY * std::abs(double(int64_t(l) - tolerance)) / double(std::max(tolerance, 1));
    std::vector<double> rbuf(size), nbuf(size), corr(size), energy(n_region + 1);
    std::vector<std::complex<double>> rspec(n_bins), nspec(n_bins);
    const pocketfft::shape_t shape = {size};
    const pocketfft::stride_t rstride = {ptrdiff_t(sizeof(double))}, cstride = {ptrdiff_t(sizeof(std::complex<double>))};
    int64_t prev = -1;
    for (int64_t k = 0; k < n_frames; ++k) {
        int64_t c = nominal[k];
        if (prev >= 0) {
            const double *natural = xp.data() + (prev + hop - half);
            const double *region = xp.data() + (c - tolerance - half);
            std::fill(rbuf.begin(), rbuf.end(), 0.0);
            std::fill(nbuf.begin(), nbuf.end(), 0.0);
            std::copy(region, region + n_region, rbuf.begin());
            std::copy(natural, natural + window, nbuf.begin());
            pocketfft::r2c(shape, rstride, cstride, 0, pocketfft::FORWARD, rbuf.data(), rspec.data(), 1.0);
            pocketfft::r2c(shape, rstride, cstride, 0, pocketfft::FORWARD, nbuf.data(), nspec.data(), 1.0);
            for (size_t i = 0; i < n_bins; ++i) {
                const double ar = rspec[i].real(), ai = rspec[i].imag();
                const double br = nspec[i].real(), bi = -nspec[i].imag();
                rspec[i] = {ar * br - ai * bi, ar * bi + ai * br};
            }
            pocketfft::c2r(shape, cstride, rstride, 0, pocketfft::BACKWARD, rspec.data(), corr.data(),
                           1.0 / double(size));
            energy[0] = 0.0;
            for (size_t i = 0; i < n_region; ++i) energy[i + 1] = energy[i] + region[i] * region[i];
            double nn = 0.0;
            for (int64_t i = 0; i < window; ++i) nn += natural[i] * natural[i];
            size_t best = 0;
            double best_v = 0.0;
            for (size_t l = 0; l < n_lag; ++l) {
                const double e = energy[l + window] - energy[l];
                const double v = corr[l] / std::sqrt(e * nn + 1e-12) - penalty[l];
                if (l == 0 || v > best_v || (std::isnan(v) && !std::isnan(best_v))) {
                    best = l;
                    best_v = v;
                }
            }
            c = c - tolerance + int64_t(best);
        }
        const double *src = xp.data() + (c - half);
        double *dst = y.data() + k * hop;
        for (int i = 0; i < window; ++i) dst[i] += win[i] * src[i];
        prev = c;
    }
    return std::vector<double>(y.begin() + half, y.begin() + half + n_out);
}

}  // namespace dubplan
