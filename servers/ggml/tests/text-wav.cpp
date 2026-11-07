// UTF-8 decoding of ill-formed input (Python errors="replace": one U+FFFD per maximal subpart) and the PCM16 WAV
// encoder (NaN -> 0, clip, x * 32767 rounded half to even).
#include "wav.h"
#include "zonos3-text.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

static int failures = 0;

static void check(bool ok, const char * what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

static void check_decode(const std::string & in, const std::u32string & want, const char * what) {
    check(utf8_to_u32(in) == want, what);
    check(utf8_length(in) == want.size(), what);
}

static int16_t sample(const std::vector<uint8_t> & wav, size_t i) {
    int16_t v;
    memcpy(&v, wav.data() + 44 + 2 * i, 2);
    return v;
}

int main() {
    check_decode("\xC0\xAF", U"\uFFFD\uFFFD", "C0 AF -> 2 x U+FFFD");
    check_decode("\xED\xA0\x80", U"\uFFFD\uFFFD\uFFFD", "ED A0 80 -> 3 x U+FFFD");
    check_decode("\xE2\x82", U"\uFFFD", "E2 82 -> 1 x U+FFFD");
    check_decode("\xF0\x9F\x98\x80", U"\U0001F600", "F0 9F 98 80 -> U+1F600");
    check(utf8_from_u32(U"\U0001F600") == "\xF0\x9F\x98\x80", "encode U+1F600");

    const float inf = std::numeric_limits<float>::infinity();
    const std::vector<float> special = {std::nanf(""), inf, -inf, 2.0f, -2.0f, 0.0f};
    const std::vector<uint8_t> wav = wav_pcm16_bytes(special, 44100);
    check(wav.size() == 44 + 2 * special.size(), "header 44 bytes");
    check(memcmp(wav.data(), "RIFF", 4) == 0 && memcmp(wav.data() + 8, "WAVE", 4) == 0, "RIFF/WAVE tags");
    const int16_t want[] = {0, 32767, -32767, 32767, -32767, 0};
    for (size_t i = 0; i < special.size(); ++i) check(sample(wav, i) == want[i], "NaN/inf/clip samples");

    std::vector<float> ties;
    std::vector<int> ks;
    for (int k = 0; k <= 2000; ++k) {
        const float v = (k + 0.5f) / 32767.0f;
        if (v * 32767.0f != k + 0.5f) continue;
        ties.push_back(v);
        ks.push_back(k);
    }
    check(ties.size() >= 10, "at least 10 exact float32 ties");
    const std::vector<uint8_t> tw = wav_pcm16_bytes(ties, 44100);
    for (size_t i = 0; i < ties.size(); ++i) {
        const int even = ks[i] % 2 == 0 ? ks[i] : ks[i] + 1;
        check(sample(tw, i) == even, "tie rounds half to even");
    }

    printf("%zu ties checked\n", ties.size());
    if (failures) return 1;
    printf("PASS\n");
    return 0;
}
