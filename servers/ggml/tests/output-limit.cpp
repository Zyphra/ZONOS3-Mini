// The output stage's soft limiter (zonos3_soft_limit, zonos3.tts.audio.output_pcm): samples at or below the knee of
// the generation registry's output_limit_dbfs are untouched, louder ones bend monotonically and symmetrically into full
// scale with slope 1 at the knee.
#include "zonos3-generation.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(cond, label) do { if (!(cond)) { fprintf(stderr, "FAIL: %s (%s:%d)\n", label, __FILE__, __LINE__); exit(1); } } while (0)

int main() {
    const float knee = (float) std::pow(10.0, zonos3_generation().output_limit_dbfs / 20.0);
    CHECK(knee > 0.5f && knee < 1.0f, "the registry knee is below full scale");

    std::vector<float> quiet;
    for (int i = -1000; i <= 1000; ++i) quiet.push_back(knee * (float) i / 1000.0f);
    std::vector<float> q = quiet;
    zonos3_soft_limit(q.data(), q.size());
    for (size_t i = 0; i < q.size(); ++i) CHECK(q[i] == quiet[i], "samples at or below the knee are unchanged");

    std::vector<float> loud;
    for (int i = 1; i <= 2000; ++i) loud.push_back(knee + 7.0f * (float) i / 2000.0f);
    std::vector<float> y = loud, neg;
    zonos3_soft_limit(y.data(), y.size());
    for (float v : loud) neg.push_back(-v);
    zonos3_soft_limit(neg.data(), neg.size());
    for (size_t i = 0; i < y.size(); ++i) {
        CHECK(y[i] > knee && y[i] <= 1.0f && y[i] <= loud[i], "peaks bend into full scale");
        CHECK(i == 0 || y[i] >= y[i - 1], "monotonic");
        CHECK(neg[i] == -y[i], "symmetric");
    }
    float just = knee + 1e-4f;
    zonos3_soft_limit(&just, 1);
    CHECK(std::fabs(just - (knee + 1e-4f)) < 1e-6f, "slope 1 at the knee");
    float overshoot = 1.2f;
    zonos3_soft_limit(&overshoot, 1);
    CHECK(overshoot < 1.0f, "a typical overshoot stays inside full scale");
    printf("PASS\n");
    return 0;
}
