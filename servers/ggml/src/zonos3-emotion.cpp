// Emotion / accent directions: port of model/src/zonos3/tts/emotion.py (see zonos3-emotion.h).
#include "zonos3-emotion.h"

#include "ggml.h"
#include "gguf.h"
#include "json.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>

using nlohmann::json;

namespace {

bool lookup(const zonos3_emotion_calibration & c, const std::string & speaker, const std::string & key, double & out) {
    if (!speaker.empty()) {
        const auto s = c.by_speaker.find(speaker);
        if (s != c.by_speaker.end()) {
            const auto e = s->second.find(key);
            if (e != s->second.end()) { out = e->second; return true; }
        }
    }
    const auto e = c.defaults.find(key);
    if (e != c.defaults.end()) { out = e->second; return true; }
    return false;
}

// Python's repr of a list of str: ['a', 'b'].
std::string py_list(const std::vector<std::string> & items) {
    std::string out = "[";
    for (size_t i = 0; i < items.size(); ++i) out += (i ? ", '" : "'") + items[i] + "'";
    return out + "]";
}

// torch.linalg.vector_norm of an fp32 vector (accumulated in double, rounded to fp32).
float norm(const std::vector<float> & v) {
    double s = 0.0;
    for (float x : v) s += (double) x * x;
    return (float) std::sqrt(s);
}

// _rescale_to
void rescale_to(std::vector<float> & v, float target) {
    const float cur = norm(v);
    if (cur > 1e-8f) {
        const float k = target / cur;
        for (float & x : v) x *= k;
    }
}

}  // namespace

double zonos3_emotion_calibration::strength(const std::string & speaker, const std::string & emotion) const {
    double s;
    return lookup(*this, speaker, emotion, s) ? s : global_default;
}

double zonos3_emotion_calibration::pole_strength(const std::string & speaker, const std::string & emotion,
                                                 double weight) const {
    double s;
    if (weight < 0 && lookup(*this, speaker, emotion + "-", s)) return s;
    return strength(speaker, emotion);
}

std::vector<std::string> zonos3_emotion_calibration::disabled(const std::string & speaker,
                                                              const std::vector<std::string> & emotions) const {
    std::vector<std::string> out;
    for (const auto & e : emotions) {
        double s;
        if (strength(speaker, e) == 0.0) out.push_back(e);
        else if (lookup(*this, speaker, e + "-", s) && s == 0.0) out.push_back(e + "-");
    }
    return out;
}

bool zonos3_emotion_calibration::parse(const std::string & json_text, zonos3_emotion_calibration & out, std::string & err) {
    out = zonos3_emotion_calibration();
    try {
        const json data = json::parse(json_text);
        if (data.contains("by_speaker"))
            for (const auto & sp : data.at("by_speaker").items())
                for (const auto & e : sp.value().items()) out.by_speaker[sp.key()][e.key()] = e.value().get<double>();
        if (data.contains("default"))
            for (const auto & e : data.at("default").items()) out.defaults[e.key()] = e.value().get<double>();
        out.global_default = data.value("global_default", 1.0);
    } catch (const std::exception & e) {
        err = std::string("invalid emotion calibration: ") + e.what();
        return false;
    }
    return true;
}

const std::vector<float> * zonos3_directions::find(const std::string & name) const {
    for (size_t i = 0; i < names.size(); ++i) if (names[i] == name) return &named[i];
    for (size_t i = 0; i < axes.size(); ++i) if (axes[i] == name) return &axis[i];
    return nullptr;
}

std::vector<std::string> zonos3_directions::controls() const {
    std::vector<std::string> out(names);
    out.insert(out.end(), axes.begin(), axes.end());
    return out;
}

bool zonos3_directions_load(zonos3_directions & d, const char * path, std::string & err) {
    d = zonos3_directions();
    ggml_context * data = nullptr;
    gguf_context * gf = gguf_init_from_file(path, {false, &data});
    if (!gf) { err = std::string("cannot read directions '") + path + "'"; return false; }
    bool ok = true;
    auto fail = [&](const std::string & message) {
        if (ok) err = std::string(path) + ": " + message;
        ok = false;
    };
    auto key = [&](const char * name, gguf_type type) -> int64_t {
        const int64_t id = gguf_find_key(gf, name);
        if (id < 0 || gguf_get_kv_type(gf, id) != type) { fail(std::string("missing or invalid ") + name); return -1; }
        return id;
    };
    auto str = [&](const char * name) -> std::string {
        const int64_t id = key(name, GGUF_TYPE_STRING);
        return id < 0 ? "" : gguf_get_val_str(gf, id);
    };
    // gguf-py drops empty arrays and empty strings on write: an absent names / axes / calibration key is empty.
    auto optional_str = [&](const char * name) -> std::string {
        return gguf_find_key(gf, name) < 0 ? "" : str(name);
    };
    auto str_array = [&](const char * name, std::vector<std::string> & out) {
        if (gguf_find_key(gf, name) < 0) return;
        const int64_t id = key(name, GGUF_TYPE_ARRAY);
        if (id < 0) return;
        if (gguf_get_arr_type(gf, id) != GGUF_TYPE_STRING) { fail(std::string(name) + " is not a string array"); return; }
        for (size_t i = 0; i < gguf_get_arr_n(gf, id); ++i) out.emplace_back(gguf_get_arr_str(gf, id, i));
    };
    auto vec = [&](const std::string & name, size_t n, std::vector<float> & out) {
        const ggml_tensor * t = ggml_get_tensor(data, name.c_str());
        if (!t || t->type != GGML_TYPE_F32 || (size_t) ggml_nelements(t) != n) {
            fail("missing or invalid tensor " + name + " (" + std::to_string(n) + " f32 values)");
            return;
        }
        const float * v = (const float *) t->data;
        out.assign(v, v + n);
    };
    if (str("general.architecture") != "zonos3_directions") fail("not a zonos3_directions GGUF");
    d.space = str("zonos3_directions.space");
    if (ok && d.space != "raw" && d.space != "lda" && d.space != "proj" && d.space != "attnres_source")
        fail("unknown direction space " + d.space);
    if (const int64_t id = key("zonos3_directions.source_index", GGUF_TYPE_INT32); id >= 0) d.source_index = gguf_get_val_i32(gf, id);
    if (const int64_t id = key("zonos3_directions.dim", GGUF_TYPE_UINT32); id >= 0) d.dim = (int) gguf_get_val_u32(gf, id);
    if (const int64_t id = key("zonos3_directions.input_dim", GGUF_TYPE_UINT32); id >= 0) d.input_dim = (int) gguf_get_val_u32(gf, id);
    str_array("zonos3_directions.names", d.names);
    str_array("zonos3_directions.axes", d.axes);
    const std::string calibration = optional_str("zonos3_directions.calibration");
    d.tiers_json = optional_str("zonos3_directions.tiers");
    d.presets_json = optional_str("zonos3_directions.presets");
    if (d.tiers_json.empty()) d.tiers_json = "{}";
    if (d.presets_json.empty()) d.presets_json = "{}";
    if (const int64_t id = key("zonos3_directions.ref_base_norm", GGUF_TYPE_FLOAT32); id >= 0) d.ref_base_norm = gguf_get_val_f32(gf, id);
    if (ok && (d.dim <= 0 || d.input_dim <= 0)) fail("invalid dim / input_dim");
    if (ok && d.space == "attnres_source" && d.source_index < 0) fail("attnres_source directions need source_index >= 0");
    if (ok && !calibration.empty()) {
        std::string why;
        if (zonos3_emotion_calibration::parse(calibration, d.calibration, why)) d.calibrated = true;
        else fail(why);
    }
    for (size_t i = 0; ok && i < d.names.size(); ++i) {
        d.named.emplace_back();
        vec("dir.named." + std::to_string(i), (size_t) d.dim, d.named.back());
    }
    for (size_t i = 0; ok && i < d.axes.size(); ++i) {
        d.axis.emplace_back();
        vec("dir.axis." + std::to_string(i), (size_t) d.dim, d.axis.back());
    }
    if (ok && d.space == "lda") {
        d.lda_dim = d.dim;
        vec("dir.lda.weight", (size_t) d.lda_dim * d.input_dim, d.lda_weight);
        vec("dir.lda.bias", (size_t) d.lda_dim, d.lda_bias);
        vec("dir.lda.pinv", (size_t) d.input_dim * d.lda_dim, d.lda_pinv);
    }
    if (ok && d.space != "lda" && d.input_dim != d.dim && d.space == "raw") fail("raw directions need input_dim == dim");
    gguf_free(gf);
    ggml_free(data);
    if (!ok) { d = zonos3_directions(); return false; }
    if (d.empty()) fprintf(stderr, "zonos3: emotion directions %s contain no directions\n", path);
    return true;
}

void zonos3_emotion_calibrate(const zonos3_directions & d, const std::string & speaker, zonos3_emotion_sliders & sliders,
                              double & valence, double & arousal) {
    if (!d.calibrated) return;
    for (auto & s : sliders) s.second = s.second * d.calibration.pole_strength(speaker, s.first, s.second);
    valence = valence * d.calibration.pole_strength(speaker, "valence", valence);
    arousal = arousal * d.calibration.pole_strength(speaker, "arousal", arousal);
}

bool zonos3_emotion_combine(const zonos3_directions & d, const zonos3_emotion_sliders & sliders, double valence,
                            double arousal, bool strict, std::vector<float> & delta, bool & requested, std::string & err) {
    delta.assign((size_t) d.dim, 0.0f);
    requested = false;
    // delta += w * vec in fp32 (torch: the Python float scalar is applied as fp32).
    auto accumulate = [&](double weight, const std::vector<float> & v) {
        const float w = (float) weight;
        for (size_t i = 0; i < delta.size(); ++i) {
            const float term = w * v[i];
            delta[i] += term;
        }
        requested = true;
    };
    for (const auto & s : sliders) {
        if (s.second == 0.0) continue;
        const std::vector<float> * v = d.find(s.first);
        if (!v) {
            const std::string msg = "Unknown emotion direction '" + s.first + "'. Available: " + py_list(d.controls()) + ".";
            if (strict) { err = msg; return false; }
            fprintf(stderr, "zonos3: %s\n", msg.c_str());
            continue;
        }
        accumulate(s.second, *v);
    }
    const std::pair<const char *, double> axes[] = {{"valence", valence}, {"arousal", arousal}};
    for (const auto & a : axes) {
        if (a.second == 0.0) continue;
        const std::vector<float> * v = nullptr;
        for (size_t i = 0; i < d.axes.size(); ++i) if (d.axes[i] == a.first) v = &d.axis[i];
        if (!v) {
            if (strict) { err = std::string("Emotion axis '") + a.first + "' is not available."; return false; }
            continue;
        }
        accumulate(a.second, *v);
    }
    return true;
}

bool zonos3_emotion_apply(const zonos3_directions & d, const std::vector<float> & base,
                          const zonos3_emotion_sliders & sliders, double valence, double arousal, double strength,
                          bool preserve_norm, bool strict, std::vector<float> & out, std::string & err) {
    out = base;
    if (d.empty() || d.hidden()) return true;
    if ((int) base.size() != d.input_dim) {
        err = "Speaker embedding dim " + std::to_string(base.size()) + " does not match emotion directions input dim " +
              std::to_string(d.input_dim) + ".";
        return false;
    }
    std::vector<float> delta;
    bool requested = false;
    if (!zonos3_emotion_combine(d, sliders, valence, arousal, strict, delta, requested, err)) return false;
    if (!requested || strength == 0.0) return true;
    const float s = (float) strength;
    if (d.space == "lda") {
        // _apply_lda: lda_base = W x + b; lda_new = lda_base + s*delta (renormalised); raw = W+ (lda_new - b).
        const int L = d.lda_dim, I = d.input_dim;
        std::vector<float> lda_base(L), lda_new(L);
        for (int r = 0; r < L; ++r) {
            double acc = 0.0;
            for (int c = 0; c < I; ++c) acc += (double) d.lda_weight[(size_t) r * I + c] * base[c];
            lda_base[r] = (float) acc + d.lda_bias[r];
            lda_new[r] = lda_base[r] + s * delta[r];
        }
        if (preserve_norm) rescale_to(lda_new, norm(lda_base));
        for (int r = 0; r < L; ++r) lda_new[r] -= d.lda_bias[r];
        for (int c = 0; c < I; ++c) {
            double acc = 0.0;
            for (int r = 0; r < L; ++r) acc += (double) d.lda_pinv[(size_t) c * L + r] * lda_new[r];
            out[c] = (float) acc;
        }
        return true;
    }
    for (size_t i = 0; i < out.size(); ++i) {
        const float term = s * delta[i];
        out[i] = base[i] + term;
    }
    if (preserve_norm) rescale_to(out, norm(base));
    return true;
}

bool zonos3_emotion_hidden_delta(const zonos3_directions & d, const zonos3_emotion_sliders & sliders, double valence,
                                 double arousal, double strength, bool strict, std::vector<float> & out,
                                 std::string & err) {
    out.clear();
    if (d.empty() || !d.hidden()) return true;
    std::vector<float> delta;
    bool requested = false;
    if (!zonos3_emotion_combine(d, sliders, valence, arousal, strict, delta, requested, err)) return false;
    if (!requested || strength == 0.0) return true;
    const float s = (float) strength;
    out.resize(delta.size());
    for (size_t i = 0; i < delta.size(); ++i) out[i] = s * delta[i];
    return true;
}

bool zonos3_accent_steering(const zonos3_directions & d, const std::string & accent, double strength,
                            const std::string & speaker, bool use_calibration, std::vector<float> * speaker_embedding,
                            std::vector<float> & hidden, std::string & err) {
    hidden.clear();
    const std::vector<float> * v = d.find(accent);
    if (!v) { err = "unknown accent direction '" + accent + "'"; return false; }
    const double dose = d.calibrated && use_calibration ? d.calibration.strength(speaker, accent) : 1.0;
    if (strength == 0.0 || dose == 0.0) return true;
    if (d.space == "raw") {
        if (!speaker_embedding) {
            err = "Accent '" + accent + "' is a speaker-embedding direction and needs speaker_mode=embedding.";
            return false;
        }
        std::vector<float> moved;
        if (!zonos3_emotion_apply(d, *speaker_embedding, {{accent, strength * dose}}, 0.0, 0.0, 1.0, true, true, moved, err))
            return false;
        *speaker_embedding = std::move(moved);
        return true;
    }
    // (strength * dose) * direction: a Python float times an fp32 tensor.
    const float k = (float) (strength * dose);
    hidden.resize(v->size());
    for (size_t i = 0; i < v->size(); ++i) hidden[i] = k * (*v)[i];
    return true;
}

bool zonos3_steer_add_delta(const std::vector<int> & sources, int hidden_size, int source,
                            const std::vector<float> & delta, std::vector<float> & layout) {
    const auto it = std::find(sources.begin(), sources.end(), source);
    if (it == sources.end() || (int) delta.size() != hidden_size) return false;
    if (layout.empty()) layout.assign(sources.size() * (size_t) hidden_size, 0.0f);
    if (layout.size() != sources.size() * (size_t) hidden_size) return false;
    float * slot = layout.data() + (size_t) (it - sources.begin()) * hidden_size;
    for (int i = 0; i < hidden_size; ++i) slot[i] += delta[i];
    return true;
}
