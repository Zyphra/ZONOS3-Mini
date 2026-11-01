// Emotion / accent directions (emotion.gguf, accent-directions.gguf; `tools/convert_gguf.py directions`): exact port of
// EmotionCalibration, _combine_delta, apply_emotion and emotion_hidden_delta (model/src/zonos3/tts/emotion.py).
//
// Where a delta lands depends on the direction space:
//   raw             the speaker embedding moves (zonos3_emotion_apply; norm kept), before the prompt is built
//   lda             the embedding moves inside the model's LDA space and maps back through the pseudo-inverse
//   proj            zonos3_emotion_hidden_delta -> zonos3_conditioning::speaker_delta (added to the projected
//                   speaker row; sgl _forward_model)
//   attnres_source  zonos3_emotion_hidden_delta -> every decode step's hidden delta of Full AttnRes bank source
//                   `source_index` (zonos3_steer_add_delta into zonos3_slot::steer; the batch's steer sources must
//                   include it)
//
// GGUF layout (arch "zonos3_directions"): zonos3_directions.{space: str, source_index: i32 (-1 = none), dim: u32,
// input_dim: u32, names: [str], axes: [str], calibration: str (calibration.json or ""), tiers: str (JSON object),
// presets: str (JSON object), ref_base_norm: f32 (NaN = none)}; f32 tensors dir.named.<i> / dir.axis.<i> [dim] and,
// for the lda space, dir.lda.weight (numpy [lda_dim, input_dim]), dir.lda.bias [lda_dim], dir.lda.pinv
// (numpy [input_dim, lda_dim]). gguf-py writes no empty array or string, so an absent names / axes / calibration /
// tiers / presets key reads as empty.
#pragma once

#include <cmath>
#include <map>
#include <string>
#include <utility>
#include <vector>

// Per-speaker, per-emotion raw dose chosen offline (calibration.json). A dose of 0 disables the emotion for that
// speaker; "<name>-" keys carry a separate dose for the negative pole of a bipolar axis.
struct zonos3_emotion_calibration {
    std::map<std::string, std::map<std::string, double>> by_speaker;
    std::map<std::string, double> defaults;  // Python `default`
    double global_default = 1.0;
    // `speaker` empty = no speaker key.
    double strength(const std::string & speaker, const std::string & emotion) const;
    double pole_strength(const std::string & speaker, const std::string & emotion, double weight) const;
    std::vector<std::string> disabled(const std::string & speaker, const std::vector<std::string> & emotions) const;
    static bool parse(const std::string & json_text, zonos3_emotion_calibration & out, std::string & err);
};

struct zonos3_directions {
    std::string space = "raw";                  // raw | lda | proj | attnres_source
    int source_index = -1;                      // Full AttnRes bank source (attnres_source)
    int dim = 0, input_dim = 0;                 // direction length; speaker-embedding length fed to the model
    std::vector<std::string> names, axes;       // named emotions / axes, manifest order
    std::vector<std::vector<float>> named, axis;// [dim] each, parallel to names / axes
    std::string tiers_json = "{}", presets_json = "{}";  // UI metadata (capabilities)
    float ref_base_norm = NAN;
    bool calibrated = false;                    // calibration.json was present
    zonos3_emotion_calibration calibration;
    int lda_dim = 0;
    std::vector<float> lda_weight, lda_bias, lda_pinv;  // [lda_dim, input_dim], [lda_dim], [input_dim, lda_dim] row-major

    bool hidden() const { return space == "proj" || space == "attnres_source"; }
    bool empty() const { return names.empty() && axes.empty(); }
    const std::vector<float> * find(const std::string & name) const;  // named first, then axes; null if unknown
    std::vector<std::string> controls() const;                        // names + axes
};

bool zonos3_directions_load(zonos3_directions & d, const char * path, std::string & err);

// Slider weights in request order (Python dict order matters for the fp32 accumulation).
using zonos3_emotion_sliders = std::vector<std::pair<std::string, double>>;

// Calibrated mode (sgl _apply_request_emotion): every weight w becomes w * pole_strength(speaker, name, w), likewise
// valence and arousal. No-op without calibration.
void zonos3_emotion_calibrate(const zonos3_directions & d, const std::string & speaker, zonos3_emotion_sliders & sliders,
                              double & valence, double & arousal);

// _combine_delta: Σ weight·direction over the requested sliders and axes. Unknown names fail when `strict`
// (Python's ValueError message in `err`), else are skipped with a warning.
bool zonos3_emotion_combine(const zonos3_directions & d, const zonos3_emotion_sliders & sliders, double valence,
                            double arousal, bool strict, std::vector<float> & delta, bool & requested, std::string & err);
// apply_emotion: the speaker embedding moved by the raw/lda directions (unchanged for hidden spaces or when nothing
// is requested).
bool zonos3_emotion_apply(const zonos3_directions & d, const std::vector<float> & base,
                          const zonos3_emotion_sliders & sliders, double valence, double arousal, double strength,
                          bool preserve_norm, bool strict, std::vector<float> & out, std::string & err);
// emotion_hidden_delta: strength * Σ weight·direction for proj / attnres_source directions; `out` empty = none.
bool zonos3_emotion_hidden_delta(const zonos3_directions & d, const zonos3_emotion_sliders & sliders, double valence,
                                 double arousal, double strength, bool strict, std::vector<float> & out,
                                 std::string & err);

// The direction part of sgl _apply_steering_accent for an accent-directions set: dose = the voice's calibrated
// strength (1 without calibration or with `use_calibration` off); nothing happens when strength or dose is 0. A raw set
// moves *speaker_embedding (apply_emotion, norm kept; an error without one), a hidden set returns
// strength * dose * direction in `hidden` (empty otherwise). False with Python's ValueError message in `err`.
bool zonos3_accent_steering(const zonos3_directions & d, const std::string & accent, double strength,
                            const std::string & speaker, bool use_calibration, std::vector<float> * speaker_embedding,
                            std::vector<float> & hidden, std::string & err);

// Adds `delta` [hidden_size] to the slot of `source` in a per-source steering layout ([sources.size() * hidden_size],
// zonos3_slot::steer); an empty layout starts at zero. False when `source` is not among `sources`.
bool zonos3_steer_add_delta(const std::vector<int> & sources, int hidden_size, int source,
                            const std::vector<float> & delta, std::vector<float> & layout);
