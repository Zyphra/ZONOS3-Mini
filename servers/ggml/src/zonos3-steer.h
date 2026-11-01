// Steering directions (steer.gguf): raw-space controls move the speaker embedding, hidden-space controls add
// to Full AttnRes bank sources of generated frames. Semantics follow minisgl zonos2/tts/steering.py.
#pragma once
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>
struct zonos3_model;

constexpr int    ZONOS3_STEER_RAMP_FRAMES = 3;        // 0.2 s at 15 frames/s
constexpr double ZONOS3_STEER_FRAMES_PER_CHAR = 1.0;  // default control_span_rate

struct zonos3_steer_direction {
    std::string id, set, name;     // id = "<set>.<name>"
    bool axis = false, hidden = false;
    int source = -1;               // bank source (hidden only)
    std::vector<float> vec;        // [speaker_dim] raw | [hidden_size] hidden
};
struct zonos3_steer_set { std::string name; bool hidden = false; int source = -1; };
// Tempo-steered speaking-rate buckets past the condition row (bundle `rate_extension`, zonos3/tts/steering.py
// RateExtension): the pace direction, the row edges they continue, the nominal edges past the row and, per TTS
// language, the buckets measured safe with their dose; reference_sides ("slower" / "faster") also serve
// reference-mode voices.
struct zonos3_steer_rate_extension {
    std::string direction;
    std::vector<double> row_edges, below, above;
    std::map<std::string, std::map<int, double>> doses;
    std::vector<std::string> reference_sides{"slower", "faster"};
};
struct zonos3_steer_bundle {
    std::string name;
    uint32_t speaker_dim = 0, hidden_size = 0;
    std::vector<zonos3_steer_set> sets;
    std::vector<zonos3_steer_direction> dirs;                               // file order
    std::map<std::string, std::vector<std::pair<double, double>>> timing;  // id -> (dose, factor), dose-sorted
    std::string timing_json, ui_json;                                       // verbatim from the file
    std::map<std::string, size_t> index;                                    // id -> dirs[i] (zonos3_steer_index)
    std::vector<int> sources;                                               // ascending distinct hidden sources
    std::optional<zonos3_steer_rate_extension> rate_extension;              // zonos3_steer.rate_extension ("" = none)
};
void zonos3_steer_index(zonos3_steer_bundle & b);
bool zonos3_steer_load(zonos3_steer_bundle & b, const char * path, const zonos3_model & model, std::string & err);
std::string zonos3_steer_controls_json(const zonos3_steer_bundle & b, int embedding_dim);
double zonos3_steer_duration_factor(const zonos3_steer_bundle & b, const std::map<std::string, double> & doses);
int zonos3_steer_text_chars(const std::string & utf8);   // Unicode code points ((c & 0xC0) != 0x80 bytes)

struct zonos3_control_span { int start = 0, end = 0; std::map<std::string, double> controls; };
struct zonos3_steering {
    std::vector<int> sources;   // layout of every delta: [sources.size() * hidden_size]
    int hidden_size = 0;
    std::vector<float> hidden;  // constant delta; empty = none
    // frames [start, end) of the character-rate schedule; chars = the request's character span [char_start, char_end)
    // (ResolvedControls.span_chars, for control_span_follow "reading")
    struct span {
        int start = 0, end = 0, ramp = ZONOS3_STEER_RAMP_FRAMES;
        std::vector<float> delta;
        int char_start = 0, char_end = 0;
    };
    std::vector<span> spans;    // sorted, non-overlapping
    bool hidden_active() const { return !hidden.empty() || !spans.empty(); }
    void delta_at(int frames, std::vector<float> & out) const;  // out.clear() when the delta is zero
};
bool zonos3_steer_resolve(const zonos3_steer_bundle & b, const std::map<std::string, double> & controls,
                          const std::vector<zonos3_control_span> & spans, int text_chars, double frames_per_char,
                          std::vector<float> * speaker_embedding,   // in/out; nullptr = no embedding-mode speaker
                          zonos3_steering & out, std::string & err);
