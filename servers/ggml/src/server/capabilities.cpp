#include "server/capabilities.h"

#include "server/dub.h"
#include "server/edit.h"
#include "server/http.h"
#include "server/generate.h"
#include "server/phonemize.h"
#include "server/projects.h"
#include "server/speakers.h"
#include "server/textnorm.h"
#include "server/util.h"

#include "zonos3-languages.h"
#include "zonos3-markup.h"

#include <cmath>
#include <filesystem>

namespace zs {

using json = nlohmann::json;

// The matched codec of the release architecture (zonos3.codec = "mossdual"; the converter checks params.audio_codec).
static const char * AUDIO_CODEC = "opus_dual_3n_600k_sched600k_r4096_step600000";
static const char * SPEAKER_EMBEDDING_CODEC = "ownmect_s2";

std::vector<std::string> laci_languages(const ServerState & s) {
    std::vector<std::string> out;   // sorted(config.laci_heads): std::map iterates in key order
    for (const auto & [lang, heads] : s.laci) out.push_back(lang);
    return out;
}

// sgl _condition_spec.
static json condition_tokens(const ServerState & s) {
    const auto & hp = s.model.hp;
    if (hp.n_cond == 0) return nullptr;
    json edges = json::object(), units = json::object();
    std::string layout = "[reference prefix | speaker row]";
    for (size_t i = 0; i < hp.cond_names.size(); ++i) {
        json e = json::array();
        if (i < hp.cond_edges.size()) for (float v : hp.cond_edges[i]) e.push_back(float_decimal(v));
        edges[hp.cond_names[i]] = e;
        if (i < hp.cond_units.size()) units[hp.cond_names[i]] = hp.cond_units[i];
        layout += "[" + hp.cond_names[i] + "]";
    }
    return {{"enabled", true}, {"names", hp.cond_names}, {"flags", json::object()}, {"count", (int) hp.n_cond},
            {"n_buckets", (int) hp.n_cond_buckets}, {"unknown_index", (int) hp.n_cond_buckets}, {"edges", edges},
            {"units", units}, {"layout", layout + " BOS text EOS"}};
}

// sgl _silence_prefill_spec (the matched dual codec has codec-matched silence frames).
static json silence_prefill(const ServerState & s) {
    const int hop = dac_samples_per_frame(s.dac);
    const double rate = zonos3_generation().sample_rate;
    const int frames = s.silence_prefill_frames;
    return {{"available", true}, {"default_frames", frames}, {"default_on", s.silence_prefill_on},
            {"default_seconds", std::round(frames * hop / rate * 1000.0) / 1000.0}, {"frame_seconds", hop / rate},
            {"codec", AUDIO_CODEC},
            {"note", "codec-matched silence frames open the audio stream before the model starts (their audio is not "
                     "emitted); a request without silence_prefill_frames gets the server default, silence_prefill_frames=0 "
                     "turns it off and silence_prefill_frames=N sets the length"}};
}

// sgl _decoder_capability: the iSTFT head decoder (dac.gguf) behind the matched codec.
static json decoder(const ServerState & s, const std::string & dac_name) {
    return {{"kind", "istft_mr"}, {"label", "Multi-resolution iSTFT head (" + dac_name + "); causal"},
            {"checkpoint", dac_name}, {"expect_step", nullptr}, {"headroom_db", (double) s.dac.headroom_db},
            {"causal", true}, {"codec", AUDIO_CODEC}};
}

json capabilities(const ServerState & s) {
    const auto & hp = s.model.hp;
    const auto modes = checkpoint_speaker_modes(s);
    const bool embedding = hp.speaker_dim > 0;
    const bool cfg = !modes.empty() && s.batch_slots >= 2;
    const zonos3_generation_registry & limits = zonos3_generation();
    json caps = {
        {"text_tokenizer_spec", json::parse(s.model.tok.spec_json)},
        {"text_normalization_enabled", s.text_normalizer != nullptr},
        {"text_norm_languages", zonos3_server_language_codes()},
        {"accents", server_accents(s)},
        {"speaker_enabled", !modes.empty()},
        {"speaker_conditioning_type", "raw_reference_prefix"},
        {"speaker_embedding_dim", embedding ? json(hp.speaker_dim) : json(nullptr)},
        {"speaker_embedding_codec", embedding ? json(SPEAKER_EMBEDDING_CODEC) : json(nullptr)},
        {"n_codebooks", (int) hp.n_codebooks},
        {"codebook_sizes", std::vector<int>(hp.n_codebooks, (int) hp.codebook_size)},
        {"audio_codec", AUDIO_CODEC},
        {"max_tokens", s.max_frames},
        {"stream_decode_modes", {"prefix", "whole"}},
        {"stream_decode_default", s.stream_decode_default},
        {"silence_prefill", silence_prefill(s)},
        // `trained`: the checkpoint was fine-tuned on nonverbal markup (GGUF zonos3.tag_cfg_trained; sgl reads config.yaml).
        {"tag_cfg", {{"available", s.batch_slots >= 2}, {"trained", s.model.hp.tag_cfg_trained},
                     {"tags", zonos3_nonverbal_markup_tags()},
                     {"min_scale", limits.tag_cfg_min_scale}, {"max_scale", limits.tag_cfg_max_scale}, {"default_scale", 1.0}}},
        {"decode_context", {{"available", true}, {"default_frames", limits.decode_context_frames}, {"max_frames", 450},
                            {"note", "decode the generated frames behind the last decode_context_frames of the raw-reference prompt "
                                     "(their audio dropped), so the first generated frame is not the decoder's position 0; "
                                     "no-op without a reference prompt"}}},
        {"speaker_modes", modes},
        {"speaker_mode_default", modes.empty() ? json(nullptr) : json(modes[0])},
        {"decoder", decoder(s, s.dac_name)},
        {"condition_tokens", condition_tokens(s)},
        {"speaker_audio_upload", !modes.empty() && (speaker_encoder_available(s) || reference_audio_available(s))},
        {"speaker_cfg_supported", cfg},
        {"speaker_cfg_modes", cfg ? json(modes) : json::array()},
        {"speaker_cfg_unconditional", "remove_reference_prefix_or_speaker_row_condition_rows_unknown_keep_text_and_generated_history"},
        {"speaker_cfg_formula", "uncond + scale * (cond - uncond)"},
        {"laci_languages", laci_languages(s)},
        {"dub", dub_available(s)},
        {"speaker_embedding_upload", embedding},
        {"speaker_embedding_cache", !modes.empty()},
        {"speaker_embedding_blend", embedding},
        {"default_voices_enabled", !s.default_voice_dir.empty()},
        {"projects", s.projects ? s.projects->capabilities() : json{{"enabled", false}}},
    };
    for (const json & part : {edit_capabilities(s), phoneme_capabilities(s)})
        for (auto it = part.begin(); it != part.end(); ++it) caps[it.key()] = it.value();
    return caps;
}

void register_capability_routes(httplib::Server & svr, ServerState & s) {
    svr.Get("/v2/health", [](const httplib::Request &, httplib::Response & res) { send_json(res, {{"status", "ok"}}); });
    svr.Get("/v1/models", [&s](const httplib::Request &, httplib::Response & res) {
        send_json(res, {{"object", "list"},
                        {"data", json::array({json{{"id", "zonos3"}, {"object", "model"}, {"created", 0}, {"owned_by", "zyphra"},
                                                   {"root", std::filesystem::path(s.model_path).filename().string()}}})}});
    });
    svr.Get("/v2/capabilities", guarded([&s](const httplib::Request &, httplib::Response & res) { send_json(res, capabilities(s)); }));
    svr.Get("/v2/controls", guarded([&s](const httplib::Request &, httplib::Response & res) {
        if (!s.steer_loaded) fail(404, "This server has no steering bundle (place steer.gguf next to the backbone or pass --steer).");
        res.set_content(s.controls_doc, "application/json");
    }));
}

}  // namespace zs
