// POST /v2/generate and POST /v1/audio/speech: a GenerateRequest (or SpeechRequest) resolved against the loaded model,
// scheduled, and answered as streamed / buffered float32 PCM or a buffered 16-bit WAV.
#pragma once

#include "server/state.h"

#include "httplib.h"
#include "json.hpp"

#include <optional>
#include <string>
#include <map>
#include <memory>
#include <vector>

namespace zs {

// A validated GenerateRequest resolved into what the scheduler runs (sgl _tts_generate_msg, the tokenizer worker and
// TTSScheduler._schedule_prefill): the request settings, the conditional prompt and, for CFG, its twin.
struct GeneratePlan {
    GenReq gr;
    zonos3_prompt prompt;
    zonos3_prompt twin;               // n_rows == 0: no CFG twin
    // "" | "speaker" (speaker row / reference prefix dropped) | "tag" (markup stripped) | "emotion" (same prompt,
    // no emotion / steering delta)
    std::string cfg_kind;
    std::vector<float> speaker_delta; // proj-space emotion delta on the projected speaker row (empty = none)
    std::string span_follow;          // X-Control-Span-Follow: "" (no control_spans) | "rate" | "reading"
};

// Throws HttpError (fail) with sgl's status and detail for an invalid request.
GeneratePlan plan_generate(ServerState & s, const nlohmann::json & body, const std::optional<std::string> & session);

// The job of a planned request: the slot-capacity check, the decoder settings and, for LACI / span follow, the
// alignment probe (attach_probe). fail() 400 when it cannot run; submit it with submit_job.
std::shared_ptr<ReqJob> make_job(ServerState & s, GeneratePlan && plan);

// A LACI take of the dub (the edit/dub worker's synthesize call, server/edit_dub.h). `request` is a GenerateRequest
// body (schema-valid; laci=true, stream=false); `duration` the take's LACI duration target. `scale` puts the take on an
// extended speaking-rate scale (zonos3.dub.plan.RateScale):
// in a steered bucket (one of `doses`) the take's hidden delta gains `doses[bucket]` of the bundle direction
// `direction` (steer.gguf rate_extension); the take carries its delta in every bucket so a rate-adjusting retake can
// switch buckets. Without `scale` retakes move along the checkpoint's speaking-rate row.
struct LaciTake {
    nlohmann::json request;
    zonos3_laci::DurationTarget duration;
    struct Scale {
        std::vector<double> edges;   // bucket edges (UTF-8 bytes / s); buckets first..first + edges.size()
        int first = 0;
        int bucket = 0;              // the take's bucket on the scale
        std::map<int, double> doses; // steered buckets -> tempo dose
        std::string direction;       // steering-bundle direction id of the tempo dose
    };
    std::optional<Scale> scale;
};
// Validates and plans a take like /v2/generate (fail(): HttpError 400 for a bad voice / control / text / scale).
// Plan every take first, then submit_job each and wait_take (the takes interleave with other requests).
std::shared_ptr<ReqJob> plan_laci_take(ServerState & s, const LaciTake & take, const std::optional<std::string> & session);
struct TakeResult {
    std::vector<float> pcm;   // 44.1 kHz mono float32 with the decoder gain (empty: no audio)
    nlohmann::json laci;      // LaciSession.record()
};
// Blocks until the job finishes; throws std::runtime_error when the generation failed.
TakeResult wait_take(const std::shared_ptr<ReqJob> & job);

// The silence-prefill default (sgl _silence_prefill_spec) from the environment: ZONOS3_SILENCE_PREFILL_FRAMES (default
// 3; unparsable = 3) frames, off when ZONOS3_SILENCE_PREFILL_DEFAULT is 0/false/no/off. False when out of range.
bool configure_silence_prefill(ServerState & s, std::string & err);

// Loads the emotion directions (emotion.gguf), accent adapters (accents.gguf; attached as model.accents) and accent
// directions (accent-directions.gguf) recorded in `s` (empty path = none) and derives s.steer_sources (sgl
// _steering_sources; the steering bundle, when loaded, adopts that layout). Call after the backbone and the steering
// bundle are loaded. False with the reason in `err`.
bool load_emotion_accents(ServerState & s, std::string & err);
// sgl _server_accents: the adapter accents, then the accent directions (named, then axes).
std::vector<std::string> server_accents(const ServerState & s);

// Request pieces shared with /v2/edit (sgl helpers; fail() with sgl's 400 / 500 details):
// _normalize_tts_request_language (body["language"], default en_us).
std::string resolve_language(const nlohmann::json & body);
// _resolve_condition_buckets (condition_buckets / condition_values): one bucket per condition row, n_cond_buckets =
// unknown.
std::vector<int32_t> resolve_condition_buckets(const ServerState & s, const nlohmann::json & body);
// The tokenizer's TTSTextNormalizer step: `text` unchanged unless requested and a normalizer runs; a worker failure is
// a 500 "text normalization failed: ...".
std::string normalize_text(ServerState & s, const std::string & text, const std::string & language, bool requested);

void register_generate_routes(httplib::Server & svr, ServerState & s);

}  // namespace zs
