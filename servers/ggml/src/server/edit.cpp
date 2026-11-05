#include "compat.h"

#include "server/edit.h"

#include "server/http.h"
#include "server/pyhost.h"
#include "server/schema.h"
#include "server/speakers.h"

namespace zs {

using json = nlohmann::json;

namespace {

// The request (schema-checked) through the Python host.
json forward(ServerState & s, const char * op, const httplib::Request & req, const char * schema_name) {
    const json body = schema::parse_body(req, schema_name);
    if (!s.pyhost) {
        if (!zonos3_supports_editing(s.model))
            fail(400, "The loaded checkpoint does not support speech editing (needs span_end_id and marker rows).");
        fail(503, "speech editing needs the Python host (--worker-python)");
    }
    return s.pyhost->request(op, {{"request", body}});
}

}  // namespace

// speech_editing: an edit checkpoint, the Python host with FuseAlign (fusealign_hub in the --worker-python
// interpreter), the ASR interpreter (ZONOS3_ASR_PYTHON / .venv-asr) and the reference encoder (the clip's codes).
json edit_capabilities(const ServerState & s) {
    const json m = edit_model(s);
    const bool editing = m["max_spans"].get<int>() > 0;
    const bool available = s.pyhost && s.fusealign_installed && asr_installed(s) && reference_audio_available(s);
    // The registry's speech-edit clip limit bounds transcribe / align and every edit (zonos3.tts.edit_runtime).
    const double max_seconds = zonos3_generation().edit_max_seconds;
    return {{"speech_editing", editing && available}, {"edit_max_spans", m["max_spans"]},
            {"edit_frame_rate", editing ? m["frame_rate"] : json()}, {"edit_max_seconds", max_seconds},
            {"edit_align_max_seconds", max_seconds}};
}

void register_edit_routes(httplib::Server & svr, ServerState & s) {
    svr.Post("/v2/edit/transcribe", keepalive([&s](const httplib::Request & req, httplib::Response & res) {
        send_json(res, forward(s, "edit_transcribe", req, "EditTranscribeRequest"));
    }));
    svr.Post("/v2/edit/align", keepalive([&s](const httplib::Request & req, httplib::Response & res) {
        send_json(res, forward(s, "edit_align", req, "EditAlignRequest"));
    }));
    svr.Post("/v2/edit", keepalive([&s](const httplib::Request & req, httplib::Response & res) {
        send_json(res, forward(s, "edit", req, "EditRequest"));
    }));
}

}  // namespace zs
