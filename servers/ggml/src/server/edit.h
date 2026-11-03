// Speech editing: POST /v2/edit/transcribe, /v2/edit/align, /v2/edit and the edit capabilities (sgl tts_edit*).
//
// MMS-1B-all (ASR text and zsa token normalization) runs in the z-speech-alignment interpreter (ZSA_ROOT, ZSA_PYTHON)
// as JsonLinesWorker processes over <python-src>/zonos3/workers/zsa_mms.py, one per language (ZONOS3_MMS_DEVICE, at
// most ZONOS3_MMS_MAX_WORKERS); every word timing comes from the server's FuseAlign worker (server/fusealign.h). Each
// edited region is one span-end-terminated generation through the scheduler (src/zonos3-edit.h ports
// zonos3/tts/edit.py and the planning of edit_align.py), and the spliced codes are decoded once.
#pragma once

#include "server/state.h"

#include "httplib.h"
#include "json.hpp"

namespace zs {

// {speech_editing, edit_max_spans, edit_frame_rate, edit_max_seconds, edit_align_max_seconds}
nlohmann::json edit_capabilities(const ServerState & s);
void register_edit_routes(httplib::Server & svr, ServerState & s);

}  // namespace zs
