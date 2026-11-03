// Speech editing: POST /v2/edit/transcribe, /v2/edit/align, /v2/edit and the edit capabilities (sgl tts_edit*).
//
// Transcripts come from the server's ASR worker (Qwen3-ASR, server/asr.h) and every word timing from its FuseAlign
// worker (server/fusealign.h), through the long-form orchestration of server/edit_align.h; the edit tokens of the word
// diff and the verify scores are computed here (src/zonos3-textmatch.h). Each edited region is one span-end-terminated
// generation through the scheduler (src/zonos3-edit.h ports zonos3/tts/edit.py and the planning of edit_align.py), and
// the spliced codes are decoded once.
#pragma once

#include "server/state.h"

#include "httplib.h"
#include "json.hpp"

namespace zs {

// {speech_editing, edit_max_spans, edit_frame_rate, edit_max_seconds, edit_align_max_seconds}
nlohmann::json edit_capabilities(const ServerState & s);
void register_edit_routes(httplib::Server & svr, ServerState & s);

}  // namespace zs
