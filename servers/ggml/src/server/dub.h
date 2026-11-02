// Dubbing routes (/v2/dub/*): a port of model/src/zonos3/dub/runtime.py (DubRuntime) and the sgl /v2/dub routes over
// dub-plan (zonos3/dub/plan.py): the reference / dub stores, ffmpeg decode and mux, the alignment worker (a
// JsonLinesWorker over <python-src>/zonos3/workers/dub_align.py) and take synthesis through the scheduler
// (plan_laci_take) with PaceMonitor duration targets; the steer.gguf rate_extension feeds the steered rate buckets.
#pragma once

#include "server/state.h"

#include "httplib.h"
#include "json.hpp"

namespace zs {

bool dub_available(const ServerState & s);   // sgl _dub_available: LACI heads and a speaking-rate condition row
nlohmann::json dub_capabilities(const ServerState & s);
void register_dub_routes(httplib::Server & svr, ServerState & s);

}  // namespace zs
