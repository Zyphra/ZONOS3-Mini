// Dubbing routes (/v2/dub/*) over the edit/dub worker (server/edit_dub.h: zonos3.dub.runtime.DubRuntime): uploads are
// stored in a temporary directory and handed over by path; the worker's take synthesis runs on the scheduler.
#pragma once

#include "server/state.h"

#include "httplib.h"
#include "json.hpp"

namespace zs {

bool dub_available(const ServerState & s);   // LACI heads, a speaking-rate condition row and --worker-python
void register_dub_routes(httplib::Server & svr, ServerState & s);

}  // namespace zs
