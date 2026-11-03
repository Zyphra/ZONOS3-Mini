// Dubbing (/v2/dub/*): the routes of model/src/zonos3/server/dub.py over the edit/dub worker (server/edit_dub.h), which
// runs zonos3.dub.runtime.DubRuntime and synthesizes every take through this server's scheduler (LACI duration
// targets, the take slots, the steer.gguf rate_extension).
#include "compat.h"

#include "server/dub.h"

#include "server/capabilities.h"
#include "server/edit_dub.h"
#include "server/http.h"
#include "server/schema.h"
#include "server/util.h"

#include <cstdio>
#include <filesystem>
#include <set>

namespace zs {

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

constexpr int64_t MAX_VIDEO_BYTES = int64_t(1) << 30;   // zonos3.dub.plan.MAX_VIDEO_BYTES

// A scratch directory removed with its contents (tempfile.TemporaryDirectory).
struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() / ("dub-" + random_hex(16));
        fs::create_directories(path);
    }
    ~TempDir() { std::error_code ec; fs::remove_all(path, ec); }
    TempDir(const TempDir &) = delete;
    TempDir & operator=(const TempDir &) = delete;
    std::string file(const std::string & name) const { return (path / name).string(); }
};

// zonos3.server.dub make_router's ready().
EditDubWorker & dub_ready(ServerState & s) {
    if (!dub_available(s))
        fail(400, "Dubbing is not enabled on this server: it needs --laci-heads, a checkpoint with a speaking-rate "
                  "condition row and --worker-python (the FuseAlign forced aligner)");
    return *s.edit_dub;
}

const std::set<std::string> UPLOAD_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".wav", ".flac", ".mp3", ".ogg"};

// zonos3.server.dub.upload: the request body (a file named `name`) in `dir`; 413 past the dub's upload cap.
std::string dub_upload(const httplib::Request & req, const std::string & name, const TempDir & dir) {
    if ((int64_t) req.body.size() > MAX_VIDEO_BYTES) fail(413, "upload too large");
    const std::string suffix = lower_ascii(fs::path(name).extension().string());
    const std::string path = dir.file("input" + (UPLOAD_SUFFIXES.count(suffix) ? suffix : std::string(".bin")));
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) fail(500, "cannot write a temporary upload file");
    const bool ok = fwrite(req.body.data(), 1, req.body.size(), f) == req.body.size();
    if (fclose(f) != 0 || !ok) fail(500, "cannot write a temporary upload file");
    return path;
}

std::string param(const httplib::Request & req, const char * key, const char * d) {
    return req.has_param(key) ? req.get_param_value(key) : std::string(d);
}

// zonos3.server.dub.download_stem: re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("._") or "video"
std::string download_stem(const std::string & name) {
    const std::string stem = fs::path(name).stem().string();
    std::string out;
    bool run = false;
    for (char c : stem) {
        const bool ok = isalnum((unsigned char) c) || c == '.' || c == '_' || c == '-';
        if (ok) { out += c; run = false; }
        else if (!run) { out += '_'; run = true; }
    }
    const size_t a = out.find_first_not_of("._"), b = out.find_last_not_of("._");
    out = a == std::string::npos ? "" : out.substr(a, b - a + 1);
    return out.empty() ? "video" : out;
}

json session_json(const httplib::Request & req) {
    const auto session = session_id(req);
    return session ? json(*session) : json();
}

}  // namespace

// LACI heads, a speaking-rate condition row and the edit/dub worker.
bool dub_available(const ServerState & s) {
    return s.edit_dub && !laci_languages(s).empty() && !speaking_rate_edges(s).empty();
}

void register_dub_routes(httplib::Server & svr, ServerState & s) {
    svr.Get("/v2/dub/capabilities", guarded([&s](const httplib::Request &, httplib::Response & res) {
        if (!s.edit_dub) fail(503, "dubbing needs the edit/dub worker (--worker-python)");
        json caps = s.edit_dub->request("dub_capabilities", json::object());
        caps["enabled"] = dub_available(s);
        send_json(res, caps);
    }));
    svr.Post("/v2/dub/ref", keepalive([&s](const httplib::Request & req, httplib::Response & res) {
        EditDubWorker & w = dub_ready(s);
        TempDir tmp;
        const std::string path = dub_upload(req, param(req, "name", "audio"), tmp);
        send_json(res, w.request("dub_ref", {{"path", path}}));
    }));
    svr.Post(R"(/v2/dub/lips/([^/]+))", keepalive([&s](const httplib::Request & req, httplib::Response & res) {
        EditDubWorker & w = dub_ready(s);
        TempDir tmp;
        const std::string video = dub_upload(req, param(req, "name", "video"), tmp);
        send_json(res, w.request("dub_lips", {{"ref_id", std::string(req.matches[1])}, {"video", video}}));
    }));
    svr.Post("/v2/dub/transcribe", keepalive([&s](const httplib::Request & req, httplib::Response & res) {
        const json body = schema::parse_body(req, "DubTranscribeRequest");
        EditDubWorker & w = dub_ready(s);
        send_json(res, w.request("dub_transcribe", {{"ref_id", body["ref_id"]}, {"ref_lang", body.value("ref_lang", json())}}));
    }));
    svr.Post("/v2/dub/analyze", keepalive([&s](const httplib::Request & req, httplib::Response & res) {
        const json body = schema::parse_body(req, "DubAnalyzeRequest");
        send_json(res, dub_ready(s).request("dub_analyze", {{"request", body}}));
    }));
    svr.Post("/v2/dub/generate", keepalive([&s](const httplib::Request & req, httplib::Response & res) {
        const json body = schema::parse_body(req, "DubGenerateRequest");
        send_json(res, dub_ready(s).request("dub_generate", {{"request", body}, {"session", session_json(req)}}));
    }));
    svr.Post("/v2/dub/regenerate", keepalive([&s](const httplib::Request & req, httplib::Response & res) {
        const json body = schema::parse_body(req, "DubRegenerateRequest");
        send_json(res, dub_ready(s).request("dub_regenerate", {{"request", body}, {"session", session_json(req)}}));
    }));
    // The video with the dub as its audio track (raw video bytes in, MP4 out).
    svr.Post(R"(/v2/dub/mux/([^/]+))", guarded([&s](const httplib::Request & req, httplib::Response & res) {
        EditDubWorker & w = dub_ready(s);
        const std::string name = param(req, "name", "video");
        TempDir tmp;
        const std::string video = dub_upload(req, name, tmp);
        const std::string out = tmp.file("out.mp4");
        w.request("dub_mux", {{"dub_id", std::string(req.matches[1])}, {"video", video}, {"out", out}});
        std::string data;
        if (!read_file(out, data)) fail(500, "ffmpeg wrote no video");
        res.set_header("Content-Disposition", "attachment; filename=\"" + download_stem(name) + "_dub.mp4\"");
        res.set_content(data, "video/mp4");
    }));
}

}  // namespace zs
