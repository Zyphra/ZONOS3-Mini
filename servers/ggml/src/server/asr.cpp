#include "server/asr.h"

#include <cstdlib>
#include <filesystem>
#include <stdexcept>

namespace zs {

using json = nlohmann::json;
namespace fs = std::filesystem;

static const char * ASR_UNAVAILABLE = "ASR is unavailable: ";

AsrWorker::AsrWorker(std::string python_src) : python_src_(std::move(python_src)) {
    const char * py = getenv("ZONOS3_ASR_PYTHON");
    python_ = py && *py ? std::string(py)
                        : (fs::path(python_src_) / ".." / ".." / ".venv-asr" / "bin" / "python").lexically_normal().string();
    argv_ = {python_, (fs::path(python_src_) / "zonos3" / "workers" / "asr.py").string()};
    const char * device = getenv("ZONOS3_ASR_DEVICE");
    if (device && *device) { argv_.push_back("--device"); argv_.push_back(device); }
}

bool AsrWorker::installed() const {
    std::error_code ec;
    return fs::is_regular_file(python_, ec);
}

// One request, on a worker started on first use and again after it exited (a call that meets an exited worker is
// retried once on a fresh one), as zonos3.tts.edit_align._JsonLinesWorker.call.
json AsrWorker::call(json request) {
    if (!installed())
        throw AsrError(std::string(ASR_UNAVAILABLE) + "interpreter not found: " + python_ + " (run tools/setup_asr_env.sh)");
    if (!worker_) {
        std::map<std::string, std::optional<std::string>> env{{"PYTHONPATH", python_src_},
                                                              {"PYTHONDONTWRITEBYTECODE", std::string("1")},
                                                              {"VIRTUAL_ENV", std::nullopt}};
        if (!getenv("HF_HUB_OFFLINE")) env["HF_HUB_OFFLINE"] = std::string("1");
        worker_ = std::make_unique<JsonLinesWorker>(argv_, std::move(env));
    }
    request["id"] = ++next_id_;
    json response;
    for (int attempt = 0;; ++attempt) {
        try {
            response = worker_->call(request);
            break;
        } catch (const WorkerError & e) {
            if (attempt > 0) throw AsrError(std::string(ASR_UNAVAILABLE) + e.what());
        }
    }
    if (response.value("status", "") != "ok") {
        auto text = [&](const char * k) { return response.contains(k) && response[k].is_string() ? response[k].get<std::string>() : std::string(); };
        if (response.contains("invalid") && response["invalid"].is_boolean() && response["invalid"].get<bool>())
            throw std::invalid_argument("ASR failed: " + (!text("message").empty() ? text("message") : text("error")));
        throw AsrError(ASR_UNAVAILABLE + (text("error").empty() ? std::string("unknown error") : text("error")));
    }
    return response;
}

void AsrWorker::ping() {
    if (model_) return;
    const json r = call({{"op", "ping"}});
    if (!(r.contains("model") && r["model"].is_string() && r.contains("languages") && r["languages"].is_array()))
        throw AsrError(std::string(ASR_UNAVAILABLE) + "malformed ping answer: " + r.dump());
    languages_ = r["languages"].get<std::vector<std::string>>();
    model_ = r["model"].get<std::string>();
}

std::vector<AsrResult> AsrWorker::transcribe(const std::vector<AsrItem> & items) {
    std::vector<AsrResult> out;
    if (items.empty()) return out;
    json sent = json::array();
    for (const auto & it : items) {
        json item = {{"audio", it.audio}};
        if (it.start) item["start"] = *it.start;
        if (it.end) item["end"] = *it.end;
        item["language"] = it.language ? json(*it.language) : json();
        sent.push_back(std::move(item));
    }
    json response;
    {
        std::lock_guard<std::mutex> lk(m_);
        response = call({{"op", "transcribe"}, {"items", sent}});
    }
    const json & results = response.contains("results") ? response["results"] : json();
    if (!results.is_array() || results.size() != items.size())
        throw AsrError(std::string(ASR_UNAVAILABLE) + "malformed transcribe answer");
    for (const auto & r : results) {
        if (!r.is_object() || !r.contains("text") || !r["text"].is_string())
            throw AsrError(std::string(ASR_UNAVAILABLE) + "malformed transcribe answer");
        out.push_back({r["text"].get<std::string>(),
                       r.contains("language") && r["language"].is_string() ? r["language"].get<std::string>() : std::string()});
    }
    return out;
}

std::string AsrWorker::model() {
    std::lock_guard<std::mutex> lk(m_);
    ping();
    return *model_;
}

std::vector<std::string> AsrWorker::languages() {
    std::lock_guard<std::mutex> lk(m_);
    ping();
    return languages_;
}

void AsrWorker::close() {
    std::lock_guard<std::mutex> lk(m_);
    if (worker_) worker_->close();
}

}  // namespace zs
