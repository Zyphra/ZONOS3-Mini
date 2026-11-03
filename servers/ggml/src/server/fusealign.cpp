#include "server/fusealign.h"

#include <filesystem>

namespace zs {

using json = nlohmann::json;

FuseAligner::FuseAligner(std::string python, std::string python_src)
    : python_(std::move(python)), python_src_(std::move(python_src)) {}

// A new worker process, pinged for its checkpoint id.
void FuseAligner::start() {
    const std::string script = (std::filesystem::path(python_src_) / "zonos3" / "workers" / "fusealign.py").string();
    worker_ = std::make_unique<JsonLinesWorker>(
        std::vector<std::string>{python_, script},
        std::map<std::string, std::optional<std::string>>{{"PYTHONPATH", python_src_},
                                                          {"PYTHONDONTWRITEBYTECODE", std::string("1")},
                                                          {"VIRTUAL_ENV", std::nullopt}});
    const json ping = worker_->call({{"id", ++next_id_}, {"op", "ping"}});
    if (!(ping.value("status", "") == "ok" && ping.contains("model") && ping["model"].is_string()))
        throw WorkerError("the FuseAlign worker did not answer its ping: " + ping.dump());
    model_ = ping["model"].get<std::string>();
}

json FuseAligner::call(json request) {
    json response;
    for (int attempt = 0;; ++attempt) {
        try {
            if (!worker_) start();
            request["id"] = ++next_id_;
            response = worker_->call(request);
            break;
        } catch (const WorkerError & e) {
            worker_.reset();
            if (attempt > 0) throw FuseAlignError(std::string("FuseAlign worker: ") + e.what());   // restarted once already
        }
    }
    if (response.value("status", "") != "ok") {
        auto text = [&](const char * k) { return response.contains(k) && response[k].is_string() ? response[k].get<std::string>() : std::string(); };
        if (response.contains("invalid") && response["invalid"].is_boolean() && response["invalid"].get<bool>())
            throw std::invalid_argument(!text("message").empty() ? text("message") : text("error"));
        throw FuseAlignError("FuseAlign failed: " + (text("error").empty() ? response.dump() : text("error")));
    }
    return response;
}

std::vector<std::vector<FuseAlignWord>> FuseAligner::align(const std::vector<FuseAlignItem> & items) {
    std::vector<std::vector<FuseAlignWord>> out(items.size());
    json sent = json::array();
    std::vector<size_t> index;
    for (size_t i = 0; i < items.size(); ++i) {
        if (items[i].tokens.empty()) continue;
        sent.push_back({{"audio", items[i].audio}, {"tokens", items[i].tokens},
                        {"language", items[i].language.empty() ? json() : json(items[i].language)}});
        index.push_back(i);
    }
    if (sent.empty()) return out;
    json response;
    {
        std::lock_guard<std::mutex> lk(m_);
        response = call({{"op", "align"}, {"items", sent}});
    }
    const json & results = response.contains("results") ? response["results"] : json();
    if (!results.is_array() || results.size() != index.size())
        throw FuseAlignError("FuseAlign worker sent a malformed align answer");
    for (size_t n = 0; n < index.size(); ++n) {
        const json & words = results[n].at("words");
        if (!words.is_array() || words.size() != items[index[n]].tokens.size())
            throw FuseAlignError("FuseAlign worker sent a word count that does not match the tokens");
        auto & dst = out[index[n]];
        for (const auto & w : words)
            dst.push_back({w.at("word").get<std::string>(), w.at("start").get<double>(), w.at("end").get<double>(),
                           w.at("presence").get<double>(), w.at("placed").get<bool>()});
    }
    return out;
}

std::string FuseAligner::model() {
    std::lock_guard<std::mutex> lk(m_);
    for (int attempt = 0; !worker_; ++attempt) {
        try {
            start();
        } catch (const WorkerError & e) {
            worker_.reset();
            if (attempt > 0) throw FuseAlignError(std::string("FuseAlign worker: ") + e.what());
        }
    }
    return model_;
}

void FuseAligner::close() {
    std::lock_guard<std::mutex> lk(m_);
    worker_.reset();
}

}  // namespace zs
