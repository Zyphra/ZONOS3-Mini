#include "server/fusealign.h"

#include <cstdlib>
#include <filesystem>

namespace zs {

using json = nlohmann::json;

static const char * UNAVAILABLE = "FuseAlign alignment is unavailable: ";

FuseAligner::FuseAligner(std::string python, std::string python_src)
    : argv_{python, (std::filesystem::path(python_src) / "zonos3" / "workers" / "fusealign.py").string()},
      python_src_(std::move(python_src)) {}

// One request, on a worker started on first use and again after it exited (a call that meets an exited worker is
// retried once on a fresh one), as zonos3.tts.edit_align._JsonLinesWorker.call.
json FuseAligner::call(json request) {
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
            if (attempt > 0) throw FuseAlignError(std::string(UNAVAILABLE) + e.what());
        }
    }
    if (response.value("status", "") != "ok") {
        auto text = [&](const char * k) { return response.contains(k) && response[k].is_string() ? response[k].get<std::string>() : std::string(); };
        if (response.contains("invalid") && response["invalid"].is_boolean() && response["invalid"].get<bool>())
            throw std::invalid_argument("FuseAlign alignment failed: " + (!text("message").empty() ? text("message") : text("error")));
        throw FuseAlignError(UNAVAILABLE + (text("error").empty() ? std::string("unknown error") : text("error")));
    }
    return response;
}

std::vector<std::vector<FuseAlignWord>> FuseAligner::align(const std::vector<FuseAlignItem> & items) {
    std::vector<std::vector<FuseAlignWord>> out(items.size());
    json sent = json::array();
    std::vector<size_t> index;
    for (size_t i = 0; i < items.size(); ++i) {
        if (items[i].tokens.empty()) continue;
        json item = {{"audio", items[i].audio}, {"tokens", items[i].tokens}};
        if (items[i].utterances) {
            json utterances = json::array();
            for (const auto & u : *items[i].utterances)
                utterances.push_back({{"text", u.text}, {"start", u.start}, {"end", u.end}});
            item["utterances"] = std::move(utterances);
        }
        sent.push_back(std::move(item));
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
        throw FuseAlignError(std::string(UNAVAILABLE) + "malformed align answer");
    for (size_t n = 0; n < index.size(); ++n) {
        const json & words = results[n].at("words");
        if (!words.is_array() || words.size() != items[index[n]].tokens.size())
            throw FuseAlignError(std::string(UNAVAILABLE) + "the worker sent a word count that does not match the tokens");
        auto & dst = out[index[n]];
        for (const auto & w : words)
            dst.push_back({w.at("word").get<std::string>(), w.at("start").get<double>(), w.at("end").get<double>(),
                           w.at("presence").get<double>(), w.at("placed").get<bool>()});
    }
    return out;
}

std::vector<std::vector<std::pair<double, double>>> FuseAligner::segments(const std::vector<std::string> & audios) {
    std::vector<std::vector<std::pair<double, double>>> out;
    if (audios.empty()) return out;
    json sent = json::array();
    for (const auto & a : audios) sent.push_back({{"audio", a}});
    json response;
    {
        std::lock_guard<std::mutex> lk(m_);
        response = call({{"op", "segments"}, {"items", sent}});
    }
    const json & results = response.contains("results") ? response["results"] : json();
    if (!results.is_array() || results.size() != audios.size())
        throw FuseAlignError(std::string(UNAVAILABLE) + "malformed segments answer");
    for (const auto & r : results) {
        const json & chunks = r.contains("chunks") ? r["chunks"] : json();
        if (!chunks.is_array()) throw FuseAlignError(std::string(UNAVAILABLE) + "malformed segments answer");
        auto & dst = out.emplace_back();
        for (const auto & c : chunks) dst.emplace_back(c.at(0).get<double>(), c.at(1).get<double>());
    }
    return out;
}

std::string FuseAligner::model() {
    std::lock_guard<std::mutex> lk(m_);
    if (!model_) {
        const json r = call({{"op", "ping"}});
        if (!(r.contains("model") && r["model"].is_string()))
            throw FuseAlignError(std::string(UNAVAILABLE) + "malformed ping answer: " + r.dump());
        model_ = r["model"].get<std::string>();
    }
    return *model_;
}

void FuseAligner::close() {
    std::lock_guard<std::mutex> lk(m_);
    if (worker_) worker_->close();
}

}  // namespace zs
