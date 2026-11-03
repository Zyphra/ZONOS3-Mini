#include "server/textnorm.h"

#include <filesystem>

namespace zs {

TextNormalizer::TextNormalizer(std::string python, std::string python_src)
    : python_(std::move(python)), python_src_(std::move(python_src)) {}

std::string TextNormalizer::normalize(const std::string & text, const std::string & language) {
    std::lock_guard<std::mutex> lk(m_);
    const nlohmann::json request = {{"id", ++next_id_}, {"text", text}, {"language", language}};
    nlohmann::json response;
    for (int attempt = 0;; ++attempt) {
        if (!worker_) {
            const std::string script = (std::filesystem::path(python_src_) / "zonos3" / "workers" / "textnorm.py").string();
            worker_ = std::make_unique<JsonLinesWorker>(
                std::vector<std::string>{python_, script},
                std::map<std::string, std::optional<std::string>>{{"PYTHONPATH", python_src_}, {"VIRTUAL_ENV", std::nullopt}});
        }
        try {
            response = worker_->call(request);
            break;
        } catch (const WorkerError &) {
            worker_.reset();
            if (attempt > 0) throw;   // restarted once already
        }
    }
    if (response.contains("error")) throw WorkerError("text normalization failed: " + response["error"].dump());
    if (!response.contains("text") || !response["text"].is_string()) throw WorkerError("text normalization worker sent no text");
    return response["text"].get<std::string>();
}

void TextNormalizer::close() {
    std::lock_guard<std::mutex> lk(m_);
    worker_.reset();
}

}  // namespace zs
