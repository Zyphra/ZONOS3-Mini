#include "server/edit_align.h"

#include "server/util.h"

#include <set>

namespace zs {

AsrWorker & asr_worker(const ServerState & s) {
    if (!s.asr) throw AsrError("ASR is unavailable: the server has no zonos3/workers/asr.py (--python-src)");
    return *s.asr;
}

FuseAligner & fusealign_worker(const ServerState & s) {
    if (!s.aligner) throw FuseAlignError("FuseAlign alignment is unavailable: start the server with --worker-python");
    return *s.aligner;
}

bool needs_anchors(double seconds, const std::vector<std::string> & tokens) {
    if (seconds > LONG_FORM_SECONDS) return true;
    size_t bytes = tokens.empty() ? 0 : tokens.size() - 1;
    for (const auto & t : tokens) bytes += t.size();
    return bytes > MAX_TEXT_BYTES;
}

namespace {

struct ChunkUtterance {
    FuseAlignUtterance u;
    std::string language;
};

// Per audio the Qwen3-ASR transcripts of its segments chunks (forced to its language or detected) in time order,
// empty texts dropped: one segments and one transcribe call for all of them (edit_align.chunk_utterances).
std::vector<std::vector<ChunkUtterance>> chunk_utterances(const ServerState & s, const std::vector<std::string> & audios,
                                                          const std::vector<std::optional<std::string>> & languages) {
    std::vector<std::vector<ChunkUtterance>> out;
    if (audios.empty()) return out;
    const auto chunks = fusealign_worker(s).segments(audios);
    std::vector<AsrItem> items;
    for (size_t a = 0; a < audios.size(); ++a)
        for (const auto & [start, end] : chunks[a]) items.push_back({audios[a], start, end, languages[a]});
    const std::vector<AsrResult> results = items.empty() ? std::vector<AsrResult>{} : asr_worker(s).transcribe(items);
    size_t next = 0;
    for (const auto & cs : chunks) {
        auto & utterances = out.emplace_back();
        for (const auto & [start, end] : cs) {
            const AsrResult & r = results[next++];
            if (!r.text.empty()) utterances.push_back({{r.text, start, end}, r.language});
        }
    }
    return out;
}

std::vector<FuseAlignUtterance> plain(const std::vector<ChunkUtterance> & utterances) {
    std::vector<FuseAlignUtterance> out;
    for (const auto & u : utterances) out.push_back(u.u);
    return out;
}

}  // namespace

Transcript transcribe_audio(const ServerState & s, const std::string & audio, double seconds,
                            const std::optional<std::string> & language) {
    if (seconds <= LONG_FORM_SECONDS) {
        const AsrResult r = asr_worker(s).transcribe({{audio, std::nullopt, std::nullopt, language}})[0];
        return {r.text, r.language, std::nullopt};
    }
    const auto utterances = chunk_utterances(s, {audio}, {language})[0];
    const std::string reported = utterances.empty() ? std::string() : utterances[0].language;
    const std::string name = language ? *language : strip(reported.substr(0, reported.find(',')));
    static const std::set<std::string> UNSPACED = {"Japanese", "Chinese", "Cantonese"};
    const std::string joiner = UNSPACED.count(name) ? "" : " ";
    std::string text;
    for (size_t k = 0; k < utterances.size(); ++k) text += (k ? joiner : "") + utterances[k].u.text;
    return {text, reported, plain(utterances)};
}

std::vector<std::vector<FuseAlignWord>> align_words(const ServerState & s, const std::vector<AlignWordsItem> & items) {
    std::vector<bool> is_long(items.size(), false);
    std::vector<size_t> missing;
    std::vector<std::string> audios;
    std::vector<std::optional<std::string>> languages;
    for (size_t k = 0; k < items.size(); ++k) {
        const auto & it = items[k];
        is_long[k] = !it.tokens.empty() && needs_anchors(it.seconds, it.tokens);
        if (is_long[k] && !it.utterances) {
            missing.push_back(k);
            audios.push_back(it.audio);
            languages.push_back(it.language);
        }
    }
    const auto made = chunk_utterances(s, audios, languages);
    std::vector<FuseAlignItem> requests;
    size_t m = 0;
    for (size_t k = 0; k < items.size(); ++k) {
        FuseAlignItem r{items[k].audio, items[k].tokens, std::nullopt};
        if (is_long[k]) r.utterances = items[k].utterances ? *items[k].utterances : plain(made[m++]);
        requests.push_back(std::move(r));
    }
    return fusealign_worker(s).align(requests);
}

}  // namespace zs
