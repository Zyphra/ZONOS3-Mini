#include "align-text.h"

#include "dub-plan.h"
#include "zonos3-edit.h"

namespace aligntext {

std::vector<std::string> align_tokens(const std::string & text, const std::string & lang) {
    if (!dubplan::unspaced(lang)) return zonos3_edit::split_words(text);
    std::vector<std::string> out;
    for (const auto & u : dubplan::text_units(text, lang)) {
        std::string token;   // the unit with its whitespace removed
        for (const auto & piece : zonos3_edit::split_words(u.text)) token += piece;
        if (!token.empty()) out.push_back(std::move(token));
    }
    return out;
}

}  // namespace aligntext
