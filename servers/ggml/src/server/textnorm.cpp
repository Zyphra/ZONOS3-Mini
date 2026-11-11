#include "server/textnorm.h"

#include "zonos3-languages.h"
#include "zonos3-markup.h"
#include "zyphra_textnorm.h"

#include <memory>

namespace zs {

namespace {

std::string owned(char * s) {
    const std::unique_ptr<char, decltype(&zyphra_textnorm_free)> out(s, &zyphra_textnorm_free);
    return out.get();
}

} // namespace

void TextNormalizer::Delete::operator()(zyphra_textnorm * tn) const {
    zyphra_textnorm_delete(tn);
}

TextNormalizer::TextNormalizer() {
    std::string tags;
    for (const std::string & tag : zonos3_nonverbal_markup_tags()) tags += (tags.empty() ? "" : "\n") + tag;
    tn_.reset(zyphra_textnorm_new(tags.data(), tags.size()));
}

std::string TextNormalizer::normalize(const std::string & text, const std::string & language) const {
    for (const auto & l : zonos3_server_languages())
        if (l.code == language)
            return owned(zyphra_textnorm_normalize(tn_.get(), text.data(), text.size(), l.nemo.data(), l.nemo.size()));
    return owned(zyphra_textnorm_collapse_line_breaks(text.data(), text.size()));
}

} // namespace zs
