// The words a forced aligner places: a port of model/src/zonos3/tts/align_text.py (shared by speech editing and
// dubbing). Text is UTF-8.
#pragma once

#include <string>
#include <vector>

namespace aligntext {

// align_tokens: for the unspaced languages (dubplan::UNSPACED: ja, zh, yue) the unit texts of dubplan::text_units with
// their whitespace removed (empty ones dropped), else the whitespace words (Python str.split()). `lang` is a
// Whisper-style code.
std::vector<std::string> align_tokens(const std::string & text, const std::string & lang);

}  // namespace aligntext
