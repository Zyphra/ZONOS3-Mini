// Text matching of speech editing: a port of model/src/zonos3/tts/textmatch.py (normalize and char_error_rate from
// Zyphra's z-speech-alignment, MIT License, Copyright (c) 2026 Sofian Mejjoute: licenses/z-speech-alignment-MIT.txt;
// and the word-error DP of the verify), with Python's str semantics: lengths count Unicode code points, \w is
// Python's (letters, numbers and _), str.upper()/str.lower() are the full case mappings (src/unicode-data.h, Unicode
// 15.0 = Python 3.12). Text is UTF-8.
#pragma once

#include <string>
#include <vector>

namespace zonos3_textmatch {

// Upper-case; keep word characters and ', everything else (and _) becomes one space; curly quotes fold to ' first.
std::string normalize(const std::string & text);
// normalize(text).split(): the tokens a word or transcript is compared by.
std::vector<std::string> tokens(const std::string & text);
// Levenshtein over the normalized characters (word breaks dropped) / their count; 0 for an empty normalized reference,
// 1 for an empty hypothesis.
double char_error_rate(const std::string & reference, const std::string & hypothesis);
// Levenshtein over tokens (substitutions, insertions and deletions of whole tokens).
int word_errors(const std::vector<std::string> & reference, const std::vector<std::string> & hypothesis);

// Python str.upper() / str.lower() (lower() with the Final_Sigma rule).
std::string py_upper(const std::string & text);
std::string py_lower(const std::string & text);

}  // namespace zonos3_textmatch
