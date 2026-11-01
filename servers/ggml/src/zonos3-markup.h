// Nonverbal (NV) markup of the nonverbal fine-tune recipe: a port of model/src/zonos3/tts/markup.py.
//
// Recognized markers are point tags ("<laugh/>") and span tags ("<shouting>", "</shouting>") of NONVERBAL_MARKUP_TAGS,
// matching the Python pattern </?(?:tag|...)\s*/?> . Text is UTF-8; character offsets are Unicode code points, as
// Python str indices.
#pragma once

#include <string>
#include <utility>
#include <vector>

// laugh, breath, sigh, cough, panting, shouting, crying (zonos3.tts.markup.NONVERBAL_MARKUP_TAGS order).
const std::vector<std::string> & zonos3_nonverbal_markup_tags();
// Whether text contains a recognized nonverbal annotation.
bool zonos3_has_nonverbal_markup(const std::string & text);
// Alternate text / marker pieces: even indices are the (possibly empty) text around the markers, odd ones the markers,
// so their concatenation is `text`.
std::vector<std::string> zonos3_split_nonverbal_markup(const std::string & text);
// Code-point ranges [start, end) of the recognized markers.
std::vector<std::pair<int, int>> zonos3_nonverbal_marker_spans(const std::string & text);
// Remove the markers (keeping their enclosed spoken words) and collapse whitespace: " ".join(sub(" ", text).split()).
std::string zonos3_strip_nonverbal_markup(const std::string & text);
