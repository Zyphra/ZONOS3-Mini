// Speech editing: a port of model/src/zonos3/tts/edit.py (the edit prompt layout, fill extraction and splicing) and
// of the transcript-diff / window planning / verify parts of model/src/zonos3/tts/edit_align.py.
//
// Codes are row-major int32 matrices: ordinary codec frames [frames, n_codebooks], prompt rows
// [rows, n_codebooks + 1]. Text is UTF-8; lengths and edit distances count Unicode code points, as Python str does.
// Invalid input throws std::invalid_argument with the Python ValueError message.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace zonos3_edit {

struct Span { int start = 0, length = 0; };   // (start_frame, length_frames)

// ---------------------------------------------------------------------------------------------------------- edit.py
struct Layout {
    int n_codebooks = 10;
    int audio_pad_id = 4097;   // EOA = audio_pad_id - 1
    int text_vocab = 8196;     // the text pad; edit marker j sits at text_vocab + 1 + j
    int span_end_id = 4098;
};

void validate_edit_spans(const std::vector<Span> & spans, int total_frames, int max_spans);
// Delay-stacked frames (+ one all-`end_id` row when end_id >= 0) followed by K-1 pad rows, text column = text pad.
std::vector<int32_t> sheared_block(const int32_t * frames, int n_frames, const Layout & l, int end_id);
// The training edit layout up to the next infill marker: text rows, the unmasked context (markers at the spans) ending
// on EOA, every generated fill ending on span-end, then marker len(fills).
std::vector<int32_t> edit_prompt_tokens(const std::vector<int32_t> & text_rows, const std::vector<int32_t> & codes,
                                        const std::vector<Span> & spans,
                                        const std::vector<std::vector<int32_t>> & fills, const Layout & l);
struct Fill {
    std::vector<int32_t> frames;   // [n, n_codebooks]
    bool ended = false;            // stopped on span-end
};
// `delayed_rows` are the generated delay-stacked rows without the finishing row [n, n_codebooks]; eos_frame < 0 = none.
Fill extract_fill(const std::vector<int32_t> & delayed_rows, int eos_frame, int audio_pad_id,
                  const std::vector<int> & codebook_sizes);
std::vector<int32_t> splice_edit(const std::vector<int32_t> & codes, int n_codebooks, const std::vector<Span> & spans,
                                 const std::vector<std::vector<int32_t>> & fills);

// ---------------------------------------------------------------------------------------------------- edit_align.py
// Server language code -> the Whisper-style code of the edit's workers (MMS_LANGUAGES: the MMS adapter, FuseAlign's
// tokenization and long-form anchoring); nullopt for a language speech editing does not serve.
std::optional<std::string> mms_language(const std::string & language);
std::string mms_language_error(const std::string & language);   // the ValueError message of mms_language

struct AlignedWord {
    std::string word;
    double start = 0, end = 0;
    std::optional<double> confidence;
    std::vector<std::string> tokens;
};

bool is_group_token(const std::string & token);
double token_chars(const std::string & token);
bool is_marker_token(const std::string & token);
std::string canonical_markup(const std::string & text);

// target_normalize_request: the texts to send to the MMS normalize and how to assemble the answer into
// [target tokens, tokens of target word 0, of word 1, ...] over the whitespace words of the target.
struct TargetNormalize {
    std::vector<std::string> texts;
    bool plain = true;                                 // texts = [target] + words
    std::vector<std::vector<std::string>> pieces;      // per whitespace word (when not plain)
    std::vector<std::vector<std::string>> assemble(const std::vector<std::vector<std::string>> & normalized) const;
};
TargetNormalize target_normalize_request(const std::string & target_text);

std::vector<Span> seconds_to_spans(const std::vector<std::pair<double, double>> & pairs, int total_frames,
                                   double frame_rate);

// difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes(); tag: 'e'qual, 'r'eplace, 'd'elete, 'i'nsert.
struct Opcode { char tag; int i1, i2, j1, j2; };
std::vector<Opcode> sequence_opcodes(const std::vector<std::string> & a, const std::vector<std::string> & b);

struct EditDiff { std::vector<Span> spans; std::vector<int> new_target_tokens; };
EditDiff diff_spans(const std::vector<AlignedWord> & words, const std::vector<std::string> & target_tokens,
                    int total_frames, double frame_rate, double margin_seconds, int min_gap_frames, int max_spans);

// edit_words: caller-supplied word timings ({word, start, end, confidence}) with their zsa tokens, checked.
struct WordIn { std::string word; double start = 0, end = 0; std::optional<double> confidence; };
std::vector<AlignedWord> edit_words(const std::vector<WordIn> & words,
                                    const std::vector<std::vector<std::string>> & word_tokens,
                                    const std::vector<std::string> & original_tokens, double duration);
// aligned_words: FuseAlign's words (confidence = presence) with each word's zsa tokens; words without tokens
// (punctuation only) are dropped, then each overlap of neighbours is split at its midpoint (in order). Throws when the
// lists differ in length.
std::vector<AlignedWord> aligned_words(const std::vector<WordIn> & words,
                                       const std::vector<std::vector<std::string>> & word_tokens);

struct EditWindow {
    int start = 0, end = 0;
    std::vector<int> regions;
    std::string text;
    std::vector<Span> spans(const std::vector<Span> & all) const;   // window-relative
};
EditWindow whole_clip_window(const std::vector<Span> & spans, int total_frames, const std::string & target_text);
std::vector<EditWindow> plan_edit_windows(const std::vector<Span> & spans, const std::vector<AlignedWord> & words,
                                          const std::vector<std::string> & target_words,
                                          const std::vector<std::vector<std::string>> & target_word_tokens,
                                          int total_frames, double frame_rate, double context_seconds,
                                          const std::optional<std::string> & target_text);
std::vector<double> expected_fill_seconds(const std::vector<Span> & spans, const std::vector<AlignedWord> & words,
                                          const std::vector<std::string> & target_words,
                                          const std::vector<std::vector<std::string>> & target_word_tokens,
                                          double frame_rate);

struct VerifyGroup { std::string ipa; std::optional<std::string> heard; };
struct Verify { int word_errors = 0, ref_words = 0; double cer = 0; std::vector<VerifyGroup> groups; };
Verify wildcard_verify(const std::vector<std::string> & reference_tokens,
                       const std::vector<std::string> & hypothesis_tokens);
std::string fill_groups(const std::string & text, const std::vector<std::optional<std::string>> & heard);

// Whitespace words of `text` (Python str.split()).
std::vector<std::string> split_words(const std::string & text);

}  // namespace zonos3_edit
