// zonos3-tokenizer.h — SuperBPE text tokenizer (HF `tokenizers` BPE JSON, byte-level, no normalizer).
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Pronunciation groups of a phoneme-token checkpoint (text_tokenizer_spec.phonemes; zonos3/tts/superbpe.py): a group
// `open_marker` IPA `close_marker` encodes as open_id, one id per IPA code point (open_id + 2 + index in `symbols`),
// close_id.
struct zonos3_phoneme_spec {
    std::string open_marker = "\xEE\x80\x80", close_marker = "\xEE\x80\x81";  // U+E000 / U+E001
    int32_t open_id = -1, close_id = -1;
    std::vector<std::string> symbols;   // single code points (UTF-8)
};

struct zonos3_tokenizer {
    std::unordered_map<std::string, int32_t> vocab;       // byte-level symbol string -> tokenizer id
    std::unordered_map<std::string, int32_t> merge_rank;  // key = a + '\x01' + b
    std::vector<std::pair<std::string, int32_t>> added;   // literal special tokens
    int32_t offset = 4, bos = 2, eos = 3;
    std::string spec_json;            // zonos3.text_tokenizer_spec as stored in the GGUF
    bool has_phonemes = false;
    zonos3_phoneme_spec phonemes;     // valid when has_phonemes
};

// Parse a `tokenizers` JSON (model.type BPE, merges as [a,b] arrays). Model ids are
// tokenizer id + offset; bos/eos are model ids. Returns false and sets err on any other shape.
bool zonos3_tokenizer_load(zonos3_tokenizer & tok, const std::string & json, int32_t offset,
                           int32_t bos, int32_t eos, std::string & err);
// Parse and check zonos3.text_tokenizer_spec (superbpe.validate_spec and load_spec's extra-row check) into `tok`.
bool zonos3_tokenizer_set_spec(zonos3_tokenizer & tok, const std::string & spec_json, uint32_t text_vocab,
                               uint32_t text_extra_tokens, std::string & err);

// Model text ids for `utf8`, including the leading BOS and trailing EOS (plain SuperBPE, no pronunciation groups).
std::vector<int32_t> zonos3_tokenize(const zonos3_tokenizer & tok, const std::string & utf8);
// superbpe.payload_ids (no BOS/EOS): SuperBPE, with pronunciation groups as phoneme ids on a phoneme-token checkpoint.
// False with Python's ValueError text in `err` ("Unbalanced phoneme group markers", "Unbalanced or empty phoneme
// group", "Phones outside the inventory: [...]").
bool zonos3_payload_ids(const zonos3_tokenizer & tok, const std::string & utf8, std::vector<int32_t> & out,
                        std::string & err);
// superbpe.text_ids: [BOS] payload_ids [EOS].
bool zonos3_text_ids(const zonos3_tokenizer & tok, const std::string & utf8, std::vector<int32_t> & out,
                     std::string & err);
// The server's _validate_phoneme_groups: "" when `utf8` has no group markers or valid groups, else the 400 message
// ("This checkpoint has no phoneme tokens: ..." / "Pronunciation groups: <payload_ids error>").
std::string zonos3_check_phoneme_groups(const zonos3_tokenizer & tok, const std::string & utf8);
