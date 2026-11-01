// Per-codebook sampler: repetition penalty, temperature, top-k, top-p, min-p, multinomial.
#pragma once

#include "zonos3.h"

#include <deque>
#include <random>
#include <vector>

struct zonos3_sampler {
    zonos3_sampling sp;
    std::mt19937 rng;
    int n_cb;
    int vocab;
    // Special ids (zonos3/tts/sampler.py sample_tts): ids below codebook_size are codec codes; the audio pad is never
    // sampled; stop_id (EOA, or span-end for edit infill) is the only other special id a request may sample.
    int codebook_size, pad_id, stop_id;
    // min_frames (sgl TTSSamplingParams.min_frames): the stop id in codebook c is masked while fewer than
    // min_frames + c frames were generated.
    int min_frames = 0;
    int generated = 0;                 // frames accepted so far
    std::vector<std::deque<int>> hist; // recent tokens per codebook (for rep penalty)

    zonos3_sampler(const zonos3_sampling & s, const zonos3_hparams & hp);
    int  sample(const float * logits, int cb);          // logits length == vocab
    void accept(const std::vector<int> & frame);        // push a sampled frame into history
};
