#pragma once
#include <filesystem>
#include <vector>

// The ffmpeg used for audio: $ZONOS3_FFMPEG, one next to the executable (dist layout / macOS Resources), else
// "ffmpeg" on PATH.
std::filesystem::path audio_ffmpeg_path();

// Decode mono PCM with the configured ffmpeg; failures return an empty vector.
std::vector<float> audio_decode_file(const char * path, int sample_rate = 44100, int max_seconds = 60);
