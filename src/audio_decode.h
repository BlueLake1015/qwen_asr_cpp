// audio_decode.h — decode any video/audio file to a 16 kHz mono 16-bit WAV by
// shelling out to the external `ffmpeg` command. We hand the resulting WAV to
// the CrispASR engine, so video containers (mp4, mkv, ...) and any codec
// ffmpeg understands all work, and the engine always gets the format it wants.
#pragma once

#include <string>
#include <vector>

namespace qasr {

constexpr int kSampleRate = 16000;  // 16 kHz
constexpr int kChannels   = 1;      // mono

struct DecodeStatus {
    bool ok = false;
    std::string error;
};

// Decode `input_path` into a 16 kHz mono s16 WAV written to `wav_out_path`.
// `ffmpeg_bin` is the ffmpeg executable (default "ffmpeg", resolved via PATH).
DecodeStatus decode_to_wav(const std::string & input_path,
                           const std::string & wav_out_path,
                           const std::string & ffmpeg_bin = "ffmpeg");

// Decode `input_path` into an in-memory 16 kHz mono float buffer ([-1,1]) via
// ffmpeg (raw s16le read from a pipe). Used by the per-segment aligned path.
DecodeStatus decode_to_samples(const std::string & input_path,
                               std::vector<float> & out_samples,
                               const std::string & ffmpeg_bin = "ffmpeg");

}  // namespace qasr
