#include "audio_decode.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include <sys/wait.h>
#include <unistd.h>

namespace qasr {

// We invoke ffmpeg via fork/exec (not popen/system) so arbitrary input paths
// are passed as a single argv entry — no shell, no quoting/injection concerns.
DecodeStatus decode_to_wav(const std::string & input_path,
                           const std::string & wav_out_path,
                           const std::string & ffmpeg_bin) {
    DecodeStatus st;

    pid_t pid = ::fork();
    if (pid < 0) {
        st.error = "fork() failed";
        return st;
    }

    if (pid == 0) {
        // ---- child: run ffmpeg, write a 16 kHz mono s16 WAV ----
        // Keep ffmpeg quiet on success; its stderr is inherited for diagnostics.
        char sr[16];
        std::snprintf(sr, sizeof(sr), "%d", kSampleRate);
        char ac[8];
        std::snprintf(ac, sizeof(ac), "%d", kChannels);

        const char * argv[] = {
            ffmpeg_bin.c_str(),
            "-nostdin",
            "-hide_banner",
            "-loglevel", "error",
            "-y",                          // overwrite output
            "-i", input_path.c_str(),
            "-vn",                         // drop any video stream
            "-ac", ac,
            "-ar", sr,
            "-c:a", "pcm_s16le",
            "-f", "wav",
            wav_out_path.c_str(),
            nullptr,
        };
        ::execvp(ffmpeg_bin.c_str(), const_cast<char * const *>(argv));
        std::fprintf(stderr, "qasr: failed to exec '%s': %s\n",
                     ffmpeg_bin.c_str(), std::strerror(errno));
        _exit(127);
    }

    // ---- parent: wait for ffmpeg ----
    int status = 0;
    while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {}

    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        int code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
        st.error = "ffmpeg exited with status " + std::to_string(code) +
                   " (is ffmpeg installed and the input file valid?)";
        return st;
    }

    st.ok = true;
    return st;
}

DecodeStatus decode_to_samples(const std::string & input_path,
                               std::vector<float> & out_samples,
                               const std::string & ffmpeg_bin) {
    DecodeStatus st;
    out_samples.clear();

    int pipefd[2];
    if (::pipe(pipefd) != 0) { st.error = "pipe() failed"; return st; }

    pid_t pid = ::fork();
    if (pid < 0) {
        ::close(pipefd[0]); ::close(pipefd[1]);
        st.error = "fork() failed";
        return st;
    }
    if (pid == 0) {
        ::close(pipefd[0]);
        ::dup2(pipefd[1], STDOUT_FILENO);
        ::close(pipefd[1]);
        char sr[16]; std::snprintf(sr, sizeof(sr), "%d", kSampleRate);
        char ac[8];  std::snprintf(ac, sizeof(ac), "%d", kChannels);
        const char * argv[] = {
            ffmpeg_bin.c_str(), "-nostdin", "-hide_banner", "-loglevel", "error",
            "-i", input_path.c_str(), "-vn", "-f", "s16le", "-acodec", "pcm_s16le",
            "-ac", ac, "-ar", sr, "pipe:1", nullptr,
        };
        ::execvp(ffmpeg_bin.c_str(), const_cast<char * const *>(argv));
        std::fprintf(stderr, "qasr: failed to exec '%s': %s\n", ffmpeg_bin.c_str(), std::strerror(errno));
        _exit(127);
    }

    ::close(pipefd[1]);
    std::vector<uint8_t> raw;
    uint8_t buf[1 << 16];
    for (;;) {
        ssize_t n = ::read(pipefd[0], buf, sizeof(buf));
        if (n < 0) { if (errno == EINTR) continue; break; }
        if (n == 0) break;
        raw.insert(raw.end(), buf, buf + n);
    }
    ::close(pipefd[0]);

    int status = 0;
    while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        int code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
        st.error = "ffmpeg exited with status " + std::to_string(code);
        return st;
    }

    const size_t ns = raw.size() / sizeof(int16_t);
    out_samples.resize(ns);
    const int16_t * pcm = reinterpret_cast<const int16_t *>(raw.data());
    for (size_t i = 0; i < ns; ++i) out_samples[i] = static_cast<float>(pcm[i]) / 32768.0f;

    st.ok = true;
    return st;
}

}  // namespace qasr
