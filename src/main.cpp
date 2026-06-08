// qwen-asr — video/audio -> transcription using Qwen3-ASR.
//
// Pipeline:  external ffmpeg (decode any media -> 16 kHz mono WAV)
//            -> CrispASR engine (Qwen3-ASR + optional Qwen3-ForcedAligner)
//            -> txt / srt / vtt.
//
// The inference engine is CrispASR (vendored at third_party/CrispASR): a pure
// C++/ggml speech runtime, no Python. This program owns argument parsing, the
// mandatory ffmpeg decode front-end, temp-file management, and invoking the
// engine with the right flags. We exec the `crispasr` binary rather than link
// it, so we inherit its model auto-download, forced-aligner routing, and
// subtitle writers for free.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <vector>

#include "audio_decode.h"
#include "crispasr_capi.h"

#ifndef CRISPASR_BIN
#define CRISPASR_BIN "crispasr"  // overridden by the Makefile to an abs path
#endif

namespace {

const char * kVersion = "qwen-asr 0.1.0 (Qwen3-ASR via CrispASR)";

#ifndef MODELS_DIR
#define MODELS_DIR "models"  // overridden by the Makefile to an abs path
#endif

struct Args {
    std::string input;
    std::string output;                 // empty -> print transcript to stdout
    std::string model;                  // explicit GGUF path / "auto"; empty ->
                                        // resolve from models_dir by size+quant
    std::string backend;                // empty -> auto-detect from GGUF metadata
                                        // (for -m auto we default it to qwen3)
    std::string aligner;                // explicit ForcedAligner GGUF; empty ->
                                        // resolve from models_dir by quant
    std::string language;               // e.g. en, zh, ko; empty -> model default
    std::string ffmpeg = "ffmpeg";
    std::string crispasr = CRISPASR_BIN;
    std::string models_dir = MODELS_DIR;
    std::string size = "1.7b";          // 0.6b | 1.7b (for model auto-resolution)
    std::string asr_quant = "q8_0";     // quantization config for the ASR model
    std::string aligner_quant = "q8_0"; // quantization config for the aligner
    int threads = 4;
    bool split_on_punct = true;         // nicer subtitle line breaks

    // --- segmentation / streaming ---
    std::string mode = "offline";       // offline | fixed | vad
    double segment_seconds = 10.0;      // fixed mode: sliding-window length (s)
    double segment_step = 0.0;          // fixed mode: window step (s); 0 -> engine default
    bool stream_json = false;           // emit --stream-json partial/final events
    bool realtime = false;              // pace ffmpeg with -re (simulate a live source)
    double vad_threshold = -1.0;        // vad mode: speech prob threshold (<0 -> default)
    int vad_min_silence_ms = -1;        // vad mode: min silence to cut (<0 -> default)
    int vad_min_speech_ms = -1;         // vad mode: min speech to keep (<0 -> default)

    // --- approach C: per-segment live + forced aligner (linked C-ABI) ---
    bool aligned = false;               // fixed/vad: align each segment for accurate timing
    std::string vad_model;              // VAD GGUF/bin for vad-mode segmentation (default: cached silero)

    bool help = false;
    bool version = false;
};

void usage(const char * prog) {
    std::fprintf(stderr,
        "%s\n\n"
        "Transcribe a video/audio file with Qwen3-ASR (C++/ggml via CrispASR).\n\n"
        "Usage:\n"
        "  %s <input-media> [-o out.srt] [options]\n\n"
        "Required:\n"
        "  <input-media>          any file ffmpeg can decode (mp4, mkv, wav, mp3, ...)\n\n"
        "Options:\n"
        "  -o, --output <path>    output file; extension picks format:\n"
        "                         .srt -> SubRip, .vtt -> WebVTT, else plain text.\n"
        "                         Omit to print the transcript to stdout.\n"
        "  -m, --model <path>     explicit Qwen3-ASR GGUF, or 'auto' to auto-download.\n"
        "                         Default: <models-dir>/qwen3-asr-<size>-<asr-quant>.gguf\n"
        "                         if present, else 'auto'.\n"
        "      --aligner <path>   explicit Qwen3-ForcedAligner GGUF (precise timestamps).\n"
        "                         Default: <models-dir>/qwen3-forced-aligner-0.6b-\n"
        "                         <aligner-quant>.gguf if present.\n"
        "      --size <s>         model size for auto-resolution: 0.6b | 1.7b (default 1.7b)\n"
        "      --asr-quant <q>    ASR quantization: q8_0 (default) | q4_k | q5_0 | f16\n"
        "      --aligner-quant <q> aligner quantization: q8_0 (default) | q4_k | q5_0 | f16\n"
        "      --models-dir <d>   where to look for model GGUFs (default: ./models)\n"
        "      --backend <name>   force backend: qwen3 | qwen3-1.7b (default: auto-detect)\n"
        "  -l, --language <code>  language hint (en, zh, ko, ...); default: auto\n"
        "  -t, --threads <n>      inference threads (default 4)\n"
        "      --no-split         do not split subtitle lines on punctuation\n"
        "      --ffmpeg <path>    ffmpeg executable (default: ffmpeg on PATH)\n"
        "      --crispasr <path>  crispasr engine binary (default: vendored build)\n"
        "\n"
        "Segmentation mode:\n"
        "      --mode <m>         offline (default) | fixed | vad\n"
        "                          offline: decode whole file -> one pass -> aligned\n"
        "                                   .srt/.vtt/.txt (uses the forced aligner)\n"
        "                          fixed:   STREAM ffmpeg -> engine, fixed-length windows\n"
        "                          vad:     STREAM ffmpeg -> engine, VAD-adaptive segments\n"
        "                         Streaming modes emit incremental text/JSON to stdout (or -o\n"
        "                         file); no SRT/aligner (engine streaming limitation).\n"
        "      --segment-seconds <s> fixed-mode window length (default 10)\n"
        "      --segment-step <s>    fixed-mode window step (default: engine ~3s)\n"
        "      --vad-threshold <f>   vad-mode speech threshold 0..1 (default ~0.5)\n"
        "      --min-silence-ms <n>  vad-mode silence to split a segment\n"
        "      --min-speech-ms <n>   vad-mode minimum speech to keep\n"
        "      --json             streaming: emit --stream-json partial/final events\n"
        "      --realtime         pace input at 1x (simulate a live source)\n"
        "      --aligned          fixed/vad: per-segment forced-aligner pass for accurate\n"
        "                         timestamps (ASR + aligner loaded ONCE, no reload). Emits\n"
        "                         a real SRT/VTT incrementally. Implied by --aligner.\n"
        "      --vad-model <path> VAD model for vad-mode segmentation in --aligned\n"
        "                         (default: cached ggml-silero in ~/.cache/crispasr)\n"
        "  -h, --help             show this help        --version\n\n"
        "Examples:\n"
        "  %s talk.mp4 -o talk.srt -l en                # offline: aligned SRT\n"
        "  %s talk.mp4 --mode vad -l en                 # streaming, VAD-adaptive, live stdout\n"
        "  %s talk.mp4 --mode fixed --segment-seconds 8 # streaming, fixed 8s windows\n",
        kVersion, prog, prog, prog, prog);
}

const char * take(int & i, int argc, char ** argv, const char * flag) {
    if (i + 1 >= argc) {
        std::fprintf(stderr, "error: %s requires a value\n", flag);
        return nullptr;
    }
    return argv[++i];
}

bool parse_args(int argc, char ** argv, Args & a) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        const char * v = nullptr;
        auto need = [&](const char * f) { v = take(i, argc, argv, f); return v != nullptr; };

        if (arg == "-h" || arg == "--help") a.help = true;
        else if (arg == "--version") a.version = true;
        else if (arg == "-o" || arg == "--output") { if (!need("--output")) return false; a.output = v; }
        else if (arg == "-m" || arg == "--model") { if (!need("--model")) return false; a.model = v; }
        else if (arg == "--backend") { if (!need("--backend")) return false; a.backend = v; }
        else if (arg == "--aligner" || arg == "-am") { if (!need("--aligner")) return false; a.aligner = v; }
        else if (arg == "--size") { if (!need("--size")) return false; a.size = v; }
        else if (arg == "--asr-quant") { if (!need("--asr-quant")) return false; a.asr_quant = v; }
        else if (arg == "--aligner-quant") { if (!need("--aligner-quant")) return false; a.aligner_quant = v; }
        else if (arg == "--models-dir") { if (!need("--models-dir")) return false; a.models_dir = v; }
        else if (arg == "--mode") { if (!need("--mode")) return false; a.mode = v; }
        else if (arg == "--segment-seconds") { if (!need("--segment-seconds")) return false; a.segment_seconds = std::atof(v); }
        else if (arg == "--segment-step") { if (!need("--segment-step")) return false; a.segment_step = std::atof(v); }
        else if (arg == "--vad-threshold") { if (!need("--vad-threshold")) return false; a.vad_threshold = std::atof(v); }
        else if (arg == "--min-silence-ms") { if (!need("--min-silence-ms")) return false; a.vad_min_silence_ms = std::atoi(v); }
        else if (arg == "--min-speech-ms") { if (!need("--min-speech-ms")) return false; a.vad_min_speech_ms = std::atoi(v); }
        else if (arg == "--json") a.stream_json = true;
        else if (arg == "--realtime") a.realtime = true;
        else if (arg == "--aligned") a.aligned = true;
        else if (arg == "--vad-model") { if (!need("--vad-model")) return false; a.vad_model = v; }
        else if (arg == "-l" || arg == "--language" || arg == "--lang") { if (!need("--language")) return false; a.language = v; }
        else if (arg == "-t" || arg == "--threads") { if (!need("--threads")) return false; a.threads = std::atoi(v); }
        else if (arg == "--no-split") a.split_on_punct = false;
        else if (arg == "--ffmpeg") { if (!need("--ffmpeg")) return false; a.ffmpeg = v; }
        else if (arg == "--crispasr") { if (!need("--crispasr")) return false; a.crispasr = v; }
        else if (!arg.empty() && arg[0] == '-' && arg != "-") {
            std::fprintf(stderr, "error: unknown option '%s'\n", arg.c_str());
            return false;
        } else {
            if (!a.input.empty()) {
                std::fprintf(stderr, "error: multiple inputs ('%s' and '%s')\n",
                             a.input.c_str(), arg.c_str());
                return false;
            }
            a.input = arg;
        }
    }
    return true;
}

// "out.srt" -> base "out", flag "-osrt". Empty/"-"/no-ext -> plain text.
struct OutSpec { std::string base; const char * flag; std::string ext; };

OutSpec output_spec(const std::string & path) {
    auto ends = [&](const char * s) {
        size_t n = std::strlen(s);
        return path.size() >= n && path.compare(path.size() - n, n, s) == 0;
    };
    if (ends(".srt")) return {path.substr(0, path.size() - 4), "-osrt", "srt"};
    if (ends(".vtt")) return {path.substr(0, path.size() - 4), "-ovtt", "vtt"};
    auto dot = path.find_last_of('.');
    std::string base = (dot == std::string::npos) ? path : path.substr(0, dot);
    return {base, "-otxt", "txt"};
}

bool file_exists(const std::string & path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

// Resolve the ASR model from models_dir by size+quant (default policy: only use
// files under models/). An explicit -m <path> or `-m auto` is honored as-is.
// Returns false (fatal) if no local model is found and none was given.
bool resolve_model(Args & a) {
    if (a.model.empty()) {
        std::string cand = a.models_dir + "/qwen3-asr-" + a.size + "-" + a.asr_quant + ".gguf";
        if (file_exists(cand)) {
            a.model = cand;
        } else {
            std::fprintf(stderr,
                "error: ASR model not found: %s\n"
                "       Run scripts/get_model.sh (downloads into %s), or pass -m <file.gguf> / -m auto.\n",
                cand.c_str(), a.models_dir.c_str());
            return false;
        }
    }
    std::fprintf(stderr, "      ASR model: %s\n", a.model.c_str());
    return true;
}

// Resolve the forced-aligner path from models_dir by quant (models/-only).
void resolve_aligner(Args & a) {
    if (a.aligner.empty()) {
        std::string cand = a.models_dir + "/qwen3-forced-aligner-0.6b-" + a.aligner_quant + ".gguf";
        if (file_exists(cand)) a.aligner = cand;
    }
    std::fprintf(stderr, "      aligner:   %s\n",
                 a.aligner.empty() ? "(none — timestamps will be coarse)" : a.aligner.c_str());
}

// Resolve the VAD model (models/-only; --vad-model overrides). Empty if absent.
std::string resolve_vad_model(const Args & a) {
    if (!a.vad_model.empty()) return a.vad_model;
    std::string p = a.models_dir + "/ggml-silero-v6.2.0.bin";
    if (file_exists(p)) return p;
    return "";
}

int run(const std::vector<std::string> & argv_s) {
    std::vector<char *> argv;
    argv.reserve(argv_s.size() + 1);
    for (const auto & s : argv_s) argv.push_back(const_cast<char *>(s.c_str()));
    argv.push_back(nullptr);

    pid_t pid = ::fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        ::execvp(argv[0], argv.data());
        std::fprintf(stderr, "qasr: failed to exec '%s': %s\n", argv[0], std::strerror(errno));
        _exit(127);
    }
    int status = 0;
    while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

std::vector<char *> to_cargv(const std::vector<std::string> & v) {
    std::vector<char *> out;
    out.reserve(v.size() + 1);
    for (const auto & s : v) out.push_back(const_cast<char *>(s.c_str()));
    out.push_back(nullptr);
    return out;
}

// Run `ffmpeg | crispasr --stream`: ffmpeg decodes the media to s16le PCM on its
// stdout, which is piped to crispasr's stdin. crispasr streams transcripts to
// `cr_stdout_fd` (a file, or -1 to inherit our stdout). Returns crispasr's exit.
int run_pipeline(const std::vector<std::string> & ff_argv,
                 const std::vector<std::string> & cr_argv,
                 int cr_stdout_fd) {
    int p[2];
    if (::pipe(p) != 0) { std::perror("qasr: pipe"); return -1; }

    pid_t ff = ::fork();
    if (ff < 0) { std::perror("qasr: fork"); return -1; }
    if (ff == 0) {
        ::dup2(p[1], STDOUT_FILENO);
        ::close(p[0]); ::close(p[1]);
        auto av = to_cargv(ff_argv);
        ::execvp(av[0], av.data());
        std::fprintf(stderr, "qasr: exec ffmpeg failed: %s\n", std::strerror(errno));
        _exit(127);
    }

    pid_t cr = ::fork();
    if (cr < 0) { std::perror("qasr: fork"); return -1; }
    if (cr == 0) {
        ::dup2(p[0], STDIN_FILENO);
        if (cr_stdout_fd >= 0) ::dup2(cr_stdout_fd, STDOUT_FILENO);
        ::close(p[0]); ::close(p[1]);
        auto av = to_cargv(cr_argv);
        ::execvp(av[0], av.data());
        std::fprintf(stderr, "qasr: exec crispasr failed: %s\n", std::strerror(errno));
        _exit(127);
    }

    ::close(p[0]); ::close(p[1]);
    int st_ff = 0, st_cr = 0;
    while (::waitpid(ff, &st_ff, 0) < 0 && errno == EINTR) {}
    while (::waitpid(cr, &st_cr, 0) < 0 && errno == EINTR) {}
    return WIFEXITED(st_cr) ? WEXITSTATUS(st_cr) : -1;
}

// ffmpeg argv that decodes `input` to 16 kHz mono s16le PCM on stdout (pipe:1).
std::vector<std::string> ffmpeg_pcm_argv(const Args & a) {
    std::vector<std::string> v = {a.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error"};
    if (a.realtime) v.push_back("-re");           // pace at 1x to mimic a live feed
    v.insert(v.end(), {
        "-i", a.input,
        "-vn", "-f", "s16le", "-acodec", "pcm_s16le",
        "-ac", "1", "-ar", "16000", "pipe:1",
    });
    return v;
}

// crispasr argv for streaming (fixed window or VAD-adaptive).
std::vector<std::string> crispasr_stream_argv(const Args & a) {
    std::vector<std::string> v = {a.crispasr, "--stream", "-m", a.model,
                                  "-t", std::to_string(a.threads)};
    if (!a.backend.empty()) { v.push_back("--backend"); v.push_back(a.backend); }
    if (!a.language.empty()) { v.push_back("-l"); v.push_back(a.language); }
    if (a.stream_json) v.push_back("--stream-json");

    if (a.mode == "vad") {
        v.push_back("--vad");
        std::string vm = resolve_vad_model(a);  // models/-only; avoids cache auto-download
        if (!vm.empty()) { v.push_back("--vad-model"); v.push_back(vm); }
        if (a.vad_threshold >= 0) { v.push_back("--vad-threshold"); v.push_back(std::to_string(a.vad_threshold)); }
        if (a.vad_min_silence_ms >= 0) { v.push_back("--vad-min-silence-duration-ms"); v.push_back(std::to_string(a.vad_min_silence_ms)); }
        if (a.vad_min_speech_ms >= 0) { v.push_back("--vad-min-speech-duration-ms"); v.push_back(std::to_string(a.vad_min_speech_ms)); }
    } else {  // fixed
        v.push_back("--stream-length");
        v.push_back(std::to_string((long)(a.segment_seconds * 1000)));
        if (a.segment_step > 0) {
            v.push_back("--stream-step");
            v.push_back(std::to_string((long)(a.segment_step * 1000)));
        }
    }
    return v;
}

// ---- offline mode: decode whole file -> one engine pass -> aligned subtitle ----
int run_offline(Args & a) {
    if (!resolve_model(a)) return 1;
    resolve_aligner(a);

    const bool to_stdout = a.output.empty() || a.output == "-";

    std::string tmpdir = "/tmp/qwen_asr_" + std::to_string(::getpid());
    if (::mkdir(tmpdir.c_str(), 0700) != 0 && errno != EEXIST) {
        std::fprintf(stderr, "error: cannot create temp dir %s\n", tmpdir.c_str());
        return 1;
    }
    std::string wav = tmpdir + "/audio.wav";

    // 1) Decode media -> 16 kHz mono WAV via external ffmpeg.
    std::fprintf(stderr, "[1/2] decoding %s via ffmpeg...\n", a.input.c_str());
    qasr::DecodeStatus dec = qasr::decode_to_wav(a.input, wav, a.ffmpeg);
    if (!dec.ok) {
        std::fprintf(stderr, "error: decode failed: %s\n", dec.error.c_str());
        ::rmdir(tmpdir.c_str());
        return 1;
    }

    // 2) Transcribe with CrispASR (Qwen3-ASR + optional ForcedAligner).
    OutSpec out = to_stdout ? OutSpec{tmpdir + "/out", "-otxt", "txt"}
                            : output_spec(a.output);

    // Backend: honor explicit --backend, else auto-detect from GGUF metadata.
    // Only "-m auto" needs an explicit backend so the engine knows what to fetch.
    std::string backend = a.backend;
    if (backend.empty() && a.model == "auto") backend = "qwen3";

    std::vector<std::string> cmd = { a.crispasr };
    if (!backend.empty()) { cmd.push_back("--backend"); cmd.push_back(backend); }
    cmd.insert(cmd.end(), {
        "-m", a.model,
        "-f", wav,
        out.flag,
        "-of", out.base,
        "-t", std::to_string(a.threads),
    });
    if (!a.aligner.empty()) { cmd.push_back("-am"); cmd.push_back(a.aligner); }
    if (!a.language.empty()) { cmd.push_back("-l"); cmd.push_back(a.language); }
    if (a.split_on_punct && out.ext != "txt") cmd.push_back("-sp");

    std::fprintf(stderr, "[2/2] transcribing with %s (backend=%s)...\n",
                 a.crispasr.c_str(), backend.empty() ? "auto" : backend.c_str());
    int rc = run(cmd);
    if (rc != 0) {
        std::fprintf(stderr,
            "error: crispasr engine failed (exit %d).\n"
            "       binary: %s\n"
            "       If it's missing, run 'make' to build it, or pass --crispasr <path>.\n",
            rc, a.crispasr.c_str());
        ::unlink(wav.c_str());
        ::rmdir(tmpdir.c_str());
        return 1;
    }

    std::string produced = out.base + "." + out.ext;
    if (to_stdout) {
        if (FILE * f = std::fopen(produced.c_str(), "rb")) {
            char buf[1 << 14];
            size_t n;
            while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) std::fwrite(buf, 1, n, stdout);
            std::fclose(f);
        }
        ::unlink(produced.c_str());
    } else {
        std::fprintf(stderr, "wrote %s\n", produced.c_str());
    }

    ::unlink(wav.c_str());
    ::rmdir(tmpdir.c_str());
    return 0;
}

// ---- streaming modes (fixed / vad): ffmpeg PCM piped live into crispasr ----
int run_streaming(Args & a) {
    if (!resolve_model(a)) return 1;
    if (a.backend.empty() && a.model == "auto") a.backend = "qwen3";
    if (!a.aligner.empty())
        std::fprintf(stderr, "note: --aligner is ignored in streaming modes "
                             "(engine streaming has no forced-aligner path)\n");

    const bool to_stdout = a.output.empty() || a.output == "-";
    int out_fd = -1;
    if (!to_stdout) {
        OutSpec spec = output_spec(a.output);
        if (spec.ext != "txt")
            std::fprintf(stderr, "note: streaming output is %s, not subtitles; writing %s\n",
                         a.stream_json ? "JSON-Lines" : "plain text", a.output.c_str());
        out_fd = ::open(a.output.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (out_fd < 0) {
            std::fprintf(stderr, "error: cannot open output %s: %s\n",
                         a.output.c_str(), std::strerror(errno));
            return 1;
        }
    }

    std::fprintf(stderr, "[stream:%s] ffmpeg -> crispasr (live, %s)...\n",
                 a.mode.c_str(),
                 a.mode == "vad" ? "VAD-adaptive segments"
                                 : ("fixed " + std::to_string(a.segment_seconds) + "s windows").c_str());

    std::vector<std::string> ff = ffmpeg_pcm_argv(a);
    std::vector<std::string> cr = crispasr_stream_argv(a);
    int rc = run_pipeline(ff, cr, out_fd);
    if (out_fd >= 0) ::close(out_fd);

    if (rc != 0) {
        std::fprintf(stderr, "error: streaming pipeline failed (crispasr exit %d)\n", rc);
        return 1;
    }
    if (!to_stdout) std::fprintf(stderr, "wrote %s\n", a.output.c_str());
    return 0;
}

// ---- approach C: per-segment live + forced aligner, both models resident ----

std::string srt_time_cs(int64_t cs, bool vtt) {
    if (cs < 0) cs = 0;
    int64_t ms = cs * 10;
    int64_t h = ms / 3600000; ms %= 3600000;
    int64_t m = ms / 60000;   ms %= 60000;
    int64_t s = ms / 1000;    ms %= 1000;
    char b[32];
    std::snprintf(b, sizeof(b), vtt ? "%02lld:%02lld:%02lld.%03lld" : "%02lld:%02lld:%02lld,%03lld",
                  (long long)h, (long long)m, (long long)s, (long long)ms);
    return b;
}

std::vector<std::string> tokenize_ws(const std::string & s) {
    std::vector<std::string> w;
    std::string cur;
    for (char c : s) {
        if (std::isspace(static_cast<unsigned char>(c))) { if (!cur.empty()) { w.push_back(cur); cur.clear(); } }
        else cur += c;
    }
    if (!cur.empty()) w.push_back(cur);
    return w;
}

std::string trim_copy(const std::string & s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

int run_aligned(Args & a) {
    if (!resolve_model(a)) return 1;
    if (a.model == "auto") {
        std::fprintf(stderr,
            "error: --aligned needs a local ASR model (not -m auto).\n"
            "       Run scripts/get_model.sh or pass -m <file.gguf>.\n");
        return 1;
    }
    resolve_aligner(a);
    if (a.aligner.empty()) {
        std::fprintf(stderr,
            "error: --aligned needs a forced-aligner GGUF.\n"
            "       Run scripts/get_model.sh or pass --aligner <file.gguf>.\n");
        return 1;
    }

    // 1) Decode whole file -> in-memory 16 kHz mono PCM via external ffmpeg.
    std::fprintf(stderr, "[decode] %s via ffmpeg...\n", a.input.c_str());
    std::vector<float> pcm;
    qasr::DecodeStatus dec = qasr::decode_to_samples(a.input, pcm, a.ffmpeg);
    if (!dec.ok || pcm.empty()) {
        std::fprintf(stderr, "error: decode failed: %s\n", dec.error.c_str());
        return 1;
    }
    const int SR = qasr::kSampleRate;
    const int total = static_cast<int>(pcm.size());

    // 2) Load ASR + aligner ONCE (resident across all segments).
    std::fprintf(stderr, "[load] ASR=%s  aligner=%s (loaded once)\n",
                 a.model.c_str(), a.aligner.c_str());
    crispasr_session * sess = crispasr_session_open(a.model.c_str(), a.threads);
    if (!sess) { std::fprintf(stderr, "error: failed to open ASR session\n"); return 1; }
    crispasr_paligner * aln = crispasr_paligner_open(a.aligner.c_str(), a.threads);
    if (!aln) { std::fprintf(stderr, "error: failed to open aligner\n"); crispasr_session_close(sess); return 1; }

    // 3) Build segments (sample ranges).
    std::vector<std::pair<int,int>> segs;  // (start, n)
    if (a.mode == "vad") {
        std::string vm = resolve_vad_model(a);
        if (vm.empty()) {
            std::fprintf(stderr, "error: no VAD model. Pass --vad-model <path>, or run a\n"
                                 "       '--mode vad' once to cache ggml-silero.\n");
            crispasr_paligner_close(aln); crispasr_session_close(sess); return 1;
        }
        float * spans = nullptr;
        int n = crispasr_vad_slices(vm.c_str(), pcm.data(), total, SR,
                                    a.vad_threshold,                       // <=0 -> model default
                                    a.vad_min_speech_ms >= 0 ? a.vad_min_speech_ms : 250,
                                    a.vad_min_silence_ms >= 0 ? a.vad_min_silence_ms : 100,
                                    30,                                    // speech pad ms
                                    a.segment_seconds,                     // cap each chunk
                                    a.threads, &spans);
        for (int i = 0; i < n; ++i) {
            int s0 = static_cast<int>(spans[2*i] * SR);
            int s1 = static_cast<int>(spans[2*i+1] * SR);
            s0 = std::max(0, std::min(s0, total));
            s1 = std::max(s0, std::min(s1, total));
            if (s1 > s0) segs.emplace_back(s0, s1 - s0);
        }
        if (spans) crispasr_vad_free(spans);
        std::fprintf(stderr, "[vad] %zu speech segments\n", segs.size());
    } else {  // fixed
        int L = std::max(1, static_cast<int>(a.segment_seconds * SR));
        for (int off = 0; off < total; off += L)
            segs.emplace_back(off, std::min(L, total - off));
        std::fprintf(stderr, "[fixed] %zu windows of %.1fs\n", segs.size(), a.segment_seconds);
    }

    // 4) Output sink: SRT/VTT/txt. Incremental — flush per cue.
    const bool to_stdout = a.output.empty() || a.output == "-";
    OutSpec spec = to_stdout ? OutSpec{"", "-osrt", "srt"} : output_spec(a.output);
    const bool vtt = spec.ext == "vtt";
    const bool txt = spec.ext == "txt";
    FILE * out = to_stdout ? stdout : std::fopen(a.output.c_str(), "w");
    if (!out) { std::fprintf(stderr, "error: cannot open %s\n", a.output.c_str());
                crispasr_paligner_close(aln); crispasr_session_close(sess); return 1; }
    if (vtt) std::fputs("WEBVTT\n\n", out);

    // 5) Per-segment loop: transcribe (resident) -> align (resident) -> emit.
    int idx = 1;
    for (size_t si = 0; si < segs.size(); ++si) {
        const float * p = pcm.data() + segs[si].first;
        int n = segs[si].second;

        crispasr_session_result * r = crispasr_session_transcribe(sess, p, n);
        std::string text;
        if (r) {
            int nseg = crispasr_session_result_n_segments(r);
            for (int k = 0; k < nseg; ++k) {
                const char * t = crispasr_session_result_segment_text(r, k);
                if (t) { if (!text.empty()) text += ' '; text += t; }
            }
            crispasr_session_result_free(r);
        }
        text = trim_copy(text);
        if (text.empty()) continue;

        int64_t off_cs = static_cast<int64_t>(segs[si].first) * 100 / SR;
        int64_t seg_end_cs = static_cast<int64_t>(segs[si].first + n) * 100 / SR;

        std::vector<std::string> words = tokenize_ws(text);
        int64_t cue_t0 = off_cs, cue_t1 = seg_end_cs;
        if (!words.empty()) {
            std::vector<const char *> wptr(words.size());
            for (size_t i = 0; i < words.size(); ++i) wptr[i] = words[i].c_str();
            std::vector<int64_t> t0(words.size()), t1(words.size());
            int arc = crispasr_paligner_align(aln, wptr.data(), static_cast<int>(words.size()),
                                              p, n, off_cs, t0.data(), t1.data());
            if (arc == 0) { cue_t0 = t0.front(); cue_t1 = t1.back(); }
        }
        if (cue_t1 <= cue_t0) cue_t1 = cue_t0 + 50;  // guard

        if (txt) {
            std::fprintf(out, "%s\n", text.c_str());
        } else {
            if (!vtt) std::fprintf(out, "%d\n", idx);
            std::fprintf(out, "%s --> %s\n%s\n\n",
                         srt_time_cs(cue_t0, vtt).c_str(), srt_time_cs(cue_t1, vtt).c_str(), text.c_str());
        }
        std::fflush(out);
        // live progress on stderr (so stdout stays clean when piping)
        std::fprintf(stderr, "[%s --> %s] %s\n",
                     srt_time_cs(cue_t0, false).c_str(), srt_time_cs(cue_t1, false).c_str(), text.c_str());
        ++idx;
    }

    if (out != stdout) { std::fclose(out); std::fprintf(stderr, "wrote %s\n", a.output.c_str()); }
    crispasr_paligner_close(aln);
    crispasr_session_close(sess);
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    Args a;
    if (const char * e = std::getenv("QASR_CRISPASR")) a.crispasr = e;
    if (const char * e = std::getenv("QASR_MODELS_DIR")) a.models_dir = e;
    if (!parse_args(argc, argv, a)) { usage(argv[0]); return 2; }
    if (a.help) { usage(argv[0]); return 0; }
    if (a.version) { std::printf("%s\n", kVersion); return 0; }
    if (a.input.empty()) { std::fprintf(stderr, "error: no input media given\n\n"); usage(argv[0]); return 2; }
    if (a.threads < 1) a.threads = 1;

    if (a.mode == "offline") return run_offline(a);
    if (a.mode == "fixed" || a.mode == "vad") {
        // --aligned (or an explicit --aligner) => per-segment linked path with
        // both models resident (accurate timestamps). Else engine --stream.
        if (a.aligned || !a.aligner.empty()) return run_aligned(a);
        return run_streaming(a);
    }

    std::fprintf(stderr, "error: unknown --mode '%s' (use offline | fixed | vad)\n", a.mode.c_str());
    return 2;
}
