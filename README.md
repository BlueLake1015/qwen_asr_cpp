# qwen_asr_cpp

Video/audio → transcription using the **Qwen3-ASR** model, in **C/C++** with a
**ggml**-based inference engine. No Python at build or runtime; media decoding is
delegated to the external **ffmpeg** command.

```
offline:  ffmpeg (decode whole file -> WAV) -> CrispASR (Qwen3-ASR + ForcedAligner) -> srt/vtt/txt
fixed:     ffmpeg PCM ──pipe──> crispasr --stream (fixed rolling window)   -> live text/JSON
vad:       ffmpeg PCM ──pipe──> crispasr --stream --vad (cut at silence)   -> live text/JSON
```

- **Input**: anything ffmpeg can decode — `.mp4`, `.mkv`, `.mov`, `.wav`, `.mp3`, `.m4a`, …
- **Modes**: `offline` (batch, aligned subtitles) + streaming `fixed` / `vad` (see [Modes](#modes---mode))
- **Output**: plain text (default / stdout), **SRT**, or **VTT** (offline); live text / JSON (streaming)
- **Model**: [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) (latest open Qwen ASR series, Jan 2026; 52 languages)
- **Timestamps**: [Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) for precise subtitle timing
- **Engine**: [CrispASR](https://github.com/CrispStrobe/CrispASR) (pure C++/ggml, no Python), vendored as a submodule
- **Build**: GNU Make (drives the engine's CMake build)

This project is a small C++ orchestrator: it owns the **ffmpeg decode front-end**,
temp-file handling, and a focused `qwen-asr` CLI, then runs the CrispASR engine
(which provides Qwen3-ASR inference, the forced aligner, model auto-download, and
the SRT/VTT writers). We **exec** the engine binary rather than link it, so all
of that comes for free.

---

## 1. Prerequisites

| Tool | Why | Install (Ubuntu) |
|------|-----|------------------|
| `g++` (C++17) | compile engine + wrapper | `sudo apt install g++` |
| `make` | build entry point | `sudo apt install make` |
| `cmake` | builds the vendored engine | `sudo apt install cmake` |
| `ffmpeg` | **runtime** media decode | `sudo apt install ffmpeg` |
| `git`, `curl` | submodules + model download | `sudo apt install git curl` |
| NVIDIA GPU + CUDA toolkit + `g++-12` | **GPU inference (default)** | `sudo apt install nvidia-cuda-toolkit g++-12` |

No Python anywhere. The default build is **GPU (CUDA)** — see [§Notes](#notes--limitations);
use `make GPU=off` for a CPU-only build.

---

## 2. Build

```bash
git clone --recursive <this-repo> qwen_asr_cpp
cd qwen_asr_cpp

# if you didn't clone with --recursive:
make submodules

make            # builds the CrispASR engine + our app -> build/qwen-asr
```

The first build compiles the engine (the heavy step). `make help` lists targets.

---

## 3. Get models

The ASR model auto-downloads on first run (`-m auto` → `~/.cache/crispasr`).
For the **forced aligner** (precise SRT timing) and an explicit/offline setup,
fetch the pre-converted GGUFs (no Python — just `curl`):

```bash
scripts/get_model.sh                 # 1.7B ASR + 0.6B aligner (q4_k) into models/
SIZE=0.6b scripts/get_model.sh       # smaller/faster 0.6B ASR
QUANT=q8_0 scripts/get_model.sh      # higher quality / larger
ALL=1 scripts/get_model.sh           # every quant variant of both models
```

This pulls, e.g.:
- `models/qwen3-asr-1.7b-q4_k.gguf` (from `cstr/qwen3-asr-1.7b-GGUF`)
- `models/qwen3-forced-aligner-0.6b-q4_k.gguf` (from `cstr/qwen3-forced-aligner-0.6b-GGUF`)

> **Model format matters.** Use the **`cstr/*-GGUF`** weights (single-file,
> CrispASR-native). The `ggml-org/Qwen3-ASR-*-GGUF` files are in llama.cpp's
> split `mmproj` format and are **not loadable by this engine** — they're for
> `llama-server`, not CrispASR.

---

## 4. Run

After `scripts/get_model.sh`, the models in `models/` are picked up
automatically — no need to pass `-m`/`--aligner`:

```bash
# SRT with precise aligner timestamps. Defaults: size 1.7B, q8_0 ASR + q8_0 aligner.
build/qwen-asr talk.mp4 -o talk.srt -l en

# Plain text to stdout
build/qwen-asr talk.mp4

# Configure quantization per model (q8_0 is the default for both)
build/qwen-asr talk.mp4 -o talk.srt --asr-quant q4_k --aligner-quant q8_0

# Smaller/faster 0.6B, 8 threads
build/qwen-asr interview.mkv -o out.vtt --size 0.6b -t 8

# Explicit paths still work, and override everything
build/qwen-asr talk.mp4 -o talk.srt \
  -m models/qwen3-asr-1.7b-q8_0.gguf \
  --aligner models/qwen3-forced-aligner-0.6b-q8_0.gguf
```

The ASR model is resolved as `models/qwen3-asr-<size>-<asr-quant>.gguf` and the
aligner as `models/qwen3-forced-aligner-0.6b-<aligner-quant>.gguf`; if the ASR
file is absent it falls back to `-m auto` (engine download). The backend is
auto-detected from the GGUF, so a 1.7B file just works.

### Modes (`--mode`)

| Mode | Data path | Output | Timestamps |
|------|-----------|--------|------------|
| `offline` (default) | decode whole file → one engine pass | `.srt` / `.vtt` / `.txt` | forced aligner (accurate) |
| `fixed` | **live**: ffmpeg PCM → `crispasr --stream`, fixed rolling window | incremental text / JSON to stdout (or `-o` file) | rolling-window |
| `vad` | **live**: ffmpeg PCM → `crispasr --stream --vad`, segments at silence | incremental text / JSON to stdout (or `-o` file) | per-utterance (`t0`/`t1`) |
| `fixed`/`vad` **+ `--aligned`** | **per-segment** (linked C-ABI): segment → transcribe → force-align, both models **resident** | incremental `.srt`/`.vtt`/`.txt` | **forced aligner, accurate** |

```bash
# Offline (batch) — aligned subtitles
qwen-asr talk.mp4 -o talk.srt -l en

# Fixed-length streaming — 8 s rolling window, live partials on stdout
qwen-asr talk.mp4 --mode fixed --segment-seconds 8 -l en

# VAD-adaptive streaming — cut at silence; structured partial/final events
qwen-asr talk.mp4 --mode vad --json -l en

# Simulate a live source (pace input at 1x) and tune VAD
qwen-asr stream.mkv --mode vad --realtime --min-silence-ms 300 --vad-threshold 0.5

# Per-segment + forced aligner: accurate SRT, both models loaded ONCE (no reload)
qwen-asr talk.mp4 --mode vad --aligned -l en -o talk.srt \
  -m models/qwen3-asr-1.7b-q8_0.gguf \
  --aligner models/qwen3-forced-aligner-0.6b-q8_0.gguf
```

Streaming knobs: `--segment-seconds` / `--segment-step` (fixed); `--vad-threshold`,
`--min-silence-ms`, `--min-speech-ms` (vad); `--json` (JSON-Lines partial/final/
silence events); `--realtime` (feed at 1×, mimicking a live capture).

The plain `fixed`/`vad` modes use the engine's `--stream` path (live text/JSON, no
aligner). Add **`--aligned`** (or pass `--aligner`) to switch to a per-segment
pipeline that links the engine's C-ABI: it segments the audio (fixed windows or
VAD), then for each segment runs transcription **and** the forced aligner with
**both models loaded once and kept resident** — no per-segment model reload — and
writes an accurate `.srt`/`.vtt` incrementally. This needs local model files
(not `-m auto`).

#### Engine patch (required for `--aligned`)

The resident aligner relies on a small shim added to the vendored engine
(`crispasr_paligner_open/align/close`). It's kept as an explicit, idempotent
patch under [patches/](patches/) rather than auto-applied, so a `git submodule
update` (which resets the submodule and drops the shim) is followed by:

```bash
make patch        # re-applies patches/0001-persistent-qwen3-aligner.patch
make              # rebuild
```

The repo currently ships with the patch already applied to the submodule
working tree. `make patch` is a no-op if it's already there. The offline and
plain streaming modes do **not** need the patch.

### Options

```
<input-media>          file ffmpeg can decode                (required)
-o, --output <path>    .srt -> SubRip, .vtt -> WebVTT, else plain text;
                       omit -> transcript to stdout
-m, --model <path>     explicit Qwen3-ASR GGUF, or 'auto'. Default: resolved
                       from --models-dir by --size + --asr-quant, else 'auto'
    --aligner <path>   explicit Qwen3-ForcedAligner GGUF. Default: resolved
                       from --models-dir by --aligner-quant
    --size <s>         0.6b | 1.7b (default 1.7b) — for model auto-resolution
    --asr-quant <q>    ASR quantization:     q8_0 (default) | q4_k | q5_0 | f16
    --aligner-quant <q> aligner quantization: q8_0 (default) | q4_k | q5_0 | f16
    --models-dir <d>   model GGUF directory (default: <repo>/models; $QASR_MODELS_DIR)
    --backend <name>   force backend: qwen3 | qwen3-1.7b (default: auto-detect)
-l, --language <code>  language hint (en, zh, ko, ...); default: auto
-t, --threads <n>      inference threads (default 4)
    --no-split         don't split subtitle lines on punctuation
    --ffmpeg <path>    ffmpeg executable (default: ffmpeg on PATH)
    --crispasr <path>  engine binary (default: vendored build; or $QASR_CRISPASR)
-h, --help             help        --version
```

Example SRT output:

```
1
00:00:04,160 --> 00:00:06,160
Who can tell what the future will be like?

2
00:00:07,200 --> 00:00:20,000
NASA scientists have combined data from several climate models ...
```

---

## 5. How it works

| Stage | Code | Notes |
|-------|------|-------|
| Decode | [src/audio_decode.cpp](src/audio_decode.cpp) | offline: `fork`/`exec`s ffmpeg → 16 kHz mono WAV. Streaming: ffmpeg PCM piped to the engine's stdin |
| Inference | `third_party/CrispASR` | Qwen3-ASR (Whisper-style audio encoder + Qwen3 LLM) on ggml; forced aligner (offline); `--stream`/`--vad` (streaming) |
| Orchestration / CLI | [src/main.cpp](src/main.cpp) | arg parsing, model/quant resolution, and one of `run_offline` (WAV → engine → subtitle file), `run_streaming` (ffmpeg \| crispasr --stream), or `run_aligned` (per-segment, linked C-ABI, models resident) |
| Linked C-ABI | [src/crispasr_capi.h](src/crispasr_capi.h) | `--aligned` path links `libcrispasr.so`: `crispasr_session_*` (ASR, load once), `crispasr_vad_slices` (segment), `crispasr_paligner_*` (resident aligner shim) |

Offline hands the engine a decoded WAV + flags
(`-m <model> -am <aligner> -osrt -of <base>`). Streaming builds a
`ffmpeg … -f s16le - | crispasr --stream …` pipeline in C++ (a pipe between two
forked children), so audio is fed live and transcripts stream out as they
finalize. Either way ffmpeg handles all container/codec decoding.

---

## 6. Tests

[tests/run_modes.sh](tests/run_modes.sh) exercises every mode end-to-end against
one input — a **media file or a stream URL** — and prints a PASS/FAIL summary
(exit code = number of failures).

```bash
make                              # build first
tests/run_modes.sh                # default input: tests/fixtures/test_en_30s.mp4
tests/run_modes.sh path/to/clip.mkv
tests/run_modes.sh udp://239.0.0.1:5000        # URL -> live streaming modes only
LANG_HINT=zh tests/run_modes.sh tests/fixtures/test_zh_30s.mp4
```

Coverage:
- **file** input → offline (txt), offline (SRT), `fixed` stream, `vad` stream (JSON),
  `fixed --aligned` (SRT), `vad --aligned` (SRT).
- **URL** input → `fixed` and `vad` streaming only (offline/aligned need a finite file);
  each is sampled for `URL_WINDOW` seconds (default 40).

It auto-picks a fast local **0.6B** model (or `-m auto`) and the
forced-aligner from `models/`; the two `--aligned` cases are **skipped** (not
failed) if no local model + aligner are present — run `scripts/get_model.sh`
first to include them. Overrides: `QASR`, `MODEL`, `ALIGNER`, `LANG_HINT`,
`THREADS`, `TIMEOUT`, `URL_WINDOW`.

Imported test media lives under [tests/fixtures/](tests/fixtures/) (small 30 s
clips, committed) and `tests/data/` (large clips, git-ignored).

## 7. Layout

```
qwen_asr_cpp/
├── Makefile                 # build entry point
├── src/
│   ├── audio_decode.{h,cpp} # external-ffmpeg decode (WAV file + in-memory PCM)
│   ├── crispasr_capi.h      # C-ABI decls for the linked --aligned path
│   └── main.cpp             # CLI + orchestration (offline / streaming / aligned)
├── scripts/
│   └── get_model.sh         # download cstr GGUFs (no Python)
├── patches/
│   ├── 0001-persistent-qwen3-aligner.patch  # the --aligned engine shim
│   └── apply.sh             # `make patch` runs this (explicit, idempotent)
└── third_party/
    └── CrispASR/            # ggml ASR engine (submodule; ggml in-tree)
        └── src/crispasr_aligner.cpp  # patched with the persistent-aligner shim
```

---

## Notes & limitations

- **GPU by default (CUDA).** `make` builds the engine with `-DGGML_CUDA=ON`, and
  all inference (ASR, aligner, VAD) runs on the GPU — `use_gpu`/`flash_attn` are
  on by default in the engine. Needs an NVIDIA GPU + the CUDA toolkit; CUDA 12.0's
  `nvcc` rejects gcc≥13, so the build pins `g++-12` as the CUDA host compiler and
  targets `sm_89` (RTX 40xx). Override with `make CUDA_ARCH=<arch>` for other GPUs.
  Build CPU-only with `make GPU=off`. (Vulkan/Metal: pass the matching `-DGGML_*`
  flag — see CrispASR docs.)
- **Models:** `-m auto` grabs the 0.6B model; `scripts/get_model.sh` defaults to
  **1.7B** (best accuracy). Both are auto-detected — just point `-m` at the file.
- Throughput: on an **RTX 4090 (CUDA)** the 1.7B q8 model runs **~5× realtime**
  (30 s clip in ~6 s). CPU-only (`make GPU=off`) is ~0.1× realtime for 1.7B — GPU
  is strongly recommended and is the default.
