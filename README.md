# qwen_asr_cpp

Video / audio → transcription using the **Qwen3-ASR** model, in **C/C++** on a
**ggml** inference engine running on the **GPU** by default. No Python at build or
runtime; media decoding is delegated to the external **ffmpeg** command.

```
                ┌─────────────────────────── qwen-asr (our C++ orchestrator) ──────────────────────────┐
 input file ──▶ │ ffmpeg (decode any container/codec → 16 kHz mono PCM/WAV)                              │
 or stream URL  │      │                                                                                 │
                │      ├─ offline ─▶ exec crispasr  ─▶ Qwen3-ASR + Qwen3-ForcedAligner ─▶ .srt/.vtt/.txt │
                │      ├─ fixed   ─▶ pipe ─▶ crispasr --stream (rolling window)        ─▶ live text/JSON  │
                │      ├─ vad     ─▶ pipe ─▶ crispasr --stream --vad (cut at silence)  ─▶ live text/JSON  │
                │      └─ --aligned ─▶ link libcrispasr.so: per-segment transcribe+align (models resident)│
                └───────────────────────────────────────────────────────────────────────────────────────┘
                                              CrispASR engine (ggml, CUDA) — vendored submodule
```

| | |
|---|---|
| **Input** | anything ffmpeg decodes — `.mp4 .mkv .mov .wav .mp3 .m4a …` — or a live URL (`udp://`, `rtp://`, `http://`) |
| **Output** | plain text, **SRT**, **VTT** (offline / `--aligned`); live text or JSON-Lines (streaming) |
| **ASR model** | [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) 1.7B (default) or 0.6B — latest open Qwen ASR series, 52 languages |
| **Timestamps** | [Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) CTC forced alignment |
| **Engine** | [CrispASR](https://github.com/CrispStrobe/CrispASR) (pure C++/ggml, no Python), vendored submodule; ggml in-tree |
| **Compute** | **GPU/CUDA by default** (RTX 4090: ~20× realtime inference on 1.7B q8, greedy); `make GPU=off` for CPU |
| **Build** | GNU Make (drives the engine's CMake build) |

---

## ▶ TL;DR

```bash
git clone --recursive <repo> qwen_asr_cpp && cd qwen_asr_cpp
sudo apt install -y g++ g++-12 make cmake ffmpeg git curl nvidia-cuda-toolkit
make                       # builds CUDA engine + our app  ->  build/qwen-asr
scripts/get_model.sh       # fetches 1.7B ASR + 0.6B aligner + VAD into models/
build/qwen-asr tests/fixtures/test_en_30s.mp4 -o out.srt -l en   # transcribe → SRT
tests/run_modes.sh         # run the full mode test suite
```

---

## 1. Prerequisites

| Tool | Why | Install (Ubuntu) |
|------|-----|------------------|
| `g++` (C++17) | compile engine + wrapper | `sudo apt install g++` |
| `make`, `cmake` | build entry point + engine build | `sudo apt install make cmake` |
| `ffmpeg` | **runtime** media decode | `sudo apt install ffmpeg` |
| `git`, `curl` | submodules + model download | `sudo apt install git curl` |
| NVIDIA GPU + CUDA toolkit + `g++-12` | **GPU inference (default build)** | `sudo apt install nvidia-cuda-toolkit g++-12` |

No Python anywhere. The default build targets the **GPU (CUDA)**; `make GPU=off`
produces a CPU-only build with no CUDA dependency.

---

## 2. Build

```bash
git clone --recursive <repo> qwen_asr_cpp
cd qwen_asr_cpp
make submodules         # only if you didn't clone with --recursive
make                    # CrispASR engine (CUDA) + our app -> build/qwen-asr
```

The first `make` compiles the vendored engine including ggml's CUDA kernels — the
slow step (minutes). Subsequent `make` only relinks our small wrapper.

| Target | Effect |
|--------|--------|
| `make` | engine + app (GPU build) |
| `make GPU=off` | CPU-only build (no CUDA) |
| `make CUDA_ARCH=86` | target a different GPU arch (default `89` = RTX 40xx) |
| `make engine` | build only the engine binary + `libcrispasr.so` |
| `make submodules` | fetch the CrispASR submodule (+ nested ggml) |
| `make patch` | (re)apply the `--aligned` engine shim (see [§8](#8-the---aligned-fast-path)) |
| `make clean` / `clean-all` | remove our `build/` / also the engine build |

Build knobs: `GPU` (`cuda`/`off`), `CUDA_ARCH`, `CUDA_HOST_CXX` (default `g++-12`),
`JOBS`, `CXX`, `CXXFLAGS`.

---

## 3. Models

The app loads models **only from `models/`** (no network, no cache). Fetch the
pre-converted GGUFs once — pure `curl`, no Python:

```bash
scripts/get_model.sh                 # 1.7B ASR + 0.6B aligner + silero VAD, all q8_0
SIZE=0.6b scripts/get_model.sh       # 0.6B ASR instead of 1.7B
ASR_QUANT=q4_k scripts/get_model.sh  # per-model quant override (q4_k|q5_0|q8_0|f16)
ALL=1 scripts/get_model.sh           # every quant variant of both models
```

Populates `models/`:

| File | From | Role |
|------|------|------|
| `qwen3-asr-1.7b-q8_0.gguf` | `cstr/qwen3-asr-1.7b-GGUF` | ASR (Whisper-style encoder + Qwen3 1.7B LLM) |
| `qwen3-forced-aligner-0.6b-q8_0.gguf` | `cstr/qwen3-forced-aligner-0.6b-GGUF` | CTC forced aligner (timestamps) |
| `ggml-silero-v6.2.0.bin` | `ggml-org/whisper-vad` | Silero VAD (vad-mode segmentation) |

> [!IMPORTANT]
> **Use the `cstr/*-GGUF` weights** (single-file, CrispASR-native). The
> `ggml-org/Qwen3-ASR-*-GGUF` files use llama.cpp's split `mmproj` layout and are
> **not loadable by this engine** (verified: `required tensor 'audio.conv.1.weight'
> not found`) — those are for `llama-server`. The forced aligner is published only
> at 0.6B, so it stays 0.6B regardless of the ASR size.

Model resolution (when `-m`/`--aligner` aren't given explicitly):
`models/qwen3-asr-<size>-<asr-quant>.gguf` and
`models/qwen3-forced-aligner-0.6b-<aligner-quant>.gguf`. Missing ASR model → hard
error pointing at `get_model.sh` (it does **not** silently auto-download). Pass
`-m auto` explicitly to opt into the engine's own downloader.

---

## ▶ 4. Running the binary

> [!TIP]
> With models in `models/`, the common case is just:
> ```bash
> build/qwen-asr INPUT -o OUT.srt -l en
> ```
> Defaults: 1.7B ASR (q8_0) + 0.6B aligner (q8_0), GPU, offline mode → aligned SRT.

```bash
# Offline → accurate SRT (forced aligner auto-applied from models/)
build/qwen-asr talk.mp4 -o talk.srt -l en

# Plain transcript to stdout
build/qwen-asr talk.mp4

# WebVTT, 0.6B model, 8 threads
build/qwen-asr interview.mkv -o out.vtt --size 0.6b -t 8

# Per-model quantization (both default q8_0)
build/qwen-asr talk.mp4 -o talk.srt --asr-quant q4_k --aligner-quant q8_0

# Explicit model/aligner paths override resolution
build/qwen-asr talk.mp4 -o talk.srt \
  -m models/qwen3-asr-1.7b-q8_0.gguf \
  --aligner models/qwen3-forced-aligner-0.6b-q8_0.gguf
```

### Modes (`--mode`)

| Mode | Data path | Live? | Output | Timestamps |
|------|-----------|:----:|--------|------------|
| `offline` *(default)* | decode whole file → one engine pass | no | `.srt`/`.vtt`/`.txt` | forced aligner (accurate) |
| `fixed` | ffmpeg PCM → `crispasr --stream` (rolling window) | yes | text / JSON to stdout (or `-o`) | rolling-window |
| `vad` | ffmpeg PCM → `crispasr --stream --vad` (cut at silence) | yes | text / JSON to stdout (or `-o`) | per-utterance `t0`/`t1` |
| `fixed`/`vad` **+ `--aligned`** | per-segment, linked C-ABI; models **resident** | yes | incremental `.srt`/`.vtt` | **forced aligner (accurate)** |

```bash
# Fixed-length streaming — 8 s rolling window, live partials on stdout
build/qwen-asr talk.mp4 --mode fixed --segment-seconds 8 -l en

# VAD-adaptive streaming — structured partial/final/silence JSON events
build/qwen-asr talk.mp4 --mode vad --json -l en

# Simulate a live source (pace at 1×) and tune VAD
build/qwen-asr in.mkv --mode vad --realtime --min-silence-ms 300 --vad-threshold 0.5

# Per-segment + forced aligner: accurate SRT, both models loaded ONCE (no reload)
build/qwen-asr talk.mp4 --mode vad --aligned -l en -o talk.srt

# Live transcription of an MPEG-TS multicast stream
build/qwen-asr udp://239.0.0.1:5000 --mode vad --json -l en
```

### All options

```
<input-media>           file ffmpeg can decode, or a stream URL          (required)
-o, --output <path>     .srt → SubRip, .vtt → WebVTT, else plain text;
                        omit → transcript to stdout
-m, --model <path>      explicit ASR GGUF, or 'auto'. Default: models/ by --size+--asr-quant
    --aligner <path>    explicit ForcedAligner GGUF. Default: models/ by --aligner-quant
    --size <s>          0.6b | 1.7b           (default 1.7b)  — model auto-resolution
    --asr-quant <q>     q8_0 (default) | q4_k | q5_0 | f16
    --aligner-quant <q> q8_0 (default) | q4_k | q5_0 | f16
    --models-dir <d>    model dir (default: <repo>/models; or $QASR_MODELS_DIR)
    --backend <name>    force backend: qwen3 | qwen3-1.7b   (default: auto-detect from GGUF)
-l, --language <code>   language hint (en, zh, ko, …)        (default: auto)
-t, --threads <n>       inference/host threads               (default 4)
-bs, --beam-size <n>    decode beam width; 1 = greedy (default, fast). >1 enables
                        beam search: ~Nx slower decode for little ASR accuracy gain
--mode <m>              offline (default) | fixed | vad
--segment-seconds <s>   fixed-mode window length             (default 10)
--segment-step <s>      fixed-mode window step               (default: engine ~3s)
--vad-threshold <f>     vad speech probability threshold     (default ~0.5)
--min-silence-ms <n>    vad silence that splits a segment
--min-speech-ms <n>     vad minimum speech kept
--vad-max-segment <s>   vad+aligned: cap each speech chunk (s); 0 (default) → use
                        --segment-seconds. Lower = less VRAM, slight accuracy cost
--vad-model <path>      VAD model (default: models/ggml-silero-v6.2.0.bin)
--aligned               fixed/vad: per-segment forced-aligner pass (resident models, accurate SRT)
--json                  streaming: emit --stream-json partial/final/silence events
--realtime              pace input at 1× (simulate a live source)
--ffmpeg <path>         ffmpeg executable                    (default: ffmpeg on PATH)
--crispasr <path>       engine binary                        (default: vendored; or $QASR_CRISPASR)
-h, --help     --version
```

Example SRT (offline / `--aligned`):

```srt
1
00:00:04,160 --> 00:00:06,160
Who can tell what the future will be like?

2
00:00:07,200 --> 00:00:20,000
NASA scientists have combined data from several climate models ...
```

Example streaming JSON (`--mode vad --json`):

```json
{"type":"partial","utterance_id":1,"text":"who can tell what the","t0":4.16,"t1":5.20}
{"type":"final","utterance_id":1,"text":"Who can tell what the future will be like?","t0":4.16,"t1":6.16}
{"type":"silence","t":6.84}
```

---

## 🧪 5. Testing

> [!TIP]
> One script runs **every mode** end-to-end against one input (file **or** stream
> URL) and prints a PASS/FAIL summary; its exit code is the failure count.
> ```bash
> tests/run_modes.sh                       # default: tests/fixtures/test_en_30s.mp4
> tests/run_modes.sh path/to/clip.mkv      # any media file
> tests/run_modes.sh udp://239.0.0.1:5000  # URL → live streaming modes only
> LANG_HINT=zh tests/run_modes.sh tests/fixtures/test_zh_30s.mp4
> ```

What it covers:

| Input | Cases run |
|-------|-----------|
| **file** | offline (txt), offline (SRT), `fixed` stream, `vad` stream (JSON), `fixed --aligned` (SRT), `vad --aligned` (SRT) |
| **URL** | `fixed` + `vad` streaming only (offline/aligned need a finite file); each sampled for `URL_WINDOW`s |

- Uses the app's default models (`models/` 1.7B ASR + 0.6B aligner). The two
  `--aligned` cases are **skipped** (not failed) if those files are absent — run
  `scripts/get_model.sh` first.
- Assertions: exit code, non-empty output, `-->` present for SRT, `"type"` for JSON.
- Env overrides: `QASR`, `MODEL`, `ALIGNER`, `LANG_HINT`, `THREADS`, `TIMEOUT`
  (default 1800 s), `URL_WINDOW` (default 40 s).

Sample run (GPU):

```
  PASS: offline -> text
  PASS: offline -> SRT (timestamps present)
  PASS: fixed-stream (exit=0, produced output)
  PASS: vad -> stream (JSON partial/final events)
  PASS: fixed --aligned -> SRT (resident models, timestamps present)
  PASS: vad --aligned -> SRT (resident models, timestamps present)
 result: 6 passed, 0 failed, 0 skipped
```

Test media lives in [tests/fixtures/](tests/fixtures/) (small 30 s clips,
committed) and `tests/data/` (large clips, git-ignored).

---

## 📦 6. Offline / air-gapped deployment

Everything (build, install, runtime) works with **no internet**. Gather Ubuntu
`.deb`s on an online host, install on the target — see [offline/README.md](offline/README.md).

```bash
# On an ONLINE machine (same Ubuntu release + CPU arch as the target):
offline/download_debs.sh        # → offline/debs/ (build tools + ffmpeg + CUDA toolkit + deps, ~2.2 GB)
scripts/get_model.sh            # → models/ (GGUFs + VAD)

# Copy the whole repo (incl. third_party/ submodules + offline/debs/ + models/) to the target, then:
offline/install.sh              # dpkg-installs the bundle, verifies the toolchain (no network)
make                            # build the CUDA engine + app  (or: make GPU=off)
```

The NVIDIA **driver** is assumed already present on the GPU target (not bundled);
the CUDA **toolkit** is. No Python at any stage.

## 7. Architecture

Two cooperating layers: a small C++ **orchestrator** we own, and the vendored
**CrispASR** ggml engine.

| Stage | Code | Detail |
|-------|------|--------|
| Decode | [src/audio_decode.cpp](src/audio_decode.cpp) | `fork`/`exec`s ffmpeg (no shell → no quoting/injection). Offline/aligned: decode to a 16 kHz mono s16 WAV / in-memory float PCM. Streaming: ffmpeg writes raw `s16le` to a pipe consumed by the engine. |
| Orchestration / CLI | [src/main.cpp](src/main.cpp) | arg parsing, model/quant resolution, temp handling, and one of three drivers below. |
| Inference | `third_party/CrispASR` | Qwen3-ASR (Whisper-style audio encoder → Qwen3 LLM decoder) on ggml + CUDA; CTC forced aligner; Silero VAD; SRT/VTT/JSON writers; streaming. |
| Linked C-ABI | [src/crispasr_capi.h](src/crispasr_capi.h) | Declarations for the `--aligned` path that links `libcrispasr.so`. |

**Three execution drivers** in `main.cpp`:

- `run_offline` — decode whole file → WAV → **exec** `crispasr -f wav -m <asr> -am <aligner> -osrt -of <base>`. The engine does VAD/chunking internally and the forced aligner produces accurate cue timing.
- `run_streaming` — build a `ffmpeg … -f s16le - | crispasr --stream …` pipeline in C++ (a `pipe(2)` between two forked children) so audio is fed live and transcripts stream out incrementally. `fixed` → `--stream-length/--stream-step`; `vad` → `--stream --vad --vad-model models/…`.
- `run_aligned` — **links** `libcrispasr.so` (not exec). Loads the ASR session **and** the aligner **once**, decodes to PCM, segments (fixed windows or `crispasr_vad_slices`), then per segment: `crispasr_session_transcribe()` → tokenize → `crispasr_paligner_align()` → emit an SRT/VTT cue incrementally. No per-segment model reload.

Why exec for offline/streaming but link for `--aligned`: exec'ing the CLI gives the
engine's downloader, writers, VAD, and streaming for free; the resident
per-segment loop needs the model to stay in memory across calls, which requires
the in-process C-ABI.

---

## 8. The `--aligned` fast path

The engine's stock aligner reloads the GGUF on every call. For per-segment
streaming that's wasteful, so we keep both models resident:

- **ASR**: `crispasr_session_open()` once → `crispasr_session_transcribe(pcm)` per segment.
- **Aligner**: a small shim added to the engine — `crispasr_paligner_open/align/close` — keeps the aligner's `qwen3_asr_context` alive across calls. It's a ~60-line addition to [third_party/CrispASR/src/crispasr_aligner.cpp](third_party/CrispASR/src/crispasr_aligner.cpp).
- **VAD**: `crispasr_vad_slices(models/ggml-silero…, pcm)` for adaptive segmentation.

Because the engine is a submodule (a `git submodule update` resets it and drops
the shim), the change is carried as an explicit, idempotent patch:

```bash
make patch     # applies patches/0001-persistent-qwen3-aligner.patch (no-op if already applied)
make           # rebuild
```

The repo ships with the patch already applied to the submodule working tree.
Offline and plain streaming modes don't need it.

Verified behavior — both models load once, then segments stream:

```
[load] ASR=models/qwen3-asr-1.7b-q8_0.gguf  aligner=models/qwen3-forced-aligner-0.6b-q8_0.gguf (loaded once)
[vad] 3 speech segments
[00:00:04,160 --> 00:00:12,320] Who can tell what the future will be like? NASA scientists...
[00:00:12,190 --> 00:00:19,950] An intergovernmental panel on climate change...
```

---

## 9. Performance & GPU

- **GPU by default (CUDA).** `make` configures the engine with `-DGGML_CUDA=ON`;
  ASR, aligner, and VAD all run on the GPU (`use_gpu`/`flash_attn` default on in
  the engine; `crispasr_session_open` and the aligner shim request GPU).
- CUDA 12.0's `nvcc` rejects gcc ≥ 13, so the build pins `g++-12` as the CUDA host
  compiler and targets `sm_89` (RTX 40xx); override with `make CUDA_ARCH=<n>`.
- Vulkan / Metal: pass the matching `-DGGML_VULKAN=ON` / `-DGGML_METAL=ON` (see
  CrispASR docs) — the `crispasr_capi`/exec paths are backend-agnostic.

- **Decoding defaults to greedy (`--beam-size 1`).** The engine's own default is
  5-way beam search, which does ~5× the decode work for little ASR accuracy gain;
  the wrapper overrides it to greedy for speed (opt back in with `--beam-size N`).
- **VRAM levers.** Biggest: `--asr-quant q8_0` (halves weight memory, also faster)
  and dropping `--aligner` when you don't need word timestamps (−~1.8 GB). To trim
  the resident compute buffer in `--mode vad`, lower `--vad-max-segment` (caps the
  longest speech chunk). See [CPP_VS_PYTHON.md](CPP_VS_PYTHON.md) for a full VRAM
  breakdown and the C++-vs-Python comparison.

| Config | 30 s clip | Realtime factor |
|--------|-----------|-----------------|
| 1.7B q8, RTX 4090, **greedy** | ~1.5 s | **~20×** |
| 1.7B f16, RTX 4090, greedy | ~1.9 s | ~16× |
| 1.7B q8, RTX 4090, beam=5 *(old default)* | ~3.9 s | ~8× |
| 1.7B q4_k, CPU (`GPU=off`) | ~280 s | ~0.1× |
| 0.6B q4_k, CPU | ~140 s | ~0.2× |

> GPU rows are **engine inference** time (model load ~2–3 s excluded); q8 is both
> fastest and smallest (it's bandwidth-bound, so 1-byte q8 beats 2-byte f16). CPU
> rows are approximate end-to-end. See [CPP_VS_PYTHON.md](CPP_VS_PYTHON.md) for
> beam-vs-greedy, C++-vs-Python, and VRAM benchmarks (30 s clip + 96 min film).

> A harmless `ggml_cuda_init: CUDA minor version mismatch — compiled 12.0, runtime
> 12.9` warning appears because apt's toolkit is 12.0 while the driver is 12.9;
> install a newer CUDA toolkit to silence it.

---

## 10. Layout

```
qwen_asr_cpp/
├── Makefile                  # build entry point (GPU default; engine + app)
├── src/
│   ├── audio_decode.{h,cpp}  # external-ffmpeg decode (WAV file + in-memory PCM)
│   ├── crispasr_capi.h       # C-ABI decls for the linked --aligned path
│   └── main.cpp              # CLI + 3 drivers (offline / streaming / aligned)
├── scripts/
│   └── get_model.sh          # download cstr GGUFs + VAD into models/ (no Python)
├── tests/
│   ├── run_modes.sh          # per-mode integration tests (file or stream URL)
│   ├── fixtures/             # small 30 s clips (committed)
│   └── data/                 # large clips (git-ignored)
├── patches/
│   ├── 0001-persistent-qwen3-aligner.patch   # the --aligned engine shim
│   └── apply.sh              # `make patch` runs this (explicit, idempotent)
├── models/                   # GGUFs + VAD (git-ignored; via get_model.sh)
└── third_party/
    └── CrispASR/             # ggml ASR engine (submodule; ggml in-tree; patched)
```

---

## 11. Notes & limitations

- **models/-only by default.** No silent cache/auto-download; `-m auto` is an
  explicit opt-in. The forced aligner is 0.6B-only (upstream).
- **Streaming vs aligner.** Plain `fixed`/`vad` (engine `--stream`) emit live
  text/JSON with VAD/window timing but **cannot** use the forced aligner or write
  SRT — that's an engine streaming limitation. Use `--aligned` (resident
  per-segment) or `offline` for aligner-accurate subtitles.
- Long media is transcribed in one pass per segment; the engine chunks internally
  to bound memory. Very long offline files rely on the engine's own chunking.
