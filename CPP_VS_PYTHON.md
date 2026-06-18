# C++ vs Python — Qwen3-ASR engine notes & benchmarks

Notes on non-obvious C++ engine (ggml/CrispASR) behaviors and head-to-head
performance/accuracy comparisons against the reference Python implementation
(`video_ai_subtitle`, which runs Qwen3-ASR via the `qwen-asr` package on
PyTorch/transformers). Covers beam size, and speed/VRAM/accuracy benchmarks on a
30 s clip and a 96 min film (with forced aligner and VAD).

---

## Beam size (`-bs` / `--beam-size`) and why it matters vs. Python

### What it is
Beam size controls how many candidate transcriptions the decoder explores in
parallel, token by token:

- **`beam-size 1` = greedy decoding.** At each step take the single
  highest-probability token. One hypothesis, minimum work.
- **`beam-size N` = beam search.** Keep the `N` best partial sequences alive at
  every step, expand and prune to the top `N` by cumulative probability, then
  emit the best complete sequence. Hedges against locally-good-but-globally-bad
  tokens — but costs roughly `N`x the decode compute.

### The mismatch we found
The two implementations defaulted to **different decoding strategies**, which
made early "C++ is slower" comparisons unfair:

| Implementation | Default decoding |
|---|---|
| **C++ engine (CrispASR)** | `--beam-size 5` (5-way beam search) |
| **Python `qwen-asr`** | greedy — `model.generate(...)` with no `num_beams` (i.e. beam=1) |

So the C++ engine was doing ~5x the decode work per step *and* comparing itself
against a greedy Python baseline.

### Measured impact (RTX 4090, `tests/fixtures/test_en_30s.mp4`, 30 s audio, f16, inference only)

| Config | Inference time | Realtime factor |
|---|---|---|
| C++ f16, **beam=5** (old engine default) | ~6.7–7.1 s | ~4.3x |
| **C++ f16, greedy (beam=1)** | **~1.9 s** | **~16x** |
| Python bf16, greedy (reference) | ~4.1 s (30 s-equiv) | ~7.4x |

Switching the C++ path to greedy is a ~3.7x speedup, and it makes C++ f16
**~2x faster than the Python bf16 baseline** — at matched precision (both
16-bit) and matched decoding (both greedy).

Two things drive the beam-search cost:
1. **~Nx decode compute** — it advances `N` hypotheses per step instead of 1.
2. **It defeats CUDA-graph reuse** — beam search feeds a varying number of
   tokens per step, so the ggml compute graph changes shape and the CUDA backend
   re-captures the graph every step (~156 re-captures per run) instead of
   replaying it. (Greedy produces a stable single-token graph.)

> Note: CUDA Graphs themselves were measured to be **neutral** here
> (greedy ran 2.04 s with graphs on vs 2.05 s with them disabled). The win is
> the decode strategy, not the graph flag.

### Does greedy hurt accuracy?
No — not for clean ASR with this model. On the 30 s clip the greedy and beam=5
transcripts were **identical**, and greedy matched the Python bf16 output
word-for-word (62/63 tokens byte-identical; the lone difference was a number
*formatting* choice, `twenty-first` vs `21st`, not a recognition error). A strong
model on clean audio rarely needs beam search to find the right sequence.

### What changed in this repo
`src/main.cpp` now defaults the wrapper to **greedy (`beam_size = 1`)** and
passes `-bs <n>` through to the engine on both the offline and streaming paths.
Beam search remains available as an explicit opt-in:

```sh
# fast, default — greedy, matches Python's decoding
./build/qwen-asr in.mp4 -o out.srt --asr-quant f16 -l en

# opt back into beam search (slower; only for noisy/ambiguous audio)
./build/qwen-asr in.mp4 -o out.srt --asr-quant f16 -l en --beam-size 5
```

### Recommendation
- Use **greedy (beam=1)** by default. It is the fastest path, matches the Python
  reference's decoding, and produces equivalent transcripts.
- Reach for **beam search (`--beam-size 4–5`)** only when audio is noisy or
  ambiguous enough that the extra hypotheses measurably improve accuracy —
  accept the ~Nx decode slowdown when you do.

---

## C++ (ggml/CrispASR) vs Python (`qwen-asr`, transformers) — benchmarks

All on one **RTX 4090**, same model (**Qwen3-ASR-1.7B**), greedy decoding both
sides. C++ runs **f16** weights (+ **f16** forced aligner); Python runs
**bf16** weights (+ **bf16** forced aligner). "Inference" excludes model load;
"wall" is full end-to-end. VRAM peak sampled via `nvidia-smi` during the run.

### Short clip — `test_en_30s.mp4` (30 s, no aligner, inference only)

| | C++ f16 (greedy) | Python bf16 (greedy) |
|---|---|---|
| Inference | **~1.9 s** | ~3.5 s (26 s VAD-trimmed) |
| Per-second-of-audio | **0.063 s/s** | 0.136 s/s |
| Realtime factor | ~16× | ~7.4× |

C++ is ~2× faster here. Transcript identical except one number-formatting token
(`twenty-first` vs `21st`). The win is the compiled ggml decode loop vs Python's
`transformers.generate()` host overhead — both are host-bound (GPU ~60–75 %),
not GPU-bound, for this small model + short decode.

### Feature film — `nightofthelivingdead_1968_en.mp4` (95.9 min), with forced aligner

Two comparisons. The **VAD-on-both** row is the true apples-to-apples (matches
the Python pipeline's architecture: decode → Silero VAD over whole file →
transcribe + align each segment).

| Run | Inference | Wall | VRAM peak | Audio processed | Notes |
|---|---|---|---|---|---|
| **C++ offline** (no VAD, full audio) | 350 s | 353 s | **15.8 GB** | 5752.8 s (all) | transcribes silence too |
| **C++ vad** (Silero) | **197 s** | **248 s** | 14.9 GB | speech only | transcribe 154 s + align 43 s |
| **Python vad** (Silero) | 295 s | 370 s | **10.2 GB** | 3096.6 s (speech) | reference pipeline |

C++ vad inference is **measured**, not estimated — `run_aligned` is instrumented
(run with `QASR_TIMING=1`). Of the 248 s wall: decode = 9 s, model load = 3 s,
VAD = 39 s, and **transcribe + align = 197 s** (154 s ASR + 43 s alignment).
The phases sum to ~248 s, so the wall is fully accounted for.

**Speed:** on the inference itself (transcribe + align, both VAD-gated),
**C++ 197 s vs Python 295 s — ~1.5× faster**. End-to-end it's the same ratio
(C++ 248 s vs Python 370 s), helped further by much faster model load
(3 s vs 20.6 s). Python's VAD is its key efficiency lever; the C++ `offline`
mode lacks VAD and wastes time transcribing silence (use `--mode vad`, see below).

**VRAM:** C++ uses **~1.5× more** (14.9–15.8 GB vs 10.2 GB) — f16 weights + f16
aligner (1.8 GB) + ggml's larger KV/compute buffers vs Python's tighter bf16
footprint. Consistent trade: C++ buys speed with memory.

**Accuracy / result equivalence:** with VAD on both sides, word counts match
(C++ 6919 vs Python 6898 tokens) and transcripts are **~93.5 % identical**
(93.4 % of words matched). The residual ~6.5 % is genuine recognition divergence
on this noisy 1968 film (faint/overlapping dialogue) plus f16-vs-bf16 token flips
and different VAD segment boundaries — **not a systematic quality gap**. (In the
offline run C++ had ~600 extra words; that was purely coverage — it transcribed
silence Python's VAD skipped, not better recognition.)

### Using VAD in C++

C++ supports VAD via `--mode vad`. For an aligned SRT (the equivalent of the
Python pipeline), combine it with the forced aligner — this runs whole-file
Silero VAD, then transcribes + aligns each speech segment:

```sh
./build/qwen-asr in.mp4 -o out.srt --asr-quant f16 \
    --aligner models/qwen3-forced-aligner-0.6b-f16.gguf \
    --mode vad --vad-model models/ggml-silero-v6.2.0.bin -l en
```

Note: the engine's **streaming** path (`--mode vad` *without* `--aligned`) has
**no forced-aligner support** — it emits incremental text/JSON only. So
"streaming VAD + transcribe + **align**" is not available; for aligned output use
the whole-file VAD path above (which also matches Python's batch architecture).

### Segment length — C++ vs Python

Both run Silero VAD, but they turn speech regions into segments differently:

| | C++ (`run_aligned`) | Python (`WhisperSegmenter`) |
|---|---|---|
| Strategy | **cap only** — split speech at the cap, no merging | **aggregate toward a target**, then cap |
| Knob(s) | `--vad-max-segment` (default 0 → `--segment-seconds`, 10 s) | `SegmentConfig`: `target_seconds`=28, `min_seconds`=8, `max_seconds`=34 ([config.py](../video_ai_subtitle/src/vas/config.py)) |
| Hard ceiling | the cap (~10 s) | `max_seconds` (34 s) |
| Typical length | skews **short** (≤ cap) | skews **long** (~target 28 s) |
| Exposed as | CLI flag | config/preset default (no `vas` CLI flag) |

The closest equivalent of the C++ cap is Python's **`max_seconds`** (both are the
per-segment ceiling). But Python additionally *merges* VAD regions up toward
`target_seconds`, which C++ does not — C++ only caps. This is exactly why the
same 96 min film yielded **C++ 734→528 short segments (~10 s cap)** vs
**Python 519 regions → 101 chunks (~28 s)**.

Consequences:
- To make C++ segments resemble Python's ~28 s, **raise** `--vad-max-segment`
  toward ~30 (not lower it) — C++ has no "aggregate-to-target" behavior.
- **Segment length is not Python's VRAM driver.** Python runs *longer* segments
  (~28 s) than C++ (~10 s) yet uses *less* VRAM (10.2 vs 14.9 GB), because its live
  activations are tiny (~0.26 GB) and its footprint is dominated by the allocator,
  not segment size. So `--vad-max-segment` trims C++'s worst-case compute buffer (a
  real but modest lever); on Python, shrinking `max_seconds`/`target_seconds` mostly
  reduces allocator high-water/fragmentation rather than live tensors.

### Takeaways
- **Speed:** C++ f16 ~1.5× faster end-to-end (and ~2× on short clips) at matched
  precision and decoding.
- **Memory:** C++ costs ~1.5× the VRAM.
- **Quality:** equivalent (~93.5 % word agreement; differences are hard-audio +
  precision, not systematic).
- **Always use `--mode vad`** for files with silence/music — without it the C++
  offline mode transcribes everything and loses its speed advantage on wall clock.

---

## VRAM peak usage — breakdown, causes, and how to reduce it

On the 96 min film with the forced aligner, peak VRAM was **C++ 14.9 GB**
(15.8 GB in offline mode) vs **Python 10.2 GB** — far above the ~5 GB you might
expect from "a 1.7B model". This section explains where it goes and how to cut it.

### Why it's higher than the "~5 GB" intuition
That ~5 GB figure was a *single 30 s clip, ASR only, no aligner*. The film runs
add two things: a **second resident model** (the forced aligner) and
**framework working/reserved memory** sized to the longest segment.

### Usage breakdown (measured + by subtraction)

| Component | What it is | Python (10.2 GB) | C++ (14.9 GB) |
|---|---|---|---|
| ASR weights | the 1.7B model | 4.08 GB (bf16) | ~4.4 GB (f16) |
| Aligner weights | 2nd model (0.6B fwd-aligner) | 1.84 GB (bf16) | ~1.8 GB (f16) |
| KV cache | past-token keys/values | small/dynamic | 0.45 GB (`max_ctx=4096`) |
| Live activations | encoder/MLP/attention scratch + logits | **~0.26 GB** (measured, 32 s seg) | (inside compute buffer) |
| CUDA context + cuBLAS/kernels | runtime + GEMM workspace | ~0.5 GB | ~0.5–1 GB |
| **Framework reservation** | **reserved-but-not-live scratch** | **~3.5 GB** | **~7–8 GB** |

Key insight: **beyond weights (~5.9 GB) and the small KV cache, very little is
actually *live*** — a single 32 s segment's activations were only **0.26 GB**.
The bulk of the gap is **reserved scratch memory** the frameworks hold.

### Why the reservation exists (and is needed)
Real GPU allocation (`cudaMalloc`/`cudaFree`) is **slow (~ms) and synchronizes
the device** — it stalls the stream until prior work finishes. A forward pass
allocates/frees thousands of tensors (per layer, per token), so doing real
driver allocs each time would cripple throughput. Instead both frameworks use a
**caching allocator**: grab big slabs once, then sub-allocate/recycle in user
space (fast, no sync). "Freed" tensors return to the pool, not the driver — so
`nvidia-smi` shows *reserved*, not *live*, memory. The pool must cover the
**peak simultaneous** working set of the heaviest segment.

The two frameworks differ in strategy, which is why **C++ > Python** here:
- **ggml (C++):** plans the whole graph ahead and reserves **one worst-case
  compute buffer** (sized to the longest segment), held for the whole run.
  → high but **flat and deterministic**; scales with max segment length
  (offline 28 s slices → 15.8 GB; VAD shorter → 14.9 GB).
- **PyTorch (Python):** allocates dynamically; the cached pool grows to the
  **high-water mark** across all segments, plus fragmentation from varied sizes.
  → lower here, but creeps up over a varied workload.

### Ways to reduce VRAM

**1. Shrink weights (biggest lever, ~half the footprint):**
- **C++ — quantize** (ggml has a real quantized path; Python's Qwen3-ASR does not):
  - `--asr-quant q8_0` → ASR 4.4 → **2.4 GB**, near-lossless, *and faster than f16*.
  - `--asr-quant q4_k` → ~1.3 GB (small accuracy cost).
  - `--aligner-quant q8_0`/`q4_k` → aligner 1.8 → ~1.0 / ~0.6 GB.
- **Drop the aligner** if word-level timestamps aren't required: **−1.8 GB** +
  its activations (Python `qwen3-asr-noalign` preset; C++ omit `--aligner`).
- **Smaller model:** Qwen3-ASR **0.6B** (`--size 0.6b` / `SIZE=0.6b get_model.sh`)
  → ~⅓ the weight VRAM, faster, some accuracy loss.

**2. Shrink the framework reservation:**
- **Python:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (lowest-risk,
  reclaims fragmentation, often 1–3 GB); `torch.cuda.empty_cache()` between
  segments (trims peak, adds re-malloc latency); **uniform/shorter segments**
  (less fragmentation).
- **C++:** **cap segment length** with **`--vad-max-segment <s>`** in `--mode vad`
  (default 0 → falls back to `--segment-seconds`, i.e. unchanged behavior). The
  resident compute buffer is sized to the longest segment, so a shorter cap shrinks
  it (offline ~28 s → 15.8 GB vs VAD ~10 s → 14.9 GB). Returns diminish below ~10 s
  and word accuracy near segment edges drops, so it's a *modest* lever — quantize
  or drop the aligner for the big wins. Note: **`offline` mode can't be capped**
  (the engine chunks internally with no CLI knob) — use `--mode vad` to control it.
  Also engine-side: lower KV `max_ctx` or quantize the KV cache via
  `CRISPASR_KV_QUANT` (f16 → q8/q4).

**3. Already applied:** flash attention is on both sides (keeps attention
activations from scaling with sequence length²).

### C++ vs Python — VRAM summary
- **Python uses less** (10.2 vs 14.9 GB) because PyTorch's allocator is tighter
  here than ggml's worst-case arena, and bf16 ≈ f16 on weights.
- **But C++ can go much lower than Python** if you allow quantization: q8_0
  ASR+aligner drops weights from ~5.9 GB to ~3.4 GB (→ ~11–12 GB peak, and
  faster), and q4_k lower still — an option Python's bf16-only Qwen3-ASR doesn't
  have.
- **Consistent trade:** at equal (f16/bf16) precision C++ is faster but reserves
  more memory; quantization flips C++ to *both* faster and smaller, at a small
  accuracy cost.
