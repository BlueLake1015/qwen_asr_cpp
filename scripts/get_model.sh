#!/usr/bin/env bash
# get_model.sh — download the Qwen3-ASR GGUFs into models/ (pre-converted; NO
# Python, NO conversion — just curl). These are the weights the CrispASR engine
# consumes directly.
#
#   * ASR model:      cstr/qwen3-asr-<size>-GGUF
#   * Forced aligner: cstr/qwen3-forced-aligner-0.6b-GGUF   (precise timestamps)
#
# Note: `qwen-asr ... -m auto` already auto-downloads a default ASR model to
# ~/.cache/crispasr on first run. This script is for an explicit/offline setup
# and is the easiest way to get the forced-aligner GGUF the SRT path wants.
#
# Usage:
#   scripts/get_model.sh                 # 1.7B ASR + 0.6B aligner, both q8_0
#   SIZE=0.6b scripts/get_model.sh       # smaller/faster 0.6B ASR
#   ASR_QUANT=q4_k scripts/get_model.sh  # configure ASR / aligner quant separately
#   ALIGNER_QUANT=f16 scripts/get_model.sh
#   QUANT=q4_k scripts/get_model.sh      # set both at once
#   ALL=1 scripts/get_model.sh           # every quant variant of both models
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="$ROOT/models"
SIZE="${SIZE:-1.7b}"                       # 1.7b (best accuracy) | 0.6b
QUANT="${QUANT:-q8_0}"                     # master default for both
ASR_QUANT="${ASR_QUANT:-$QUANT}"           # ASR quantization     (default q8_0)
ALIGNER_QUANT="${ALIGNER_QUANT:-$QUANT}"   # aligner quantization (default q8_0)
ALL="${ALL:-0}"                            # 1 -> grab every variant below

ASR_REPO="cstr/qwen3-asr-${SIZE}-GGUF"
ALN_REPO="cstr/qwen3-forced-aligner-0.6b-GGUF"

# Published variants per repo (the aligner is only at 0.6B).
ASR_VARIANTS="f16 q4_k q8_0"
ALN_VARIANTS="f16 q4_k q5_0 q8_0"

mkdir -p "$MODELS"

fetch() {  # repo, file
    local out="$MODELS/$2"
    if [[ -s "$out" ]]; then echo ">> already have $2"; return; fi
    echo ">> downloading $2"
    curl -L --fail --retry 3 -o "$out" "https://huggingface.co/$1/resolve/main/$2"
}

if [[ "$ALL" == "1" ]]; then
    for q in $ASR_VARIANTS; do fetch "$ASR_REPO" "qwen3-asr-${SIZE}-${q}.gguf"; done
    for q in $ALN_VARIANTS; do fetch "$ALN_REPO" "qwen3-forced-aligner-0.6b-${q}.gguf"; done
else
    fetch "$ASR_REPO" "qwen3-asr-${SIZE}-${ASR_QUANT}.gguf"
    fetch "$ALN_REPO" "qwen3-forced-aligner-0.6b-${ALIGNER_QUANT}.gguf"
fi

# Silero VAD model — used by vad-mode segmentation. Kept in models/ so the app
# never has to auto-download into a cache.
fetch "ggml-org/whisper-vad" "ggml-silero-v6.2.0.bin"

echo
echo "Models in $MODELS:"
ls -lh "$MODELS"/*.gguf
echo
echo "Run (qwen-asr defaults to size=$SIZE, q8_0 for both):"
echo "  $ROOT/build/qwen-asr <input> -o out.srt -l en"
