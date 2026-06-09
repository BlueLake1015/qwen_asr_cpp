#!/usr/bin/env bash
# download_debs.sh — gather every .deb needed to build & run qwen_asr_cpp into
# offline/debs/, for installation on an air-gapped machine.
#
# RUN THIS ON AN ONLINE MACHINE whose Ubuntu release + CPU architecture MATCH the
# offline target (e.g. both Ubuntu 24.04 / amd64). It downloads the full recursive
# dependency closure of the packages below, so the target needs no network.
#
# Does NOT include the NVIDIA *driver* — the GPU target is assumed to already have
# a working driver (`nvidia-smi` runs). Only the CUDA *toolkit* (nvcc + libs) and
# build/runtime packages are bundled.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DEBS="$HERE/debs"

# CUDA toolkit package. The GPU build needs CUDA >= 12 for Ada/sm_89 (RTX 40xx).
# Distro `nvidia-cuda-toolkit` is 12.0 on 24.04 (OK) but only 11.5 on 22.04 (too
# old for sm_89) — so prefer NVIDIA's repo `cuda-toolkit-12-4` when available.
# Override with CUDA_PKG=<name>, or CUDA_PKG=none to skip (CPU-only target).
CUDA_PKG="${CUDA_PKG:-auto}"
if [ "$CUDA_PKG" = auto ]; then
    if apt-cache show cuda-toolkit-12-4 >/dev/null 2>&1; then CUDA_PKG=cuda-toolkit-12-4
    elif apt-cache show nvidia-cuda-toolkit >/dev/null 2>&1; then CUDA_PKG=nvidia-cuda-toolkit
    else CUDA_PKG=none; fi
fi

# Top-level packages we explicitly need:
#   g++, g++-12, gcc-12  -> compilers (g++-12 is the CUDA host compiler)
#   make, cmake          -> build
#   ffmpeg               -> runtime media decode
#   git, curl, ca-certificates -> submodules / model download / TLS
#   $CUDA_PKG            -> nvcc + CUDA dev/runtime libs (cublas, cudart)
PKGS=(g++ g++-12 gcc-12 make cmake ffmpeg git curl ca-certificates)
[ "$CUDA_PKG" != none ] && PKGS+=("$CUDA_PKG")
echo ">> CUDA toolkit package: ${CUDA_PKG} (override with CUDA_PKG=...)"

. /etc/os-release
ARCH="$(dpkg --print-architecture)"
echo ">> online host: Ubuntu ${VERSION_ID} (${VERSION_CODENAME}) / ${ARCH}"
echo ">> target MUST match this release + arch"

mkdir -p "$DEBS"
sudo apt-get update

echo ">> resolving recursive dependency closure ..."
# Drop NVIDIA *driver*/compute packages: the GPU target already provides them, and
# apt-cache --recurse otherwise pulls every libnvidia-compute-NNN alternative.
mapfile -t DEPS < <(
    apt-cache depends --recurse --no-recommends --no-suggests \
        --no-conflicts --no-breaks --no-replaces --no-enhances "${PKGS[@]}" \
    | grep -E '^[a-zA-Z0-9]' | grep -v -- '<' \
    | grep -vE '^(libnvidia-|nvidia-(compute|driver|kernel|dkms|utils|settings)|xserver-xorg-video-nvidia|libgl1-nvidia)' \
    | grep -vE '^(libav(codec|format|filter|device|util|resample)-extra|liboss4-salsa-asound2|libjack0)' \
    | sort -u
)
echo ">> ${#DEPS[@]} packages in closure (driver libs excluded); downloading .debs ..."

cd "$DEBS"
fail=0
for p in "${DEPS[@]}"; do
    apt-get download "$p" 2>/dev/null || { echo "   skip (virtual/unavailable): $p"; fail=$((fail+1)); }
done

# Manifest + a local apt index so the target can resolve with apt if it prefers.
dpkg-scanpackages --multiversion . /dev/null > Packages 2>/dev/null || true
gzip -kf Packages 2>/dev/null || true
ls -1 *.deb 2>/dev/null > "$HERE/manifest.txt" || true
echo "${VERSION_ID} ${VERSION_CODENAME} ${ARCH}" > "$HERE/target.txt"

n=$(ls -1 *.deb 2>/dev/null | wc -l)
size=$(du -sh "$DEBS" 2>/dev/null | cut -f1)
echo ">> done: $n .debs ($size) in $DEBS  (skipped: $fail virtual)"
echo ">> copy the whole repo (with third_party/ submodule populated) + offline/debs/"
echo "   to the target, then run: offline/install.sh"
