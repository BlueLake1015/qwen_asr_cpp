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

# Top-level packages we explicitly need:
#   g++, g++-12, gcc-12  -> compilers (g++-12 is the CUDA host compiler)
#   make, cmake          -> build
#   ffmpeg               -> runtime media decode
#   git, curl, ca-certificates -> submodules / model download / TLS
#   nvidia-cuda-toolkit  -> nvcc + CUDA dev/runtime libs (cublas, cudart)
PKGS=(g++ g++-12 gcc-12 make cmake ffmpeg git curl ca-certificates nvidia-cuda-toolkit)

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
