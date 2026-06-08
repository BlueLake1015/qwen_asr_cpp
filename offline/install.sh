#!/usr/bin/env bash
# install.sh — set up the build/runtime environment on an OFFLINE machine from the
# bundled .debs, then verify the toolchain. No network required.
#
# Prereq: copy the whole qwen_asr_cpp repo (with third_party/ submodules already
# populated — they ship checked out) AND offline/debs/ to this machine. The .debs
# must have been gathered by offline/download_debs.sh on a matching Ubuntu/arch.
#
# After this, build with `make` and fetch models (offline note in offline/README.md).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
DEBS="$HERE/debs"

[ -d "$DEBS" ] && ls "$DEBS"/*.deb >/dev/null 2>&1 || {
    echo "error: no .debs in $DEBS — run offline/download_debs.sh on an online host first." >&2
    exit 1
}

# Warn on Ubuntu/arch mismatch (debs are release+arch specific).
if [ -f "$HERE/target.txt" ]; then
    . /etc/os-release
    want="$(cat "$HERE/target.txt")"
    have="${VERSION_ID} ${VERSION_CODENAME} $(dpkg --print-architecture)"
    [ "$want" = "$have" ] || echo "WARNING: bundle built for [$want] but this host is [$have] — install may fail."
fi

n=$(ls -1 "$DEBS"/*.deb | wc -l)
echo ">> installing $n packages from $DEBS (offline) ..."
# Two dpkg passes resolve intra-set ordering (unpack-all then configure-all);
# the closure is self-contained, so no downloads are attempted.
sudo dpkg -i "$DEBS"/*.deb 2>/dev/null || true
sudo dpkg -i "$DEBS"/*.deb || {
    echo "error: dpkg could not configure all packages (missing dep / arch mismatch)." >&2
    echo "       Check that download_debs.sh ran on a matching Ubuntu release + arch." >&2
    exit 1
}

echo ">> verifying toolchain ..."
miss=0
for t in g++ g++-12 make cmake ffmpeg git curl nvcc; do
    if command -v "$t" >/dev/null 2>&1; then echo "   ok: $t"; else echo "   MISSING: $t"; miss=$((miss+1)); fi
done
echo ">> GPU driver check:"; nvidia-smi -L 2>/dev/null || echo "   (no nvidia-smi — install the NVIDIA driver separately; needed for GPU inference)"

[ "$miss" -eq 0 ] && echo ">> environment ready. Next: cd $ROOT && make" \
                  || { echo ">> $miss tool(s) missing — see warnings above." >&2; exit 1; }
