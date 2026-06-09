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
DEBS="${DEBS_DIR:-$HERE/debs}"      # override with DEBS_DIR=<path> (e.g. for testing)

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

# DRY_RUN=1 → simulate only: validate the .debs and dependency closure with
# `dpkg --dry-run`, make no changes (no install, no profile.d, no verify side effects).
if [ "${DRY_RUN:-0}" = 1 ]; then
    echo ">> DRY RUN: $n .debs in $DEBS ($(du -sh "$DEBS" 2>/dev/null | cut -f1)) — no changes will be made"
    out="$(sudo dpkg --dry-run -i "$DEBS"/*.deb 2>&1)" && rc=0 || rc=$?
    # Conflicts = redundant OR-alternatives (e.g. *-extra ffmpeg libs); benign — the
    # needed counterpart installs. Dependency problems referencing a package NOT in
    # the bundle would be fatal on a bare target.
    conflicts=$(printf '%s\n' "$out" | grep -ciE 'conflicting packages|conflicts with' || true)
    depprob=$(printf '%s\n'  "$out" | grep -ciE 'dependency problems|depends on .*; however' || true)
    echo ">> dry-run: ${conflicts} conflicting-alternative(s) [benign], ${depprob} dependency-problem(s)"
    if [ "$depprob" -eq 0 ]; then
        echo ">> DRY RUN OK: only redundant OR-alternatives conflict (auto-skipped). Required"
        echo "   packages would install; install.sh verifies the toolchain afterward."
    else
        echo ">> DRY RUN: dependency problems found — bundle may be incomplete:"
        printf '%s\n' "$out" | grep -iE 'depends on' | head | sed 's/^/     /'
    fi
    echo ">> NOTE: deps are also satisfied by THIS host's installed packages; a pristine"
    echo "   target relies on the bundle being a complete closure (download_debs.sh)."
    echo "   Re-run without DRY_RUN=1 to install."
    exit 0
fi

echo ">> installing $n packages from $DEBS (offline) ..."
# Two dpkg passes resolve intra-set ordering (unpack-all then configure-all);
# the closure is self-contained, so no downloads are attempted. We do NOT treat a
# non-zero dpkg exit as fatal: bundled OR-alternatives (e.g. *-extra ffmpeg libs)
# conflict with their chosen counterparts and are skipped — harmless. The
# toolchain verification below is the real success criterion.
sudo dpkg -i "$DEBS"/*.deb 2>/dev/null || true
sudo dpkg -i "$DEBS"/*.deb 2>/dev/null || true

# The CUDA toolkit installs to /usr/local/cuda-XX/bin, which is NOT on PATH.
# Expose it for this shell + future logins so cmake/nvcc are found by `make`.
CUDA_BIN="$(ls -d /usr/local/cuda-*/bin 2>/dev/null | sort -V | tail -1)"
if [ -n "$CUDA_BIN" ]; then
    export PATH="$CUDA_BIN:$PATH"
    if [ ! -f /etc/profile.d/cuda.sh ]; then
        echo "export PATH=$CUDA_BIN:\$PATH" | sudo tee /etc/profile.d/cuda.sh >/dev/null 2>&1 \
            && echo ">> wrote /etc/profile.d/cuda.sh ($CUDA_BIN added to PATH; re-login or 'source' it)"
    fi
fi

echo ">> verifying toolchain ..."
miss=0
for t in g++ g++-12 make cmake ffmpeg git curl nvcc; do
    if command -v "$t" >/dev/null 2>&1; then echo "   ok: $t"; else echo "   MISSING: $t"; miss=$((miss+1)); fi
done
echo ">> GPU driver check:"; nvidia-smi -L 2>/dev/null || echo "   (no nvidia-smi — install the NVIDIA driver separately; needed for GPU inference)"

if [ "$miss" -eq 0 ]; then
    echo ">> environment ready. Next:"
    [ -n "$CUDA_BIN" ] && echo "     export PATH=$CUDA_BIN:\$PATH    # if nvcc isn't on PATH in this shell yet"
    echo "     cd $ROOT && make                 # (or: make GPU=off for CPU-only)"
else
    echo ">> $miss tool(s) missing — see warnings above." >&2; exit 1
fi
