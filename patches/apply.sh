#!/usr/bin/env bash
# apply.sh — (re)apply the local patches to the vendored CrispASR submodule.
#
# The --aligned per-segment path needs a persistent forced-aligner that the
# stock engine doesn't expose. We carry that as a patch here instead of forking
# the submodule, so a `git submodule update` (which resets the submodule to its
# pinned commit) can be followed by `patches/apply.sh` to restore the shim.
#
# Idempotent: skips patches that are already applied.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENGINE="$ROOT/third_party/CrispASR"
PATCHES=("$ROOT/patches/0001-persistent-qwen3-aligner.patch")

if [[ ! -d "$ENGINE/.git" && ! -f "$ENGINE/.git" ]]; then
    echo "error: engine submodule not present; run 'make submodules' first." >&2
    exit 1
fi

for p in "${PATCHES[@]}"; do
    name="$(basename "$p")"
    if git -C "$ENGINE" apply --reverse --check "$p" >/dev/null 2>&1; then
        echo ">> $name: already applied"
    elif git -C "$ENGINE" apply --check "$p" >/dev/null 2>&1; then
        git -C "$ENGINE" apply "$p"
        echo ">> $name: applied"
    else
        echo "error: $name does not apply cleanly (submodule changed upstream?)." >&2
        echo "       Re-generate it: cd $ENGINE && git diff > $p" >&2
        exit 1
    fi
done
echo "Patches OK."
