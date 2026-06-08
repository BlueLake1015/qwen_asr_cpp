#!/usr/bin/env bash
# run_modes.sh — exercise every qwen-asr mode against one input.
#
# Usage:
#   tests/run_modes.sh [INPUT]
#     INPUT  a media file (mp4/mkv/wav/mp3/m4a/...) OR a stream URL
#            (udp://, rtp://, http(s)://, ...). Default: the simplest bundled
#            fixture (tests/fixtures/test_en_30s.mp4).
#
# Models: the app loads only from models/ (1.7B ASR + 0.6B aligner by default).
# This harness uses those defaults — run scripts/get_model.sh first. The 1.7B
# model is slow on CPU, so the full suite takes a while; raise TIMEOUT if needed
# or override MODEL with a lighter quant for quick local runs.
#
# Env overrides:
#   QASR=<path>        qwen-asr binary            (default: build/qwen-asr)
#   MODEL=<gguf>       override ASR model         (default: app resolves models/ 1.7B)
#   ALIGNER=<gguf>     override forced-aligner     (default: app resolves models/ 0.6B)
#   LANG_HINT=<code>   language hint              (default: en; use zh for the zh fixture)
#   THREADS=<n>        inference threads          (default: 4)
#   TIMEOUT=<sec>      per-file-mode timeout      (default: 1800)
#   URL_WINDOW=<sec>   how long to sample a live URL per streaming mode (default: 40)
#
# Modes covered:
#   file input : offline(txt), offline(srt), fixed(stream), vad(stream),
#                fixed --aligned (srt), vad --aligned (srt)
#   URL input  : fixed(stream), vad(stream)   [offline/aligned need a finite file]
#
# Exit code = number of failed cases (0 = all good). SKIP doesn't fail.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${QASR:-$ROOT/build/qwen-asr}"
INPUT="${1:-$ROOT/tests/fixtures/test_en_30s.mp4}"
LANG_HINT="${LANG_HINT:-en}"
THREADS="${THREADS:-4}"
TIMEOUT="${TIMEOUT:-1800}"
URL_WINDOW="${URL_WINDOW:-40}"
OUTDIR="$(mktemp -d /tmp/qwen_asr_test.XXXXXX)"

pass=0; fail=0; skip=0
ok()  { echo "  PASS: $*"; pass=$((pass+1)); }
ko()  { echo "  FAIL: $*"; fail=$((fail+1)); }
sk()  { echo "  SKIP: $*"; skip=$((skip+1)); }

[[ -x "$BIN" ]] || { echo "error: $BIN not found — run 'make' first." >&2; exit 2; }
if [[ ! "$INPUT" =~ ^[a-zA-Z][a-zA-Z0-9+.-]*:// ]] && [[ ! -f "$INPUT" ]]; then
    echo "error: input '$INPUT' is neither a file nor a URL." >&2; exit 2
fi

is_url=0
[[ "$INPUT" =~ ^[a-zA-Z][a-zA-Z0-9+.-]*:// ]] && is_url=1

# Models live in models/ — the app resolves the 1.7B ASR + 0.6B aligner + silero
# VAD from there. We only pass flags when MODEL/ALIGNER are explicitly overridden.
MODEL_FLAGS=();   [[ -n "${MODEL:-}" ]]   && MODEL_FLAGS=(-m "$MODEL")
ALIGNER_FLAGS=(); [[ -n "${ALIGNER:-}" ]] && ALIGNER_FLAGS=(--aligner "$ALIGNER")

have_asr=0
{ [[ -n "${MODEL:-}" ]] || ls "$ROOT"/models/qwen3-asr-1.7b-*.gguf >/dev/null 2>&1; } && have_asr=1
have_aligner=0
{ [[ -n "${ALIGNER:-}" ]] || ls "$ROOT"/models/qwen3-forced-aligner-0.6b-*.gguf >/dev/null 2>&1; } && have_aligner=1

echo "=============================================================="
echo " qwen-asr mode tests"
echo "   input   : $INPUT  ($([[ $is_url -eq 1 ]] && echo URL || echo file))"
echo "   binary  : $BIN"
echo "   model   : ${MODEL:-<app default: models/ 1.7B>}"
echo "   aligner : ${ALIGNER:-<app default: models/ 0.6B>}"
echo "   lang    : $LANG_HINT   threads: $THREADS   out: $OUTDIR"
echo "=============================================================="
[[ $have_asr -eq 0 ]] && echo " WARNING: no models/qwen3-asr-1.7b-*.gguf — run scripts/get_model.sh"

# Run a streaming-mode case: accept clean exit (file) or timeout (live URL),
# and require some transcription output.
stream_case() {
    local name="$1" win="$2"; shift 2
    local out="$OUTDIR/$name.out" log="$OUTDIR/$name.log" rc
    timeout "$win" "$BIN" "$@" >"$out" 2>"$log"; rc=$?
    if { [[ $rc -eq 0 ]] || [[ $rc -eq 124 ]]; } && { [[ -s "$out" ]] || grep -q '\[' "$log"; }; then
        ok "$name (exit=$rc, produced output)"
    else
        ko "$name (exit=$rc, no output — see $log)"
    fi
}

if [[ $is_url -eq 0 ]]; then
    # ---- offline: plain text to stdout ----
    if timeout "$TIMEOUT" "$BIN" "$INPUT" "${MODEL_FLAGS[@]}" -l "$LANG_HINT" -t "$THREADS" \
            >"$OUTDIR/offline.txt" 2>"$OUTDIR/offline.log" && [[ -s "$OUTDIR/offline.txt" ]]; then
        ok "offline -> text"
    else
        ko "offline -> text (see $OUTDIR/offline.log)"
    fi

    # ---- offline: SRT (forced aligner auto-applied from models/) ----
    if timeout "$TIMEOUT" "$BIN" "$INPUT" -o "$OUTDIR/offline.srt" "${MODEL_FLAGS[@]}" "${ALIGNER_FLAGS[@]}" \
            -l "$LANG_HINT" -t "$THREADS" >"$OUTDIR/offline_srt.log" 2>&1 \
            && grep -q -- '-->' "$OUTDIR/offline.srt"; then
        ok "offline -> SRT (timestamps present)"
    else
        ko "offline -> SRT (see $OUTDIR/offline_srt.log)"
    fi

    # ---- streaming: fixed window ----
    stream_case "fixed-stream" "$TIMEOUT" "$INPUT" --mode fixed --segment-seconds 8 \
        "${MODEL_FLAGS[@]}" -l "$LANG_HINT" -t "$THREADS"

    # ---- streaming: VAD-adaptive, JSON events ----
    if timeout "$TIMEOUT" "$BIN" "$INPUT" --mode vad --json "${MODEL_FLAGS[@]}" -l "$LANG_HINT" -t "$THREADS" \
            >"$OUTDIR/vad.jsonl" 2>"$OUTDIR/vad.log" && grep -q '"type"' "$OUTDIR/vad.jsonl"; then
        ok "vad -> stream (JSON partial/final events)"
    else
        ko "vad -> stream (see $OUTDIR/vad.log)"
    fi

    # ---- per-segment + forced aligner (models resident, accurate SRT) ----
    if [[ $have_asr -eq 1 && $have_aligner -eq 1 ]]; then
        for m in fixed vad; do
            out="$OUTDIR/$m-aligned.srt"
            extra=(); [[ "$m" == fixed ]] && extra=(--segment-seconds 10)
            if timeout "$TIMEOUT" "$BIN" "$INPUT" --mode "$m" --aligned "${extra[@]}" \
                    "${MODEL_FLAGS[@]}" "${ALIGNER_FLAGS[@]}" -l "$LANG_HINT" -t "$THREADS" -o "$out" \
                    >"$OUTDIR/$m-aligned.log" 2>&1 && grep -q -- '-->' "$out"; then
                ok "$m --aligned -> SRT (resident models, timestamps present)"
            else
                ko "$m --aligned -> SRT (see $OUTDIR/$m-aligned.log)"
            fi
        done
    else
        sk "fixed --aligned / vad --aligned (need models/ ASR + aligner; run scripts/get_model.sh)"
    fi
else
    echo "(URL input: testing live streaming modes only; sampling ${URL_WINDOW}s each)"
    stream_case "fixed-stream-url" "$URL_WINDOW" "$INPUT" --mode fixed --segment-seconds 8 \
        "${MODEL_FLAGS[@]}" -l "$LANG_HINT" -t "$THREADS"
    stream_case "vad-stream-url" "$URL_WINDOW" "$INPUT" --mode vad --json \
        "${MODEL_FLAGS[@]}" -l "$LANG_HINT" -t "$THREADS"
fi

echo "=============================================================="
echo " result: $pass passed, $fail failed, $skip skipped"
echo " artifacts: $OUTDIR"
echo "=============================================================="
exit "$fail"
