#!/usr/bin/env bash
# Builds and runs the real onevoice CLI binary (src/main.rs) against a
# WAV file, as a real CLI app -- not `cargo run`. Requires models/ to
# already be populated (see scripts/setup_models.sh) and configs/ to
# describe the language pair(s) you're running (see configs/README.md).
#
# Usage:
#   scripts/run_pipeline.sh
#   # no args -> builds --release, runs against the committed eval
#   # fixture (data/eval/ptt_sample_vi_48k.wav), source=vi target=en,
#   # writes to /tmp/onevoice_out.wav, logs at INFO level.
#
#   scripts/run_pipeline.sh <input.wav> <output.wav> [-- <extra onevoice args>]
#   # your own input/output, still builds --release first. Anything
#   # after `--` is passed straight through to the binary, e.g.:
#   scripts/run_pipeline.sh in.wav out.wav -- --source vi --target en
#
# Env vars:
#   ONEVOICE_LOG   RUST_LOG value (default "info")
#   ONEVOICE_PROFILE   cargo build profile: "release" (default) or "debug"

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PROFILE="${ONEVOICE_PROFILE:-release}"
export RUST_LOG="${ONEVOICE_LOG:-info}"

INPUT_WAV="${1:-data/eval/ptt_sample_vi_48k.wav}"
OUTPUT_WAV="${2:-/tmp/onevoice_out.wav}"
EXTRA_ARGS=()
if [ "$#" -gt 2 ]; then
    if [ "$3" = "--" ]; then
        EXTRA_ARGS=("${@:4}")
    else
        EXTRA_ARGS=("${@:3}")
    fi
fi

if [ ! -d models/asr/vi ] || [ -z "$(ls -A models/asr/vi 2>/dev/null)" ]; then
    echo "error: models/ isn't populated yet." >&2
    echo "  run: scripts/setup_models.sh" >&2
    exit 1
fi

case "$PROFILE" in
    release)
        echo "building (release)..."
        cargo build --release
        BIN="target/release/onevoice"
        ;;
    debug)
        echo "building (debug)..."
        cargo build
        BIN="target/debug/onevoice"
        ;;
    *)
        echo "error: ONEVOICE_PROFILE must be 'release' or 'debug', got '$PROFILE'" >&2
        exit 1
        ;;
esac

# bash 3.2 (macOS default) errors on `${EXTRA_ARGS[*]}` under `set -u` when
# the array is empty; the `+` guard expands to nothing in that case.
echo "running: $BIN $INPUT_WAV $OUTPUT_WAV ${EXTRA_ARGS[*]+"${EXTRA_ARGS[*]}"}"
echo
exec "$BIN" "$INPUT_WAV" "$OUTPUT_WAV" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
