#!/usr/bin/env bash
# Bundles the current models/ tree into a single .tar.gz for one-shot
# distribution, so users can `setup_models.sh` with just curl + tar --
# no conda/torch/jupyter/uroman/HF-Hub needed (that whole toolchain is
# only for *rebuilding* the models, see setup_models.sh rebuild).
#
# Workflow:
#   1. scripts/pack_models.sh                 # -> dist/onevoice-models.tar.gz
#   2. upload it to a GitHub Release (the script prints the exact
#      `gh release create` command), or any static host
#   3. point setup_models.sh at it via ONEVOICE_MODELS_URL, or bake the
#      release URL into DEFAULT_MODELS_URL there
#
# Usage:
#   scripts/pack_models.sh [output.tar.gz]     # default: dist/onevoice-models.tar.gz

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT="${1:-dist/onevoice-models.tar.gz}"
mkdir -p "$(dirname "$OUT")"

# Sanity: refuse to pack an incomplete models/ tree -- catches the
# "packed before setup finished" mistake before it ships a broken bundle.
required=(
    models/vad/model.onnx
    models/asr/vi/encoder_model.onnx
    models/asr/en-zh-ko/encoder_model.onnx
    models/mt/vi-en/encoder_model.onnx
    models/tts/vi/vi_VN-vais1000-medium.onnx
    models/tts/en/en_US-lessac-medium.onnx
    models/tts/zh/zh_CN-huayan-medium.onnx
    models/tts/ko/mms_tts_kor.onnx
    models/tts/espeak-ng-data/phontab
)
missing=()
for f in "${required[@]}"; do
    [ -f "$f" ] || missing+=("$f")
done
if [ "${#missing[@]}" -gt 0 ]; then
    echo "error: models/ is incomplete -- run scripts/setup_models.sh rebuild first." >&2
    printf '  missing: %s\n' "${missing[@]}" >&2
    exit 1
fi

echo "packing models/ -> $OUT ..."
# Archive paths are relative to the repo root (models/...), so extracting
# at the repo root restores the tree in place. Skip .gitkeep placeholders.
tar --exclude='models/**/.gitkeep' --exclude='models/.gitkeep' \
    -czf "$OUT" models/

size="$(du -h "$OUT" | cut -f1)"
sha="$(shasum -a 256 "$OUT" | cut -d' ' -f1)"
echo
echo "created $OUT ($size)"
echo "sha256: $sha"
echo
echo "next -- upload to a GitHub Release (tag 'models-v1' assumed by setup_models.sh):"
echo "  gh release create models-v1 \"$OUT\" \\"
echo "    --title 'OneVoice model bundle v1' \\"
echo "    --notes 'Prebuilt ONNX artifacts for all stages (vad/asr/mt/tts, VI+EN+ZH+KO). sha256: $sha'"
echo
echo "if you use a different tag/host, point setup_models.sh at it via"
echo "  ONEVOICE_MODELS_URL=<url> scripts/setup_models.sh"
