#!/usr/bin/env bash
# Downloads the ONNX model artifacts scripts/run_pipeline.sh needs into
# models/<stage>/. See models/README.md and models/*/MANIFEST.md for
# what each artifact is and why it was picked.
#
# vad, asr/vi, asr/en-zh-ko, tts/vi are pre-converted ONNX already on
# the HF Hub -- these stages are just a download (needs the
# `huggingface_hub` Python package, nothing else -- no PyTorch, no
# Jupyter). mt/vi-en is the odd one out: Helsinki-NLP only ships
# PyTorch weights for opus-mt-vi-en, so that stage still needs a real
# export (optimum + a custom tokenizer-surgery step for Marian's
# sentencepiece tokenizer) -- this script shells out to the notebook
# that does it (needs the full audio-lab env + Jupyter). Lifting that
# into a standalone script is tracked in docs/TODO.md.
#
# denoise has no model file at all -- RNNoise's weights are compiled
# into the `nnnoiseless` Rust crate, see models/denoise/MANIFEST.md.
#
# Usage:
#   scripts/setup_models.sh              # everything (vad, asr, tts, mt)
#   scripts/setup_models.sh vad asr tts  # skip the notebook-based MT export
#   scripts/setup_models.sh mt           # just the MT export
#
# Env vars:
#   ONEVOICE_HF_PYTHON   python3 to use for the HF downloads (default:
#                         python3 on PATH -- must have huggingface_hub
#                         installed, e.g. via `conda activate audio-lab`)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PYTHON="${ONEVOICE_HF_PYTHON:-python3}"

if ! "$PYTHON" -c "import huggingface_hub" >/dev/null 2>&1; then
    echo "error: '$PYTHON -c \"import huggingface_hub\"' failed." >&2
    echo "  pip install huggingface_hub   (or: conda activate audio-lab)" >&2
    exit 1
fi

download_vad() {
    echo "=== vad ==="
    mkdir -p models/vad
    "$PYTHON" - <<'PYEOF'
from huggingface_hub import hf_hub_download
import shutil
src = hf_hub_download("onnx-community/silero-vad", "onnx/model.onnx")
shutil.copyfile(src, "models/vad/model.onnx")
print("models/vad/model.onnx")
PYEOF
}

download_asr() {
    echo "=== asr/vi ==="
    mkdir -p models/asr/vi
    "$PYTHON" - <<'PYEOF'
from huggingface_hub import hf_hub_download
import shutil
repo = "huuquyet/PhoWhisper-base"
files = {
    "onnx/encoder_model_quantized.onnx": "encoder_model.onnx",
    "onnx/decoder_model_merged_quantized.onnx": "decoder_model_merged.onnx",
    "tokenizer.json": "tokenizer.json",
    "preprocessor_config.json": "preprocessor_config.json",
    "generation_config.json": "generation_config.json",
}
for remote, local in files.items():
    src = hf_hub_download(repo, remote)
    shutil.copyfile(src, f"models/asr/vi/{local}")
print("models/asr/vi/*")
PYEOF

    echo "=== asr/en-zh-ko ==="
    mkdir -p models/asr/en-zh-ko
    "$PYTHON" - <<'PYEOF'
from huggingface_hub import hf_hub_download
import shutil
repo = "onnx-community/whisper-small"
files = {
    "onnx/encoder_model_quantized.onnx": "encoder_model.onnx",
    "onnx/decoder_model_merged_quantized.onnx": "decoder_model_merged.onnx",
    "tokenizer.json": "tokenizer.json",
    "preprocessor_config.json": "preprocessor_config.json",
    "generation_config.json": "generation_config.json",
}
for remote, local in files.items():
    src = hf_hub_download(repo, remote)
    shutil.copyfile(src, f"models/asr/en-zh-ko/{local}")
print("models/asr/en-zh-ko/*")
PYEOF
}

download_tts() {
    echo "=== tts/vi ==="
    mkdir -p models/tts/vi
    "$PYTHON" - <<'PYEOF'
from huggingface_hub import hf_hub_download
import shutil
repo = "rhasspy/piper-voices"
voice = "vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium"
for ext in (".onnx", ".onnx.json"):
    src = hf_hub_download(repo, f"{voice}{ext}")
    shutil.copyfile(src, f"models/tts/vi/vi_VN-vais1000-medium{ext}")
print("models/tts/vi/*")
PYEOF
}

export_mt() {
    echo "=== mt/vi-en (via notebook -- needs Jupyter + the audio-lab kernel) ==="
    "$REPO_ROOT/scripts/run_notebook.sh" 03_mt
}

stages=("$@")
if [ "$#" -eq 0 ]; then
    stages=(vad asr tts mt)
fi

for stage in "${stages[@]}"; do
    case "$stage" in
        vad) download_vad ;;
        asr) download_asr ;;
        tts) download_tts ;;
        mt) export_mt ;;
        *)
            echo "error: unknown stage '$stage' (expected: vad, asr, tts, mt)" >&2
            exit 1
            ;;
    esac
    echo
done

echo "done. models/ artifacts:"
find models -type f \( -name '*.onnx' -o -name '*.onnx.json' \) | sort
