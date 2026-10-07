#!/usr/bin/env bash
# Populates models/<stage>/ with the ONNX artifacts scripts/run_pipeline.sh
# needs. Two ways to do it:
#
#   1. BUNDLE (default) -- download one prebuilt .tar.gz and extract it.
#      Needs only `curl` + `tar`. No conda/torch/jupyter/HF-Hub. This is
#      the easy path for anyone who just wants to run the pipeline.
#
#   2. REBUILD -- fetch/export every model from its original source.
#      Needs the full audio-lab env (huggingface_hub for vad/asr; plus
#      torch/transformers/uroman for tts/ko's ONNX export; plus jupyter
#      for the mt/vi-en export). `pip install piper-tts` is auto-invoked
#      if needed to source tts/espeak-ng-data (Piper's runtime phoneme
#      data, required by the vi/en/zh Rust synthesis path -- see
#      src/stages/tts.rs). This is how the bundle itself is made (then
#      scripts/pack_models.sh zips models/ up).
#
# denoise has no model file -- RNNoise's weights are compiled into the
# `nnnoiseless` Rust crate (models/denoise/MANIFEST.md).
#
# Usage:
#   scripts/setup_models.sh                 # download + extract the bundle
#   scripts/setup_models.sh bundle          # same, explicit
#   scripts/setup_models.sh rebuild         # rebuild all stages from source
#   scripts/setup_models.sh rebuild vad asr # rebuild just these stages
#   scripts/setup_models.sh vad asr         # (shorthand for `rebuild vad asr`)
#
# Env vars:
#   ONEVOICE_MODELS_URL   if set, download the bundle from this URL with
#                          curl (use for a public/custom host). If unset,
#                          the default fetches the GitHub Release asset
#                          below with `gh` (works on the private repo).
#   ONEVOICE_MODELS_REPO  GitHub repo for the release (default below)
#   ONEVOICE_MODELS_TAG   release tag to pull (default below)
#   ONEVOICE_HF_PYTHON    python3 for the rebuild path (default: python3).
#                          Needs huggingface_hub for vad/asr; the tts/mt
#                          stages additionally need the full audio-lab env.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DEFAULT_MODELS_REPO="MinLee0210/OneVoice_2026"
DEFAULT_MODELS_TAG="models-v2"
BUNDLE_NAME="onevoice_models_v2.tar.gz"
PYTHON="${ONEVOICE_HF_PYTHON:-python3}"

# ---------------------------------------------------------------------------
# Bundle path (default): download one archive, extract in place.
# The repo is private, so the default path uses `gh release download`
# (which carries the user's GitHub auth) rather than plain curl -- an
# unauthenticated curl of a private release asset just 404s. Setting
# ONEVOICE_MODELS_URL switches to curl for a public/custom host.
# ---------------------------------------------------------------------------
download_bundle() {
    local tmp archive
    tmp="$(mktemp -d -t onevoice-models.XXXXXX)"
    archive="$tmp/$BUNDLE_NAME"

    if [ -n "${ONEVOICE_MODELS_URL:-}" ]; then
        command -v curl >/dev/null 2>&1 || { echo "error: 'curl' not found." >&2; rm -rf "$tmp"; exit 1; }
        echo "downloading model bundle (curl):"
        echo "  $ONEVOICE_MODELS_URL"
        if ! curl -fL --retry 3 --progress-bar -o "$archive" "$ONEVOICE_MODELS_URL"; then
            rm -rf "$tmp"
            echo "error: bundle download failed from $ONEVOICE_MODELS_URL" >&2
            exit 1
        fi
    else
        local repo="${ONEVOICE_MODELS_REPO:-$DEFAULT_MODELS_REPO}"
        local tag="${ONEVOICE_MODELS_TAG:-$DEFAULT_MODELS_TAG}"
        if ! command -v gh >/dev/null 2>&1; then
            echo "error: 'gh' not found (needed to download from the private release)." >&2
            echo "  install GitHub CLI + 'gh auth login', or set ONEVOICE_MODELS_URL to a" >&2
            echo "  reachable bundle, or rebuild from source: scripts/setup_models.sh rebuild" >&2
            rm -rf "$tmp"
            exit 1
        fi
        echo "downloading model bundle (gh): $repo release $tag ($BUNDLE_NAME)"
        if ! gh release download "$tag" --repo "$repo" --pattern "$BUNDLE_NAME" --output "$archive" --clobber; then
            rm -rf "$tmp"
            echo "error: 'gh release download' failed." >&2
            echo "  - is the release published? (scripts/pack_models.sh prints the create command)" >&2
            echo "  - are you 'gh auth login'-ed with access to $repo?" >&2
            echo "  - or rebuild from source: scripts/setup_models.sh rebuild" >&2
            exit 1
        fi
    fi

    echo "extracting into models/ ..."
    tar -xzf "$archive" -C "$REPO_ROOT"
    rm -rf "$tmp"
    echo "done."
    list_artifacts
}

# ---------------------------------------------------------------------------
# Rebuild path: fetch/export each model from its original source.
# ---------------------------------------------------------------------------
require_hf_python() {
    if ! "$PYTHON" -c "import huggingface_hub" >/dev/null 2>&1; then
        echo "error: '$PYTHON -c \"import huggingface_hub\"' failed." >&2
        echo "  pip install huggingface_hub   (or: conda activate audio-lab)" >&2
        exit 1
    fi
}

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
    # The device is bidirectional, so it speaks every target language:
    # VI, EN, ZH (Mandarin) via Piper voices (pre-made ONNX, plain
    # download), and KO via MMS-TTS (needs a one-time ONNX export -- so
    # this stage needs the full audio-lab env, not just huggingface_hub,
    # unlike vad/asr). See experiments/04_tts.ipynb for the validation.
    echo "=== tts/vi, tts/en, tts/zh (Piper, direct download) ==="
    "$PYTHON" - <<'PYEOF'
from huggingface_hub import hf_hub_download
import shutil, os
repo = "rhasspy/piper-voices"
voices = {
    "vi": "vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium",
    "en": "en/en_US/lessac/medium/en_US-lessac-medium",
    "zh": "zh/zh_CN/huayan/medium/zh_CN-huayan-medium",
}
for lang, voice in voices.items():
    os.makedirs(f"models/tts/{lang}", exist_ok=True)
    base = os.path.basename(voice)
    for ext in (".onnx", ".onnx.json"):
        src = hf_hub_download(repo, f"{voice}{ext}")
        shutil.copyfile(src, f"models/tts/{lang}/{base}{ext}")
    print(f"models/tts/{lang}/{base}.onnx")
PYEOF

    echo "=== tts/ko (MMS-TTS, exported to ONNX) ==="
    "$PYTHON" - <<'PYEOF'
import os, torch, onnxruntime as ort
from transformers import VitsModel, AutoTokenizer
repo = "facebook/mms-tts-kor"
os.makedirs("models/tts/ko", exist_ok=True)
tok = AutoTokenizer.from_pretrained(repo)          # is_uroman=True -> needs `uroman`
model = VitsModel.from_pretrained(repo).eval()
model.noise_scale = 0.0                             # deterministic (VITS is stochastic by default)
model.noise_scale_duration = 0.0
ids = tok("경보가 울리면 즉시 기계를 멈추세요.", return_tensors="pt")["input_ids"]
onnx_path = "models/tts/ko/mms_tts_kor.onnx"
torch.onnx.export(
    model, (ids,), onnx_path, dynamo=False,
    input_names=["input_ids"], output_names=["waveform"],
    dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "waveform": {0: "batch", 1: "samples"}},
    opset_version=15,
)
ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])  # smoke-test it loads
tok.save_pretrained("models/tts/ko")
print("models/tts/ko/mms_tts_kor.onnx")
PYEOF

    echo "=== tts/espeak-ng-data (Piper's runtime phoneme data, needed by vi/en/zh) ==="
    "$PYTHON" -c "import piper" >/dev/null 2>&1 || "$PYTHON" -m pip install --quiet piper-tts
    "$PYTHON" - <<'PYEOF'
import os, shutil, piper
src = os.path.join(os.path.dirname(piper.__file__), "espeak-ng-data")
dst = "models/tts/espeak-ng-data"
if os.path.exists(dst):
    shutil.rmtree(dst)
shutil.copytree(src, dst)
print(f"models/tts/espeak-ng-data/ ({len(os.listdir(dst))} entries)")
PYEOF
}

export_mt() {
    echo "=== mt/en-vi (via scripts/edge/export_en_vi_mt.py -- needs the full audio-lab env) ==="
    "$PYTHON" "$REPO_ROOT/scripts/edge/export_en_vi_mt.py"
}

rebuild() {
    require_hf_python
    local stages=("$@")
    if [ "${#stages[@]}" -eq 0 ]; then
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
    list_artifacts
}

list_artifacts() {
    echo "models/ artifacts:"
    find models -type f \( -name '*.onnx' -o -name '*.onnx.json' \) | sort
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
mode="${1:-bundle}"
case "$mode" in
    bundle)
        download_bundle
        ;;
    rebuild)
        shift
        rebuild "$@"
        ;;
    vad | asr | tts | mt)
        # shorthand: bare stage names imply a rebuild of that subset
        rebuild "$@"
        ;;
    *)
        echo "error: unknown argument '$mode'" >&2
        echo "usage: scripts/setup_models.sh [bundle | rebuild [vad asr tts mt]]" >&2
        exit 1
        ;;
esac
