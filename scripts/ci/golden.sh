#!/usr/bin/env bash
# Golden regression check. Run after models/ is populated (the nightly
# workflow fetches the models-v3 bundle first — see
# .github/workflows/nightly-golden.yml).
#
# The model-backed Rust tests are the primary golden (they assert exact
# transcriptions/translations against committed fixtures); this adds coarse
# quality floors so a silent model/preprocessing regression fails CI.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MIN_MT_BLEU="${MIN_MT_BLEU:-25}"   # vi->en, opus-100 n=100
MAX_ASR_WER="${MAX_ASR_WER:-15}"   # PhoWhisper, VIVOS clean, n=20

echo "== Rust model-backed golden tests =="
cargo test --release -- --ignored

echo "== MT BLEU floor (vi->en >= $MIN_MT_BLEU) =="
uv run --no-project --python 3.12 --with torch --with transformers \
  --with sentencepiece --with sacrebleu --with datasets \
  python scripts/evaluate_mt.py --n 100 --legs vi-en --out /tmp/golden_mt.md
bleu="$(grep -E '^\| vi-en \|' /tmp/golden_mt.md | head -1 | awk -F'|' '{gsub(/ /,"",$4); print $4+0}')"
echo "vi-en BLEU = $bleu"
awk -v b="$bleu" -v m="$MIN_MT_BLEU" 'BEGIN { exit (b+0 >= m+0) ? 0 : 1 }' \
  || { echo "FAIL: BLEU $bleu < $MIN_MT_BLEU"; exit 1; }

echo "== ASR WER ceiling (clean <= $MAX_ASR_WER%) =="
uv run --no-project --python 3.12 --with torch --with "transformers>=4.46" \
  --with jiwer --with datasets --with soundfile \
  python scripts/evaluate_asr.py --num-samples 20 --snr clean --models vinai/PhoWhisper-base \
  --out /tmp/golden_asr.md
wer="$(grep -E '\| clean \|' /tmp/golden_asr.md | head -1 | awk -F'|' '{gsub(/[ %]/,"",$4); print $4+0}')"
echo "PhoWhisper clean WER = ${wer}%"
awk -v w="$wer" -v m="$MAX_ASR_WER" 'BEGIN { exit (w+0 <= m+0) ? 0 : 1 }' \
  || { echo "FAIL: WER $wer% > $MAX_ASR_WER%"; exit 1; }

# COMET is the judged MT metric but its fastBPE native build fails on macOS,
# so run it only where it builds (Linux CI) — gate via GOLDEN_COMET=1.
if [ "${GOLDEN_COMET:-0}" = "1" ]; then
  echo "== COMET (judged metric) =="
  uv run --no-project --python 3.12 --with "numpy>=2" --with torch \
    --with transformers --with sentencepiece --with sacrebleu --with datasets \
    --with unbabel-comet \
    python scripts/evaluate_mt.py --n 200 --comet --legs vi-en --out /tmp/golden_comet.md
fi

echo "golden OK"
