# VI CTC fine-tuning recipe (2026-10-12)

Follow-up to the CTC sweep (`docs/reports/asr_ctc_sweep.md`). The runtime now
has a real CTC backend (`src/stages/ctc.rs`, ASR ~57 ms vs ~360 ms Whisper),
but the only **Apache-2.0** VI CTC checkpoint (`dragonSwing/wav2vec2-base-vietnamese`)
is weak (clean 10.4% / 5 dB 23.0%), while the strong ones are CC-BY-NC and
cannot ship. This recipe fine-tunes the Apache-2.0 one to close the gap.

## Goal / acceptance

- Target: **≤7% clean WER and ≤15% at 5 dB** on the VIVOS slice → then CTC
  becomes the default fast backend and the ~300 ms end-to-end win is unlocked.
- Gate: re-run `scripts/evaluate_asr.py` and `scripts/evaluate_asr_noise.py`
  before switching the runtime default.

## Run it (needs a GPU)

```sh
uv run --python 3.12 --with torch --with "transformers>=4.46" \
  --with datasets --with jiwer --with soundfile --with librosa --with accelerate \
  python scripts/edge/finetune_ctc_vi.py \
    --base dragonSwing/wav2vec2-base-vietnamese \
    --out models/asr/vi-ctc-ft \
    --epochs 8 --batch 8 --accum 2 --lr 3e-4 \
    --num-train 3000 --num-eval 200 --noise
```

- **Hardware:** 1 GPU, ≥8 GB (T4/Colab fine; bf16/fp16). `--num-train 3000`
  is a few minutes–~1 h depending on GPU; scale to the full VIVOS (~15 h) or
  Common Voice VI for a real gain.
- **Data:** VIVOS (`quocanh34/viet_vivos`, train split) + optional ESC-50
  industrial-noise augmentation (chain-saw/engine/siren/…) mixed at random
  SNR. `--noise` mixes at 0–15 dB with p=0.7.
- **Base settings:** feature encoder frozen, `ctc_loss_reduction="mean"`,
  char-level tokenizer from the base checkpoint (Wav2Vec2Processor).

## Output & drop-in

The script saves the HF weights **and** exports the runtime layout
(`model.onnx`, `vocab.json`, `preprocessor_config.json`, `MANIFEST.md`) to
`--out`. Drop-in:

```sh
cp -r models/asr/vi-ctc-ft/model.onnx \
      models/asr/vi-ctc-ft/vocab.json \
      models/asr/vi-ctc-ft/preprocessor_config.json models/asr/vi-ctc/
# then the runtime auto-selects CTC for asr_model_dir="asr/vi-ctc" (Asr::load)
```

## Evaluate

```sh
# quality (must beat the 10.4% / 23.0% baseline)
uv run --no-project --python 3.12 --with torch --with "transformers>=5.13.0" \
  --with jiwer --with datasets --with soundfile --with librosa \
  python scripts/evaluate_asr.py --num-samples 200 --snr clean 5 \
  --models models/asr/vi-ctc-ft --out docs/reports/asr_ctc_ft.md

# real-noise robustness
python scripts/evaluate_asr_noise.py --model models/asr/vi-ctc-ft
```

## Empirical note (2026-10-12, Mac MPS)

A first run (`--lr 3e-4 --epochs 3 --num-train 1000`) **regressed**: clean WER
10.38% → 16.94%, 5 dB 22.95% → 25.14% (harness-normalized, same 20-utt test
slice). The base is already converged, so the XLSR-from-scratch LR (3e-4) is
far too high. **Use a gentle LR (≤1e-5)**, more in-domain data, and early-stop
on normalized WER. (Training works on Apple-MPS; ~1.5 s/step, ~10 min/epoch
for 1000 utts. The ONNX export must run after `model.to("cpu")`.)

## Licensing note

`dragonSwing/wav2vec2-base-vietnamese` is Apache-2.0 (fine). VIVOS is public
research data; **ESC-50 is CC-BY-NC** — using it as *training augmentation*
for a commercial model is a gray area, so for the shipping model train
without `--noise` (or with a commercially-licensed noise source) and keep the
ESC-50 arm for internal robustness measurement only.

## If it plateaus

- More in-domain data (Common Voice VI, factory recordings) beats more epochs.
- Try a larger Apache-2.0 base (`CuongLD/wav2vec2-large-xlsr-vietnamese`) —
  better ceiling, bigger/slower.
- If no Apache-2.0 CTC reaches the bar, keep Whisper as the shipped backend
  and treat CTC as an internal benchmark (the `Asr` enum already supports both).
