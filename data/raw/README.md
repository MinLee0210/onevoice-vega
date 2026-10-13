# data/raw — downloaded samples (gitignored except this file)

Small, curated **samples** for the OneVoice use-cases. The audio/text itself
is gitignored (`data/raw/**`); regenerate any time with:

```sh
uv run --no-project --python 3.12 --with datasets --with soundfile \
  python scripts/fetch_samples.py --speech-n 5 --parallel-n 100 --noise-n 12
```

## Layout

```
data/raw/
├── speech/<vi|en|zh|ko>/   N short utterances (16 kHz mono WAV) + transcripts.tsv
├── parallel/               VI/EN/ZH/KO sentence pairs (OPUS-100): en-vi, en-zh, en-ko
│                           + phomt-test.vi-en.tsv (higher-quality VI-EN benchmark)
└── noise/                  industrial/machinery clips (ESC-50): chainsaw, engine,
                            hand_saw, helicopter, siren, vacuum_cleaner,
                            washing_machine, train
```

## Use them offline in the eval scripts

The harnesses accept these local samples (no HF streaming needed):

```sh
# MT on the local PhoMT VI-EN slice (all 6 legs also work from the OPUS TSVs)
python scripts/evaluate_mt.py --legs vi-en --n 200 --parallel-dir data/raw/parallel --phomt

# ASR on the local VI speech samples
python scripts/evaluate_asr.py --models vinai/PhoWhisper-base --num-samples 5 \
  --speech-dir data/raw/speech/vi

# Noise robustness with the local machinery clips (no ESC-50 download)
python scripts/evaluate_asr_noise.py --speech-dir data/raw/speech/vi --noise-dir data/raw/noise
```

## Sources & licenses (verify before redistributing)

| Use-case | Dataset | Notes |
|---|---|---|
| VI speech | `quocanh34/viet_vivos` (VIVOS) | public Vietnamese research corpus |
| EN/ZH/KO speech | `google/fleurs` (`en_us`, `cmn_hans_cn`, `ko_kr`) | FLEURS, CC-BY-4.0 |
| Parallel sentences | `Helsinki-NLP/opus-100` (`en-vi`, `en-zh`, `en-ko`) | OPUS, mixed per-corpus licenses |
| VI-EN benchmark | `tuandunghcmut/PhoMT` (test split) | PhoMT (EMNLP 2021); original is CC-BY-NC-ND — internal eval only |
| Industrial noise | `ashraq/esc50` (ESC-50) | **CC-BY-NC** — internal eval only |

## Why these

- **speech/**: drives ASR (PhoWhisper/Whisper-small), VAD, and TTS round-trips;
  four languages × the same use-case pairing the device supports.
- **parallel/**: MT quality/glossary work (the VI↔EN/ZH/KO directions; the
  ZH/KO pairs pivot through EN).
- **noise/**: the noise-robustness eval (`scripts/evaluate_asr_noise.py`) —
  real machinery sounds, not just AWGN.

For bulk corpora (full VIVOS/Common Voice/PhoMT/MUSAN) keep them here too —
fetch is scripted, never committed. See `data/README.md`.
