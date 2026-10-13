# data

```
data/
├── glossary/   small, versioned VI/EN/ZH/KO parallel termbase (checked
│               into git — this is source, not a bulk dataset). Consumed
│               by experiments/03_mt.ipynb (LoRA fine-tune) and
│               src/stages/glossary.rs.
├── eval/       eval sets: PhoMT + domain test set (BLEU), WER-vs-SNR
│               noise-mixed sets. Small curated files may be tracked;
│               bulk audio is gitignored — keep a download/build script.
│               Includes `vad_sample_vi.wav` (~76KB, one real VIVOS
│               utterance, 16kHz) — a committed test fixture for
│               src/stages/vad.rs's integration test — and
│               `ptt_sample_vi_48k.wav` (same utterance, resampled to
│               48kHz) for end-to-end `cargo run` testing of the full
│               pipeline (RNNoise's fixed input rate, see
│               src/stages/denoise.rs). Neither is a bulk dataset.
└── raw/        raw corpora and curated samples (VIVOS/FLEURS speech,
                OPUS-100 parallel sentences, ESC-50 industrial noise).
                Gitignored — never commit raw audio. Populate with
                `python scripts/fetch_samples.py` (see raw/README.md).
```

Anything above a few MB belongs in `raw/` or `eval/` and stays out of
git; document how to regenerate it instead of committing it.
