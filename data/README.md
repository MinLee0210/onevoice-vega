# data

```
data/
├── glossary/   small, versioned VI/EN/ZH/KO parallel termbase (checked
│               into git — this is source, not a bulk dataset). Consumed
│               by notebook/03_mt (LoRA fine-tune) and src/crates/glossary.
├── eval/       eval sets: PhoMT + domain test set (BLEU), WER-vs-SNR
│               noise-mixed sets. Small curated files may be tracked;
│               bulk audio is gitignored — keep a download/build script.
└── raw/        raw corpora (VIVOS, Common Voice VI, MUSAN, factory
                field recordings). Gitignored — never commit raw audio.
                Keep fetch scripts in scripts/, not the data itself.
```

Anything above a few MB belongs in `raw/` or `eval/` and stays out of
git; document how to regenerate it instead of committing it.
