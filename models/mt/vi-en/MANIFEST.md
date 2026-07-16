# mt/vi-en

- Source checkpoint: Helsinki-NLP/opus-mt-vi-en
- License: Apache 2.0 (Opus-MT / Helsinki-NLP — safe to commercialize,
  unlike NLLB's CC-BY-NC; see CLAUDE.md license trap note)
- Exported: 2026-07-16
- Precision: fp32 (baseline export, not yet quantized — INT8 via AI Hub
  Workbench is the documented upgrade path in CLAUDE.md's model stack
  table; revisit before flashing to device, current size is ~500MB,
  well over the ~300MB/pair memory budget target)
- Decode: greedy (beam=1), matches CLAUDE.md's latency-driven decode
  strategy
- Eval at export time: exact-match against PyTorch reference on 4 VI
  test sentences (1 general + 3 industrial-glossary-flavored) — see
  experiments/03_mt.ipynb section 3. No BLEU/domain benchmark run yet
  (tracked in docs/TODO.md).
- Known gap: generic model mistranslates domain terms (e.g. "xe nâng" /
  forklift -> "lift") — this is what the glossary engine + LoRA
  fine-tune are meant to fix, not yet applied to this artifact.

Files:
- `encoder_model.onnx`
- `decoder_model_merged.onnx` (covers both the no-cache first step and
  the with-cache subsequent steps via `use_cache_branch`)
- `tokenizer.json` (Rust `tokenizers`-crate-compatible; see notebook
  section 4 for how it was derived from the slow MarianTokenizer)
