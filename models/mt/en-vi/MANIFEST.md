# mt/en-vi

- Source checkpoint: Helsinki-NLP/opus-mt-en-vi
- License: Apache 2.0 (Opus-MT / Helsinki-NLP — safe to commercialize,
  unlike NLLB's CC-BY-NC; see CLAUDE.md license trap note)
- Exported: 2026-07-18
- Precision: fp32 (baseline export, not yet quantized — INT8 via AI Hub
  Workbench is the documented upgrade path in CLAUDE.md's model stack
  table; revisit before flashing to device, current size is ~500MB,
  well over the ~300MB/pair memory budget target)
- Decode: greedy (beam=1), matches CLAUDE.md's latency-driven decode
  strategy

Files:
- `encoder_model.onnx`
- `decoder_model_merged.onnx` (covers both the no-cache first step and
  the with-cache subsequent steps via `use_cache_branch`)
- `tokenizer.json` (Rust `tokenizers`-crate-compatible; derived from slow MarianTokenizer)
