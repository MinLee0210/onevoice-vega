# models

Exported `.onnx` files and QNN context binaries — the handoff artifact
between `experiments/` (Python export) and `src/` (Rust runtime). This is
the language-agnostic interface point from CLAUDE.md.

Binaries are gitignored (too large for git); only this README and the
per-stage manifest files are tracked. Regenerate via
`scripts/setup_models.sh` (plain HF Hub downloads for `vad`/`asr`/`tts`;
shells out to `scripts/run_notebook.sh 03_mt` for `mt`, which still
needs a real optimum export — see that script's header and
`docs/TODO.md`).

```
models/
├── denoise/   no model file — RNNoise's weights are compiled into the
│              `nnnoiseless` Rust crate, see denoise/MANIFEST.md
├── vad/
├── asr/       one subfolder per language (vi/, en-zh-ko/, ...)
├── mt/        one subfolder per language pair (vi-en/, vi-zh/, vi-ko/, ...)
└── tts/       one subfolder per language
```

Filenames follow each artifact's `optimum`/upstream export convention
as-is (`encoder_model.onnx`, `decoder_model_merged.onnx`, `model.onnx`,
...) rather than a project-invented scheme — see each stage's
`MANIFEST.md` for exactly what's there. Keep a `MANIFEST.md` alongside
each artifact recording: source checkpoint, export/packaging date,
quantization scheme, and eval metrics (WER/BLEU) at export time — this
is what `src/stages/*` expects to find when it loads from
`models/<stage>/`.
