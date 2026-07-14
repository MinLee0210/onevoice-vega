# models

Exported `.onnx` files and QNN context binaries — the handoff artifact
between `notebook/` (Python export) and `src/` (Rust runtime). This is
the language-agnostic interface point from CLAUDE.md.

Binaries are gitignored (too large for git); only this README and the
per-stage manifest files are tracked. Regenerate via `notebook/05_quantization_export/`.

```
models/
├── denoise/
├── vad/
├── asr/       one subfolder per language (vi/, en-zh-ko/, ...)
├── mt/        one subfolder per language pair (vi-en/, vi-zh/, vi-ko/, ...)
└── tts/       one subfolder per language
```

Naming convention: `<stage>-<lang(s)>-<precision>.onnx`, e.g.
`asr-vi-w8a16.onnx`, `mt-vi-en-int8.onnx`. Keep a `MANIFEST.md` (or
`.json`) alongside each artifact recording: source checkpoint, export
date, quantization scheme, and eval metrics (WER/BLEU) at export time —
this is what `src/` crates read to know what they're loading.
