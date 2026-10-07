# denoise

- License: BSD-3-Clause (`nnnoiseless` crate; RNNoise weights compiled in)

No model file here, deliberately — unlike every other stage.

RNNoise (CLAUDE.md's denoise baseline) never shipped an official ONNX
export; the Rust `nnnoiseless` crate (BSD-3-Clause) reimplements it
natively with the trained weights compiled directly into the crate. See
`src/stages/denoise.rs` module docs for the full rationale, including
why this stage also does its own 48kHz -> 16kHz resampling.

If DeepFilterNet3 (CLAUDE.md's "better on machinery noise" upgrade
path) is adopted later, that one *does* export to ONNX
(`enc.onnx`/`erb_dec.onnx`/`df_dec.onnx`, bundled as a tar.gz per its
own `export.py`) — those files would land here.
