# scripts

Glue scripts that don't belong in a notebook (one-shot, CLI-driven) or
in a Rust crate (build/deploy tooling, not runtime logic).

Expected as the project reaches Phase 3 (prototype build):
- AI Hub Workbench export/quantization CLI wrappers
- QNN SDK conversion (`.onnx` -> QNN context binary)
- Deploy-to-device (push binary + models to Rubik Pi 3 over adb/scp)
- Benchmark harnesses (latency, WER-vs-SNR sweep) invoked from CI or
  manually against real hardware

Nothing here yet — add scripts as these workflows solidify rather than
scaffolding unused ones.
