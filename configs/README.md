# configs

Declarative config consumed by both `notebook/` (eval/export) and
`src/` (runtime) — keep it data, not code, so both sides read the same
source of truth.

- `language-pairs.toml` — the 6 directions (VI↔EN, VI↔ZH, VI↔KO):
  which ASR/MT/TTS model each pair loads, lazy-load grouping for the
  ~2GB memory budget.
- `latency-budget.toml` — per-stage latency allocation against the
  1.2s (stretch 0.8s) release-to-first-audio target, for the runtime's
  instrumented timing / live latency HUD.
