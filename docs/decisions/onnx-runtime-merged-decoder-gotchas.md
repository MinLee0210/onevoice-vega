# ONNX Runtime merged-decoder gotchas (seq2seq KV-cache decode)

Found while building the MT stage (vi-en, `experiments/03_mt.ipynb` +
`src/stages/mt.rs`). Recorded here because every other seq2seq/autoregressive
stage that gets exported the same way (ASR's decoder, in particular) will
hit the same shape of graph and can hit the same bugs.

## Context

`optimum`'s ONNX export of a HuggingFace seq2seq model (`text2text-generation-with-past`
task) produces an encoder + a **merged** decoder (`decoder_model_merged.onnx`)
with a `use_cache_branch` boolean input that switches between two `If`
branches: "no cache yet, compute everything fresh" (step 0) and "reuse
cached KV" (step 1+).

## Gotcha 1 — cross-attention KV outputs are only valid on step 0

The merged decoder's `present.*.encoder.key`/`present.*.encoder.value`
outputs (the cross-attention KV cache) are correctly populated **only**
when `use_cache_branch = false` (the first call). On later calls
(`use_cache_branch = true`), those same output names come back with a
malformed/garbage shape — re-reading them every step and feeding them
back in corrupts the KV cache and the decoder graph fails a few steps in
with an ONNX Runtime shape-broadcast error inside the exported `If` node.

**Fix:** capture `present.*.encoder.*` once, right after the step-0 call,
and hold those tensors fixed for the rest of decoding. Never overwrite
them from later steps' outputs. `optimum`'s own `ORTDecoderForSeq2Seq.forward`
(`optimum/onnxruntime/modeling_seq2seq.py`) does exactly this — it's just
not obvious from the ONNX graph's I/O signature alone, and the naive
"read every declared output every step" implementation looks correct
until it silently breaks a few tokens into generation.

`present.*.decoder.*` (self-attention KV) has no such restriction — that
one is safe to re-read and grow every step, as expected.

## Gotcha 2 — default execution provider can silently break zero-length tensors

Step 0 needs dummy past-KV inputs shaped `[batch, heads, 0, head_dim]`
(zero-length sequence axis) for every `past_key_values.*` input the graph
declares, including the ones the `If` node's false-branch won't actually
read.

On the machine this was developed on (macOS/arm64), the Python
`onnxruntime` package auto-selects `CoreMLExecutionProvider` over
`CPUExecutionProvider` when no `providers=[...]` is passed to
`InferenceSession`. `CoreMLExecutionProvider` mishandles these
zero-length tensors — hard failure inside the exported `If` node's fused
`MatMul`, even though the exact same feed dict replayed through a
CPU-only session succeeds. Disabling ONNX Runtime graph optimizations did
*not* fix it; forcing `providers=["CPUExecutionProvider"]` did.

**Not currently a Rust concern:** the `ort` crate only compiles in the
execution providers whose Cargo feature is explicitly enabled (this
project only enables the default, i.e. CPU) — so `src/stages/mt.rs`
never had this problem. Worth re-checking if a `coreml`/`qnn`/other EP
feature gets turned on later: pin the provider explicitly rather than
trusting whatever the runtime picks by default.

## How this was found

Isolated by replaying `optimum`'s exact captured ONNX feed dict (via a
monkey-patched `session.run`) through a fresh plain `InferenceSession` —
it succeeded with byte-identical inputs, which ruled out a shape/dtype
bug in the manual reimplementation and pointed at either the execution
provider or the multi-step reuse of the cross-attention cache. Full
walk-through, including the working manual greedy-decode loop, is in
`experiments/03_mt.ipynb` section 6.
