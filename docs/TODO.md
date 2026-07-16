# TODO

## Phase 2 — technical spec submission (due Jul 2026)
- [ ] Verify OneVoice IP/equity terms in FAQ (compare to QVIC precedent)
- [x] Write technical spec draft (docs/proposals/) — done outside this repo
- [x] Write business viability pitch (docs/proposals/) — done outside this repo
- [x] Competitor landscape research — see
      docs/decisions/competitor-landscape-2026-07.md. Key finding: no
      genuinely-offline industrial PTT translator exists among
      competitors found (Weavix Walt, Relay both require LTE/WiFi,
      degrade to plain voice radio when offline) — "zero connectivity
      required" is a real, differentiated claim, not just a spec.
- [ ] **Directly verify the Vbee/Qualcomm offline-edge-device claim**
      (CLAUDE.md → Contest/business context) — this research could not
      independently confirm it; everything findable about Vbee online
      is their cloud TTS SaaS product, not an edge device. Don't repeat
      the claim in Phase 2/3 materials without confirming it first.
- [ ] Business model needs an explicit answer for Phase 2/3 pitch —
      every competitor found monetizes via subscription/data-plan tied
      to cloud connectivity, which OneVoice's offline architecture
      structurally can't do. See competitor-landscape doc's
      recommendation #6 (hardware margin? enterprise licensing? paid
      glossary/vocab-pack updates?).

## Open decisions to close out (see docs/decisions/)
- [ ] MADLAD-single-model vs. Opus-MT-per-pair — benchmark on device
- [~] PhoWhisper post-quantization WER — partially validated: the
      *pre-made* int8 dynamic-quantized ONNX (huuquyet/PhoWhisper-base,
      CPU) shows zero measurable WER loss vs. fp32 on a 10-sample VIVOS
      slice. Still open: this isn't the same thing as W8A16 on the
      actual Hexagon NPU via AI Hub Workbench — see the QCS6490 Whisper
      deployment risk below, that path isn't proven yet.
- [ ] VI↔KO / VI↔ZH: direct vs. EN-pivot — measure quality delta
- [x] Rust vs. C++ for runtime — `ort` crate confirmed working (builds
      clean, no cmake dependency, real ONNX inference verified) on CPU;
      see docs/decisions/onnx-runtime-merged-decoder-gotchas.md. QNN
      execution provider still untested on real Rubik Pi 3 hardware —
      that part of the smoke test is still open.
- [ ] Factory noise data — synthesize VIVOS/Common Voice + MUSAN mixes
- [ ] **QCS6490 Whisper/PhoWhisper deployment risk** — CLAUDE.md assumes
      "Whisper-Small (AI Hub, pre-quantized)" is a ready path; it isn't,
      as of writing. See
      docs/decisions/qcs6490-whisper-deployment-risk.md — needs a plan
      before Phase 3 hardware bring-up (options: AIMET-ONNX from source,
      re-check AI Hub Workbench for updates, fall back to the QNN EP
      directly against a non-AI-Hub quantized ONNX like the ones used
      in experiments/02_asr.ipynb).

## MT stage (vi-en) — baseline done, follow-ups
- [x] Opus-MT vi-en: PyTorch baseline, ONNX export, tokenizer.json
      conversion, real `ort`-based Rust port in src/stages/mt.rs — all
      validated end-to-end (see experiments/03_mt.ipynb, tests in
      src/stages/mt.rs). Gotchas hit along the way logged in
      docs/decisions/onnx-runtime-merged-decoder-gotchas.md.
- [ ] INT8 quantize (current export is fp32, ~500MB — over the
      ~300MB/pair memory budget target)
- [ ] Repeat for the other 5 directions (en-vi, vi-zh, zh-vi, vi-ko, ko-vi)
- [x] **Wire the glossary engine into MT decode (2026-07-16)** — done via
      decoder logit biasing, not placeholder substitution (that was
      tested and rejected earlier — see
      docs/decisions/glossary-runtime-enforcement-2026-07.md for both).
      `MtEngine::translate` now runs a cheap unbiased pass first and
      only re-decodes with a bias toward missing glossary terms when
      the unbiased pass actually got one wrong. Validated in
      experiments/03_mt.ipynb (sections 7-7d, three real bugs found and
      fixed along the way) then ported to `src/stages/mt.rs`, with
      exact-match Rust tests against the same reference outputs. Result:
      50% -> 100% term accuracy on a 10-sentence vi-en carrier set, at a
      real fluency cost on corrected sentences (dropped articles, minor
      grammar roughness) — acceptable for now, not native-speaker
      reviewed. vi-en only; the other 5 pairs have no MT model ported
      yet. LoRA fine-tune (CLAUDE.md's original plan) is still the
      better long-term fix for fluency, not started.
- [ ] Port `MtEngine` to the QNN execution provider once running on
      real Rubik Pi 3 hardware (currently CPU-only)

## ASR stage (vi, en-zh-ko) — baseline done, follow-ups
- [x] PhoWhisper-base (VI) and Whisper-small (EN/ZH/KO): used pre-made
      ONNX exports directly, no conversion — huuquyet/PhoWhisper-base
      and onnx-community/whisper-small both already ship
      encoder/decoder ONNX (fp32 *and* several pre-quantized variants)
      and a proper fast tokenizer.json. Validated against PyTorch and
      against ground truth (WER 2.2% VI/10 VIVOS samples, ONNX
      int8 == fp32 exactly; EN WER ~26-27% on 5 LibriSpeech-dummy
      samples but that's mostly casing/punctuation normalization
      noise against un-normalized references, not real errors — see
      experiments/02_asr.ipynb).
- [x] Rust port (`src/stages/asr.rs`) — done. Used the `mel_spec` crate
      for the log-mel feature extraction, but its output didn't actually
      match Whisper's preprocessing out of the box (two real bugs found
      and fixed: natural log instead of log10, zero-padding instead of
      reflect-padding at the signal edges — see
      docs/decisions/mel-spec-crate-whisper-compat-bugs.md). Validated
      numerically against `WhisperFeatureExtractor`, then end-to-end —
      the Rust-computed features fed through the real ONNX model produce
      the exact expected transcription. Real test in src/stages/asr.rs
      against a committed WAV fixture, passes.
- [x] **Latency concern from the 30s-padded encoder — now measured, not
      just theorized** (2026-07-16, CPU dev machine, NOT the Hexagon
      NPU target): for a ~1.7s utterance producing 16 output tokens,
      encoder ≈130-310ms, decoder loop ≈250-405ms — the autoregressive
      **decoder is the bigger cost of the two**, not the fixed-size
      encoder pass, confirming CLAUDE.md's own "ASR decoder is the
      bottleneck" assumption with real numbers for the first time. Full
      ASR stage ≈420-720ms; MT ≈90-240ms; denoise+VAD+capture <15ms
      combined. Through-MT total ≈530-970ms (real run-to-run variance
      on a shared dev machine, not device-representative). This is real
      evidence for prioritizing VietASR (CTC, no decode loop) below —
      removing the decoder loop entirely could plausibly save more than
      removing the 30s-padding does.
- [ ] ZH/KO not actually transcription-tested yet (only EN via notebook,
      only VI via the real Rust port) — low priority, same model/code
      path already validated for EN and VI respectively.
- [ ] Resolve the QCS6490 deployment risk above before committing to
      this artifact for the real device.
- [ ] **Prototype VietASR (13M param CTC model) as a VI alternative** —
      found in the 2026-07-15 research pass, see
      docs/decisions/pipeline-improvement-research-2026-07.md. No
      autoregressive decode loop at all (CTC, not seq2seq) — directly
      attacks the latency bottleneck CLAUDE.md calls out. WER not yet
      confirmed vs. PhoWhisper, ONNX export not yet confirmed either —
      needs a real validation pass before trusting it, not a swap-in.
- [ ] Bookmark Qwen3-ASR-0.6B (Apache-2.0, 52 languages incl. one model
      for EN/ZH/KO+VI) for later — too large for the current budget now
      (~8x PhoWhisper-base), but worth rechecking if a smaller
      distillation ships.

## TTS stage (vi) — license research done, implementation next
- [x] Found Piper VI voices (rhasspy/piper-voices on HF) — but two
      license issues found before touching implementation, see
      docs/decisions/piper-vi-voice-license-trap.md:
      1. Only `vi_VN-vais1000-medium` (CC BY 4.0) is commercially
         usable; `vivos-x_low` is CC-BY-NC-SA (blocked), `25hours_single`
         is unlicensed (unusable). Use vais1000-medium — also happens
         to be the best quality tier available.
      2. Piper's phonemizer (`piper-phonemize`/`espeak-ng`) is GPL-3.0.
         Unresolved question (never answered by upstream maintainers
         either): does embedding it in our on-device binary GPL-taint
         the whole product? Not a blocker for prototyping, is a blocker
         for the commercialization pitch — needs a real answer (legal
         read or a non-GPL Vietnamese phonemizer) before Phase 3
         architecture commits to it.
- [x] Validate vais1000-medium synthesis quality (notebook) — done,
      see experiments/04_tts.ipynb (round-trip TTS→ASR intelligibility
      check, 3/4 test sentences verbatim)
- [x] Research non-GPL phonemizer alternatives for Vietnamese — dead
      end, see docs/decisions/pipeline-improvement-research-2026-07.md.
      OpenPhonemizer doesn't support VI and is archived; Kyutai Pocket
      TTS doesn't support VI yet; sherpa-onnx's own fix is
      external-process isolation, not a real license resolution. The
      GPL question in piper-vi-voice-license-trap.md still needs an
      actual legal read, not more searching.
- [ ] Rust port — deliberately not started (2026-07-16): would need the
      GPL-3.0 phonemizer in `src/`'s real dependencies, which is a
      license decision for the shipped binary, not something to do
      silently while chasing an end-to-end demo. See the pipeline
      wiring section below — the binary now stops cleanly at this
      boundary with a clear message instead of a raw panic.
- [ ] Human-listener QA pass on the synthesized audio (only an ASR
      round-trip proxy has been done so far, see experiments/04_tts.ipynb)

## Denoise + VAD stages — implemented, follow-ups
- [x] Denoise: `nnnoiseless` crate (pure Rust RNNoise reimplementation,
      BSD-3-Clause, weights compiled in — no ONNX file, unlike every
      other stage). Real implementation in src/stages/denoise.rs, not
      just notebook validation — RNNoise's fixed 48kHz/480-sample-frame
      requirements meant capture now needs to be 48kHz, with a
      windowed-sinc resample down to 16kHz on the way out for
      VAD/ASR. 2 unit tests (length/rate check, lowpass frequency
      response check), both pass, no real-model file needed.
- [x] VAD: official Silero VAD ONNX (onnx-community/silero-vad, MIT),
      no conversion needed. Real implementation in src/stages/vad.rs
      (stateful, 512-sample chunks, trims leading/trailing silence
      only). Integration test uses a real committed speech sample
      (`data/eval/vad_sample_vi.wav`) — a synthetic tone doesn't work,
      Silero VAD is a trained speech/non-speech classifier, not an
      energy detector (found this the hard way, see
      docs/notes/NOTE.md). Test passes against the real model.
- [x] `experiments/01_denoise_vad.ipynb` written and run (was documented
      in experiments/README.md's expected structure but never actually
      created until now). VAD section is a full real validation, same
      pattern as ASR/TTS. Denoise section is deliberately
      documentation-first: tried `pyrnnoise` (real C RNNoise via Python)
      and the `deepfilternet` PyPI package as potential Python-side
      baselines, both are broken in this environment (`pyrnnoise`: an
      unmaintained transitive dependency, `av.option` doesn't exist
      anymore; `deepfilternet`: imports `torchaudio.backend`, an API
      `torchaudio` has since removed, no compatible older `torchaudio`
      available) — both failures are captured directly in the notebook,
      not hidden.
- [ ] DeepFilterNet3 as the denoise upgrade path (CLAUDE.md: "better on
      machinery noise") — a legitimate pre-exported ONNX exists
      ([soniqo/DeepFilterNet3-ONNX](https://huggingface.co/soniqo/DeepFilterNet3-ONNX),
      MIT, validated against PyTorch to ~1e-6), found and inspected in
      experiments/01_denoise_vad.ipynb, but its input is DeepFilterNet's
      own ERB + STFT spectral features, not raw audio — using it means
      implementing that feature extraction (and the inverse ERB-mask +
      deep-filtering reconstruction on the output side) from scratch,
      comparable in scope to ASR's mel-spectrogram work. Not started;
      not yet evaluated against nnnoiseless's RNNoise on the 75-85dB
      factory noise target either.
- [ ] Quantized Silero VAD variants (int8/uint8/q4/fp16/bnb4 all exist
      in the source repo, same pattern as Whisper) — not yet evaluated.
- [ ] Neither stage ported to the QNN execution provider yet (CPU only,
      same status as MT/ASR).

## End-to-end pipeline wiring — done through MT, stops cleanly at TTS
- [x] `src/stages/audio_io.rs` implemented for real — reads/writes WAV
      files rather than live mic/speaker (no Rubik Pi in hand yet), but
      it's the real `AudioCapture` used by `Pipeline`, not a mock.
      `cargo run -- <input.wav> <output.wav>` now runs the actual
      `Pipeline::run_once()`: capture → denoise → VAD trim → ASR → MT
      all execute for real, logged at INFO level. Confirmed with a real
      48kHz VI test fixture (`data/eval/ptt_sample_vi_48k.wav`,
      resampled from the existing VIVOS sample) — produced a correct,
      genuine ASR transcription and MT translation through the real
      binary, not a test harness.
- [x] TTS is the deliberate stopping point, not an accident: turned its
      `todo!()` into a clean `Result::Err` (was a raw panic before) so
      the binary now exits gracefully with a clear message pointing at
      docs/decisions/piper-vi-voice-license-trap.md, having already
      logged the transcription/translation it did produce. Chose not to
      add the GPL phonemizer to get real audio out — that's a
      real license decision for a shipped binary, not one to make
      silently while chasing a milestone.
- [x] Real CLI (`clap`, derive API) — `cargo run -- --help` for full
      usage. `--source`/`--target` (default vi/en) and
      `--configs-dir`/`--models-dir`/`--data-dir` overrides, on top of
      the positional input/output WAV paths.
- [x] Per-stage latency instrumentation added and run (see the ASR
      section above for the actual numbers) — this now measures
      performance, not just composition. Still CPU-dev-machine-only,
      not the Hexagon NPU target hardware.
- [ ] Live mic/speaker capture (cpal, gated on a real PTT button) once
      hardware exists — `audio_io.rs`'s file-based implementation is
      the one piece designed to be swapped out, everything else in
      `Pipeline` stays the same.
- [x] **`scripts/` populated (2026-07-16)** — three real, tested scripts,
      not scaffolding: `run_notebook.sh` (headless notebook execution,
      renamed from `run-experiments.sh`), `run_pipeline.sh` (builds
      `--release` and runs the real `onevoice` binary — the intended
      "run it like a real CLI app" entry point), `setup_models.sh`
      (downloads the 4 pre-made-ONNX stages directly from the HF Hub,
      no Jupyter needed; shells out to `run_notebook.sh 03_mt` for the
      one stage — MT — that still needs a real `optimum` export).
- [ ] `RUST_LOG=info` output is very noisy — `ort`'s own session/arena
      logging is emitted at INFO alongside `onevoice`'s own spans,
      burying the actual pipeline logs in ONNX Runtime allocator
      chatter. Worth filtering `ort` down to WARN while keeping
      `onevoice`'s spans at INFO, not yet done.

## Future ideas (not scoped yet)
- [ ] TTS voice cloning — let the output voice match the speaker instead
      of a fixed Piper voice per language. Check feasibility on-device
      (NPU budget, latency vs. the ≤1.2s target) before committing;
      likely a post-Phase-2 stretch feature, not baseline.
      Candidate: VoxCPM (https://github.com/OpenBMB/VoxCPM) — evaluate
      license, model size, and on-device (QNN/ONNX) feasibility.
