# TODO

## Plan (2026-08, post-submission) — QUAVIO/Vega two carriers

The Phase 2 submission expanded the project from a single handheld into a
two-carrier platform: **Vega Hand** (PTT handheld, one-to-one) and
**Vega Air** (drone-mounted multilingual broadcast, one-to-many). Research on
both carriers is in
docs/decisions/vega-air-and-feature-research-2026-08.md. Immediate execution
order below; the first five are the do-now engineering steps (no hardware /
no new models needed), the rest are Phase 3+.

### First 5 steps (execute now)
1. [x] Add CI (GitHub Actions): `cargo fmt --check`, `cargo clippy -- -D warnings`,
      `cargo test`, `shellcheck` — encode the pre-commit checks so regressions
      can't merge silently.
2. [x] Fix `scripts/setup_models.sh` model-version mismatch (`models-v1` tag /
      `onevoice-models.tar.gz` vs. README's `models-v2` / `onevoice_models_v2.tar.gz`).
3. [x] VAD empty-utterance early-out in `src/pipeline.rs` — skip ASR/MT/TTS when
      Silero finds no speech (currently ASR pads silence to 30s and emits garbage).
4. [x] Extract shared `src/stages/seq2seq.rs` — dedup the identical KV-cache /
      tensor plumbing duplicated across `asr.rs` and `mt.rs` (LayerCache,
      tensor_input, extract_owned, argmax, present-state extraction).
5. [x] `DecoderSpec::from_session` — derive decoder num_layers/num_heads/head_dim
      from the ONNX graph instead of hardcoding 6/8/64. Fixes a latent bug:
      `asr/en-zh-ko` uses whisper-small (12 layers / 12 heads) but `asr.rs`
      hardcodes PhoWhisper-base's 6/8.

### From the 2026-10 deep research (scoring is BLEU+COMET, TTS MOS, RTF, latency; see docs/decisions/deep-research-2026-10.md)
Organisers' dates: prototype submission Sep 2026, field testing Oct 2026, finale Nov 2026.
- [ ] COMET + chrF + BLEU eval harness (Python, offline) and a small held-out
      vi↔en/zh/ko set — COMET is scored by the organisers and is unmeasured here.
- [ ] Replace Korean TTS (MMS-TTS is CC-BY-NC 4.0, see THIRD_PARTY_LICENSES.md) and
      run a native-speaker MOS check on all four voices.
- [ ] Distil a teacher MT (HY-MT1.5 / TranslateGemma, check licences) into Opus-MT
      or a LoRA using a hard-example glossary set.
- [ ] Airplane-mode audit: no network use anywhere (model downloads, ort, espeak data).
- [x] Real-noise eval (2026-10-12): `scripts/evaluate_asr_noise.py` mixes ESC-50
      industrial sounds (chainsaw/engine/siren/...) — PhoWhisper 11.62% at 5 dB
      (≤15% holds; AWGN was conservative). docs/reports/asr_noise_real.md.
- [ ] Noise eval still needs a no-denoise arm *in Rust* (Python RNNoise/DFN
      bindings broken; denoising can hurt ASR, arXiv 2512.17562).

### Then (Phase 3+, needs hardware / models / design)
- [ ] **Vega Air carrier** — broadcast-tuned TTS (slower rate, higher
      articulation for outdoor SPL), sequential VI→EN→ZH→KO broadcast loop,
      MAVLink/UART telemetry reader. Acoustic parity bar: ≥120 dB @1 m, ≤300 g
      (see drone-PA section of the research doc).
- [ ] **Vega Air two-way stretch** — ego-noise cancellation. Do NOT promise for
      Phase 3; research says prefer *conditioning* over *subtraction* and add a
      near-rotor reference mic. Train/eval on DroneAudioSet.
- [ ] **Hand feature pack (cheap, copyable)** — haptic/vibration feedback
      (unclaimed by all competitors), SOS trigger (long/triple-press PTT),
      multi-state LED semantics, high-SPL speaker + EQ, translation-log + USB
      export, lanyard/belt-clip.
- [x] Export the remaining MT directions (2026-10-12) — en-vi direct plus the
      EN-pivot legs (en-zh, zh-en, ko-en, en-ko); all 6 directions now wired
      via `MtChain` (VI↔ZH/KO pivot through EN). Quality measured in
      docs/reports/mt_quality.md. Caveat: the `tc-big-en-ko` checkpoint is
      broken (BLEU 0.1) so VI→KO is wired but not yet usable.
- [ ] Replace the broken en→ko model (or use M2M-100 for KO) before relying
      on VI→KO. Also COMET not yet measured (BLEU/chrF only).
- [ ] Prototype VietASR (13M CTC, no autoregressive loop) as VI latency fix.
- [ ] QNN/Hexagon smoke test (week 1 of Phase 3 hardware bring-up).
- [ ] **Test edge devices on Qualcomm AI Hub** — compile + profile every pipeline
      stage (VAD, ASR encoder/decoder, MT, TTS) on the hosted devices, starting with
      `Dragonwing RB3 Gen 2 Vision Kit` (QCS6490, our target; Rubik Pi 3 class), then
      the other Dragonwing/Snapdragon parts as fallbacks and comparison points
      (`Dragonwing Q-6690 MTP`, `IQ-9075 EVK`, `QCS8550 (Proxy)`, Snapdragon X). Produce a
      model × device matrix (inference ms, peak memory, context-binary size, compile
      status) and check it against `configs/latency-budget.toml`. Use calibration audio
      with noisy Vietnamese for full-INT8 (W8A16 does not compile on this HTP). Work in
      tooling already exists in `scripts/qaihub/benchmark_devices.py` and
      `benchmark_qnn.py`; run it and record results in `docs/reports/`. They feed the
      field-test device choice and the
      "Dragonwing-native" pitch. A hosted AI Hub run is not a substitute for running on the
      real Rubik Pi 3, so keep the on-device smoke test above.
- [ ] Business model answer (hardware margin / enterprise licensing / paid
      vocab packs) — still open, 25% of judging.

## Phase 2 — technical spec submission (submitted 03/08/2026)
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
      (AGENTS.md → Contest/business context) — this research could not
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
- [~] VI↔KO / VI↔ZH: direct vs. EN-pivot — **EN-pivot measured** (2026-10-12,
      docs/reports/mt_quality.md): VI→ZH ok (en-zh leg 46.5 BLEU), VI→KO
      blocked by broken tc-big-en-ko (BLEU 0.1). Direct models don't exist.
- [x] Rust vs. C++ for runtime — `ort` crate confirmed working (builds
      clean, no cmake dependency, real ONNX inference verified) on CPU;
      see docs/decisions/onnx-runtime-merged-decoder-gotchas.md. QNN
      execution provider still untested on real Rubik Pi 3 hardware —
      that part of the smoke test is still open.
- [x] **Factory noise evaluation (R10) (2026-07-18)** — implemented evaluation
      script (`scripts/evaluate_noise_robustness.py`) and generated a report
      (`docs/reports/asr_noise_robustness.md`) measuring PhoWhisper WER at 5dB/10dB/15dB SNR.
- [ ] **QCS6490 Whisper/PhoWhisper deployment risk** — AGENTS.md assumes
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
- [x] **INT8 quantize (2026-07-18)** — quantized model components to INT8
      (reducing encoder from 178MB to 44MB, and decoder from 307MB to 77MB,
      shrinking total MT model size from 485MB to 122MB, fitting well under the 300MB target).
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
      yet. LoRA fine-tune (AGENTS.md's original plan) is still the
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
      encoder pass, confirming AGENTS.md's own "ASR decoder is the
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
      attacks the latency bottleneck AGENTS.md calls out. WER not yet
      confirmed vs. PhoWhisper, ONNX export not yet confirmed either —
      needs a real validation pass before trusting it, not a swap-in.
- [x] **REJECTED: Prototype Zipformer Transducer as a VI ASR alternative** —
      Model candidate: `sherpa-onnx-zipformer-vi-30M-int8-2026-02-09`.
      Evaluated on 2026-07-18. Although it showed a 15x latency improvement
      (~20-30ms vs. ~418ms), the transcription accuracy on the standard evaluation
      fixtures (`vad_sample_vi.wav`) was unacceptable compared to PhoWhisper-base
      (e.g., transcribing *"nửa vòng trái đất hơn bảy năm"* as *"KHỔ SỞ VỚI HÀNH VI"*).
      Sticking with PhoWhisper as the baseline VI ASR model for accuracy.
- [ ] Bookmark Qwen3-ASR-0.6B (Apache-2.0, 52 languages incl. one model
      for EN/ZH/KO+VI) for later — too large for the current budget now
      (~8x PhoWhisper-base), but worth rechecking if a smaller
      distillation ships.


## TTS stage — all 4 languages real, Rust port done (2026-07-17)
The device is bidirectional, so it must speak all four target languages
(VI, EN, ZH, KO), not just Vietnamese. Stack: Piper for VI/EN/ZH
(VITS/ONNX, espeak-ng front-end), MMS-TTS for KO (VITS/ONNX, uroman
front-end — Piper has no Korean voice).
- [x] **VI** — Piper `vi_VN-vais1000-medium` (CC BY 4.0). Round-trip
      TTS→ASR: 3/4 sentences verbatim. See experiments/04_tts.ipynb
      sections 1-4. (Two VI license notes remain documented in
      docs/decisions/piper-vi-voice-license-trap.md: only vais1000-medium
      is commercially usable of the 3 VI voices; espeak-ng is GPL-3.0.
      **Neither blocks the hackathon** — GPL binds at commercial
      distribution, not a prototype; revisit at commercialization.)
- [x] **EN, ZH, KO (2026-07-16)** — added because the device translates
      both ways. EN = Piper `en_US-lessac-medium`, ZH = Piper
      `zh_CN-huayan-medium` (Mandarin), KO = `facebook/mms-tts-kor`
      (MIT, no espeak/GPL). All validated by round-trip TTS→Whisper-small
      ASR in experiments/04_tts.ipynb section 5: EN verbatim, ZH industrial
      sentence clean, KO essentially verbatim. Packaged into
      models/tts/{en,zh,ko}/. Korean needs `uroman` (Hangul→Latin) at the
      text front-end (tokenizer is is_uroman=True) and a one-time ONNX
      export (done in the notebook + inline in scripts/setup_models.sh);
      the MMS ONNX matches PyTorch to ~1e-4 with noise disabled.
- [x] **Rust port (`src/stages/tts.rs`), 2026-07-17.** `TtsEngine` is
      real for all 4 languages, not a stub — the pipeline runs end to
      end through actual synthesized audio for the first time (full
      button-release→audio latency ~550-600ms on CPU, well under
      AGENTS.md's ≤1.2s). VI/EN/ZH use the `piper-rs` crate (MIT) —
      handles VITS inference + espeak-ng phonemization internally, same
      `ort` version this project already depends on. KO (MMS) has no
      covering crate, so it's hand-rolled: the `uroman` crate (a from-
      scratch Rust reimplementation, verified byte-identical to Python's
      `uroman` output) + a direct port of
      `transformers.VitsTokenizer.prepare_for_tokenization`/`_tokenize`
      + raw `ort` inference — the ported tokenizer produces
      byte-identical `input_ids` to the real Python tokenizer (test in
      tts.rs), and end-to-end synthesis matches the exact sample count
      from experiments/04_tts.ipynb (deterministic since the ONNX export
      has noise disabled).
      Real build gotcha hit and fixed: `piper-rs`'s default
      `compile-espeak-intonations` feature fails on the crates.io
      package (a broken relative path in the vendored CMakeLists, not a
      genuinely missing file). Building with `default-features = false`
      fixes the build but also means espeak-ng's `phontab`/`phondata`/
      `phonindex` runtime data files aren't generated either (compiled
      by that same disabled step) — sourced instead from the `piper-tts`
      PyPI package's bundled copy (pure data, arch-independent) and
      shipped as `models/tts/espeak-ng-data/` via
      `scripts/setup_models.sh`. Full detail in `src/stages/tts.rs`
      module docs.
- [ ] Human-listener QA pass on the synthesized audio for all 4 languages
      (only an ASR round-trip proxy done so far, now through the real
      Rust path too, not just the Python notebook).
- [ ] ZH greeting round-trip was slightly garbled ("你好吗" → "一號嗎")
      while the industrial sentence was clean — recheck ZH voice quality
      on more sentences before trusting it for the demo.
- [ ] **Prototype license-clean and voice-cloning alternatives for Vietnamese TTS**:
      - **Viet-TTS** (`dangvansam/viet-tts`): Download ONNX model weights (`speech_tokenizer.onnx`, `speech_embedding.onnx`) and evaluate them to bypass the `espeak-ng` GPL-3.0 phonemizer dependency.
      - **VieNeu-TTS** (`pnnbao97/VieNeu-TTS`): Evaluate offline CPU-optimized voice cloning support using 3-8s of reference audio to match the speaker's voice.

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
- [ ] DeepFilterNet3 as the denoise upgrade path (AGENTS.md: "better on
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

## End-to-end pipeline wiring — done through TTS + playback (2026-07-17)
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
- [x] **TTS is no longer the stopping point (2026-07-17)** — the binary
      now runs the complete cycle and writes real synthesized audio to
      the output WAV, round-trip-ASR verified. Superseded the earlier
      "stop at text output, GPL is unresolved" call: per explicit
      direction, GPL/licensing isn't a build blocker for a hackathon
      prototype (it binds at commercial distribution, not a demo) — see
      docs/decisions/piper-vi-voice-license-trap.md, still the doc to
      read before commercialization. See the TTS stage section above
      for the Rust port itself.
- [x] Real CLI (`clap`, derive API) — `cargo run -- --help` for full
      usage. `--source`/`--target` (default vi/en) and
      `--configs-dir`/`--models-dir`/`--data-dir` overrides, on top of
      the positional input/output WAV paths.
- [x] Per-stage latency instrumentation added and run (see the ASR
      section above for the actual numbers) — this now measures
      performance, not just composition. Now covers the full cycle
      through TTS: button-release → synthesized-audio measured at
      ~550-600ms, comfortably under AGENTS.md's ≤1.2s target (and its
      0.8s stretch goal). Still CPU-dev-machine-only, not the Hexagon
      NPU target hardware — a real number on real target hardware is
      still an open question.
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
- [x] `RUST_LOG=info` output is very noisy — `ort`'s own session/arena
      logging is emitted at INFO alongside `onevoice`'s own spans,
      burying the actual pipeline logs in ONNX Runtime allocator
      chatter. Worth filtering `ort` down to WARN while keeping
      `onevoice`'s spans at INFO, completed in src/main.rs.

## Future ideas (not scoped yet)
- [ ] TTS voice cloning — let the output voice match the speaker instead
      of a fixed Piper voice per language. Check feasibility on-device
      (NPU budget, latency vs. the ≤1.2s target) before committing;
      likely a post-Phase-2 stretch feature, not baseline.
      Candidate: VoxCPM (https://github.com/OpenBMB/VoxCPM) — evaluate
      license, model size, and on-device (QNN/ONNX) feasibility.
