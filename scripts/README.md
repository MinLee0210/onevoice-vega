# scripts

Glue scripts that don't belong in a notebook (one-shot, CLI-driven) or
in a Rust crate (build/deploy tooling, not runtime logic). All three
below are real and tested — run them from the repo root.

## First time in this repo

```sh
scripts/setup_models.sh   # populate models/ (needs the audio-lab conda
                           # env active for the mt step, see below)
scripts/run_pipeline.sh   # build --release, run against the committed
                           # eval fixture, see the real ASR/MT output
```

## `setup_models.sh` — populate `models/<stage>/`

`vad`/`asr`/`tts` are pre-made ONNX already on the HF Hub — plain
downloads, only needs the `huggingface_hub` Python package, no Jupyter.
`mt` has no pre-made ONNX (Helsinki-NLP only ships PyTorch), so it
still needs a real export — this shells out to `run_notebook.sh 03_mt`,
which does need the full `audio-lab` env.

```sh
scripts/setup_models.sh              # everything (vad, asr, tts, mt)
scripts/setup_models.sh vad asr tts  # skip the notebook-based mt step
scripts/setup_models.sh mt           # just re-run the mt export
scripts/setup_models.sh vad          # just one stage

# Point at a specific python if huggingface_hub isn't on your default one
ONEVOICE_HF_PYTHON=/path/to/audio-lab/bin/python3 scripts/setup_models.sh vad
```

## `run_pipeline.sh` — build and run the real `onevoice` CLI binary

Requires `models/` to already be populated (`setup_models.sh` above).

```sh
scripts/run_pipeline.sh                              # release build,
                                                       # runs the eval fixture
scripts/run_pipeline.sh my_input.wav my_output.wav    # your own WAV
                                                       # (mono, 48kHz)

# extra onevoice flags go after --
scripts/run_pipeline.sh my_input.wav my_output.wav -- --source en --target vi

ONEVOICE_PROFILE=debug scripts/run_pipeline.sh        # debug build instead
ONEVOICE_LOG=warn scripts/run_pipeline.sh             # quieter logs
ONEVOICE_LOG=debug scripts/run_pipeline.sh            # verbose (incl. ort internals)
```

## `run_notebook.sh` — headless `experiments/*.ipynb` execution

Executes in place via the `audio-lab` Jupyter kernel (see
`experiments/README.md` -> Setup) and fails loudly if any cell errors.

```sh
scripts/run_notebook.sh                    # all notebooks, in order
scripts/run_notebook.sh 02_asr             # just one (.ipynb optional)
scripts/run_notebook.sh 01_denoise_vad 03_mt  # a specific few, in order

ONEVOICE_NB_TIMEOUT=2400 scripts/run_notebook.sh 02_asr   # longer timeout
ONEVOICE_NB_KERNEL=my-other-env scripts/run_notebook.sh   # different kernel
```

## Expected as the project reaches Phase 3 (prototype build)
- AI Hub Workbench export/quantization CLI wrappers
- QNN SDK conversion (`.onnx` -> QNN context binary)
- Deploy-to-device (push binary + models to Rubik Pi 3 over adb/scp)
- Benchmark harnesses (latency, WER-vs-SNR sweep) invoked from CI or
  manually against real hardware
