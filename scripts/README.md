# scripts

Glue scripts that don't belong in a notebook (one-shot, CLI-driven) or
in a Rust crate (build/deploy tooling, not runtime logic). All three
below are real and tested — run them from the repo root.

## First time in this repo

```sh
scripts/setup_models.sh   # download the prebuilt model bundle (curl + tar,
                           # no Python needed) and extract into models/
scripts/run_pipeline.sh   # build --release, run against the committed
                           # eval fixture, see the real ASR/MT output
```

## `setup_models.sh` — populate `models/<stage>/`

Two ways to get the models:

**Bundle (default)** — download one prebuilt `.tar.gz` and extract it.
Needs only `curl` + `tar`; no conda/torch/jupyter/HF-Hub. This is the
easy path for anyone who just wants to run the pipeline.

```sh
scripts/setup_models.sh                     # download + extract the bundle
scripts/setup_models.sh bundle              # same, explicit
ONEVOICE_MODELS_URL=<url> scripts/setup_models.sh   # non-default bundle host
```

**Rebuild** — fetch/export every model from its original source. Needs
the full `audio-lab` env (`huggingface_hub` for vad/asr; plus
torch/transformers/uroman for tts/ko's ONNX export; plus jupyter for the
mt/vi-en export). This is how the bundle itself is produced.

```sh
scripts/setup_models.sh rebuild             # all stages from source
scripts/setup_models.sh rebuild vad asr     # just these stages
scripts/setup_models.sh vad asr             # shorthand for `rebuild vad asr`

ONEVOICE_HF_PYTHON=/path/to/audio-lab/bin/python3 scripts/setup_models.sh rebuild
```

## `pack_models.sh` — build the distributable bundle

Tars the current `models/` tree into `dist/onevoice-models.tar.gz` and
prints the `gh release create` command to publish it. Run this after a
`setup_models.sh rebuild` when you want to refresh the downloadable
bundle that the default `setup_models.sh` pulls.

```sh
scripts/pack_models.sh                 # -> dist/onevoice-models.tar.gz
scripts/pack_models.sh my-bundle.tar.gz
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
