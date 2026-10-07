# experiments

Python side: model evaluation, fine-tuning, quantization, ONNX export.
Nothing here runs on-device — the output artifact of this directory is
`.onnx` / QNN context binaries dropped into `models/`, which `src/`
consumes. Keep the real-time loop out of Python entirely (see AGENTS.md
→ Tooling / language split).

Flat directory of `.ipynb` files, numbered by pipeline stage:

```
01_denoise_vad.ipynb          RNNoise/DeepFilterNet3, Silero VAD eval
02_asr.ipynb                  PhoWhisper / Whisper eval, WER-vs-SNR curves
03_mt.ipynb                   Opus-MT / MADLAD / VinAI-Translate eval, BLEU
04_tts.ipynb                  Piper (and Kyutai Pocket TTS if evaluated)
05_quantization_export.ipynb  AI Hub Workbench quantization, ONNX export,
                               QNN context binary generation
06_benchmarks.ipynb           end-to-end latency, glossary term accuracy,
                               cross-stage eval
```

Exported artifacts go to `models/<stage>/`, named and versioned per
`models/README.md`. Eval sets/termbase come from `data/`.

## Setup

Use the `audio-lab` conda env for everything here — not base/system
Python:

```sh
conda activate audio-lab
pip install -r requirements.txt
python -m ipykernel install --user --name audio-lab --display-name "Python (audio-lab)"
```

When creating or running a notebook, make sure its kernel is set to
`audio-lab` (Jupyter: Kernel -> Change Kernel -> Python (audio-lab)),
so results are reproducible across the team. To execute headlessly:

```sh
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=audio-lab 03_mt.ipynb
```
