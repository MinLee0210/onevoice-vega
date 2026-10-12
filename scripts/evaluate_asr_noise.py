#!/usr/bin/env python3
"""ASR noise robustness with *real* industrial noise, not just AWGN.

The existing `evaluate_noise_robustness.py` (and the AGENTS "≤15% at 75 dB"
claim) uses white Gaussian noise. The roadmap calls for real noise — this
mixes ESC-50 industrial categories (chainsaw, engine, hand_saw, helicopter,
siren, vacuum_cleaner, washing_machine, train) into the VIVOS signals at
controlled SNR, alongside AWGN as the reference, and reports WER for each.

Reuses the model runner + normalization from `evaluate_asr.py`.

Env:
    uv run --no-project --python 3.12 --with torch --with "transformers>=5.13.0" \\
      --with jiwer --with datasets --with soundfile --with librosa \\
      python scripts/evaluate_asr_noise.py --num-samples 30 --snr 10 5 0

Outputs docs/reports/asr_noise_real.md.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import jiwer
import librosa
import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from evaluate_asr import build_runner, load_samples, normalize_text

INDUSTRIAL = {
    "chainsaw",
    "engine",
    "hand_saw",
    "helicopter",
    "siren",
    "vacuum_cleaner",
    "washing_machine",
    "train",
}


def add_awgn(signal: np.ndarray, snr_db: int) -> np.ndarray:
    power = float(np.mean(signal**2))
    if power == 0:
        return signal
    noise = np.random.normal(0, np.sqrt(power * 10 ** (-snr_db / 10.0)), len(signal))
    return signal + noise


def mix_noise(signal: np.ndarray, noise: np.ndarray, snr_db: int) -> np.ndarray:
    n = np.resize(noise, len(signal))
    ps = float(np.mean(signal**2))
    pn = float(np.mean(n**2))
    if ps == 0 or pn == 0:
        return signal
    n = n * np.sqrt((ps * 10 ** (-snr_db / 10.0)) / pn)
    return signal + n


def load_industrial_noise(sr: int = 16_000) -> list[np.ndarray]:
    ds = load_dataset("ashraq/esc50", split="train").cast_column("audio", Audio(decode=False))
    pool: list[np.ndarray] = []
    for s in ds:
        if s["category"] not in INDUSTRIAL:
            continue
        arr, src_sr = sf.read(io.BytesIO(s["audio"]["bytes"]), dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if src_sr != sr:
            arr = librosa.resample(arr, orig_sr=src_sr, target_sr=sr)
        pool.append(arr[: sr * 5])  # 5 s clips
    return pool


def main(model_id: str, n: int, snrs: list[int], out: Path) -> int:
    samples = load_samples(n)
    print(f"loaded {len(samples)} VIVOS samples")
    pool = load_industrial_noise()
    print(f"loaded {len(pool)} industrial noise clips from ESC-50")

    run = build_runner(model_id)
    rows = [
        f"# ASR noise robustness — real industrial noise ({model_id})\n",
        "| condition | SNR | WER |",
        "|---|---|---|",
    ]

    for kind in ("awgn", "industrial"):
        for snr in [None, *snrs]:
            np.random.seed(42)
            refs, hyps = [], []
            for i, s in enumerate(samples):
                audio, sr = s["audio"], s["sr"]
                if snr is None:
                    noisy = audio
                elif kind == "awgn":
                    noisy = add_awgn(audio, snr)
                else:
                    noisy = mix_noise(audio, pool[i % len(pool)], snr)
                refs.append(normalize_text(s["transcription"]))
                hyps.append(normalize_text(run(noisy, sr)))
            wer = jiwer.wer(refs, hyps)
            label = "clean" if snr is None else f"{snr} dB"
            cond = "—" if snr is None else kind
            rows.append(f"| {cond} | {label} | {wer * 100:.2f}% |")
            print(rows[-1], flush=True)

    rows.append(
        "\nESC-50 is CC-BY-NC (internal eval only). Industrial categories: "
        + ", ".join(sorted(INDUSTRIAL))
        + "."
    )
    out.write_text("\n".join(rows) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="vinai/PhoWhisper-base")
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--snr", nargs="*", type=int, default=[10, 5, 0])
    parser.add_argument("--out", type=Path, default=Path("docs/reports/asr_noise_real.md"))
    args = parser.parse_args()
    raise SystemExit(main(args.model, args.num_samples, args.snr, args.out))
