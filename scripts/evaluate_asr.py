#!/usr/bin/env python3
"""ASR A/B/C: WER + latency + RTF across candidate models on a VI test set.

Workstream 2 of docs/decisions/improvement-roadmap-2026-10.md. Compares
PhoWhisper-base against newer, smaller/multilingual options (Qwen3-ASR-0.6B)
on the same VIVOS slice, at clean and AWGN-degraded SNRs. Runs on CPU.

Models are run through the `transformers` ASR pipeline so each model's own
preprocessing is used. Qwen3-ASR needs a recent `transformers`.

Env (no GPU needed):
    uv run --no-project --python 3.12 --with torch --with transformers \\
      --with "transformers>=4.57" --with jiwer --with datasets --with soundfile \\
      python scripts/evaluate_asr.py --num-samples 20

Outputs a table to docs/reports/asr_ab.md.
"""

from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from eval_data import load_local_speech
from transformers import pipeline

DEFAULT_MODELS = ["vinai/PhoWhisper-base", "Qwen/Qwen3-ASR-0.6B-hf"]


def add_noise(signal: np.ndarray, snr_db: int | None) -> np.ndarray:
    if snr_db is None:
        return signal
    power_signal = float(np.mean(signal**2))
    if power_signal == 0:
        return signal
    power_noise = power_signal * (10 ** (-snr_db / 10.0))
    return signal + np.random.normal(0, np.sqrt(power_noise), len(signal))


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("|", " ")  # Wav2Vec2 CTC word delimiter
    for p in [".", ",", "?", "!", ";", ":", '"', "'"]:
        text = text.replace(p, "")
    return " ".join(text.split())


def load_samples(n: int) -> list[dict]:
    """VIVOS test slice, decoded with soundfile (avoids datasets' torchcodec
    dependency). Returns dicts with `transcription`, `audio` (float32 mono),
    `sr`."""
    vivos = load_dataset("quocanh34/viet_vivos", split="test", streaming=True)
    vivos = vivos.cast_column("audio", Audio(decode=False))
    out: list[dict] = []
    for _, s in zip(range(n), vivos):
        raw = s["audio"]
        if raw.get("array") is not None:  # already decoded by datasets
            arr, sr = np.asarray(raw["array"], dtype=np.float32), raw["sampling_rate"]
        else:  # decode the raw bytes ourselves
            arr, sr = sf.read(io.BytesIO(raw["bytes"]), dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        out.append({"transcription": s["transcription"], "audio": arr, "sr": sr})
    return out


def build_runner(model_id: str):
    """Return a callable(audio: np.ndarray, sr: int) -> transcript.

    Qwen3-ASR has a bespoke multimodal API (transformers >=5.13), everything
    else goes through the standard ASR pipeline."""
    if "qwen3-asr" in model_id.lower():
        import os
        import tempfile

        import soundfile as sf
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForMultimodalLM as _M
        except ImportError:  # pragma: no cover - older fallback
            from transformers import Qwen3ASRForConditionalGeneration as _M

        proc = AutoProcessor.from_pretrained(model_id)
        mdl = _M.from_pretrained(model_id).eval()

        def run_qwen(audio: np.ndarray, sr: int) -> str:
            import torch

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio.astype(np.float32), sr)
                path = f.name
            try:
                req = proc.apply_transcription_request(audio=path, language="vi")
                req = req.to(mdl.dtype)
                with torch.no_grad():
                    out = mdl.generate(**req, max_new_tokens=256, do_sample=False)
                gen = out[:, req["input_ids"].shape[1] :]
                return proc.decode(gen, return_format="transcription_only")[0]
            finally:
                os.unlink(path)

        return run_qwen

    pipe = pipeline("automatic-speech-recognition", model=model_id, device=-1)

    def run_pipe(audio: np.ndarray, sr: int) -> str:
        kwargs = {}
        if "whisper" in model_id.lower():
            kwargs["generate_kwargs"] = {"language": "vi", "task": "transcribe"}
        out = pipe({"array": audio.astype(np.float32), "sampling_rate": sr}, **kwargs)
        return out["text"] if isinstance(out, dict) else str(out)

    return run_pipe


def main(
    models: list[str],
    n: int,
    snr_levels: list[int | None],
    out: Path,
    speech_dir: Path | None = None,
) -> int:
    if speech_dir:
        samples = load_local_speech(speech_dir)[:n]
        print(f"loaded {len(samples)} local samples from {speech_dir}")
    else:
        samples = load_samples(n)
        print(f"loaded {len(samples)} VIVOS samples")

    rows = [
        "| model | condition | WER | latency/utt | RTF | n |",
        "|---|---|---|---|---|---|",
    ]
    for model_id in models:
        print(f"\n== {model_id} ==")
        try:
            run = build_runner(model_id)
        except Exception as e:  # noqa: BLE001
            print(f"  LOAD FAILED: {type(e).__name__}: {e}")
            rows.append(f"| {model_id} | load | — | — | — | — |")
            continue

        for snr in snr_levels:
            label = "clean" if snr is None else f"{snr}dB"
            np.random.seed(42)
            refs: list[str] = []
            hyps: list[str] = []
            total_audio_s = 0.0
            start = time.time()
            for s in samples:
                audio = s["audio"]
                sr = s["sr"]
                total_audio_s += len(audio) / sr
                refs.append(normalize_text(s["transcription"]))
                hyps.append(normalize_text(run(add_noise(audio, snr), sr)))
            elapsed = time.time() - start
            wer = jiwer.wer(refs, hyps)
            lat_ms = elapsed / len(samples) * 1000.0
            rtf = elapsed / total_audio_s if total_audio_s else 0.0
            rows.append(
                f"| {model_id} | {label} | {wer * 100:.2f}% | {lat_ms:.0f} ms | {rtf:.2f} | {len(samples)} |"
            )
            print(rows[-1], flush=True)

    md = (
        "# ASR A/B (VIVOS test slice, CPU)\n\n"
        + "\n".join(rows)
        + "\n\nRTF = wall-clock / audio-seconds (lower is better; <1.0 = faster "
        + "than real time). WER is case/punctuation-normalized. PhoWhisper-base "
        + "is the current runtime model; Qwen3-ASR-0.6B (Apache-2.0) is the "
        + "multilingual candidate (one model for 30+ langs incl. VI).\n"
    )
    out.write_text(md)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--snr", nargs="*", default=["clean", "5"])
    parser.add_argument("--out", type=Path, default=Path("docs/reports/asr_ab.md"))
    parser.add_argument("--speech-dir", type=Path, default=None,
                        help="use a local dir (data/raw/speech/<lang>) instead of VIVOS")
    args = parser.parse_args()
    snrs = [None if s == "clean" else int(s) for s in args.snr]
    raise SystemExit(main(args.models, args.num_samples, snrs, args.out, args.speech_dir))
