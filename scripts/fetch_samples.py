#!/usr/bin/env python3
"""Download small, curated data samples for the OneVoice use-cases into
`data/raw/` (gitignored — this is bulk data, not source).

Use-cases covered (see AGENTS.md):
  * speech    — one short utterance set per language VI/EN/ZH/KO
                (VI: VIVOS; EN/ZH/KO: FLEURS) for ASR/VAD/TTS round-trips
  * parallel  — VI<->EN / ZH<->EN / KO<->EN sentence pairs (OPUS-100) for MT
  * noise     — industrial/machinery clips (ESC-50) for the noise-robustness work

    uv run --no-project --python 3.12 --with datasets --with soundfile \\
      python scripts/fetch_samples.py --speech-n 5 --parallel-n 100 --noise-n 12

Everything is public research data; check each corpus's license before any
redistribution (FLEURS is CC-BY-4.0, VIVOS public research, OPUS-100 mixes
OPUS corpora, ESC-50 is CC-BY-NC — internal eval only).
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEECH = {
    "vi": ("quocanh34/viet_vivos", None, "test"),
    "en": ("google/fleurs", "en_us", "test"),
    "zh": ("google/fleurs", "cmn_hans_cn", "test"),
    "ko": ("google/fleurs", "ko_kr", "test"),
}
PARALLEL = ["en-vi", "en-zh", "en-ko"]
# Higher-quality VI-EN MT benchmark than OPUS-100 (PhoMT, EMNLP 2021). The
# original is CC-BY-NC-ND; this mirror is a cleaned copy — internal eval only.
PHOMT_REPO = "tuandunghcmut/PhoMT"
NOISE_CATEGORIES = {
    "chainsaw", "engine", "hand_saw", "helicopter", "siren",
    "vacuum_cleaner", "washing_machine", "train",
}


def _audio_bytes(row: dict) -> bytes | None:
    """Raw file bytes for a `decode=False` audio column (sample rate is read
    from the file itself via soundfile, so we don't rely on the dict)."""
    a = row["audio"]
    if a.get("bytes") is not None:
        return a["bytes"]
    if a.get("path"):
        return Path(a["path"]).read_bytes()
    return None


def fetch_speech(root: Path, n: int) -> None:
    for lang, (repo, config, split) in SPEECH.items():
        out = root / "speech" / lang
        out.mkdir(parents=True, exist_ok=True)
        ds = load_dataset(repo, config, split=split) if config else load_dataset(repo, split=split)
        ds = ds.cast_column("audio", Audio(decode=False))
        lines = []
        for i in range(min(n, len(ds))):
            row = ds[i]
            data = _audio_bytes(row)
            if data is None:
                continue
            audio, sr = sf.read(io.BytesIO(data), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            name = f"{lang}_{i:03d}.wav"
            sf.write(out / name, audio, sr)
            text = row.get("transcription") or row.get("sentence") or ""
            lines.append(f"{name}\t{text}")
        (out / "transcripts.tsv").write_text("\n".join(lines) + "\n")
        print(f"  speech/{lang}: {len(lines)} utts from {repo}")


def fetch_parallel(root: Path, n: int) -> None:
    out = root / "parallel"
    out.mkdir(parents=True, exist_ok=True)
    for pair in PARALLEL:
        ds = load_dataset("Helsinki-NLP/opus-100", pair, split="test")
        a, b = pair.split("-")
        rows = [f"{ds[i]['translation'][a]}\t{ds[i]['translation'][b]}" for i in range(min(n, len(ds)))]
        (out / f"{pair}.tsv").write_text("\n".join(rows) + "\n")
        print(f"  parallel/{pair}.tsv: {len(rows)} pairs")


def fetch_phomt(root: Path, n: int) -> None:
    """A PhoMT VI-EN test slice (higher-quality MT benchmark than OPUS-100)."""
    out = root / "parallel"
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(PHOMT_REPO, split="test")
    rows = [f"{ds[i]['vi']}\t{ds[i]['en']}" for i in range(min(n, len(ds)))]
    (out / "phomt-test.vi-en.tsv").write_text("\n".join(rows) + "\n")
    print(f"  parallel/phomt-test.vi-en.tsv: {len(rows)} pairs")


def fetch_noise(root: Path, n: int) -> None:
    out = root / "noise"
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("ashraq/esc50", split="train").cast_column("audio", Audio(decode=False))
    written, per_cat = 0, {}
    for row in ds:
        cat = row["category"]
        if cat not in NOISE_CATEGORIES or per_cat.get(cat, 0) >= 2:
            continue
        data = _audio_bytes(row)
        if data is None:
            continue
        audio, sr = sf.read(io.BytesIO(data), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        sf.write(out / f"{cat}_{row['filename']}", audio, sr)
        per_cat[cat] = per_cat.get(cat, 0) + 1
        written += 1
        if written >= n:
            break
    print(f"  noise/: {written} clips across {len(per_cat)} categories")


def main(sets: list[str], speech_n: int, parallel_n: int, noise_n: int, phomt_n: int, root: Path) -> int:
    root.mkdir(parents=True, exist_ok=True)
    print(f"writing samples to {root} ({', '.join(sets)})")
    if "speech" in sets:
        print("== speech ==")
        fetch_speech(root, speech_n)
    if "parallel" in sets:
        print("== parallel ==")
        fetch_parallel(root, parallel_n)
    if "phomt" in sets:
        print("== phomt ==")
        fetch_phomt(root, phomt_n)
    if "noise" in sets:
        print("== noise ==")
        fetch_noise(root, noise_n)
    print("done")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=["speech", "parallel", "phomt", "noise"],
                        choices=["speech", "parallel", "phomt", "noise"])
    parser.add_argument("--speech-n", type=int, default=5)
    parser.add_argument("--parallel-n", type=int, default=100)
    parser.add_argument("--phomt-n", type=int, default=200)
    parser.add_argument("--noise-n", type=int, default=12)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data/raw")
    args = parser.parse_args()
    raise SystemExit(main(args.datasets, args.speech_n, args.parallel_n, args.noise_n, args.phomt_n, args.out))
