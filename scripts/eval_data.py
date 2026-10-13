"""Loaders for the locally-fetched samples under data/raw/ (see
scripts/fetch_samples.py) so the eval harnesses can run offline on the
curated samples instead of streaming from the HF Hub."""

from __future__ import annotations

from pathlib import Path


def load_local_speech(split_dir: Path) -> list[dict]:
    """Reads `*.wav` + `transcripts.tsv` from a data/raw/speech/<lang> dir."""
    import soundfile as sf  # lazy: only the audio loaders need it

    texts: dict[str, str] = {}
    tsv = split_dir / "transcripts.tsv"
    if tsv.exists():
        for line in tsv.read_text().splitlines():
            if "\t" in line:
                name, text = line.split("\t", 1)
                texts[name] = text
    out: list[dict] = []
    for wav in sorted(split_dir.glob("*.wav")):
        audio, sr = sf.read(wav, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        out.append({"audio": audio, "sr": sr, "transcription": texts.get(wav.name, "")})
    return out


def load_local_noise(noise_dir: Path) -> list[object]:
    """Reads the industrial clips from data/raw/noise/ into float arrays."""
    import soundfile as sf

    out: list = []
    for wav in sorted(noise_dir.glob("*.wav")):
        audio, _ = sf.read(wav, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        out.append(audio)
    return out


def load_local_parallel(
    parallel_dir: Path, config: str, n: int, prefer_phomt: bool = False
) -> list[dict]:
    """Reads a 2-column TSV from data/raw/parallel/ as {lang: text} dicts.

    File/lang order follows the config name (`en-vi` -> cols en, vi). With
    `prefer_phomt` and the VI-EN pair, uses the higher-quality PhoMT slice
    (`phomt-test.vi-en.tsv`, cols vi, en)."""
    if prefer_phomt and config in ("en-vi", "vi-en"):
        path, (la, lb) = parallel_dir / "phomt-test.vi-en.tsv", ("vi", "en")
    else:
        la, lb = config.split("-")
        path = parallel_dir / f"{config}.tsv"
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        if "\t" in line:
            cols = line.split("\t", 1)
            rows.append({la: cols[0], lb: cols[1]})
        if len(rows) >= n:
            break
    return rows
