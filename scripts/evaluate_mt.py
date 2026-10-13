#!/usr/bin/env python3
"""Offline MT quality eval for the OneVoice translation legs.

Judging scores BLEU + COMET on a hidden benchmark (docs/decisions/
deep-research-2026-10.md). This measures the Opus-MT legs the runtime exports
on a public held-out set (Helsinki-NLP/opus-100 test split; FLORES-200 is
gated), reporting BLEU/chrF with 95% bootstrap CIs and, with --comet, COMET
(the judged metric). It reports the *six translation legs*; direct directions
are one leg, pivot directions compose two.

    uv run --no-project --python 3.12 --with torch --with transformers \\
      --with sentencepiece --with sacrebleu --with datasets --with unbabel-comet \\
      python scripts/evaluate_mt.py --n 200 --comet --out docs/reports/mt_quality.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import sacrebleu
import torch
from transformers import MarianMTModel, MarianTokenizer

LEGS = [
    ("vi-en", "Helsinki-NLP/opus-mt-vi-en", "en-vi", "vi", "en"),
    ("en-vi", "Helsinki-NLP/opus-mt-en-vi", "en-vi", "en", "vi"),
    ("en-zh", "Helsinki-NLP/opus-mt-en-zh", "en-zh", "en", "zh"),
    ("zh-en", "Helsinki-NLP/opus-mt-zh-en", "en-zh", "zh", "en"),
    ("en-ko", "Helsinki-NLP/opus-mt-tc-big-en-ko", "en-ko", "en", "ko"),
    ("ko-en", "Helsinki-NLP/opus-mt-ko-en", "en-ko", "ko", "en"),
]

DIRECTIONS = {
    "vi-en": ["vi-en"],
    "en-vi": ["en-vi"],
    "vi-zh": ["vi-en", "en-zh"],
    "zh-vi": ["zh-en", "en-vi"],
    "vi-ko": ["vi-en", "en-ko"],
    "ko-vi": ["ko-en", "en-vi"],
}

BLEU_TOK = {"en": "13a", "vi": "13a", "zh": "zh", "ko": "13a"}
_cache: dict[str, tuple] = {}
_pairs_cache: dict[str, list[dict]] = {}


def load(model_id: str):
    if model_id not in _cache:
        tok = MarianTokenizer.from_pretrained(model_id)
        mdl = MarianMTModel.from_pretrained(model_id).eval()
        _cache[model_id] = (tok, mdl)
    return _cache[model_id]


def opus100(config: str, n: int) -> list[dict]:
    if config not in _pairs_cache:
        from datasets import load_dataset

        ds = load_dataset("Helsinki-NLP/opus-100", config, split="test")
        _pairs_cache[config] = [ds[i]["translation"] for i in range(len(ds))]
    return _pairs_cache[config][:n]


@torch.no_grad()
def translate(texts: list[str], model_id: str) -> list[str]:
    tok, mdl = load(model_id)
    out: list[str] = []
    for i in range(0, len(texts), 32):
        enc = tok(texts[i : i + 32], return_tensors="pt", padding=True, truncation=True, max_length=256)
        gen = mdl.generate(**enc, num_beams=1, max_new_tokens=256)
        out.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return out


def bootstrap_ci(score_fn, n: int, n_boot: int = 300, seed: int = 0) -> float:
    """95% bootstrap CI half-width for a corpus metric (resamples both sides)."""
    rng = np.random.default_rng(seed)
    vals = [score_fn(rng.integers(0, n, n)) for _ in range(n_boot)]
    return float(np.std(vals) * 1.96)


_comet: dict[str, object] = {}


def comet_model():
    if "m" not in _comet:
        from comet import download_model, load_from_checkpoint

        _comet["m"] = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
    return _comet["m"]


def comet_score(srcs, hyps, refs) -> float:
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
    return float(comet_model().predict(data, batch_size=8, gpus=0)["system_score"])


def main(n: int, legs: list[str], use_comet: bool, out: Path | None) -> int:
    results: dict[str, float] = {}
    rows = ["| leg | model | BLEU | chrF | COMET | n |", "|---|---|---|---|---|---|"]
    for name, model_id, config, src, tgt in LEGS:
        if legs and name not in legs:
            continue
        pairs = opus100(config, n)
        srcs = [p[src] for p in pairs]
        refs = [p[tgt] for p in pairs]
        hyp = translate(srcs, model_id)

        n_ = len(hyp)
        bleu = sacrebleu.corpus_bleu(hyp, [refs], tokenize=BLEU_TOK[tgt]).score
        chrf = sacrebleu.corpus_chrf(hyp, [refs]).score
        bleu_ci = bootstrap_ci(
            lambda idx: sacrebleu.corpus_bleu(
                [hyp[i] for i in idx], [[refs[i] for i in idx]], tokenize=BLEU_TOK[tgt]
            ).score,
            n_,
        )
        chrf_ci = bootstrap_ci(
            lambda idx: sacrebleu.corpus_chrf([hyp[i] for i in idx], [[refs[i] for i in idx]]).score,
            n_,
        )
        comet = comet_score(srcs, hyp, refs) if use_comet else None
        comet_str = f"{comet:.3f}" if comet is not None else "—"
        results[name] = bleu
        rows.append(
            f"| {name} | {model_id.split('/')[-1]} | {bleu:.1f} ±{bleu_ci:.1f} | "
            f"{chrf:.1f} ±{chrf_ci:.1f} | {comet_str} | {n_} |"
        )
        print(rows[-1], flush=True)

    rows.append("\n## Pipeline directions\n")
    rows.append("| direction | path | legs | leg BLEU |")
    rows.append("|---|---|---|---|")
    for direction, used in DIRECTIONS.items():
        if any(u not in results for u in used):
            continue
        leg_scores = " + ".join(f"{u}={results[u]:.1f}" for u in used)
        kind = "direct" if len(used) == 1 else "pivot"
        rows.append(f"| {direction} | {kind} | {'→'.join(used)} | {leg_scores} |")

    md = "# MT quality (Helsinki-NLP/opus-100 test slice)\n\n" + "\n".join(rows) + "\n"
    if out:
        out.write_text(md)
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--legs", nargs="*", default=[])
    parser.add_argument("--comet", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    raise SystemExit(main(args.n, args.legs, args.comet, args.out))
