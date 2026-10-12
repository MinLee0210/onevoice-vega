#!/usr/bin/env python3
"""Offline MT quality eval for the OneVoice translation legs (BLEU + chrF).

Judging scores BLEU + COMET on a hidden benchmark (docs/decisions/
deep-research-2026-10.md) and the repo had no BLEU number at all. This
measures the Opus-MT legs the runtime exports, on a public held-out set
(Helsinki-NLP/opus-100 test split; FLORES-200 is gated). It reports the
*six translation legs*; the direct directions are single legs and the pivot
directions (VI<->ZH/KO) are compositions of two legs.

Env (run under the experiments group or uv):
    uv run --no-project --python 3.12 --with torch --with transformers \\
      --with sentencepiece --with sacrebleu --with datasets \\
      python scripts/evaluate_mt.py --n 200 --out docs/reports/mt_quality.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sacrebleu
import torch
from transformers import MarianMTModel, MarianTokenizer

# leg, model, opus-100 config, source lang, target lang
LEGS = [
    ("vi-en", "Helsinki-NLP/opus-mt-vi-en", "en-vi", "vi", "en"),
    ("en-vi", "Helsinki-NLP/opus-mt-en-vi", "en-vi", "en", "vi"),
    ("en-zh", "Helsinki-NLP/opus-mt-en-zh", "en-zh", "en", "zh"),
    ("zh-en", "Helsinki-NLP/opus-mt-zh-en", "en-zh", "zh", "en"),
    ("en-ko", "Helsinki-NLP/opus-mt-tc-big-en-ko", "en-ko", "en", "ko"),
    ("ko-en", "Helsinki-NLP/opus-mt-ko-en", "en-ko", "ko", "en"),
]

# pipeline direction -> legs it composes
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


def main(n: int, legs: list[str], out: Path | None) -> int:
    results: dict[str, tuple[float, float]] = {}
    rows = ["| leg | model | BLEU | chrF | n |", "|---|---|---|---|---|"]
    for name, model_id, config, src, tgt in LEGS:
        if legs and name not in legs:
            continue
        pairs = opus100(config, n)
        srcs = [p[src] for p in pairs]
        refs = [p[tgt] for p in pairs]
        hyp = translate(srcs, model_id)
        bleu = sacrebleu.corpus_bleu(hyp, [refs], tokenize=BLEU_TOK[tgt]).score
        chrf = sacrebleu.corpus_chrf(hyp, [refs]).score
        results[name] = (bleu, chrf)
        rows.append(f"| {name} | {model_id.split('/')[-1]} | {bleu:.1f} | {chrf:.1f} | {len(srcs)} |")
        print(rows[-1], flush=True)

    rows.append("\n## Pipeline directions\n")
    rows.append("| direction | path | legs | leg BLEU |")
    rows.append("|---|---|---|---|")
    for direction, used in DIRECTIONS.items():
        if any(u not in results for u in used):
            continue
        leg_scores = " + ".join(f"{u}={results[u][0]:.1f}" for u in used)
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
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    raise SystemExit(main(args.n, args.legs, args.out))
