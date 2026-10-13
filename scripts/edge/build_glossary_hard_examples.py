#!/usr/bin/env python3
"""Build a "hard-example" glossary training set for MT LoRA fine-tuning.

WMT26 finding (arXiv 2609.09999): for most glossary pairs the base model
already emits the right term, so training on them teaches nothing. Keep only
the sentences where Opus-MT **contradicts** the glossary → +11% term accuracy
at fixed data.

For each term in data/glossary/termbase.tsv, scan a parallel VI→EN corpus for
source sentences containing the term; translate with the base model; keep the
pair only if the required target term is **missing** from the output.

    uv run --no-project --python 3.12 --with torch --with transformers \\
      --with sentencepiece --with datasets \\
      python scripts/edge/build_glossary_hard_examples.py --n 2000 \\
      --out data/glossary/hard_examples.jsonl

Then fine-tune with scripts/edge/finetune_mt_glossary.py.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import MarianMTModel, MarianTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL = "Helsinki-NLP/opus-mt-vi-en"


def load_term_pairs(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    lines = path.read_text().splitlines()
    header = lines[0].split("\t")
    try:
        vi_i, en_i = header.index("vi"), header.index("en")
    except ValueError as e:  # noqa: BLE001
        raise SystemExit(f"termbase missing vi/en columns: {header}") from e
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) > max(vi_i, en_i):
            pairs.append((parts[vi_i].lower(), parts[en_i].lower()))
    return pairs


@torch.no_grad()
def translate(texts: list[str], tok, mdl) -> list[str]:
    out: list[str] = []
    for i in range(0, len(texts), 32):
        enc = tok(texts[i : i + 32], return_tensors="pt", padding=True, truncation=True, max_length=256)
        gen = mdl.generate(**enc, num_beams=1, max_new_tokens=256)
        out.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return out


def main(n: int, out_path: Path) -> int:
    term_pairs = load_term_pairs(REPO_ROOT / "data/glossary/termbase.tsv")
    print(f"{len(term_pairs)} term pairs")

    from datasets import load_dataset

    ds = load_dataset("Helsinki-NLP/opus-100", "en-vi", split="test")
    pairs = [ds[i]["translation"] for i in range(min(n, len(ds)))]

    tok = MarianTokenizer.from_pretrained(MODEL)
    mdl = MarianMTModel.from_pretrained(MODEL).eval()

    # Word-boundary match — plain substring gives false positives (e.g. the
    # vi term "hàn" matches inside "thành"). Mirrors the Rust glossary fix.
    def has_term(text: str, term: str) -> bool:
        return re.search(r"\b" + re.escape(term.lower()) + r"\b", text.lower()) is not None

    # Candidates: source sentences containing a term as a whole word.
    candidates: list[tuple[str, str, str]] = []  # (vi, en_ref, en_term)
    for p in pairs:
        for vi_t, en_t in term_pairs:
            if has_term(p["vi"], vi_t):
                candidates.append((p["vi"], p["en"], en_t))
                break
    print(f"{len(candidates)} candidate sentences contain a glossary term")

    hyps = translate([c[0] for c in candidates], tok, mdl)
    hard = [
        {"source": vi, "target": ref, "term": term}
        for (vi, ref, term), hyp in zip(candidates, hyps)
        if not has_term(hyp, term)
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for ex in hard:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"kept {len(hard)}/{len(candidates)} hard examples -> {out_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data/glossary/hard_examples.jsonl")
    args = parser.parse_args()
    raise SystemExit(main(args.n, args.out))
