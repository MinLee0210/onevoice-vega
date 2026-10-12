#!/usr/bin/env python3
"""Export a Helsinki-NLP Opus-MT pair to the ONNX layout `src/stages/mt.rs`
expects (`encoder_model.onnx`, `decoder_model_merged.onnx`, `tokenizer.json`).

Generalizes the original `export_en_vi_mt.py`. Needs the full audio-lab env
(torch/optimum/sentencepiece/tokenizers).

Usage:
    python scripts/edge/export_mt.py --model Helsinki-NLP/opus-mt-en-vi --dir models/mt/en-vi
    python scripts/edge/export_mt.py --model Helsinki-NLP/opus-mt-tc-big-en-ko --dir models/mt/en-ko
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import date
from pathlib import Path

import sentencepiece as spm
import sentencepiece.sentencepiece_model_pb2 as spm_pb2
from optimum.exporters.onnx import main_export
from tokenizers import Tokenizer, decoders, pre_tokenizers
from tokenizers.models import Unigram
from tokenizers.normalizers import Precompiled, Replace, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]


def export(model_id: str, final_dir: Path) -> None:
    export_dir = final_dir / "_export"
    export_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting {model_id} -> {final_dir} ...")
    main_export(
        model_name_or_path=model_id,
        output=str(export_dir),
        task="text2text-generation-with-past",
    )

    with open(export_dir / "vocab.json") as f:
        vocab_json = json.load(f)
    vocab_size = max(vocab_json.values()) + 1
    id_to_piece = [None] * vocab_size
    for piece, idx in vocab_json.items():
        id_to_piece[idx] = piece
    assert all(p is not None for p in id_to_piece), "gap in vocab id space"

    sp = spm.SentencePieceProcessor(model_file=str(export_dir / "source.spm"))
    piece_scores = {sp.id_to_piece(i): sp.get_score(i) for i in range(sp.get_piece_size())}
    min_score = min(piece_scores.values())
    vocab_list = [(piece, piece_scores.get(piece, min_score - 1.0)) for piece in id_to_piece]
    unk_id = vocab_json["<unk>"]

    fast_tokenizer = Tokenizer(Unigram(vocab_list, unk_id=unk_id, byte_fallback=False))
    proto = spm_pb2.ModelProto()
    proto.ParseFromString((export_dir / "source.spm").read_bytes())
    charsmap = proto.normalizer_spec.precompiled_charsmap
    fast_tokenizer.normalizer = Sequence([Precompiled(charsmap), Replace(" {2,}", " ")])
    fast_tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Metaspace(replacement="\u2581", prepend_scheme="always")]
    )
    fast_tokenizer.decoder = decoders.Metaspace(replacement="\u2581", prepend_scheme="always")
    fast_tokenizer.save(str(export_dir / "tokenizer.json"))

    for name in ["encoder_model.onnx", "decoder_model_merged.onnx", "tokenizer.json"]:
        shutil.copyfile(export_dir / name, final_dir / name)

    (final_dir / "MANIFEST.md").write_text(
        f"""# {final_dir.name}

- Source checkpoint: {model_id}
- License: Opus-MT (Helsinki-NLP) — **verify** (Apache-2.0 vs CC-BY-4.0)
- Exported: {date.today().isoformat()}
- Precision: fp32 (quantize via scripts/edge/quantize_mt.py)
- Decode: greedy (beam=1)

Files: `encoder_model.onnx`, `decoder_model_merged.onnx`, `tokenizer.json`
"""
    )
    shutil.rmtree(export_dir)
    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model id")
    parser.add_argument("--dir", required=True, help="output dir under models/mt/")
    args = parser.parse_args()
    export(args.model, REPO_ROOT / args.dir)
