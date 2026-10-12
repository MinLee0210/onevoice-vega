#!/usr/bin/env python3
"""Export a CTC (Wav2Vec2-family) VI ASR model to the ONNX layout the
`CtcAsrEngine` (src/stages/ctc.rs) expects:

    models/asr/<dir>/model.onnx        # input_values [1, N] -> logits [1, T, V]
    models/asr/<dir>/vocab.json        # token -> id
    models/asr/<dir>/preprocessor_config.json
    models/asr/<dir>/MANIFEST.md

No KV cache, no autoregressive loop — a single forward pass + CTC greedy
decode (the latency argument for CTC vs Whisper).

Usage:
    python scripts/edge/export_ctc_asr.py \
        --model dragonSwing/wav2vec2-base-vietnamese --dir models/asr/vi-ctc
"""

from __future__ import annotations

import argparse
import shutil
from datetime import date
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import Wav2Vec2ForCTC

REPO_ROOT = Path(__file__).resolve().parents[2]


class Wrap(torch.nn.Module):
    def __init__(self, m: Wav2Vec2ForCTC) -> None:
        super().__init__()
        self.m = m

    def forward(self, input_values):  # noqa: ANN001, ANN201
        return self.m(input_values).logits


def export(model_id: str, final_dir: Path) -> None:
    final_dir.mkdir(parents=True, exist_ok=True)
    model = Wav2Vec2ForCTC.from_pretrained(model_id).eval()

    dummy = torch.randn(1, 16000 * 5)
    torch.onnx.export(
        Wrap(model),
        (dummy,),
        str(final_dir / "model.onnx"),
        input_names=["input_values"],
        output_names=["logits"],
        dynamic_axes={"input_values": {1: "samples"}, "logits": {1: "frames"}},
        opset_version=14,
        dynamo=False,
    )

    for f in ("vocab.json", "preprocessor_config.json", "config.json"):
        try:
            src = hf_hub_download(model_id, f)
            shutil.copyfile(src, final_dir / f)
        except Exception as e:  # noqa: BLE001
            print(f"  (could not fetch {f}: {e})")

    (final_dir / "MANIFEST.md").write_text(
        f"""# {final_dir.name}

- Source checkpoint: {model_id}
- License: Apache-2.0 (verify the card)
- Exported: {date.today().isoformat()}
- Architecture: Wav2Vec2ForCTC — single forward pass, CTC greedy decode
- ONNX: `model.onnx` (input_values [1,N] f32 -> logits [1,T,V])

Files: `model.onnx`, `vocab.json`, `preprocessor_config.json`, `config.json`
"""
    )
    print(f"done -> {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dir", required=True)
    args = parser.parse_args()
    export(args.model, REPO_ROOT / args.dir)
