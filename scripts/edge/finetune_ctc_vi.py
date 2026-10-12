#!/usr/bin/env python3
"""Fine-tune an Apache-2.0 VI CTC model to close the WER gap.

Rationale: the best VI CTC checkpoints are CC-BY-NC (can't ship); the only
Apache-2.0 one (`dragonSwing/wav2vec2-base-vietnamese`) is clean-10.4% /
5dB-23.0% (docs/reports/asr_ctc_sweep.md). Fine-tune it on VIVOS (+ optional
ESC-50 industrial-noise augmentation) so CTC can become the default fast
backend (ASR ~57 ms in the Rust pipeline vs ~360 ms Whisper).

Needs a GPU. This is a *recipe-as-code*: run it where you have hardware, then
drop the output into `models/asr/vi-ctc/` and re-run the eval scripts.

```sh
uv run --python 3.12 --with "torch" --with "transformers>=4.46" \
  --with "datasets" --with "jiwer" --with "soundfile" --with "librosa" --with "accelerate" \
  python scripts/edge/finetune_ctc_vi.py \
    --base dragonSwing/wav2vec2-base-vietnamese \
    --out models/asr/vi-ctc-ft \
    --epochs 8 --batch 8 --num-train 3000 --num-eval 200 --noise
```

Then:
    cp -r models/asr/vi-ctc-ft/* models/asr/vi-ctc/     # drop-in for the runtime
    uv run ... python scripts/evaluate_asr.py  --models models/asr/vi-ctc  ...
    uv run ... python scripts/evaluate_asr_noise.py --model-ctc models/asr/vi-ctc
"""

from __future__ import annotations

import argparse
import io
from dataclasses import dataclass
from pathlib import Path

import jiwer
import librosa
import numpy as np
import soundfile as sf
import torch
from datasets import Audio, Dataset, load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

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


@dataclass
class DataCollatorCTCPadding:
    processor: Wav2Vec2Processor
    padding: bool = True

    def __call__(self, features):
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(
            label_features, padding=self.padding, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


def load_vivos(n: int, skip: int = 0) -> list[dict]:
    ds = load_dataset("quocanh34/viet_vivos", split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    out: list[dict] = []
    for i, s in enumerate(ds):
        if i < skip:
            continue
        raw = s["audio"]
        arr, sr = sf.read(io.BytesIO(raw["bytes"]), dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        out.append({"array": arr, "sr": sr, "text": s["transcription"]})
        if len(out) >= n:
            break
    return out


def load_industrial_noise(sr: int = 16_000) -> list[np.ndarray]:
    ds = load_dataset("ashraq/esc50", split="train").cast_column("audio", Audio(decode=False))
    pool: list[np.ndarray] = []
    for s in ds:
        if s["category"] in INDUSTRIAL:
            arr, src_sr = sf.read(io.BytesIO(s["audio"]["bytes"]), dtype="float32")
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            if src_sr != sr:
                arr = librosa.resample(arr, orig_sr=src_sr, target_sr=sr)
            pool.append(arr)
    return pool


def mix(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    n = np.resize(noise, len(signal))
    ps, pn = float(np.mean(signal**2)), float(np.mean(n**2))
    if ps == 0 or pn == 0:
        return signal
    return signal + n * np.sqrt((ps * 10 ** (-snr_db / 10.0)) / pn)


def main(args: argparse.Namespace) -> int:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"device: {device}")
    processor = Wav2Vec2Processor.from_pretrained(args.base)
    model = Wav2Vec2ForCTC.from_pretrained(
        args.base,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    model.freeze_feature_encoder()  # standard Wav2Vec2 CTC fine-tune setting

    pool = load_industrial_noise() if args.noise else []

    def example(rec: dict) -> dict:
        audio = rec["array"].astype(np.float32)
        if args.noise and pool and np.random.rand() < 0.7:
            audio = mix(audio, pool[np.random.randint(len(pool))], np.random.uniform(0, 15))
        feat = processor(audio, sampling_rate=rec["sr"]).input_values[0]
        labels = processor(text=rec["text"]).input_ids
        return {"input_values": feat, "labels": labels}

    print(f"materializing {args.num_train} train + {args.num_eval} eval VIVOS utts ...")
    train = Dataset.from_list([example(r) for r in load_vivos(args.num_train)])
    evalset = Dataset.from_list(
        [example(r) for r in load_vivos(args.num_eval, skip=args.num_train)]
    )

    def compute_metrics(pred):
        pred_ids = np.argmax(pred.predictions, axis=-1)
        hyp = processor.batch_decode(pred_ids)
        labels = pred.label_ids.copy()
        labels[labels == -100] = processor.tokenizer.pad_token_id
        ref = processor.batch_decode(labels, group_tokens=False)
        return {"wer": jiwer.wer(ref, hyp)}

    targs = TrainingArguments(
        output_dir=str(Path(args.out) / "_hf"),
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup,
        num_train_epochs=args.epochs,
        fp16=(device == "cuda"),
        use_cpu=False,
        dataloader_num_workers=0,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train,
        eval_dataset=evalset,
        data_collator=DataCollatorCTCPadding(processor=processor),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    print(f"final eval WER: {metrics.get('eval_wer')}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    processor.save_pretrained(out)

    # ONNX export in the runtime's expected layout.
    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_values):
            return self.m(input_values).logits

    model.eval()
    model.to("cpu")  # ONNX export traces on CPU; weights must follow (MPS/CUDA)
    torch.onnx.export(
        Wrap(model),
        (torch.randn(1, 16000 * 5),),
        str(out / "model.onnx"),
        input_names=["input_values"],
        output_names=["logits"],
        dynamic_axes={"input_values": {1: "samples"}, "logits": {1: "frames"}},
        opset_version=14,
        dynamo=False,
    )
    (out / "MANIFEST.md").write_text(
        f"# {out.name}\n\nFine-tuned from {args.base} on {args.num_train} VIVOS "
        f"utts ({args.epochs} epochs, noise_aug={args.noise}). Apache-2.0 base; "
        "check VIVOS/ESC-50 licenses before shipping. Drop-in for "
        "`models/asr/vi-ctc/` (see src/stages/ctc.rs).\n"
    )
    print(f"done -> {out} (model.onnx, vocab.json, preprocessor_config.json)")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="dragonSwing/wav2vec2-base-vietnamese")
    parser.add_argument("--out", default="models/asr/vi-ctc-ft")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--num-train", type=int, default=3000)
    parser.add_argument("--num-eval", type=int, default=200)
    parser.add_argument("--noise", action="store_true", help="ESC-50 industrial-noise augmentation")
    raise SystemExit(main(parser.parse_args()))
