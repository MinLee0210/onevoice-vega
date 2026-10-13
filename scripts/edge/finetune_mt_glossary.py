#!/usr/bin/env python3
"""LoRA fine-tune Opus-MT on the hard-example glossary set.

Recipe (needs a GPU): train a small LoRA adapter on
`data/glossary/hard_examples.jsonl` (built by
`scripts/edge/build_glossary_hard_examples.py`), merge it, and export the
runtime layout (`encoder_model.onnx` etc. via scripts/edge/export_mt.py's
conventions). This is AGENTS.md's "LoRA fine-tune on an industrial glossary"
— the intended replacement for the decoder-logit-bias stopgap.

    uv run --python 3.12 --with torch --with transformers --with peft \\
      --with datasets --with sentencepiece --with accelerate \\
      python scripts/edge/finetune_mt_glossary.py \\
      --base Helsinki-NLP/opus-mt-vi-en \\
      --data data/glossary/hard_examples.jsonl \\
      --out models/mt/vi-en-lora --epochs 5

Acceptance: term accuracy on a **held-out** set up (the logit-bias method
already hits 100% on tuned sentences; LoRA should match it *without* the
fluency cost) and no BLEU regression (scripts/evaluate_mt.py).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    DataCollatorForSeq2Seq,
    MarianMTModel,
    MarianTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def load_jsonl(path: Path) -> Dataset:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return Dataset.from_list(rows)


def main(args: argparse.Namespace) -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = MarianTokenizer.from_pretrained(args.base)
    model = MarianMTModel.from_pretrained(args.base)

    # Target attention projections (Marian names q_proj/k_proj/v_proj/out_proj).
    lora = LoraConfig(
        task_type=TaskType.SEQ2SEQ_LM,
        r=args.r,
        lora_alpha=2 * args.r,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = load_jsonl(Path(args.data))

    def prep(batch):
        enc = tok(batch["source"], truncation=True, max_length=256)
        with tok.as_target_tokenizer():
            labels = tok(batch["target"], truncation=True, max_length=256)["input_ids"]
        enc["labels"] = labels
        return enc

    ds = ds.map(prep, batched=True, remove_columns=ds.column_names)

    targs = Seq2SeqTrainingArguments(
        output_dir=str(Path(args.out) / "_hf"),
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        fp16=(device == "cuda"),
        logging_steps=25,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=DataCollatorForSeq2Seq(tok, model=model),
    )
    trainer.train()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "MANIFEST.md").write_text(
        f"# {out.name}\n\n- Base: {args.base}\n- Fine-tune: LoRA r={args.r} on "
        f"{args.data}\n- License: Opus-MT (Helsinki-NLP) — verify; hard-example\n"
        "  data derived from opus-100 (mixed OPUS; internal training use).\n\n"
        "Next: ONNX-export (see scripts/edge/export_mt.py) + quantize "
        "(scripts/edge/quantize_mt.py), then scripts/evaluate_mt.py.\n"
    )
    print(f"done -> {out}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Helsinki-NLP/opus-mt-vi-en")
    parser.add_argument("--data", default="data/glossary/hard_examples.jsonl")
    parser.add_argument("--out", default="models/mt/vi-en-lora")
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    raise SystemExit(main(parser.parse_args()))
