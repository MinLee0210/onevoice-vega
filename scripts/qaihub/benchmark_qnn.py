#!/usr/bin/env python3
"""QCS6490-class NPU latency benchmark for the ASR stage.

The bundled pipeline ONNX is already INT8 (optimum dynamic quant), which the
QAIRT/QNN converter rejects (`exit code 255`). The proven path (matching
`qualcomm-ai-hub-community`'s QCS6490 Whisper builds) is FP32 source ->
server-side full-INT8 QNN: `--quantize_full_type int8 --quantize_io`. This
script benchmarks that path across Qualcomm edge devices.

FP32 sources live in /tmp/fp32models (download with --download).

Usage:
    python3 scripts/qaihub/benchmark_qnn.py --download              # fetch FP32 ONNX
    python3 scripts/qaihub/benchmark_qnn.py --models whisper_base_encoder
    python3 scripts/qaihub/benchmark_qnn.py                         # all, all devices
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import qai_hub  # noqa: E402

FP32_DIR = Path("/tmp/fp32models")
TAG = "onevoice-fp32"
OPTIONS = "--target_runtime precompiled_qnn_onnx --quantize_full_type int8 --quantize_io"

ENC = {"feature_size": 80, "encoder_sequence_length": 3000}
DEC_BASE = {"encoder_sequence_length / 2": 1500, "encoder_sequence_length_out": 1500}

FP32_MODELS: list[dict] = [
    {"key": "phowhisper_base_encoder", "file": "phowhisper-base-encoder-fp32.onnx", "dims": ENC},
    {"key": "whisper_base_encoder", "file": "whisper-base-encoder-fp32.onnx", "dims": ENC},
    {"key": "whisper_small_encoder", "file": "whisper-small-encoder-fp32.onnx", "dims": ENC},
    {
        "key": "phowhisper_base_decoder",
        "file": "phowhisper-base-decoder-fp32.onnx",
        "dims": DEC_BASE,
    },
    {"key": "whisper_base_decoder", "file": "whisper-base-decoder-fp32.onnx", "dims": DEC_BASE},
    {"key": "whisper_small_decoder", "file": "whisper-small-decoder-fp32.onnx", "dims": DEC_BASE},
]

HF_SOURCES = [
    ("onnx-community/whisper-base", "onnx/encoder_model.onnx", "whisper-base-encoder-fp32.onnx"),
    (
        "onnx-community/whisper-base",
        "onnx/decoder_model_merged.onnx",
        "whisper-base-decoder-fp32.onnx",
    ),
    ("onnx-community/whisper-small", "onnx/encoder_model.onnx", "whisper-small-encoder-fp32.onnx"),
    (
        "onnx-community/whisper-small",
        "onnx/decoder_model_merged.onnx",
        "whisper-small-decoder-fp32.onnx",
    ),
    ("huuquyet/PhoWhisper-base", "onnx/encoder_model.onnx", "phowhisper-base-encoder-fp32.onnx"),
    (
        "huuquyet/PhoWhisper-base",
        "onnx/decoder_model_merged.onnx",
        "phowhisper-base-decoder-fp32.onnx",
    ),
]

DEFAULT_DEVICES = [
    "Dragonwing RB3 Gen 2 Vision Kit",  # QCS6490 — target
    "Dragonwing Q-6690 MTP",
    "Dragonwing IQ-9075 EVK",
    "Arduino VENTUNO Q",
    "QCS8550 (Proxy)",
]

CSV_COLS = ["model", "device", "chipset", "status", "latency_ms", "peak_mem_mb"]


def chipset_of(device: qai_hub.Device) -> str:
    for attr in getattr(device, "attributes", []) or []:
        if attr.startswith("chipset:"):
            return attr.split(":", 1)[1]
    return "?"


def download_fp32() -> None:
    from huggingface_hub import hf_hub_download

    FP32_DIR.mkdir(parents=True, exist_ok=True)
    for repo, remote, local in HF_SOURCES:
        dst = FP32_DIR / local
        if dst.exists():
            print("have", local)
            continue
        p = hf_hub_download(repo, remote)
        import shutil

        shutil.copyfile(p, dst)
        print("got", local)


def main(args: argparse.Namespace) -> int:
    if args.download:
        download_fp32()
        return 0

    models = [m for m in FP32_MODELS if args.models is None or m["key"] in args.models]
    all_devs = {d.name: d for d in qai_hub.get_devices()}
    devices = [all_devs[n] for n in args.devices if n in all_devs]
    csv_path = Path(args.csv)

    done: set[tuple[str, str]] = set()
    if csv_path.exists():
        done = {(r["model"], r["device"]) for r in csv.DictReader(csv_path.open())}

    f = csv_path.open("a", newline="")
    w = csv.DictWriter(f, fieldnames=CSV_COLS)
    if csv_path.stat().st_size == 0:
        w.writeheader()

    uploaded: dict[str, qai_hub.Model] = {}
    for m in models:
        path = FP32_DIR / m["file"]
        if not path.exists():
            print("missing (run --download):", path)
            continue
        uploaded[m["key"]] = qai_hub.upload_model(str(path), name=f"{TAG}-{m['key']}")

    for m in models:
        if m["key"] not in uploaded:
            continue
        specs = common.auto_input_specs(FP32_DIR / m["file"], m["dims"])
        for dev in devices:
            if (m["key"], dev.name) in done:
                continue
            row = {
                "model": m["key"],
                "device": dev.name,
                "chipset": chipset_of(dev),
                "status": "?",
                "latency_ms": "",
                "peak_mem_mb": "",
            }
            cj = qai_hub.submit_compile_job(
                uploaded[m["key"]],
                device=dev,
                input_specs=specs,
                options=OPTIONS,
                name=f"{TAG}-{m['key']}-{dev.name[:16]}",
            )
            cs = common.wait_for(cj)
            if not cs.success:
                row["status"] = f"COMPILE_FAILED:{cs.code}"
                print(
                    f"{m['key']:26} {dev.name:30} -> {row['status']} {cs.message[:80]}", flush=True
                )
                w.writerow(row)
                f.flush()
                continue
            pj = qai_hub.submit_inference_job(
                cj.get_target_model(),
                device=dev,
                inputs={
                    k: v
                    for k, v in common.sample_inputs(
                        {"dims": m["dims"], "key": m["key"]}, FP32_DIR / m["file"]
                    ).items()
                },
                profile=True,
                name=f"{TAG}-{m['key']}-{dev.name[:16]}-prof",
            )
            ps = common.wait_for(pj)
            if not ps.success:
                row["status"] = f"PROFILE_FAILED:{ps.code}"
                print(f"{m['key']:26} {dev.name:30} -> {row['status']}", flush=True)
                w.writerow(row)
                f.flush()
                continue
            s = common.profile_summary(pj)
            lat, mem = s["inference_time_ms"], s["inference_peak_memory"]
            row["status"] = "OK"
            row["latency_ms"] = f"{lat:.1f}" if lat else ""
            row["peak_mem_mb"] = f"{mem / 1e6:.1f}" if mem else ""
            print(
                f"{m['key']:26} {dev.name:30} -> {row['latency_ms']:>9} ms  {row['peak_mem_mb']:>7} MB",
                flush=True,
            )
            w.writerow(row)
            f.flush()

    f.close()
    print(f"\nresults -> {csv_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--devices", nargs="*", default=DEFAULT_DEVICES)
    parser.add_argument("--csv", default="/tmp/onevoice_qnn_latency.csv")
    raise SystemExit(main(parser.parse_args()))
