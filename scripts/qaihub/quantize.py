#!/usr/bin/env python3
"""Quantize a model to a lower precision via qai-hub's PTQ and compare.

qai-hub exposes ``submit_quantize_job`` with weights/activations dtype in
{INT8, INT16, INT4} — there is **no FP8** option in the SDK. This script
lets you pick a target dtype, runs the quantize job, and reports the
source vs. quantized size; ``--compile`` additionally compiles the result
for the QCS6490 to check the target NPU actually accepts that precision.

Usage:
    python3 scripts/qaihub/quantize.py --key vad --weights int4 --compile
    python3 scripts/qaihub/quantize.py --key mt_encoder --weights int8

Notes:
  * The bundled ASR/MT models are **already INT8** (the setup script pulls
    the ``*_quantized`` ONNX exports), so this is most useful on the FP32
    models (``vad``) or to push an INT8 model to INT4.
  * Calibration data here is zeros — a placeholder. Real activation
    calibration needs representative audio / token ids (see the pipeline
    fixtures); for weights-only INT4 the size delta is still representative.
  * Whether INT4 helps at *runtime* is a different question: see the
    "why lower precision may not help" notes in the README.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import common
import qai_hub

TAG = "onevoice"


def resolve_dtype(name: str) -> qai_hub.QuantizeDtype:
    name = name.lower()
    mapping = {
        "int8": qai_hub.QuantizeDtype.INT8,
        "int16": qai_hub.QuantizeDtype.INT16,
        "int4": qai_hub.QuantizeDtype.INT4,
    }
    if name not in mapping:
        raise SystemExit(f"unknown dtype {name!r} (pick int8/int16/int4)")
    return mapping[name]


def main(key: str, weights: str, activations: str | None, compile_: bool) -> int:
    entry = next((m for m in common.MODELS if m["key"] == key), None)
    if entry is None:
        raise SystemExit(f"unknown model key {key!r} (see common.MODELS)")

    path = common.REPO_ROOT / entry["path"]
    src_bytes = path.stat().st_size

    model = qai_hub.upload_model(str(path), name=f"{TAG}-{key}")
    calib = qai_hub.upload_dataset(common.sample_inputs(entry, path), name=f"{TAG}-{key}-calib")

    w_dtype = resolve_dtype(weights)
    a_dtype = resolve_dtype(activations) if activations else qai_hub.QuantizeDtype.INT8
    print(f"quantizing {key}: weights={weights} activations={a_dtype}")

    job = qai_hub.submit_quantize_job(
        model,
        calibration_data=calib,
        weights_dtype=w_dtype,
        activations_dtype=a_dtype,
        name=f"{TAG}-{key}-{weights}",
    )
    status = common.wait_for(job)
    if not status.success:
        print(f"QUANTIZE FAILED: {status.code} — {status.message}")
        return 1

    target = job.get_target_model()
    out_path = Path(f"/tmp/{TAG}-{key}-{weights}.onnx")
    downloaded = job.download_target_model(str(out_path))
    q_bytes = Path(downloaded).stat().st_size if downloaded else None
    print(f"  source    {common.humanize_bytes(src_bytes)}")
    print(
        f"  quantized {common.humanize_bytes(q_bytes or 0)}"
        + (f"  ({q_bytes / src_bytes * 100:.0f}% of source)" if q_bytes else "")
    )

    if compile_:
        device = common.resolve_device()
        cjob = qai_hub.submit_compile_job(
            target,
            device=device,
            input_specs=common.input_specs_for(entry, path),
            options=common.COMPILE_OPTIONS,
            name=f"{TAG}-{key}-{weights}-compile",
        )
        cstatus = common.wait_for(cjob)
        print(f"  compile on QCS6490: {cstatus.code}")
        if not cstatus.success:
            print(f"    {cstatus.message}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True, help="model key (see common.MODELS)")
    parser.add_argument("--weights", default="int8", help="int8/int16/int4")
    parser.add_argument("--activations", default=None, help="int8/int16/int4 (default int8)")
    parser.add_argument("--compile", action="store_true", help="also compile for QCS6490")
    args = parser.parse_args()
    sys.exit(main(args.key, args.weights, args.activations, args.compile))
