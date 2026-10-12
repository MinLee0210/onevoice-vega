#!/usr/bin/env python3
"""Dynamic INT8-quantize every Opus-MT leg under models/mt/.

The runtime loads whatever ONNX is in each `models/mt/<pair>/` dir; the v3
bundle ships INT8 to stay within GitHub's 2 GB/asset limit (fp32 legs are
~3 GB total). Uses `onnxruntime.quantization.quantize_dynamic` (weight-only,
per-tensor) — the same dynamic-int8 scheme the v2 bundle used. Originals are
replaced in place; already-int8 legs are skipped (re-quantizing corrupts).

    uv run --no-project --python 3.12 --with onnx --with onnxruntime \
      python scripts/edge/quantize_mt.py
"""

from __future__ import annotations

from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

# int8 encoder ~47 MB, fp32 ~180 MB+ — anything smaller is already quantized.
FP32_MIN_BYTES = 120_000_000


def main() -> int:
    root = Path("models/mt")
    dirs = sorted(d for d in root.glob("*") if (d / "encoder_model.onnx").exists())
    if not dirs:
        raise SystemExit("no models/mt/*/encoder_model.onnx found")

    for save_dir in dirs:
        print(f"== {save_dir} ==")
        for component in ("encoder_model", "decoder_model_merged"):
            onnx_path = save_dir / f"{component}.onnx"
            if not onnx_path.exists():
                continue
            if onnx_path.stat().st_size < FP32_MIN_BYTES:
                print(f"  {component}: already int8, skipping")
                continue
            tmp = save_dir / f"{component}.int8.onnx"
            quantize_dynamic(
                model_input=str(onnx_path),
                model_output=str(tmp),
                weight_type=QuantType.QInt8,
                per_channel=False,
            )
            orig_mb = onnx_path.stat().st_size / 1e6
            q_mb = tmp.stat().st_size / 1e6
            tmp.replace(onnx_path)
            print(f"  {component}: {orig_mb:.0f} MB -> {q_mb:.0f} MB")

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
