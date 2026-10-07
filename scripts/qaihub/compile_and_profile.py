#!/usr/bin/env python3
"""Upload, compile, and profile every pipeline stage on the QCS6490.

For each model in ``common.MODELS``:
  1. upload the ONNX to Qualcomm AI Hub,
  2. compile it for the QCS6490 (static input shapes),
  3. profile the compiled model (inference latency, peak memory),
  4. report source size, compiled-context size, latency, memory, and an
     estimated FLOPs number.

Usage:
    python3 scripts/qaihub/compile_and_profile.py [--keys KEY ...]

Requires a configured AI Hub token (``~/.qai_hub/client.ini``, or set
``QAIHUB_API_KEY`` and run ``scripts/qaihub/configure.py``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import common
import qai_hub

TAG = "onevoice"


def main(keys: list[str] | None) -> int:
    device = common.resolve_device()
    print(f"device: {device.name}  (QCS6490, {getattr(device, 'os', '?')} OS)\n")

    models = [m for m in common.MODELS if keys is None or m["key"] in keys]
    if not models:
        print("no models selected")
        return 1

    rows: list[dict] = []
    for entry in models:
        key, path = entry["key"], common.REPO_ROOT / entry["path"]
        print(f"== {key}: {entry['stage']} ==")

        specs = common.input_specs_for(entry, path)
        gflops = common.theoretical_flops(entry.get("arch"))
        src_bytes = path.stat().st_size

        model = qai_hub.upload_model(str(path), name=f"{TAG}-{key}")
        print(
            f"  uploaded  {model.model_id}  src_size={common.humanize_bytes(src_bytes)}  "
            f"theoretical={gflops / 1e9:.2f} GFLOPs"
        )

        compile_job = qai_hub.submit_compile_job(
            model,
            device=device,
            input_specs=specs,
            options=common.COMPILE_OPTIONS,
            name=f"{TAG}-{key}-compile",
        )
        cstatus = common.wait_for(compile_job)
        row = {
            "key": key,
            "stage": entry["stage"],
            "src_mb": src_bytes / 1e6,
            "est_gflops": gflops / 1e9,
            "compile": cstatus.code,
            "ctx_mb": None,
            "latency_ms": None,
            "peak_mem_mb": None,
        }
        if not cstatus.success:
            print(f"  COMPILE FAILED: {cstatus.code} — {cstatus.message}")
            rows.append(row)
            print()
            continue

        target = compile_job.get_target_model()
        ctx_bytes = None
        ctx_path = Path(f"/tmp/{TAG}-{key}.bin")
        try:
            downloaded = compile_job.download_target_model(str(ctx_path))
            if downloaded:
                ctx_bytes = Path(downloaded).stat().st_size
        except Exception as e:  # noqa: BLE001
            print(f"  (could not download compiled context: {e})")
        row["ctx_mb"] = ctx_bytes / 1e6 if ctx_bytes else None

        profile_job = qai_hub.submit_inference_job(
            target,
            device=device,
            inputs=common.sample_inputs(entry, path),
            profile=True,
            name=f"{TAG}-{key}-profile",
        )
        pstatus = common.wait_for(profile_job)
        if pstatus.success:
            summary = common.profile_summary(profile_job)
            row["latency_ms"] = summary["inference_time_ms"]
            peak = summary["inference_peak_memory"]
            if peak is None and summary.get("inference_memory_peak_range"):
                lo, hi = summary["inference_memory_peak_range"]
                peak = (lo + hi) / 2
            row["peak_mem_mb"] = peak / 1e6 if peak else None
            print(
                f"  PROFILE    latency={common.humanize_ms(row['latency_ms'])}  "
                f"peak_mem={common.humanize_bytes(peak or 0)}  "
                f"ctx={common.humanize_bytes(ctx_bytes or 0)}"
            )
        else:
            print(f"  PROFILE FAILED: {pstatus.code} — {pstatus.message}")

        rows.append(row)
        print()

    _print_table(rows)
    return 0


def _print_table(rows: list[dict]) -> None:
    print("\n" + "=" * 100)
    print(
        f"{'model':16} {'src(MB)':>8} {'ctx(MB)':>8} {'latency':>10} "
        f"{'peakRAM(MB)':>12} {'est GFLOPs':>11} {'status':>9}"
    )
    print("-" * 100)
    for r in rows:

        def f(v, fmt):
            return fmt.format(v) if v is not None else "-"

        print(
            f"{r['key']:16} {f(r['src_mb'], '{:.1f}'):>8} {f(r['ctx_mb'], '{:.1f}'):>8} "
            f"{f(r['latency_ms'], '{:.2f}'):>10} {f(r['peak_mem_mb'], '{:.1f}'):>12} "
            f"{f(r['est_gflops'], '{:.2f}'):>11} {r['compile']:>9}"
        )
    print("=" * 100)
    print("latency/peakRAM are the NPU inference numbers (not the full button-release")
    print("latency — that includes CPU denoise/VAD/TTS + the autoregressive decode loop,")
    print("measured locally at ~740ms end-to-end via scripts/run_pipeline.sh).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keys", nargs="*", help="limit to these model keys")
    args = parser.parse_args()
    sys.exit(main(args.keys))
