#!/usr/bin/env python3
"""Run on-device inference (on AI Hub's QCS6490) for each model.

Uploads + compiles each model (qai-hub dedups server-side by content, so
re-runs reuse the cached compile), then feeds a deterministic zero input
at its static shape and reports each output's shape + sum (a cheap
drift/smoke signal, not a correctness check).

Usage:
    python3 scripts/qaihub/inference.py [--keys KEY ...]
"""

from __future__ import annotations

import argparse
import sys

import common
import qai_hub

TAG = "onevoice"


def main(keys: list[str] | None) -> int:
    device = common.resolve_device()

    models = [m for m in common.MODELS if keys is None or m["key"] in keys]
    for entry in models:
        key = entry["key"]
        path = common.REPO_ROOT / entry["path"]
        specs = common.input_specs_for(entry, path)
        inputs = common.sample_inputs(entry, path)
        print(f"== {key}: {entry['stage']} ==")
        print("   inputs: " + ", ".join(f"{n}:{arr.shape}" for n, (arr,) in inputs.items()))

        # Reuse the compiled context by name (qai-hub dedups by content).
        model = qai_hub.upload_model(str(path), name=f"{TAG}-{key}")
        cjob = qai_hub.submit_compile_job(
            model,
            device=device,
            input_specs=specs,
            options=common.COMPILE_OPTIONS,
            name=f"{TAG}-{key}-compile",
        )
        cstatus = common.wait_for(cjob)
        if not cstatus.success:
            print(f"   COMPILE FAILED: {cstatus.code} — {cstatus.message}")
            continue

        ijob = qai_hub.submit_inference_job(
            cjob.get_target_model(),
            device=device,
            inputs=inputs,
            name=f"{TAG}-{key}-infer",
        )
        istatus = common.wait_for(ijob)
        if not istatus.success:
            print(f"   INFER FAILED: {istatus.code} — {istatus.message}")
            continue

        outputs = ijob.download_output_data()
        for name, arrs in (outputs or {}).items():
            for arr in arrs:
                print(
                    f"   out {name}: shape={arr.shape} dtype={arr.dtype} sum={float(arr.sum()):.4f}"
                )
        print()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keys", nargs="*", help="limit to these model keys")
    args = parser.parse_args()
    sys.exit(main(args.keys))
