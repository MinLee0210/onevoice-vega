#!/usr/bin/env python3
"""Compile + profile every pipeline stage across Qualcomm edge devices.

Produces a model x device latency matrix (inference time, peak memory,
compiled-context size) and writes results to a CSV incrementally so a
partial run is resumable (already-done model/device pairs are skipped).

Usage:
    python3 scripts/qaihub/benchmark_devices.py [--keys KEY ...] [--devices "Name A" "Name B" ...]

The default device list is the Qualcomm SoC / IoT / automotive / compute
platforms (the "edge" targets); consumer phones are excluded. Compile
failures on a device are recorded as status=FAILED, not fatal.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import common
import qai_hub

TAG = "onevoice"

# Curated Qualcomm edge/IoT/compute platforms (excludes consumer phones).
DEFAULT_DEVICES = [
    "Dragonwing RB3 Gen 2 Vision Kit",  # QCS6490 — our target
    "Dragonwing Q-6690 MTP",  # QCM6690
    "Dragonwing IQ-9075 EVK",  # QCS9075
    "Arduino VENTUNO Q",  # QCS8275
    "QCS8550 (Proxy)",  # QCS8550
    "SA7255P ADP",  # automotive
    "SA8295P ADP",  # automotive
    "SA8775P ADP",  # automotive
    "Snapdragon 7 Gen 4 QRD",
    "Snapdragon X Elite CRD",  # windows
    "Snapdragon X Plus 8-Core CRD",  # windows
    "Snapdragon X2 Elite CRD",  # windows
]

CSV_COLS = ["model", "device", "chipset", "status", "latency_ms", "peak_mem_mb", "ctx_mb"]


def _chipset(device: qai_hub.Device) -> str:
    for attr in getattr(device, "attributes", []) or []:
        if attr.startswith("chipset:"):
            return attr.split(":", 1)[1]
    return "?"


def load_done(csv_path: Path) -> set[tuple[str, str]]:
    if not csv_path.exists():
        return set()
    with csv_path.open() as f:
        return {(r["model"], r["device"]) for r in csv.DictReader(f)}


def resolve_devices(names: list[str]) -> list[qai_hub.Device]:
    all_devs = {d.name: d for d in qai_hub.get_devices()}
    ordered = [all_devs[n] for n in names if n in all_devs]
    return ordered


def profile(job: qai_hub.InferenceJob) -> tuple[float | None, float | None]:
    s = common.profile_summary(job)
    latency = s["inference_time_ms"]
    peak = s["inference_peak_memory"]
    if peak is None and s.get("inference_memory_peak_range"):
        lo, hi = s["inference_memory_peak_range"]
        peak = (lo + hi) / 2
    return latency, peak / 1e6 if peak else None


def main(keys: list[str] | None, device_names: list[str], csv_path: Path) -> int:
    devices = resolve_devices(device_names)
    models = [m for m in common.MODELS if keys is None or m["key"] in keys]
    if not devices or not models:
        print("no devices/models selected")
        return 1

    done = load_done(csv_path)
    print(f"devices: {len(devices)}, models: {len(models)}, already done: {len(done)}")

    # Upload each model once (content-addressed, deduped server-side).
    uploaded: dict[str, qai_hub.Model] = {}
    for entry in models:
        path = common.REPO_ROOT / entry["path"]
        uploaded[entry["key"]] = qai_hub.upload_model(str(path), name=f"{TAG}-{entry['key']}")

    f = csv_path.open("a", newline="")
    writer = csv.DictWriter(f, fieldnames=CSV_COLS)
    if csv_path.stat().st_size == 0:
        writer.writeheader()

    for device in devices:
        for entry in models:
            key = entry["key"]
            if (key, device.name) in done:
                continue
            path = common.REPO_ROOT / entry["path"]
            specs = common.input_specs_for(entry, path)
            row = {
                "model": key,
                "device": device.name,
                "chipset": _chipset(device),
                "status": "?",
                "latency_ms": "",
                "peak_mem_mb": "",
                "ctx_mb": "",
            }

            cjob = qai_hub.submit_compile_job(
                uploaded[key],
                device=device,
                input_specs=specs,
                options=common.COMPILE_OPTIONS,
                name=f"{TAG}-{key}-{device.name[:20]}-compile",
            )
            cstatus = common.wait_for(cjob)
            if not cstatus.success:
                row["status"] = f"COMPILE_FAILED:{cstatus.code}"
                print(f"{device.name:32} {key:16} -> {row['status']}", flush=True)
                writer.writerow(row)
                f.flush()
                continue

            # compiled context size
            ctx_bytes = None
            tmp = Path(f"/tmp/{TAG}-{key}-{device.name[:12]}.bin")
            try:
                dl = cjob.download_target_model(str(tmp))
                if dl:
                    ctx_bytes = Path(dl).stat().st_size
            except Exception:  # noqa: BLE001
                pass
            row["ctx_mb"] = f"{ctx_bytes / 1e6:.1f}" if ctx_bytes else ""

            pjob = qai_hub.submit_inference_job(
                cjob.get_target_model(),
                device=device,
                inputs=common.sample_inputs(entry, path),
                profile=True,
                name=f"{TAG}-{key}-{device.name[:20]}-profile",
            )
            pstatus = common.wait_for(pjob)
            if not pstatus.success:
                row["status"] = f"PROFILE_FAILED:{pstatus.code}"
                print(f"{device.name:32} {key:16} -> {row['status']}", flush=True)
                writer.writerow(row)
                f.flush()
                continue

            latency, peak = profile(pjob)
            row["status"] = "OK"
            row["latency_ms"] = f"{latency:.2f}" if latency is not None else ""
            row["peak_mem_mb"] = f"{peak:.1f}" if peak is not None else ""
            print(
                f"{device.name:32} {key:16} -> {row['latency_ms'] or 'n/a':>8} ms  "
                f"{row['peak_mem_mb'] or 'n/a':>8} MB",
                flush=True,
            )
            writer.writerow(row)
            f.flush()

    f.close()
    print(f"\nresults -> {csv_path}")
    _print_matrix(csv_path, models)
    return 0


def _print_matrix(csv_path: Path, models: list[dict]) -> None:
    if not csv_path.exists():
        return
    rows = list(csv.DictReader(csv_path.open()))
    keys = [m["key"] for m in models]
    devices: list[str] = []
    for r in rows:
        if r["device"] not in devices:
            devices.append(r["device"])
    print("\nLatency (ms) matrix — model x device:")
    header = f"{'model':16} " + " ".join(f"{d[:14]:>14}" for d in devices)
    print(header)
    for key in keys:
        line = f"{key:16} "
        for d in devices:
            r = next((x for x in rows if x["model"] == key and x["device"] == d), None)
            if r and r["status"] == "OK":
                line += f"{float(r['latency_ms']):>14.1f}"
            else:
                line += f"{'-':>14}"
        print(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keys", nargs="*", help="limit to these model keys")
    parser.add_argument("--devices", nargs="*", help="limit to these device names")
    parser.add_argument("--csv", default=str(common.REPO_ROOT / "data" / "onevoice_latency_devices.csv"))
    args = parser.parse_args()
    sys.exit(main(args.keys, args.devices or DEFAULT_DEVICES, Path(args.csv)))
