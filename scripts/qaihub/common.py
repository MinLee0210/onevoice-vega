"""Shared helpers for the Qualcomm AI Hub (qai-hub) edge-AI pipeline test.

Target device is the QCS6490, which AI Hub lists as
``Dragonwing RB3 Gen 2 Vision Kit`` (chipset ``qualcomm-qcs6490``,
``os:qc_linux``, ``hexagon:v68``, ``framework:qnn``) — the same silicon /
OS family as the Rubik Pi 3 prototype.

The models under test are the NPU-bound stages of the pipeline (see
AGENTS.md → Compute split): ASR encoder/decoder and MT encoder/decoder,
plus the small Silero VAD. TTS (Piper/MMS) and RNNoise denoise are
CPU-only by design and are not NPU targets, so they are not sent to the
Hub.

All jobs run on Qualcomm's cloud; results (compiled model size, inference
latency, peak memory) are what you'd budget against the device's NPU /
memory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import qai_hub

# The QCS6490 on Qualcomm AI Hub.
DEVICE_NAME = "Dragonwing RB3 Gen 2 Vision Kit"

# QNN requires 64-bit integer I/O to be truncated to 32-bit (token ids,
# attention masks, Silero's `sr`). Token ids are < 2^31 and masks are
# 0/1, so this is always safe for this pipeline.
COMPILE_OPTIONS = "--truncate_64bit_io"

# Default values for dynamic ONNX dims we can't infer from the graph.
# Per-model overrides (see MODELS below) win over these.
_DEFAULT_DIMS = {
    "batch_size": 1,
    "decoder_sequence_length": 1,
    "past_decoder_sequence_length": 1,
}

# The pipeline stages that target the Hexagon NPU (or are small enough to
# profile there). `dims` maps ONNX dynamic-dim names -> static size used
# for compile; `input_specs` is an explicit override for models whose
# dynamic dims aren't name-keyed (Silero VAD). `arch` is the transformer
# geometry (kind, d_model, n_layers, enc_seq) used for the theoretical
# FLOPs estimate.
MODELS: list[dict] = [
    {
        "key": "vad",
        "path": "models/vad/model.onnx",
        "stage": "VAD — Silero (512-sample frames)",
        # `sr` is a rank-0 scalar (int64 sample rate); qai-hub represents a
        # scalar input as `((), "int64")` (empty shape + dtype), not `()`.
        "input_specs": {"input": (1, 512), "state": (2, 1, 128), "sr": ((), "int64")},
        "arch": None,  # tiny GRU/LSTM, negligible
    },
    {
        "key": "asr_vi_encoder",
        "path": "models/asr/vi/encoder_model.onnx",
        "stage": "ASR encoder — PhoWhisper-base (VI), 30s-padded mel",
        "dims": {"feature_size": 80, "encoder_sequence_length": 3000},
        "arch": ("enc", 512, 6, 1500),
    },
    {
        "key": "asr_vi_decoder",
        "path": "models/asr/vi/decoder_model_merged.onnx",
        "stage": "ASR decoder — PhoWhisper-base (VI), merged KV-cache",
        "dims": {
            "encoder_sequence_length / 2": 1500,
            "encoder_sequence_length_out": 1500,
        },
        "arch": ("dec", 512, 6, 1500),
    },
    {
        "key": "asr_en_encoder",
        "path": "models/asr/en-zh-ko/encoder_model.onnx",
        "stage": "ASR encoder — Whisper-small (EN/ZH/KO), 30s-padded mel",
        "dims": {"feature_size": 80, "encoder_sequence_length": 3000},
        "arch": ("enc", 768, 12, 1500),
    },
    {
        "key": "asr_en_decoder",
        "path": "models/asr/en-zh-ko/decoder_model_merged.onnx",
        "stage": "ASR decoder — Whisper-small (EN/ZH/KO), 12 layers",
        "dims": {
            "encoder_sequence_length / 2": 1500,
            "encoder_sequence_length_out": 1500,
        },
        "arch": ("dec", 768, 12, 1500),
    },
    {
        "key": "mt_encoder",
        "path": "models/mt/vi-en/encoder_model.onnx",
        "stage": "MT encoder — Opus-MT (vi-en), INT8",
        "dims": {"encoder_sequence_length": 32},
        "arch": ("enc", 512, 6, 32),
    },
    {
        "key": "mt_decoder",
        "path": "models/mt/vi-en/decoder_model_merged.onnx",
        "stage": "MT decoder — Opus-MT (vi-en), INT8, merged KV-cache",
        "dims": {"encoder_sequence_length": 32},
        "arch": ("dec", 512, 6, 32),
    },
]

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_device() -> qai_hub.Device:
    for dev in qai_hub.get_devices():
        if dev.name == DEVICE_NAME:
            return dev
    raise RuntimeError(f"device {DEVICE_NAME!r} not found on AI Hub")


# ONNX TensorProto elem_type -> qai-hub dtype string (only the ones we use).
_ELEM_TYPE_TO_DTYPE = {1: "float32", 7: "int64", 9: "bool", 11: "float64"}


def auto_input_specs(model_path: Path, dims: dict | None = None) -> dict:
    """Derive a static ``{input_name: (shape, dtype)}`` map from the graph.

    Dynamic dims are resolved by their ``dim_param`` name, falling back to
    ``_DEFAULT_DIMS`` (then 1). The dtype is read from the ONNX so qai-hub
    doesn't default int64 inputs to float32.
    """
    import onnx

    merged = {**_DEFAULT_DIMS, **(dims or {})}
    model = onnx.load(str(model_path))
    specs: dict[str, tuple] = {}
    for inp in model.graph.input:
        shape: list[int] = []
        for d in inp.type.tensor_type.shape.dim:
            if d.dim_value and d.dim_value > 0:
                shape.append(d.dim_value)
            else:
                shape.append(merged.get(d.dim_param, 1))
        dtype = _ELEM_TYPE_TO_DTYPE.get(inp.type.tensor_type.elem_type, "float32")
        specs[inp.name] = (tuple(shape) if shape else (), dtype)
    return specs


def input_specs_for(entry: dict, model_path: Path) -> dict:
    if "input_specs" in entry:
        return entry["input_specs"]
    return auto_input_specs(model_path, entry.get("dims"))


def _spec_shape(spec) -> tuple[int, ...]:
    # An input_specs value is either a plain shape tuple or `(shape, dtype)`.
    return spec[0] if (spec and isinstance(spec[0], tuple)) else spec


def sample_inputs(entry: dict, model_path: Path) -> dict[str, list[np.ndarray]]:
    """Build one deterministic sample input per model input (zeros).

    Real preprocessing (mel spectrogram, token ids) isn't needed for a
    Hub-side *latency/size* pass — zeros exercise the exact same graph at
    the exact same static shapes. For output-correctness, run the real
    Rust pipeline locally (see scripts/run_pipeline.sh). Scalars (rank-0
    inputs, e.g. Silero VAD's `sr`) are fed as 1-element arrays so the
    dataset/calibration path (h5py) accepts them.
    """
    import onnx

    model = onnx.load(str(model_path))
    specs = input_specs_for(entry, model_path)
    dtype_map = {1: np.float32, 7: np.int64, 9: np.bool_, 11: np.float64}
    out: dict[str, list[np.ndarray]] = {}
    for inp in model.graph.input:
        name = inp.name
        dt = dtype_map.get(inp.type.tensor_type.elem_type, np.float32)
        ndim = len(inp.type.tensor_type.shape.dim)
        scalar = ndim == 0 or (name in specs and not _spec_shape(specs[name]))
        if scalar:
            # rank-0 scalar (e.g. Silero VAD's `sr`): feed as a 1-element
            # array so the dataset/calibration path (h5py) accepts it.
            arr = np.array([16000 if dt == np.int64 else 0], dtype=dt)
        elif name in specs:
            arr = np.zeros(_spec_shape(specs[name]), dtype=dt)
        else:
            dims = tuple(
                d.dim_value if d.dim_value > 0 else 1 for d in inp.type.tensor_type.shape.dim
            )
            arr = np.zeros(dims, dtype=dt)
        out[name] = [arr]
    return out


def wait_for(job, timeout_s: int = 900) -> qai_hub.JobStatus:
    return job.wait(timeout=timeout_s)


def humanize_bytes(n: float | int) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def humanize_ms(n: float | int | None) -> str:
    if n is None:
        return "n/a"
    n = float(n)
    if n >= 1000:
        return f"{n / 1000:.2f} s"
    return f"{n:.2f} ms"


def theoretical_flops(arch) -> float:
    """Analytic FLOPs estimate from the transformer geometry.

    ``arch = (kind, d_model, n_layers, enc_seq)``. Per token/layer:
    encoder ≈ (12·d² + 2·seq·d) MACs (QKV+out 4d², FFN 8d², attention
    2·seq·d); decoder per step ≈ (14·d² + 2·enc_seq·d) MACs (adds
    cross-attention). ×2 for FLOPs. Standard transformer approximation —
    conv/softmax/LayerNorm omitted (negligible). Encoders report the full
    forward pass; decoders report per-token (they run in a greedy loop).
    """
    if arch is None:
        return 0.0
    kind, d, layers, seq = arch
    d2 = d * d
    if kind == "enc":
        macs = seq * layers * (12 * d2 + 2 * seq * d)
    else:
        macs = layers * (14 * d2 + 2 * seq * d)
    return 2 * macs


def profile_summary(job: qai_hub.InferenceJob) -> dict:
    """Extract the edge-AI metrics from a profiled inference job.

    ``download_profile()`` returns ``{"execution_summary": {...},
    "execution_detail": [...]}``; the numbers we care about live in the
    ``execution_summary`` sub-dict, not the top level.
    """
    profile = job.download_profile() or {}
    summary = profile.get("execution_summary", {}) or {}
    return {
        "inference_time_ms": summary.get("estimated_inference_time"),
        "inference_peak_memory": summary.get("estimated_inference_peak_memory"),
        "warm_load_time_ms": summary.get("warm_load_time"),
        "compile_time_ms": summary.get("compile_time"),
        "inference_memory_peak_range": summary.get("inference_memory_peak_range"),
    }
