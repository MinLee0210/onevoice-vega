"""Unit tests for scripts/qaihub/common.py — the pure, network-free parts.

The qai_hub cloud calls (upload/compile/profile/inference) are exercised by
the scripts themselves against the live API, not here. These tests cover
the deterministic logic: FLOPs estimates, unit formatting, ONNX input-spec
derivation, and sample-input generation (against a tiny synthetic model so
the tests don't depend on the gitignored models/ bundle).
"""

from __future__ import annotations

from pathlib import Path

import common
import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper


def make_synthetic_model(path: Path) -> Path:
    """A tiny ONNX model with a dynamic float input, an int64 input, and a
    rank-0 scalar input, so spec/sample derivation can be tested offline."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 512])
    ids = helper.make_tensor_value_info("ids", TensorProto.INT64, [1, "seq"])
    sr = helper.make_tensor_value_info("sr", TensorProto.INT64, [])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 1])
    node = helper.make_node("ReduceSum", ["x"], ["y"], keepdims=0)
    graph = helper.make_graph([node], "g", [x, ids, sr], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, str(path))
    return path


@pytest.fixture
def synthetic_model(tmp_path: Path) -> Path:
    return make_synthetic_model(tmp_path / "m.onnx")


# --- theoretical_flops -----------------------------------------------------


def test_theoretical_flops_none_is_zero():
    assert common.theoretical_flops(None) == 0.0


def test_theoretical_flops_encoder():
    d, layers, seq = 512, 6, 1500
    expected = 2 * seq * layers * (12 * d * d + 2 * seq * d)
    assert common.theoretical_flops(("enc", d, layers, seq)) == pytest.approx(expected)


def test_theoretical_flops_decoder_is_per_token():
    # A decoder reports one step, so its FLOPs are far below the encoder's
    # full-pass count for the same geometry.
    enc = common.theoretical_flops(("enc", 512, 6, 1500))
    dec = common.theoretical_flops(("dec", 512, 6, 1500))
    assert dec < enc


def test_theoretical_flops_scales_with_size():
    # whisper-small (768/12) > PhoWhisper-base (512/6) for the same seq.
    small = common.theoretical_flops(("enc", 768, 12, 1500))
    base = common.theoretical_flops(("enc", 512, 6, 1500))
    assert small > base


# --- formatting helpers ----------------------------------------------------


def test_humanize_bytes():
    assert common.humanize_bytes(0) == "0.0 B"
    assert common.humanize_bytes(1024) == "1.0 KB"
    assert common.humanize_bytes(2 * 1024 * 1024) == "2.0 MB"


def test_humanize_ms():
    assert common.humanize_ms(None) == "n/a"
    assert common.humanize_ms(37.5) == "37.50 ms"
    assert common.humanize_ms(1500) == "1.50 s"


# --- _spec_shape -----------------------------------------------------------


def test_spec_shape_plain_tuple():
    assert common._spec_shape((1, 512)) == (1, 512)


def test_spec_shape_with_dtype():
    assert common._spec_shape(((1, 512), "float32")) == (1, 512)


def test_spec_shape_scalar():
    assert common._spec_shape(((), "int64")) == ()


# --- auto_input_specs / input_specs_for ------------------------------------


def test_auto_input_specs_resolves_dynamic_dims(synthetic_model):
    specs = common.auto_input_specs(synthetic_model)
    assert specs["x"] == ((1, 512), "float32")  # batch -> 1
    assert specs["ids"] == ((1, 1), "int64")  # seq -> default 1
    assert specs["sr"] == ((), "int64")  # scalar


def test_auto_input_specs_honors_overrides(synthetic_model):
    specs = common.auto_input_specs(synthetic_model, {"batch": 4, "seq": 8})
    assert specs["x"] == ((4, 512), "float32")
    assert specs["ids"] == ((1, 8), "int64")


def test_input_specs_for_prefers_explicit(synthetic_model):
    entry = {"input_specs": {"x": (2, 3)}}
    assert common.input_specs_for(entry, synthetic_model) == {"x": (2, 3)}


def test_input_specs_for_falls_back_to_auto(synthetic_model):
    entry = {"dims": {"seq": 16}}
    specs = common.input_specs_for(entry, synthetic_model)
    assert specs["ids"] == ((1, 16), "int64")


# --- sample_inputs ---------------------------------------------------------


def test_sample_inputs_generates_all_inputs(synthetic_model):
    entry: dict = {}
    inputs = common.sample_inputs(entry, synthetic_model)
    assert set(inputs) == {"x", "ids", "sr"}
    assert inputs["x"][0].shape == (1, 512)
    assert inputs["x"][0].dtype == np.float32
    assert inputs["ids"][0].dtype == np.int64
    # scalar `sr` is fed as a 1-element int64 array (h5py-safe)
    assert inputs["sr"][0].shape == (1,)
    assert inputs["sr"][0][0] == 16000


def test_sample_inputs_are_zeros(synthetic_model):
    inputs = common.sample_inputs({}, synthetic_model)
    assert np.all(inputs["x"][0] == 0)


# --- MODELS manifest integrity --------------------------------------------


def test_model_keys_are_unique():
    keys = [m["key"] for m in common.MODELS]
    assert len(keys) == len(set(keys))


def test_model_entries_have_required_fields():
    for m in common.MODELS:
        assert {"key", "path", "stage"} <= set(m)


def test_model_arch_tuples_are_well_formed():
    for m in common.MODELS:
        arch = m.get("arch")
        if arch is None:
            continue
        kind, d, layers, seq = arch
        assert kind in ("enc", "dec")
        assert d > 0 and layers > 0 and seq > 0
