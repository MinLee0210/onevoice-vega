#!/usr/bin/env python3
"""Strip a dynamic float attention mask from an exported seq2seq ONNX.

The QCS6490 Hexagon HTP is INT8-only and rejects the float additive
attention mask that `optimum`/`transformers` bakes into Marian/Opus-MT
exports (`Expand -> Cast(float) -> Sub/Where -> Add(scores, mask)`):
compiling such a graph fails with

    Tensor '/Cast_output_0_pre_quant' has a floating-point type which is
    not supported by the targeted device.

Our runtime always feeds an all-ones attention mask (batch=1, no padding,
see `src/stages/mt.rs`), so the additive mask is identically zero and the
`Add(scores, mask)` nodes can be bypassed with no change in output. This
rewrites every `Add` fed by a mask-derived `Where` so it just passes the
attention scores through, then dead-code-eliminates the mask subgraph.

Usage:
    python3 scripts/qaihub/strip_float_attention_mask.py in.onnx out.onnx
"""

from __future__ import annotations

import sys
from pathlib import Path

import onnx


def _mask_producers(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    return {out: n for n in model.graph.node for out in n.output}


def _derives_from(producers, node, roots: set[str], seen: set[str]) -> bool:
    """True if any input of `node` transitively depends on a graph input in
    `roots` (e.g. "attention_mask" / "encoder_attention_mask")."""
    stack = [node]
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        for i in n.input:
            if i in roots:
                return True
            p = producers.get(i)
            if p is not None:
                stack.append(p)
    return False


def strip(in_path: Path, out_path: Path) -> int:
    model = onnx.load(str(in_path))
    graph = model.graph
    producers = _mask_producers(model)
    mask_roots = {"attention_mask", "encoder_attention_mask"}

    bypass: dict[str, str] = {}  # Add output -> passthrough input
    for node in graph.node:
        if node.op_type != "Add" or len(node.input) != 2:
            continue
        mask_inputs = [
            inp
            for inp in node.input
            if (p := producers.get(inp)) is not None
            and p.op_type == "Where"
            and _derives_from(producers, p, mask_roots, set())
        ]
        if len(mask_inputs) != 1:
            continue
        keep = next(i for i in node.input if i not in mask_inputs)
        bypass[node.output[0]] = keep

    if not bypass:
        print("no float attention-mask Adds found; nothing to strip")
        return 0

    # Rewire consumers of each bypassed Add output to the passthrough input.
    for node in graph.node:
        for i, inp in enumerate(node.input):
            if inp in bypass:
                node.input[i] = bypass[inp]

    # Dead-code elimination: keep only nodes reachable from graph outputs.
    keep_ids: set[int] = set()
    stack = [o.name for o in graph.output]
    while stack:
        name = stack.pop()
        n = producers.get(name)
        if n is None or id(n) in keep_ids:
            continue
        keep_ids.add(id(n))
        stack.extend(n.input)
    nodes = [n for n in graph.node if id(n) in keep_ids]
    del graph.node[:]
    graph.node.extend(nodes)

    used = {i for n in graph.node for i in n.input}
    inits = [t for t in graph.initializer if t.name in used]
    del graph.initializer[:]
    graph.initializer.extend(inits)
    vinfos = [v for v in graph.value_info if v.name in used]
    del graph.value_info[:]
    graph.value_info.extend(vinfos)

    onnx.checker.check_model(model)
    onnx.save(model, str(out_path))
    print(f"stripped {len(bypass)} float mask Add(s) -> {out_path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: strip_float_attention_mask.py in.onnx out.onnx")
    raise SystemExit(strip(Path(sys.argv[1]), Path(sys.argv[2])))
