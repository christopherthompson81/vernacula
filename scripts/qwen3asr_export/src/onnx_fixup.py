"""Post-export ONNX graph fixups for runtime compatibility."""

from __future__ import annotations

import onnx


def fix_reshape_allowzero(onnx_path: str) -> int:
    """Remove allowzero=1 from Reshape nodes for DirectML compatibility."""
    model = onnx.load(onnx_path, load_external_data=False)
    count = 0
    for node in model.graph.node:
        if node.op_type != "Reshape":
            continue
        for attr in list(node.attribute):
            if attr.name == "allowzero" and attr.i == 1:
                node.attribute.remove(attr)
                count += 1
    if count > 0:
        onnx.save(model, onnx_path)
    return count
