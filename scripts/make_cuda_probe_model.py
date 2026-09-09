#!/usr/bin/env python3
"""Regenerate the CUDA execution-provider probe graph.

The probe is the smallest model that still makes ONNX Runtime build a CUDA
kernel and put a weight on the device: a 1x1x1x1 Conv whose filter is an
initializer. It is embedded in ModelManagerService as base64 rather than shipped
as a file, because the check it backs must work on a machine that has downloaded
no models at all -- which is exactly the machine the old check, which loaded the
Parakeet preprocessor, reported as having no CUDA.

Prints the base64 to paste into ModelManagerService.CudaProbeModelBase64.

    pip install onnx onnxruntime numpy
    python scripts/make_cuda_probe_model.py
"""

import base64

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

WRAP = 96  # characters per C# string literal line


def build() -> bytes:
    weight = numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float32), name="W")
    graph = helper.make_graph(
        [helper.make_node("Conv", ["X", "W"], ["Y"], name="probe")],
        "cuda_ep_probe",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 1, 1])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 1, 1])],
        [weight],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], producer_name=""
    )
    # Older than the newest ORT understands, on purpose: the probe has to load on
    # whatever runtime the app was built against, including an older one.
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model.SerializeToString()


def main() -> None:
    payload = build()

    # Prove it loads before printing it: a blob that only fails at CUDA session
    # creation would look exactly like a machine with no CUDA.
    session = ort.InferenceSession(payload, providers=["CPUExecutionProvider"])
    out = session.run(None, {"X": np.ones((1, 1, 1, 1), dtype=np.float32)})[0]
    assert out.shape == (1, 1, 1, 1) and out.ravel()[0] == 1.0, out

    # Printed as the C# literal it becomes, continuations included, so it can be pasted verbatim.
    encoded = base64.b64encode(payload).decode()
    print(f"// {len(payload)} bytes")
    for i in range(0, len(encoded), WRAP):
        prefix = "    " if i == 0 else "        + "
        print(f'{prefix}"{encoded[i:i + WRAP]}"')


if __name__ == "__main__":
    main()
