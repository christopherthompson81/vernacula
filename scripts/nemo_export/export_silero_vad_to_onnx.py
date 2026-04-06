from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the Silero VAD v5 model to the ONNX contract used by this repository.\n"
            "\n"
            "Loads via the silero-vad pip package (run 'pip install silero-vad' first).\n"
            "Exports the inner 16 kHz GRU model directly — no sr input needed in the graph.\n"
            "\n"
            "ONNX contract:\n"
            "  inputs:  input  [batch, 512]        float32\n"
            "           state  [2, batch, 128]      float32\n"
            "  outputs: output [batch, 1]           float32  (speech probability)\n"
            "           stateN [2, batch, 128]      float32  (updated GRU state)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=17, help="Target ONNX opset (default 17)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output file")
    return parser.parse_args()


def import_dependencies() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch not found. Install it with:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            f"Original error: {exc}"
        ) from exc
    try:
        from silero_vad import load_silero_vad
    except ImportError as exc:
        raise SystemExit(
            "silero-vad not found. Install it with:\n"
            "  pip install silero-vad\n"
            f"Original error: {exc}"
        ) from exc
    return torch, load_silero_vad


def load_and_wrap(torch: Any, load_silero_vad: Any) -> tuple[Any, int, int]:
    """
    Load the Silero VAD model and return an ONNX-friendly wrapper plus shape constants.

    The outer VADRNNJITMerge forward prepends a context buffer to every window:
        x_with_context = cat([context, x], dim=-1)   # shape [B, context+512]
    and passes that to the inner GRU model.  Ignoring the context (as simply
    exporting _model with 512-sample windows does) causes near-zero probabilities
    on real speech.

    The wrapper exposes context as an explicit round-trip tensor so the C# caller
    can maintain it across windows without any hidden internal state.

    Contract:
        inputs:  input   [1, 512]       float32   — new audio window
                 state   [2, 1, 128]    float32   — GRU hidden state
                 context [1, C]         float32   — trailing C samples from previous window
        outputs: output  [1, 1]         float32   — speech probability
                 stateN  [2, 1, 128]    float32   — updated GRU state
                 contextN[1, C]         float32   — context for next window
    """
    print("Loading silero-vad model…")
    outer = load_silero_vad(onnx=False)

    if not hasattr(outer, "_model"):
        raise SystemExit(
            "Could not find '_model' attribute. "
            "This exporter targets Silero VAD v5; update or pin silero-vad if the API changed."
        )

    inner        = outer._model
    context_size = int(inner.context_size_samples)
    window_size  = 512
    print(f"context_size_samples = {context_size}, window_size = {window_size}")

    # Export the inner model directly with a full (context + window) input.
    # The caller (C#) is responsible for prepending the 64-sample context buffer
    # before each inference call — keeping the ONNX graph completely static and
    # trivially exportable from a plain ScriptModule.
    #
    # Contract:
    #   inputs:  input [1, context+512]  float32  — context prepended by caller
    #            state [2, 1, 128]       float32  — GRU state
    #   outputs: output [1, 1]           float32  — speech probability
    #            stateN [2, 1, 128]      float32  — updated state

    full_input_size = context_size + window_size  # 576

    dummy_x     = torch.zeros(1, full_input_size)
    dummy_state = torch.zeros(2, 1, 128)

    with torch.no_grad():
        out, s_new = inner(dummy_x, dummy_state)
    print(f"Forward pass OK — output:{tuple(out.shape)}, stateN:{tuple(s_new.shape)}")

    return inner, window_size, context_size


@dataclass
class ExportMetadata:
    silero_vad_version: str
    opset: int
    window_samples: int
    sample_rate: int
    state_shape: list[int]
    notes: list[str]


def get_silero_version() -> str:
    try:
        from importlib.metadata import version
        return version("silero-vad")
    except Exception:
        return "unknown"


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()
    report_path = output_path.with_suffix(output_path.suffix + ".report.json")

    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {output_path}. Re-run with --overwrite.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch, load_silero_vad = import_dependencies()
    model, window_size, context_size = load_and_wrap(torch, load_silero_vad)

    full_input_size = window_size + context_size  # 576
    dummy_x     = torch.zeros(1, full_input_size)
    dummy_state = torch.zeros(2, 1, 128)

    print(f"Exporting to {output_path} (opset {args.opset})…")
    torch.onnx.export(
        model,
        (dummy_x, dummy_state),
        str(output_path),
        input_names=["input", "state"],
        output_names=["output", "stateN"],
        dynamic_axes={
            "input":  {0: "batch"},
            "state":  {1: "batch"},
            "output": {0: "batch"},
            "stateN": {1: "batch"},
        },
        opset_version=args.opset,
        dynamo=False,
    )

    # Quick round-trip check via onnxruntime
    try:
        import onnx
        import onnxruntime as ort
        import numpy as np

        onnx.checker.check_model(str(output_path))

        sess  = ort.InferenceSession(str(output_path))
        feeds = {
            "input": np.zeros((1, full_input_size), np.float32),
            "state": np.zeros((2, 1, 128), np.float32),
        }
        out_np, state_np = sess.run(None, feeds)
        print(f"ORT round-trip OK — output:{out_np.shape}, stateN:{state_np.shape}")
    except ImportError:
        print("(onnx / onnxruntime not installed — skipping round-trip check)")

    version = get_silero_version()
    metadata = ExportMetadata(
        silero_vad_version=version,
        opset=args.opset,
        window_samples=window_size,
        sample_rate=16_000,
        state_shape=[2, 1, 128],
        notes=[
            f"Exports inner 16 kHz GRU (_model) directly with full input size {full_input_size} (context={context_size} + window=512).",
            "The caller (C# VadSegmenter) maintains a 64-sample context buffer and prepends it to each 512-sample window.",
            "Inputs: input [1,576], state [2,1,128]. Outputs: output [1,1], stateN [2,1,128].",
        ],
    )

    with report_path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(asdict(metadata), fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    print(output_path)
    print(report_path)


if __name__ == "__main__":
    main()
