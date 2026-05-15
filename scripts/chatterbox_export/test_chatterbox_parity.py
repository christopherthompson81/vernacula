#!/usr/bin/env python3
"""End-to-end parity check: ONNX export vs PyTorch reference.

Stage 0 / step E4: skeleton only. Wires up once the export script
emits real artifacts.

The parity protocol is three layers, in order of strictness:

  1. **Speech-token sequence parity (LM):** same prompt + same reference
     audio through PyTorch and through our ONNX pipeline; assert
     near-identical token sequences (modulo numerical drift in early
     positions; allow first-divergence > N tokens). Same idiom as
     `scripts/vibevoice_export/test_static_kv_parity.py`.

  2. **Decoder waveform parity:** feed identical speech-tokens +
     speaker conditioning to the PyTorch decoder and our ONNX decoder.
     Compare via mel-spectral distance — bit-exact is unrealistic for a
     vocoder.

  3. **End-to-end audio parity:** full pipeline both ways, spectral
     distance + a short listen-test sample dumped to disk for the
     investigation doc.

Layers 1 and 2 are CI-gateable. Layer 3 is informational.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from _common import (
    add_local_script_path,
    choose_onnx_providers,
    fail,
    read_export_report,
)

add_local_script_path()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx-dir", type=Path, required=True,
                   help="Directory produced by export_chatterbox_to_onnx.py")
    p.add_argument("--audio", type=Path, required=True,
                   help="Reference voice clip (any sample rate, mono or stereo)")
    p.add_argument("--text", type=str, default="The Lord of the Rings is the greatest work of literature.",
                   help="Prompt text for parity comparison")
    p.add_argument("--max-tokens", type=int, default=256,
                   help="Max LM steps for token-parity check (default: 256)")
    p.add_argument("--runtime", default="cuda", choices=["cpu", "cuda", "tensorrt"],
                   help="ONNX Runtime EP family (default: cuda)")
    p.add_argument("--no-decoder-parity", action="store_true",
                   help="Skip layer 2 (decoder waveform parity)")
    p.add_argument("--no-end-to-end", action="store_true",
                   help="Skip layer 3 (end-to-end audio comparison)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    report = read_export_report(args.onnx_dir)
    graphs = report.get("graphs_exported", [])
    if not graphs:
        fail("export-report.json reports no graphs exported yet. Run the export script first.")

    providers = choose_onnx_providers(args.runtime)
    print(f"Parity check against {args.onnx_dir}")
    print(f"  graphs available: {graphs}")
    print(f"  providers: {providers}")
    print()

    # TODO E4 — Layer 1: token-sequence parity
    print("[ ] Layer 1 (LM token sequence parity) — not implemented yet")

    # TODO E4 — Layer 2: decoder waveform parity
    if not args.no_decoder_parity:
        print("[ ] Layer 2 (decoder waveform spectral parity) — not implemented yet")

    # TODO E4 — Layer 3: end-to-end audio
    if not args.no_end_to_end:
        print("[ ] Layer 3 (end-to-end audio comparison) — not implemented yet")


if __name__ == "__main__":
    main()
