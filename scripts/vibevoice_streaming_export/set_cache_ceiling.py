#!/usr/bin/env python
"""Change a GQA package's KV cache ceiling without re-exporting or touching the weights.

The ceiling is only the buffer length declared on the decoder's past_key/value inputs; no
node holds it as a constant (GroupQueryAttention takes the live length through seqlens_k and
total_sequence_length instead). So raising it rewrites the ~1 MB graph file and the export
report, and leaves the multi-gigabyte `.onnx.data` byte-for-byte identical — which matters
because that is what a re-publish would otherwise have to re-upload.

The practical maximum is the checkpoint's trained context: 65,536 positions for the 1.5B and
131,072 for the 7B, about 68 and 137 minutes of audio at ~16 cache positions per second.
"""
import argparse, hashlib, json, shutil
from pathlib import Path

import onnx

KV_PREFIXES = ("past_key_", "past_value_", "present_key_", "present_value_")


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True, help="package directory, edited in place")
    ap.add_argument("--ceiling", type=int, required=True)
    ap.add_argument("--decoder", default="decoder_gqa.onnx")
    args = ap.parse_args()

    pkg = Path(args.package)
    graph_path = pkg / args.decoder
    data_path = pkg / (args.decoder + ".data")
    before = md5(data_path) if data_path.exists() else None

    # load_external_data=False keeps the weights on disk and out of this process entirely.
    m = onnx.load(str(graph_path), load_external_data=False)
    patched = 0
    for vi in list(m.graph.input) + list(m.graph.output):
        if not vi.name.startswith(KV_PREFIXES):
            continue
        dims = vi.type.tensor_type.shape.dim
        if len(dims) == 4 and dims[2].HasField("dim_value"):
            dims[2].dim_value = args.ceiling
            patched += 1
    if patched == 0:
        raise SystemExit("no KV buffer dimensions found; is this a GQA package?")
    onnx.save(m, str(graph_path), save_as_external_data=False)

    rep = json.loads((pkg / "export-report.json").read_text())
    rep["static_kv_max_tokens"] = args.ceiling
    (pkg / "export-report.json").write_text(json.dumps(rep, indent=1))

    after = md5(data_path) if data_path.exists() else None
    print(f"patched {patched} KV dims to {args.ceiling} (~{args.ceiling / 16.0 / 60:.0f} min of audio)")
    print(f"weights unchanged: {before == after}  ({data_path.name})")


if __name__ == "__main__":
    main()
