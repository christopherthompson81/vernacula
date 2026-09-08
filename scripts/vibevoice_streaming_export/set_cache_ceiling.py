#!/usr/bin/env python
"""Change a GQA package's KV cache ceiling without re-exporting or touching the weights.

The ceiling is only the buffer length declared on the decoder's past_key/value inputs; no
node holds it as a constant (GroupQueryAttention takes the live length through seqlens_k and
total_sequence_length instead). So raising it rewrites the ~1 MB graph file and the export
report, and leaves the multi-gigabyte `.onnx.data` byte-for-byte identical — which matters
because that is what a re-publish would otherwise have to re-upload.

The practical maximum is the checkpoint's trained context: 65,536 positions for the 1.5B and
131,072 for the 7B, about 68 and 137 minutes of audio at ~16 cache positions per second.

--dynamic replaces the fixed length with a symbolic one, so the runtime picks the buffer size
instead of the export. A fixed length is a *requirement*, not a ceiling: ORT rejects a smaller
buffer with "Got invalid dimensions for input: past_key_0", which made every run pay the
two-hour cache even for a one-minute file, and put the 7B out of reach of a 16 GB card
(issue #150). The recorded ceiling stays in the report as the checkpoint's context bound.
"""
import argparse, hashlib, json, shutil
from pathlib import Path

import onnx

KV_PREFIXES = ("past_key_", "past_value_", "present_key_", "present_value_")

# One name across every KV input and output: they are the same buffers, and giving them
# separate symbols would let ORT's shape inference believe they could differ in length.
KV_DIM_NAME = "kv_cache_len"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True, help="package directory, edited in place")
    ap.add_argument("--ceiling", type=int, required=True,
                    help="context bound recorded in the report, and the fixed buffer length "
                         "unless --dynamic is given")
    ap.add_argument("--dynamic", action="store_true",
                    help="declare the KV length symbolically so the runtime sizes the buffer")
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
        if len(dims) != 4:
            continue
        if args.dynamic:
            dims[2].dim_param = KV_DIM_NAME
            patched += 1
        elif dims[2].HasField("dim_value"):
            dims[2].dim_value = args.ceiling
            patched += 1
    if patched == 0:
        raise SystemExit("no KV buffer dimensions found; is this a GQA package?")
    onnx.save(m, str(graph_path), save_as_external_data=False)

    # Only when it actually changes: --dynamic on an already-correct ceiling should leave the
    # report byte-identical, so a re-publish uploads the graph alone.
    rep_path = pkg / "export-report.json"
    rep = json.loads(rep_path.read_text())
    if rep.get("static_kv_max_tokens") != args.ceiling:
        rep["static_kv_max_tokens"] = args.ceiling
        rep_path.write_text(json.dumps(rep, indent=1))

    after = md5(data_path) if data_path.exists() else None
    shape = KV_DIM_NAME if args.dynamic else args.ceiling
    print(f"patched {patched} KV dims to {shape}; report ceiling {args.ceiling} "
          f"(~{args.ceiling / 16.0 / 60:.0f} min of audio)")
    print(f"weights unchanged: {before == after}  ({data_path.name})")


if __name__ == "__main__":
    main()
