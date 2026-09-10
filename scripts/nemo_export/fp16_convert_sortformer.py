#!/usr/bin/env python3
"""Convert an exported Sortformer ONNX to fp16 -- one that actually loads.

`onnxconverter_common.float16.convert_float_to_float16` on its own produces a model
ONNX Runtime refuses:

    Type Error: Type parameter (T) of Optype (Mul) bound to different types
    (tensor(float16) and tensor(float)) in node (/pre_encode/conv/Mul_3)

Three things are wrong with the naive conversion, and each needs a different answer.

1. THE LENGTH ARITHMETIC MUST NOT BE CONVERTED.
   `torch.onnx.export` emits the conv output-length formula -- floor((L + 2p - k)/s) + 1,
   once per pre-encode conv layer -- as float `Add`/`Sub`/`Div`/`Cast` ops. Those 39 nodes
   are not activations; they compute a FRAME COUNT. Rounding a frame count through fp16 is
   wrong on its own terms (fp16 integers are exact only to 2048, and the intermediate
   divisions are not integers), and it is where the mixed-type errors start. They are held
   in fp32 by name, found by walking back from `chunk_pre_encode_lengths` rather than
   hardcoded, so this keeps working if the exporter's node names change.

2. THE CONVERTER DOES NOT UPDATE `Cast` NODES.
   The graph has 141 explicit `Cast`s. The converter rewrites tensor types but leaves each
   `Cast`'s `to` attribute at FLOAT, so 27 of them end up declaring they produce fp32 while
   the graph types their output fp16. Reconciled here by trusting the converted types.

3. `keep_io_types` BREAKS TENSORS THAT ARE BOTH AN OUTPUT AND AN INPUT.
   `chunk_pre_encode_embs` is a graph output AND feeds onward into the encoder. The
   converter appends a Cast to hand the caller fp32, and the internal consumer then reads
   that fp32 tensor while expecting fp16 -- the `Mul` in the error above. Internal consumers
   are rewired back to the pre-cast fp16 tensor, so the cast serves only the graph output.

`keep_io_types` is deliberate: the C# caller feeds float32 and reads float32, so the fp16
model is a drop-in for the fp32 one and `Sortformer.cs` needs no variant path.

⚠ A model that loads is not a model that is correct. fp16 was measured fastest on the
CoreML EP (22.3 ms/chunk vs 51 ms) but `chunk_pre_encode_embs` diverged to 9.8e-02, and
those embeddings feed back into the speaker cache and FIFO across chunks -- so a
single-chunk comparison cannot see the compounding. The go/no-go is end-to-end:

    python scripts/nemo_export/sortformer_fidelity_der.py --onnx <this model> ...

See issue #172 and docs/investigations/sortformer_fp16_investigation.md.
"""
from __future__ import annotations

import argparse
import os
from collections import deque

import numpy as np
import onnx
from onnx import TensorProto


LENGTH_OUTPUT = "chunk_pre_encode_lengths"


def length_path_nodes(graph) -> list[str]:
    """Names of every node feeding `chunk_pre_encode_lengths`, stopping at graph inputs
    and initializers. These stay fp32.

    Walked rather than hardcoded, so it survives the exporter renaming nodes -- and so it
    reports honestly on a graph where the length arithmetic has already been folded away
    (the CoreML variant bakes the lengths in, leaving nothing to protect).
    """
    if LENGTH_OUTPUT not in {o.name for o in graph.output}:
        raise SystemExit(
            f"graph has no `{LENGTH_OUTPUT}` output, so the length arithmetic cannot be "
            "located and held in fp32. Point --input at a Sortformer export.")
    prod = {o: n for n in graph.node for o in n.output}
    init = {i.name for i in graph.initializer}
    gin = {i.name for i in graph.input}
    names, seen, q = set(), set(), deque([LENGTH_OUTPUT])
    while q:
        name = q.popleft()
        if name in seen or name in init or name in gin:
            continue
        seen.add(name)
        node = prod.get(name)
        if node is None:
            continue
        if node.name:
            names.add(node.name)
        q.extend(node.input)
    return sorted(names)


def reconcile_casts(graph) -> int:
    """Point each `Cast`'s `to` at the type the converted graph gives its output."""
    types = {v.name: v.type.tensor_type.elem_type
             for v in list(graph.value_info) + list(graph.input) + list(graph.output)}
    fixed = 0
    for node in graph.node:
        if node.op_type != "Cast":
            continue
        attr = next(a for a in node.attribute if a.name == "to")
        declared = types.get(node.output[0])
        floats = (TensorProto.FLOAT, TensorProto.FLOAT16)
        if declared in floats and attr.i in floats and attr.i != declared:
            attr.i = declared
            fixed += 1
    return fixed


def rewire_output_casts(graph) -> int:
    """Undo `keep_io_types` for INTERNAL consumers of a graph output.

    The converter ends such a tensor with `Cast(fp16 -> fp32)` so the caller still gets
    fp32. Any node that also consumes it inside the graph wants the fp16 source instead.

    ⚠ Only fp16->fp32 casts qualify. "Produced by a Cast" is not enough: Sortformer's
    `chunk_pre_encode_lengths` output is produced by a legitimate `Cast(to=INT64)` that the
    model itself contains, and it IS consumed internally -- the attention mask is built from
    it. Rewiring past that one would hand the mask a float tensor in place of the frame
    count, which is a silent semantic change, not a dtype tidy-up.
    """
    types = {v.name: v.type.tensor_type.elem_type
             for v in list(graph.value_info) + list(graph.input) + list(graph.output)}
    outputs = {o.name for o in graph.output}
    producer = {o: n for n in graph.node for o in n.output}
    rewired = 0
    for name in outputs:
        node = producer.get(name)
        if node is None or node.op_type != "Cast":
            continue
        to = next(a.i for a in node.attribute if a.name == "to")
        src = node.input[0]
        if to != TensorProto.FLOAT or types.get(src) != TensorProto.FLOAT16:
            continue
        for consumer in graph.node:
            if consumer is node:
                continue
            for i, inp in enumerate(consumer.input):
                if inp == name:
                    consumer.input[i] = src
                    rewired += 1
    return rewired


# Ops ONNX Runtime has no fp16 CPU kernel for. Converting them produces a model that
# LOADS and then dies at the first inference:
#
#   RUNTIME_EXCEPTION ... ScatterElements ... MLFloat16 data type is not supported with
#   ScatterElements opset 16 when reduction is 'add'
#
# Held in fp32 with a cast on each side, which costs one node's worth of bandwidth. Other
# execution providers may well have the kernel, but the model has to run on CPU too, and a
# per-EP artifact is not worth one op.
UNSUPPORTED_FP16_OPS = ["ScatterElements"]


def convert(src: str, dst: str, keep_io_types: bool = True) -> None:
    from onnxconverter_common import float16

    model = onnx.load(src)
    block = length_path_nodes(model.graph)
    op_block = list(float16.DEFAULT_OP_BLOCK_LIST) + UNSUPPORTED_FP16_OPS
    print(f"[1/4] holding {len(block)} length-arithmetic nodes and "
          f"{'/'.join(UNSUPPORTED_FP16_OPS)} in fp32")
    fp16 = float16.convert_float_to_float16(
        model, keep_io_types=keep_io_types, node_block_list=block, op_block_list=op_block)
    print(f"[2/4] reconciling Cast `to` attributes ...  {reconcile_casts(fp16.graph):3d} fixed")
    print(f"[3/4] rewiring internal consumers of cast-back outputs ... "
          f"{rewire_output_casts(fp16.graph):3d} edges")
    onnx.checker.check_model(fp16, full_check=False)
    onnx.save(fp16, dst)
    print(f"[4/4] wrote {dst} ({os.path.getsize(dst) / 1e6:.0f} MB)")


def check_loads(path: str) -> bool:
    """A model that loads at one optimization level can fail at another; check all three."""
    import onnxruntime as ort
    ok = True
    for level in ("ORT_DISABLE_ALL", "ORT_ENABLE_BASIC", "ORT_ENABLE_ALL"):
        so = ort.SessionOptions()
        so.graph_optimization_level = getattr(ort.GraphOptimizationLevel, level)
        try:
            ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
            print(f"  {level:18} loads")
        except Exception as exc:
            print(f"  {level:18} FAILS: {str(exc).strip().splitlines()[-1][:150]}")
            ok = False
    return ok


def compare(fp32_path: str, fp16_path: str, seed: int = 7) -> None:
    """Single-chunk parity, on the steady-state shapes. Indicative only -- see the module
    docstring on why this cannot answer the question by itself."""
    import onnxruntime as ort

    rng = np.random.default_rng(seed)
    ref = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    got = ort.InferenceSession(fp16_path, providers=["CPUExecutionProvider"])

    shapes = {"chunk": (1, 992, 128), "spkcache": (1, 188, 512), "fifo": (1, 124, 512)}
    feed = {}
    for i in ref.get_inputs():
        if i.type == "tensor(int64)":
            buf = i.name[: -len("_lengths")]
            feed[i.name] = np.array([shapes[buf][1]], np.int64)
        else:
            feed[i.name] = rng.standard_normal(shapes[i.name]).astype(np.float32)

    a = dict(zip([o.name for o in ref.get_outputs()], ref.run(None, feed)))
    b = dict(zip([o.name for o in got.get_outputs()],
                 got.run(None, {k: feed[k] for k in [i.name for i in got.get_inputs()]})))
    print("\nsingle-chunk fp16 vs fp32 (CPU EP, steady state):")
    for key in ("spkcache_fifo_chunk_preds", "chunk_pre_encode_embs"):
        x, y = a[key].astype(np.float32), b[key].astype(np.float32)
        d = np.abs(x - y)
        print(f"  {key:26} maxAbs={d.max():.3E}  rms={np.sqrt((d ** 2).mean()):.3E}")
    print("  ⚠ indicative only: embs feed back into spkcache/FIFO, so run the DER harness.")


def drift(fp32_path: str, fp16_path: str, audio_paths, max_seconds: float) -> None:
    """Does the fp16 error COMPOUND across chunks, or stay bounded?

    That is the question the single-chunk comparison cannot answer and DER is too coarse to
    answer: `chunk_pre_encode_embs` feeds the speaker cache and FIFO, so error can
    accumulate, and a collar plus a median filter will hide a lot of per-frame movement.
    Runs the ported streaming loop twice on the same audio and diffs the raw posteriors, per
    chunk. Growth with chunk index is compounding; a trace that wanders is not.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import soundfile as sf

    import benchmark_sortformer_rtf as B

    def run(model, mel, n_chunks, stride, total):
        pipe = B.OnnxSortformerPipeline(Path(model), device="cpu", gpu_id=0,
                                        provider_kind="cpu")
        pipe.reset_state()
        return [pipe.process_chunk(i, stride, total, mel)[0] for i in range(n_chunks)]

    for path in audio_paths:
        path = Path(path)
        frames = int(max_seconds * 16000) if max_seconds else -1
        audio, sr = sf.read(str(path), dtype="float32",
                            frames=frames if frames > 0 else -1)
        if sr != 16000:
            raise SystemExit(f"{path}: expected 16 kHz, got {sr}")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        mel = B.log_mel_spectrogram(audio)
        stride = B.CHUNK_LENGTH * B.SUBSAMPLING
        n_chunks = (mel.shape[1] + stride - 1) // stride

        a = run(fp32_path, mel, n_chunks, stride, mel.shape[1])
        b = run(fp16_path, mel, n_chunks, stride, mel.shape[1])
        per_chunk = [float(np.abs(x.astype(np.float32) - y.astype(np.float32)).max())
                     for x, y in zip(a, b)]
        ca, cb = np.concatenate(a), np.concatenate(b)
        d = np.abs(ca.astype(np.float32) - cb.astype(np.float32))
        flips = int((((ca > 0.5) != (cb > 0.5)).any(axis=1)).sum())
        print(f"\n{path.stem}  {n_chunks} chunks, {len(ca)} frames")
        print(f"  per-frame preds  maxAbs={d.max():.3E}  rms={np.sqrt((d ** 2).mean()):.3E}")
        print(f"  frames whose binarized speaker set differs: {flips}/{len(ca)}")
        print("  per-chunk maxAbs: " + " ".join(f"{v:.2E}" for v in per_chunk))
    print("\n  A trend across the per-chunk figures is compounding. A trace that rises and")
    print("  falls is bounded error, and the binarized column says whether it reaches output.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="fp32 Sortformer ONNX")
    ap.add_argument("--output", required=True, help="fp16 model to write")
    ap.add_argument("--io-fp16", action="store_true",
                    help="Make the graph's inputs and outputs fp16 too. Off by default: "
                         "fp32 io keeps this a drop-in for Sortformer.cs.")
    ap.add_argument("--compare", action="store_true",
                    help="Single-chunk parity against --input afterwards.")
    ap.add_argument("--drift", nargs="+", metavar="AUDIO",
                    help="16 kHz mono files to run both models over, reporting per-chunk "
                         "posterior divergence -- whether the fp16 error compounds through "
                         "the spkcache/FIFO feedback. This is what DER is too coarse to say.")
    ap.add_argument("--max-seconds", type=float, default=90.0,
                    help="Truncate each --drift file (default 90).")
    args = ap.parse_args()

    src, dst = os.path.expanduser(args.input), os.path.expanduser(args.output)
    convert(src, dst, keep_io_types=not args.io_fp16)
    print("\nload check:")
    ok = check_loads(dst)
    if args.compare:
        compare(src, dst)
    if args.drift:
        drift(src, dst, args.drift, args.max_seconds)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
