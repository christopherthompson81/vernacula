#!/usr/bin/env python3
"""Post-process a --coreml-static-batch1 Sortformer export into a graph the
CoreML execution provider can compile as a SINGLE partition.

Run this on the output of:

    export_sortformer_nemo_to_onnx.py --coreml-static-batch1 --coreml-const-lengths

Background
----------
ORT's CoreML EP only fuses ops it fully supports; everything else stays on CPU
and splits the graph at that boundary. A conformer export straight out of NeMo
leaves two op families on CPU, and because they sit inside *every* encoder
layer they shred the graph into dozens of partitions. Each boundary is a
CPU<->CoreML copy, which is why a 94%-supported graph was still barely faster
than pure CPU.

Measured on an M5 (ORT 1.24.4, Sortformer 4spk v2.1, chunk=992/cache=188/fifo=124):

    stage                                partitions   median
    raw --coreml-static-batch1 export           71    166.0 ms
      + --coreml-const-lengths                  69         --
      + all-False Where removed                 35     97.0 ms
      + Pad -> Concat                            1     51.3 ms

    reference: CPU 164.4 ms, WebGPU 94.2 ms

The two transforms
------------------
1. `Where` (51x). With constant lengths the attention padding mask folds to a
   compile-time constant that is ALL FALSE -- the sequence is fully packed
   (188 + 124 + 124 = 436). `Where(false, -10000, x)` is exactly `x`, so these
   are identities that exist only to break up the graph. Removed by rewiring
   consumers to the data input. Numerically exact.

2. `Pad` (34x). Two single-axis constant zero-pads per layer (the relative
   position shift in self-attention, and the depthwise-conv padding). The
   CoreML EP declines Pad but supports Concat, and a constant zero-pad is
   exactly a Concat against a zero tensor. Numerically exact.

Both rewrites are value-preserving; verify with --verify (compares against the
original dynamic model on the CPU EP).

Caveats
-------
* Constant folding must run at ORT_ENABLE_BASIC. ORT_ENABLE_EXTENDED fuses ops
  the CoreML EP then rejects and makes partitioning much WORSE (69 -> 191).
* The result is steady-state only: full cache/fifo and a full-length chunk.
* Folding prunes the now-unused *_lengths inputs, so the graph takes three
  inputs (chunk, spkcache, fifo), not six.
* Partitioning behaviour is ORT-version dependent. Validated on 1.24.4.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference


def basic_fold(src: str, dst: str) -> None:
    """Constant-fold at BASIC level. EXTENDED is deliberately avoided."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.optimized_model_filepath = dst
    ort.InferenceSession(src, so, providers=["CPUExecutionProvider"])


def drop_allfalse_where(g, shapes: dict) -> int:
    init = {i.name: i for i in g.initializer}
    rewire, drop = {}, set()
    for idx, n in enumerate(g.node):
        if n.op_type != "Where" or n.input[0] not in init:
            continue
        mask = numpy_helper.to_array(init[n.input[0]])
        if mask.dtype != np.bool_ or mask.any():
            continue
        # Where BROADCASTS all three inputs. An all-False Where equals its false
        # branch only when that branch is already the broadcast shape -- otherwise
        # rewiring past it drops the broadcast and hands consumers a smaller
        # tensor, which is a silent numeric change (or a load-time shape error).
        fs, os_ = shapes.get(n.input[2]), shapes.get(n.output[0])
        if fs is None or os_ is None or any(d is None for d in fs) or any(d is None for d in os_):
            continue
        if list(fs) != list(os_):
            continue
        rewire[n.output[0]] = n.input[2]
        drop.add(idx)

    def resolve(x):
        seen = set()
        while x in rewire and x not in seen:
            seen.add(x)
            x = rewire[x]
        return x

    rewire = {k: resolve(v) for k, v in rewire.items()}
    for n in g.node:
        for i, inp in enumerate(n.input):
            if inp in rewire:
                n.input[i] = rewire[inp]
    for o in g.output:
        if o.name in rewire:
            o.name = rewire[o.name]
    kept = [n for i, n in enumerate(g.node) if i not in drop]
    del g.node[:]
    g.node.extend(kept)
    return len(drop)


def pad_to_concat(g, shapes: dict, dtypes: dict) -> int:
    init = {i.name: i for i in g.initializer}
    zero_cache: dict[tuple, str] = {}

    def zeros(shape, np_dtype):
        # Must match the padded tensor's dtype or Concat fails type inference
        # (notably after an fp16 conversion).
        key = (tuple(shape), np_dtype.str)
        if key not in zero_cache:
            name = f"coreml_zeros_{np_dtype.name}_" + "x".join(map(str, shape))
            g.initializer.append(numpy_helper.from_array(np.zeros(shape, np_dtype), name))
            zero_cache[key] = name
        return zero_cache[key]

    converted = 0
    for idx, n in enumerate(g.node):
        if n.op_type != "Pad" or len(n.input) < 2 or n.input[1] not in init:
            continue
        if next((a.s.decode() for a in n.attribute if a.name == "mode"), "constant") != "constant":
            continue
        # constant_value must be KNOWN zero. A third input that is present and
        # non-empty but is not a zero initializer (e.g. computed at runtime) is
        # disqualifying -- rewriting it to a Concat against zeros would silently
        # change the numerics.
        if len(n.input) > 2 and n.input[2]:
            if n.input[2] not in init:
                continue
            cv = numpy_helper.to_array(init[n.input[2]]).reshape(-1)
            if cv.size == 0 or float(cv[0]) != 0.0:
                continue
        ish = shapes.get(n.input[0])
        if ish is None or any(d is None for d in ish):
            continue
        dt = dtypes.get(n.input[0])
        if dt is None:
            continue
        np_dtype = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(dt))
        # Opset 18's optional `axes` input remaps what `pads` refers to. Resolving it
        # is not worth it here; skip rather than misapply the pads to the wrong axes.
        if len(n.input) > 3 and n.input[3]:
            continue
        pads = numpy_helper.to_array(init[n.input[1]]).astype(int).tolist()
        r = len(ish)
        if len(pads) != 2 * r:
            continue
        begins, ends = pads[:r], pads[r:]
        # ONNX Pad allows NEGATIVE pads, which crop rather than pad. A Concat cannot
        # express that, and a negative extent would reach np.zeros as a negative
        # dimension and abort the run.
        if any(p < 0 for p in pads):
            continue
        axes = [i for i in range(r) if begins[i] or ends[i]]
        if len(axes) != 1:
            continue
        ax = axes[0]
        parts = []
        if begins[ax]:
            s = list(ish); s[ax] = begins[ax]; parts.append(zeros(s, np_dtype))
        parts.append(n.input[0])
        if ends[ax]:
            s = list(ish); s[ax] = ends[ax]; parts.append(zeros(s, np_dtype))
        g.node[idx].CopyFrom(helper.make_node(
            "Concat", inputs=parts, outputs=list(n.output), name=n.name + "_concat", axis=ax))
        converted += 1
    return converted


def pretranspose_gemm(g) -> int:
    """Pre-transpose constant Gemm weights and set transB=1.

    THE single biggest load-time lever. CoreML's `linear` op wants the weight as
    [N, K]; an ONNX Gemm with transB=0 stores it [K, N], so ORT's CoreML EP
    synthesizes a transposed copy per node -- and it writes those synthesized
    constants INLINE in model.mil as hex-float TEXT (~16 chars per float)
    instead of into the binary weight blob. On Sortformer that produced a
    1.49 GB model.mil for 0.39 GB of actual weights, and CoreML re-parsed it on
    every load.

    Feeding the weight already transposed removes the synthesized copy, so the
    weights land in weight.bin. Measured on Sortformer/M5:

        transB=0:  cold 101.0s  warm 20.8s  model.mil 1.49 GB
        transB=1:  cold   2.2s  warm  0.2s  model.mil 0.00 GB

    Bit-exact: Gemm(A, B, transB=1) with B pre-transposed is the same operation.
    """
    init = {i.name: i for i in g.initializer}
    users: dict[str, int] = {}
    for n in g.node:
        for i in n.input:
            users[i] = users.get(i, 0) + 1

    done = 0
    for n in g.node:
        if n.op_type != "Gemm":
            continue
        attrs = {a.name: a for a in n.attribute}
        if "transB" in attrs and int(attrs["transB"].i) == 1:
            continue
        w = n.input[1]
        # Only safe in place when this initializer feeds nothing else.
        if w not in init or users.get(w, 0) != 1:
            continue
        arr = numpy_helper.to_array(init[w])
        if arr.ndim != 2:
            continue
        init[w].CopyFrom(numpy_helper.from_array(np.ascontiguousarray(arr.T), w))
        if "transB" in attrs:
            attrs["transB"].i = 1
        else:
            n.attribute.append(helper.make_attribute("transB", 1))
        done += 1
    return done


def restore_folded_outputs(g) -> int:
    """Folding can turn a graph output into a bare initializer with no producer,
    which the ONNX checker rejects. Give each one an Identity producer."""
    init = {i.name: i for i in g.initializer}
    produced = {o for n in g.node for o in n.output}
    fixed = 0
    for o in g.output:
        if o.name in produced:
            continue
        if o.name not in init:
            raise SystemExit(f"graph output {o.name} has neither producer nor initializer")
        src = o.name + "_const"
        init[o.name].name = src
        # Must go at the FRONT: the renamed initializer may feed other nodes, which
        # now read this Identity's output. ONNX requires a node to precede its
        # consumers, and check_model(full_check=False) does not catch a violation --
        # ORT rejects the model only at session creation.
        g.node.insert(0, helper.make_node("Identity", [src], [o.name], name=o.name + "_id"))
        fixed += 1
    return fixed


def prune(g) -> None:
    """Drop initializers nothing references any more, and any graph input that
    named one of them. Real graph inputs (fed by the caller) are never
    initializers, so they always survive."""
    used = {i for n in g.node for i in n.input}
    dropped = {i.name for i in g.initializer if i.name not in used}
    keep = [i for i in g.initializer if i.name in used]
    del g.initializer[:]
    g.initializer.extend(keep)
    gi = [v for v in g.input if v.name not in dropped]
    del g.input[:]
    g.input.extend(gi)
    del g.value_info[:]          # let ORT re-infer


def verify(original: str, optimized: str, tol: float) -> bool:
    import onnxruntime as ort

    rng = np.random.default_rng(7)
    NP = {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(int64)": np.int64}

    def session(path, opt=None):
        so = ort.SessionOptions()
        if opt is not None:
            so.graph_optimization_level = opt
        return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])

    def run(s, feed):
        return dict(zip([o.name for o in s.get_outputs()], s.run(None, feed)))

    ref_sess = session(original)

    # Build the feed from the model's OWN signature. The frame counts follow
    # --chunk-frames / --fixed-spkcache-frames / --fixed-fifo-frames at export time,
    # so hardcoding them here made --verify usable on exactly one configuration.
    shapes = {i.name: i.shape for i in ref_sess.get_inputs()}
    feed = {}
    for i in ref_sess.get_inputs():
        if any(not isinstance(d, int) for d in i.shape):
            raise SystemExit(
                f"--verify needs a static graph; input {i.name} has shape {i.shape}")
        np_dtype = NP.get(i.type)
        if np_dtype is None:
            raise SystemExit(f"--verify cannot synthesize input {i.name} of type {i.type}")
        if np_dtype == np.int64:
            # A `<buffer>_lengths` input. Steady state is the buffer's own frame count,
            # which is what the CoreML graph was specialized for.
            buf = i.name[: -len("_lengths")] if i.name.endswith("_lengths") else None
            if buf is None or buf not in shapes:
                raise SystemExit(f"--verify cannot infer a value for integer input {i.name}")
            feed[i.name] = np.full(i.shape, shapes[buf][1], np.int64)
        else:
            feed[i.name] = rng.standard_normal(i.shape).astype(np_dtype)

    ref = run(ref_sess, feed)

    # Feed the optimized model only what it still declares: --coreml-const-lengths
    # folds the baked lengths away, and prune() then drops them from the signature.
    opt_sess = session(optimized, ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
    opt_feed = {}
    for i in opt_sess.get_inputs():
        if i.name not in feed:
            raise SystemExit(f"optimized model expects input {i.name}, absent from the original")
        arr = feed[i.name]
        opt_feed[i.name] = arr.astype(np.float16) if i.type == "tensor(float16)" else arr
    got = run(opt_sess, opt_feed)

    ok = True
    for k in ("spkcache_fifo_chunk_preds", "chunk_pre_encode_embs"):
        a, b = ref[k], got[k]
        if a.shape != b.shape:
            print(f"  {k:28} SHAPE {a.shape} vs {b.shape}  -> FAIL")
            ok = False
            continue
        d = np.abs(a.astype(np.float32) - b.astype(np.float32))
        good = d.max() < tol
        ok &= good
        print(f"  {k:28} maxAbsDiff={d.max():.3E} rms={np.sqrt((d ** 2).mean()):.3E}"
              f"  -> {'OK' if good else 'FAIL'}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="ONNX from --coreml-static-batch1")
    ap.add_argument("--output", required=True, help="CoreML-optimized ONNX to write")
    ap.add_argument("--verify", action="store_true",
                    help="Compare against --reference on the CPU EP")
    ap.add_argument("--reference", help="Original dynamic ONNX to verify against")
    ap.add_argument("--tolerance", type=float, default=1e-3)
    ap.add_argument("--keep-intermediate", action="store_true")
    args = ap.parse_args()

    src = os.path.expanduser(args.input)
    dst = os.path.expanduser(args.output)
    folded = dst + ".basic.tmp.onnx"

    print(f"[1/5] constant-folding at ORT_ENABLE_BASIC  <- {src}")
    basic_fold(src, folded)

    inferred = folded + ".inferred.onnx"
    shape_inference.infer_shapes_path(folded, inferred)
    sm = onnx.load(inferred, load_external_data=False)
    shapes, dtypes = {}, {}
    for vi in list(sm.graph.input) + list(sm.graph.value_info) + list(sm.graph.output):
        d = vi.type.tensor_type.shape.dim
        shapes[vi.name] = [x.dim_value if x.HasField("dim_value") else None for x in d]
        dtypes[vi.name] = vi.type.tensor_type.elem_type

    m = onnx.load(folded)
    g = m.graph
    print(f"[2/5] removing all-False Where ...      {drop_allfalse_where(g, shapes):3d} removed")
    print(f"[3/5] rewriting Pad -> Concat ...       {pad_to_concat(g, shapes, dtypes):3d} converted")
    print(f"[4/5] pre-transposing Gemm weights ...  {pretranspose_gemm(g):3d} converted")
    restore_folded_outputs(g)
    prune(g)

    onnx.checker.check_model(m, full_check=False)
    onnx.save(m, dst, save_as_external_data=False)
    print(f"[5/5] wrote {dst} ({os.path.getsize(dst) / 1e6:.0f} MB), "
          f"{len(g.node)} nodes, inputs={[v.name for v in g.input]}")

    if not args.keep_intermediate:
        for f in (folded, inferred):
            try: os.remove(f)
            except OSError: pass

    if args.verify:
        if not args.reference:
            raise SystemExit("--verify needs --reference <original dynamic .onnx>")
        print("\nverifying against reference on CPU EP:")
        if not verify(os.path.expanduser(args.reference), dst, args.tolerance):
            sys.exit(1)
        print("PARITY: PASS")


if __name__ == "__main__":
    main()
