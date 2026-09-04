"""Compress model.llm.embed_tokens.weight, which MatMulNBits cannot reach.

The table is 151676 x 1024 fp32 = 621 MB and is consumed by a Gather, not a MatMul, so the
weight-only quantizer skips it by construction — it is 621 MB of the w4 model's 937 MB. Two
variants, both rewriting the graph so the dequantization happens on the GATHERED SLICE (a handful
of rows) rather than on the whole table:

  fp16 : Gather(fp16 table, ids) -> Cast(fp32)                        621 -> 310 MB
  int8 : per-row symmetric scales; Gather(int8 table) and Gather(scales),
         then Cast + Mul                                              621 -> 155 MB

Per-ROW scales for int8, not per-tensor: the IPA fine-tune retrained 5,572 rows of this table, so
it is not the uniformly-distributed, quantization-tolerant tensor an embedding usually is.

⚠ THIS LIVED IN /tmp UNTIL v7. The 472 MB browser build shipped for two months from a scratch
script with v6-shaped constants baked in, so the published artifact could not be rebuilt from this
repo — which is how a recipe becomes folklore. Source and destination are arguments now.

  python3 quantize_embedding.py int8 --src <...wo4b32.onnx> --dst <...int4.onnx>
"""
import argparse, os, time, numpy as np, onnx
from onnx import helper, numpy_helper, TensorProto

_ap = argparse.ArgumentParser()
_ap.add_argument("mode", choices=("fp16", "int8"))
_ap.add_argument("--src", required=True)
_ap.add_argument("--dst", required=True)
_A = _ap.parse_args()
mode, src, dst = _A.mode, _A.src, _A.dst
NAME = "model.llm.embed_tokens.weight"

t = time.time()
m = onnx.load(src)
g = m.graph
init = {i.name: i for i in g.initializer}
# Both lookups are model-specific, so say so rather than dying with KeyError / StopIteration on a
# graph that simply names things differently.
if NAME not in init:
    raise SystemExit(f"{src}: no initializer named {NAME} — is this an OmniVoice transformer export?")
E = numpy_helper.to_array(init[NAME]).astype(np.float32)
# ⚠ REFUSE AN ALREADY-COMPRESSED TABLE. After a first pass the initializer and its Gather both
# still exist — only the dtype changed — so "is the Gather still there" does NOT detect a repeat.
# Running twice quantizes int8-of-int8, and the size moves 471.8 -> 472.4 MB, which is invisible.
if init[NAME].data_type != TensorProto.FLOAT:
    raise SystemExit(f"{src}: {NAME} is already "
                     f"{TensorProto.DataType.Name(init[NAME].data_type)}, not fp32 — "
                     "this graph has been through quantize_embedding already.")
gather = next((n for n in g.node if n.op_type == "Gather" and n.input[0] == NAME), None)
if gather is None:
    raise SystemExit(f"{src}: {NAME} exists but no Gather consumes it")
out, ids = gather.output[0], gather.input[1]

g.initializer.remove(init[NAME])
new_nodes, new_inits = [], []
if mode == "fp16":
    new_inits.append(numpy_helper.from_array(E.astype(np.float16), NAME))
    gather.output[0] = out + "__q"
    new_nodes.append(helper.make_node("Cast", [out + "__q"], [out], to=TensorProto.FLOAT,
                                      name=gather.name + "/emb_cast"))
else:
    s = np.abs(E).max(axis=1, keepdims=True) / 127.0
    s[s == 0] = 1.0
    q = np.clip(np.rint(E / s), -127, 127).astype(np.int8)
    new_inits += [numpy_helper.from_array(q, NAME),
                  numpy_helper.from_array(s.astype(np.float32), NAME + "__scale")]
    gather.output[0] = out + "__q"
    new_nodes += [
        helper.make_node("Gather", [NAME + "__scale", ids], [out + "__s"], name=gather.name + "/emb_scale"),
        helper.make_node("Cast", [out + "__q"], [out + "__qf"], to=TensorProto.FLOAT, name=gather.name + "/emb_cast"),
        helper.make_node("Mul", [out + "__qf", out + "__s"], [out], name=gather.name + "/emb_mul"),
    ]
g.initializer.extend(new_inits)
# Insert immediately after the Gather so the graph stays topologically sorted.
idx = list(g.node).index(gather)
nodes = list(g.node)[: idx + 1] + new_nodes + list(g.node)[idx + 1 :]
del g.node[:]
g.node.extend(nodes)

# ⚠ WIPE THE STALE SIDECAR FIRST. `onnx.save` CONCATENATES to an existing external-data file
# rather than truncating it, so a second run over the same destination produces a model whose data
# file is the sum of both — 968 MB instead of 472, and it still loads, which is the dangerous part.
# quantize_lm.py already carries this guard; this script did not, and rebuilding v7 hit it.
if os.path.exists(dst + ".data"):
    os.remove(dst + ".data")
onnx.save(m, dst, save_as_external_data=True, location=os.path.basename(dst) + ".data",
          all_tensors_to_one_file=True, size_threshold=1024)
tot = os.path.getsize(dst) + os.path.getsize(dst + ".data")
print(f"emb-{mode}: {tot/1e6:.1f} MB in {time.time()-t:.0f}s -> {os.path.basename(dst)}")
