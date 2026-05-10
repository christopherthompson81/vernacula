"""Granite Speech 4.1 attention-fusion rewriter (issue #43, phase B).

Rewrites the per-layer attention pattern in a static-shape Granite
decoder.onnx, replacing the unfused
  MatMul(Q, K^T) -> Mul(scale) -> Add(mask) -> Softmax -> MatMul(V)
sequence with a single com.microsoft.MultiHeadAttention node.

Granite's attention pattern is uniform across all 40 decoder layers
(verified via Run 18 inspection). The matcher anchors on each Softmax
node and walks back/forward to identify the seven nodes per attention
block that we replace. The remaining Q/K/V projection chain, the RoPE
Mul/Add applied to Q and K, the past_kv Concat, and the GQA broadcast
Expand all stay in place — MHA wants K/V already broadcast to 16 heads
and already RoPE'd, which is exactly what those upstream nodes produce.

The matcher emits transpose+reshape adapters around MHA because MHA's
input layout is (B, S, num_heads × head_dim) flat-hidden whereas the
existing chain produces (B, num_heads, S, head_dim) split-hidden. The
adapters are constant-folded by ORT in many cases (transpose+reshape
of static dims).

USAGE (called as a post-export step):

    python scripts/granite_export/rewrite_attention_to_mha.py \\
        --input  /tmp/granite_static_eager/decoder.onnx \\
        --output /tmp/granite_static_mha/decoder.onnx \\
        [--layers 0,1,2]    # only rewrite specific layers (default: all)
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

NUM_HEADS = 16
HEAD_DIM  = 128
HIDDEN    = NUM_HEADS * HEAD_DIM  # 2048


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--input",  type=Path, required=True,
                    help="Input decoder.onnx (static-shape, eager-attention export).")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output decoder.onnx with attention rewritten to MHA.")
    ap.add_argument("--layers", type=str, default=None,
                    help="Comma-separated layer indices to rewrite (default: all).")
    return ap.parse_args()


def build_indices(graph: onnx.GraphProto):
    """Build producer / consumer indices for the graph."""
    producer: dict[str, onnx.NodeProto] = {}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for n in graph.node:
        for o in n.output:
            producer[o] = n
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    return producer, consumers


def find_attention_block(softmax: onnx.NodeProto, producer, consumers):
    """Given a Softmax node anchoring an attention block, return:
       (q_tensor, k_tensor, v_tensor, mask_tensor, output_consumers,
        nodes_to_remove)
    or raise ValueError if the surrounding pattern doesn't match."""
    # Walk back: softmax(in) <- Add(scaled, mask) <- Mul(matmul_QK, scale)
    add_mask = producer.get(softmax.input[0])
    if add_mask is None or add_mask.op_type != "Add":
        raise ValueError(f"{softmax.name}: expected Softmax input from Add, got {add_mask.op_type if add_mask else 'GRAPH_INPUT'}")

    # Identify which Add input is the scaled scores vs the mask
    add_in0_node = producer.get(add_mask.input[0])
    add_in1_node = producer.get(add_mask.input[1])
    if add_in0_node and add_in0_node.op_type == "Mul":
        scale_mul, mask_tensor = add_in0_node, add_mask.input[1]
    elif add_in1_node and add_in1_node.op_type == "Mul":
        scale_mul, mask_tensor = add_in1_node, add_mask.input[0]
    else:
        raise ValueError(f"{softmax.name}: Add doesn't have a Mul producer for scaled scores")

    scores_matmul = producer.get(scale_mul.input[0])
    if scores_matmul is None or scores_matmul.op_type != "MatMul":
        raise ValueError(f"{softmax.name}: scale Mul doesn't follow a MatMul (got {scores_matmul})")

    q_tensor = scores_matmul.input[0]
    k_transposed = scores_matmul.input[1]

    # K feeder: walk back through Transpose to get the un-transposed K
    k_transpose = producer.get(k_transposed)
    if k_transpose is None or k_transpose.op_type != "Transpose":
        raise ValueError(f"{softmax.name}: K is not transposed by a Transpose node")
    k_tensor = k_transpose.input[0]   # (B, num_heads, T, head_dim)

    # Walk forward: softmax_out -> attn_matmul (× V) -> transpose_5 -> view_9 -> linear_3
    sm_out_consumers = consumers.get(softmax.output[0], [])
    if len(sm_out_consumers) != 1 or sm_out_consumers[0].op_type != "MatMul":
        raise ValueError(f"{softmax.name}: softmax output not consumed by exactly one MatMul")
    attn_matmul = sm_out_consumers[0]
    v_tensor = attn_matmul.input[1]   # (B, num_heads, T, head_dim)

    # Walk forward two hops to find the consumer of the reshape (which is
    # the output_proj's input). After our rewrite, MHA's output replaces
    # the reshape's output.
    cur = attn_matmul.output[0]
    chain_after_attn = []
    for _ in range(3):
        cs = consumers.get(cur, [])
        if len(cs) != 1:
            break
        n = cs[0]
        chain_after_attn.append(n)
        cur = n.output[0]
        if n.op_type == "Reshape":
            break

    # Expecting: Transpose -> Reshape (then linear_3 consumes Reshape output)
    if len(chain_after_attn) < 2 or chain_after_attn[-1].op_type != "Reshape":
        raise ValueError(
            f"{softmax.name}: attn output chain doesn't terminate at Reshape "
            f"(got {[n.op_type for n in chain_after_attn]})")

    # output_proj input tensor name = the reshape's output. We rewire this.
    output_tensor = chain_after_attn[-1].output[0]
    output_consumers = consumers.get(output_tensor, [])

    nodes_to_remove = [scores_matmul, scale_mul, add_mask, softmax, attn_matmul, *chain_after_attn]
    return q_tensor, k_tensor, v_tensor, mask_tensor, output_tensor, nodes_to_remove


def make_const_int64(name: str, values: list[int]) -> onnx.TensorProto:
    return numpy_helper.from_array(np.array(values, dtype=np.int64), name=name)


def add_q_kv_adapter_nodes(prefix: str, src: str, dst: str) -> list[onnx.NodeProto]:
    """Build Transpose+Reshape that adapts (B, num_heads, S, head_dim) ->
    (B, S, num_heads × head_dim). The adapter shape constant is shared
    via the prefix-derived initializer name; caller is responsible for
    adding the initializer."""
    transposed = f"{prefix}_t"
    return [
        helper.make_node(
            "Transpose",
            inputs=[src],
            outputs=[transposed],
            name=f"{prefix}_transpose",
            perm=[0, 2, 1, 3],
        ),
        helper.make_node(
            "Reshape",
            inputs=[transposed, f"{prefix}_shape"],
            outputs=[dst],
            name=f"{prefix}_reshape",
        ),
    ]


def rewrite_layer(graph: onnx.GraphProto, softmax: onnx.NodeProto, layer_idx: int,
                  producer, consumers):
    """Rewrite one layer's attention. Returns (added_nodes, added_initializers,
    nodes_to_remove). Reuses the per-call additive mask `where_2` (or whatever
    the graph computes once and shares across all 40 layers' Add nodes) as
    MHA's relative_position_bias — that mask already encodes both
    key-padding AND causal restrictions, so we don't need to feed a separate
    key_padding_mask."""
    q, k, v, mask, output_tensor, to_remove = find_attention_block(softmax, producer, consumers)

    pfx = f"mha_l{layer_idx}"

    # Adapter shape constants (shared across Q/K/V):
    # We can't fold seq/total_len since they're symbolic. Use [-1, -1, HIDDEN]
    # — but Reshape only allows one -1. So use [B=16, -1, HIDDEN]:
    init_q = make_const_int64(f"{pfx}_q_shape", [16, -1, HIDDEN])
    init_kv = make_const_int64(f"{pfx}_kv_shape", [16, -1, HIDDEN])

    # Q adapter (B, 16, S, 128) -> (B, S, 2048)
    q_adapted = f"{pfx}_q_in"
    q_nodes = [
        helper.make_node("Transpose", inputs=[q], outputs=[f"{pfx}_q_t"],
                         name=f"{pfx}_q_transpose", perm=[0, 2, 1, 3]),
        helper.make_node("Reshape", inputs=[f"{pfx}_q_t", f"{pfx}_q_shape"],
                         outputs=[q_adapted], name=f"{pfx}_q_reshape"),
    ]

    # K adapter
    k_adapted = f"{pfx}_k_in"
    k_nodes = [
        helper.make_node("Transpose", inputs=[k], outputs=[f"{pfx}_k_t"],
                         name=f"{pfx}_k_transpose", perm=[0, 2, 1, 3]),
        helper.make_node("Reshape", inputs=[f"{pfx}_k_t", f"{pfx}_kv_shape"],
                         outputs=[k_adapted], name=f"{pfx}_k_reshape"),
    ]

    # V adapter
    v_adapted = f"{pfx}_v_in"
    v_nodes = [
        helper.make_node("Transpose", inputs=[v], outputs=[f"{pfx}_v_t"],
                         name=f"{pfx}_v_transpose", perm=[0, 2, 1, 3]),
        helper.make_node("Reshape", inputs=[f"{pfx}_v_t", f"{pfx}_kv_shape"],
                         outputs=[v_adapted], name=f"{pfx}_v_reshape"),
    ]

    # MultiHeadAttention. We pass the graph's raw attention_mask (cast to
    # int32 once at the top of the graph) as key_padding_mask, plus
    # is_unidirectional=1 to have MHA apply causal masking internally.
    # This wiring lets MHA's CUDA kernel skip masked positions in COMPUTE,
    # not just zero them post-softmax — that's the difference between
    # "correct output but slow" (relative_position_bias path) and
    # "correct AND fast". Relevant ORT MHA implementation skip happens
    # only via key_padding_mask + is_unidirectional, not via RPB.
    mha_out = f"{pfx}_out"
    scale = 1.0 / (HEAD_DIM ** 0.5)
    mha_node = helper.make_node(
        "MultiHeadAttention",
        inputs=[q_adapted, k_adapted, v_adapted, "", "mha_mask_int32"],
        outputs=[mha_out],
        name=f"{pfx}_node",
        domain="com.microsoft",
        num_heads=NUM_HEADS,
        scale=scale,
        unidirectional=1,
    )

    # Rewire output_tensor consumers to use mha_out instead.
    output_consumers = consumers.get(output_tensor, [])
    for c in output_consumers:
        for i, inp in enumerate(c.input):
            if inp == output_tensor:
                c.input[i] = mha_out

    new_nodes = q_nodes + k_nodes + v_nodes + [mha_node]
    new_inits = [init_q, init_kv]
    return new_nodes, new_inits, to_remove


def add_mask_int32(graph: onnx.GraphProto) -> str:
    """Add a Cast node that converts the existing graph-input attention_mask
    (int64) to int32 for MHA. Returns the int32 tensor name."""
    out_name = "mha_mask_int32"
    cast = helper.make_node(
        "Cast",
        inputs=["attention_mask"],
        outputs=[out_name],
        name="mha_mask_cast_int32",
        to=TensorProto.INT32,
    )
    graph.node.insert(0, cast)
    return out_name


def main() -> int:
    args = parse_args()

    print(f"Loading {args.input} ...")
    model = onnx.load(str(args.input), load_external_data=False)
    graph = model.graph
    print(f"  nodes={len(graph.node)} inputs={len(graph.input)} outputs={len(graph.output)}")

    producer, consumers = build_indices(graph)

    softmaxes = [n for n in graph.node if n.op_type == "Softmax"]
    print(f"  Found {len(softmaxes)} softmax nodes (one per layer).")

    target_layers = (
        list(range(len(softmaxes))) if args.layers is None
        else [int(x) for x in args.layers.split(",")]
    )
    print(f"  Rewriting layers: {target_layers}")

    # Cast the graph's attention_mask (int64) to int32 once for all layers'
    # MHA key_padding_mask consumers.
    add_mask_int32(graph)

    all_new_nodes: list[onnx.NodeProto] = []
    all_new_inits: list[onnx.TensorProto] = []
    all_remove: list[onnx.NodeProto] = []

    for li in target_layers:
        if li >= len(softmaxes):
            raise ValueError(f"Layer {li} out of range; graph has {len(softmaxes)} layers")
        sm = softmaxes[li]
        try:
            new_n, new_i, rem = rewrite_layer(graph, sm, li, producer, consumers)
        except ValueError as e:
            print(f"  layer {li}: SKIP — {e}")
            continue
        print(f"  layer {li}: replaced {len(rem)} nodes with MHA + 6 adapter nodes")
        all_new_nodes.extend(new_n)
        all_new_inits.extend(new_i)
        all_remove.extend(rem)

    # Apply: remove old nodes, append new nodes + initializers.
    remove_names = {n.name for n in all_remove}
    new_node_list = [n for n in graph.node if n.name not in remove_names]
    new_node_list.extend(all_new_nodes)
    del graph.node[:]
    graph.node.extend(new_node_list)
    graph.initializer.extend(all_new_inits)

    # Need com.microsoft opset
    has_msft = any(o.domain == "com.microsoft" for o in model.opset_import)
    if not has_msft:
        model.opset_import.append(helper.make_opsetid("com.microsoft", 1))

    print(f"\nSaving {args.output} ...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Save with external-data sidecar matching the input
    onnx.save(model, str(args.output), save_as_external_data=False)

    # Copy the .data sidecar over (we didn't touch weights).
    in_data = args.input.with_suffix(".onnx.data")
    out_data = args.output.with_suffix(".onnx.data")
    if in_data.exists() and not out_data.exists():
        print(f"  Copying weights sidecar {in_data} -> {out_data}")
        shutil.copy(in_data, out_data)

    print(f"  Wrote graph proto: {args.output.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
