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

Layout (Run 19 — grounded in ORT contrib spec on `main`):

  - Q is fed via Transpose+Reshape adapter into (B, S, num_heads*head_dim)
    rank-3 form. MHA requires query rank 3 or 5.
  - K and V are fed DIRECTLY in their native (B, num_heads, T, head_dim)
    rank-4 layout. The contrib spec accepts this for the `key`/`value`
    inputs (described as "past_key with shape (batch_size, num_heads,
    kv_sequence_length, head_size)"). This eliminates the K/V adapter
    Transpose+Reshape pair and any head-ordering bug in it — the most
    likely cause of iter-2's partial-correctness output.
  - The graph's per-call additive mask (`where_2`-style; bf16, encodes
    both key-padding and causal triangulation) is fed as `attention_bias`
    (input 5; renamed from `relative_position_bias` in current ORT). The
    spec confirms it is a literal element-wise add on QxK' with shape
    (batch_size or 1, num_heads or 1, S, total_S).

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

NUM_HEADS    = 16
NUM_KV_HEADS = 4
HEAD_DIM     = 128
HIDDEN       = NUM_HEADS * HEAD_DIM        # 2048
KV_HIDDEN    = NUM_KV_HEADS * HEAD_DIM     # 512


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--input",  type=Path, required=True,
                    help="Input decoder.onnx (static-shape, eager-attention export).")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output decoder.onnx with attention rewritten to MHA.")
    ap.add_argument("--layers", type=str, default=None,
                    help="Comma-separated layer indices to rewrite (default: all).")
    ap.add_argument("--mode", choices=["mha", "gqa"], default="mha",
                    help="Fusion target: mha (MultiHeadAttention, default) or gqa "
                         "(GroupQueryAttention). GQA enables compute-skip via "
                         "seqlens_k and requires a coordinated cache-layout "
                         "change in the C# driver.")
    ap.add_argument("--add-argmax", action="store_true",
                    help="Append Slice(last)+ArgMax(V) to the LM head and "
                         "replace `logits` with `next_token` (int64 [B, 1]) "
                         "as a graph output. Eliminates ~6 MB of "
                         "GPU→CPU logits transfer per step (Run 20 phase 9).")
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


def find_kv_chain_for_gqa(kv_tensor: str, producer, consumers):
    """Walk back from the K/V tensor consumed by the attention MatMul (post
    broadcast 4→16) to the underlying Concat that joins the past_kv graph
    input with the fresh K/V (post-RoPE for K, post-Transpose for V).

    Returns:
      past_kv:      name of the past_key_<L>/past_value_<L> graph input.
      fresh_kv:     name of the fresh K/V tensor (B, num_kv_heads, S, head_dim).
      present_kv:   name of the present_key_<L>/present_value_<L> graph output
                    (currently produced by a Slice on top of the Concat).
      slice_node:   the Slice node that produces present_kv; must be removed
                    so the GQA op can claim the graph-output name.
    """
    # Walk back through Transpose/Reshape/Expand/Unsqueeze chain until Concat.
    cur = kv_tensor
    for _ in range(10):
        n = producer.get(cur)
        if n is None:
            raise ValueError(f"Hit graph input {cur} before finding Concat")
        if n.op_type == "Concat":
            concat_node = n
            break
        if n.op_type not in ("Transpose", "Reshape", "Expand", "Unsqueeze"):
            raise ValueError(
                f"Unexpected op {n.op_type} ({n.name}) walking back K/V chain")
        cur = n.input[0]
    else:
        raise ValueError("K/V chain too deep; expected Concat within 10 hops")

    # Identify which Concat input is the past_kv graph input by name pattern.
    past_kv = None
    fresh_kv = None
    for inp in concat_node.input:
        if inp.startswith("past_key_") or inp.startswith("past_value_"):
            past_kv = inp
        else:
            fresh_kv = inp
    if past_kv is None or fresh_kv is None:
        raise ValueError(
            f"Concat {concat_node.name} inputs {list(concat_node.input)} "
            f"don't match expected past_kv/fresh_kv pattern")

    # Locate the present_key_<L>/present_value_<L> graph-output edge.
    # Two shapes here:
    #   - static-shapes-unified export: Concat → Slice → graph output (the
    #     sliding-window slice that pins present_kv to past_len).
    #   - dynamic-shape export: Concat output IS the graph output directly
    #     (no Slice — present_kv grows by S each call).
    present_name = None
    drop_node = None        # node to remove so GQA's output can claim the name
    for c in consumers.get(concat_node.output[0], []):
        if c.op_type == "Slice" and (
            c.output[0].startswith("present_key_") or
            c.output[0].startswith("present_value_")
        ):
            present_name = c.output[0]
            drop_node = c
            break
    if present_name is None:
        # Direct: Concat output is itself the graph output.
        concat_out = concat_node.output[0]
        if concat_out.startswith("present_key_") or concat_out.startswith("present_value_"):
            present_name = concat_out
            drop_node = concat_node
        else:
            raise ValueError(
                f"Concat {concat_node.name} output has no present_kv "
                f"Slice consumer and isn't itself a graph output")

    return past_kv, fresh_kv, present_name, drop_node


def rewrite_layer_mha(graph: onnx.GraphProto, softmax: onnx.NodeProto, layer_idx: int,
                      producer, consumers):
    """Rewrite one layer's attention to MultiHeadAttention. Returns
    (added_nodes, added_initializers, nodes_to_remove).

    Iter 5 wiring:
      - Q: Transpose+Reshape adapter to (B, S, hidden_size).
      - K, V: passed directly in (B, num_heads, T, head_dim) — ORT MHA
        accepts this rank-4 layout for the `key`/`value` inputs.
      - attention_bias: graph's `where_2` mask (bf16, additive, encodes
        causal+padding) fed unchanged as MHA input 5.
    """
    q, k, v, mask, output_tensor, to_remove = find_attention_block(softmax, producer, consumers)

    pfx = f"mha_l{layer_idx}"

    # Q must be rank 3 (B, S, hidden_size). Reshape needs the batch dim
    # explicit since it can only carry one -1.
    init_q = make_const_int64(f"{pfx}_q_shape", [0, -1, HIDDEN])

    # Q adapter (B, 16, S, 128) -> (B, S, 2048)
    q_adapted = f"{pfx}_q_in"
    q_nodes = [
        helper.make_node("Transpose", inputs=[q], outputs=[f"{pfx}_q_t"],
                         name=f"{pfx}_q_transpose", perm=[0, 2, 1, 3]),
        helper.make_node("Reshape", inputs=[f"{pfx}_q_t", f"{pfx}_q_shape"],
                         outputs=[q_adapted], name=f"{pfx}_q_reshape"),
    ]

    # K and V are already in (B, num_heads, T, head_dim) at the source
    # tensors `k` and `v`. Pass them straight through.
    mha_out = f"{pfx}_out"
    # Granite uses an unusual scale: 1/head_dim (= 0.0078125 for head_dim=128),
    # not the conventional 1/sqrt(head_dim). This is the `attention_multiplier`
    # in IBM Granite's HF config; it shows up in the unfused graph as the
    # initializer `scalar_tensor_default_2 = 0.0078125`. Passing the wrong
    # scale here (e.g., 1/sqrt(head_dim) ~= 0.0884) makes attention 11.3×
    # too peaky per layer; one layer is recoverable but it compounds across
    # 40 layers into garbled output (Run 19 bisect, 2026-05-10).
    scale = 1.0 / HEAD_DIM
    mha_node = helper.make_node(
        "MultiHeadAttention",
        inputs=[q_adapted, k, v, "", "", mask],
        outputs=[mha_out],
        name=f"{pfx}_node",
        domain="com.microsoft",
        num_heads=NUM_HEADS,
        scale=scale,
    )

    # Rewire output_tensor consumers to use mha_out instead.
    output_consumers = consumers.get(output_tensor, [])
    for c in output_consumers:
        for i, inp in enumerate(c.input):
            if inp == output_tensor:
                c.input[i] = mha_out

    new_nodes = q_nodes + [mha_node]
    new_inits = [init_q]
    return new_nodes, new_inits, to_remove


def rewrite_layer_gqa(graph: onnx.GraphProto, softmax: onnx.NodeProto, layer_idx: int,
                      producer, consumers, pass_attention_bias: bool = False):
    """Rewrite one layer's attention to GroupQueryAttention. Returns
    (added_nodes, added_initializers, nodes_to_remove).

    Layout (Run 20):
      - Q: Transpose+Reshape adapter to (B, S, hidden_size=2048) rank-3.
      - K, V: walked back through Expand(4→16) + Concat(past, fresh) to
        the fresh K/V (post-RoPE for K, post-Transpose for V) in
        (B, kv_num_heads=4, S, head_dim=128); adapted via Transpose+Reshape
        to (B, S, kv_hidden_size=512) rank-3 for GQA's key/value inputs.
      - past_key, past_value: graph inputs past_key_<L>/past_value_<L>
        feed directly into GQA (BNSH layout, max-sized buffer).
      - seqlens_k, total_sequence_length: new graph inputs (added once
        at the top level by main()), passed into every layer's GQA.
      - No attention_bias is wired: GQA does causal masking internally
        via seqlens_k. We rely on the C# driver to right-pad prompts
        and pass per-call seqlens_k that reflects the true real length
        per batch row, which excludes pad tokens from compute and
        attention without an explicit mask.
      - Output: GQA's `present_key`/`present_value` directly claim the
        graph output names `present_key_<L>`/`present_value_<L>` (the
        original Slice→graph_output path is removed and replaced).
      - do_rotary=0: RoPE is already applied in the graph (HF wrapper);
        GQA only does scoring + cache append.
      - scale=1/HEAD_DIM: Granite's attention_multiplier (Run 19 finding).
    """
    q, k, v, mask, output_tensor, to_remove = find_attention_block(softmax, producer, consumers)

    past_k, fresh_k, present_k_name, k_slice = find_kv_chain_for_gqa(k, producer, consumers)
    past_v, fresh_v, present_v_name, v_slice = find_kv_chain_for_gqa(v, producer, consumers)

    pfx = f"gqa_l{layer_idx}"

    # Q adapter: (B, 16, S, 128) -> (B, S, 2048)
    init_q = make_const_int64(f"{pfx}_q_shape", [0, -1, HIDDEN])
    q_adapted = f"{pfx}_q_in"
    q_nodes = [
        helper.make_node("Transpose", inputs=[q], outputs=[f"{pfx}_q_t"],
                         name=f"{pfx}_q_transpose", perm=[0, 2, 1, 3]),
        helper.make_node("Reshape", inputs=[f"{pfx}_q_t", f"{pfx}_q_shape"],
                         outputs=[q_adapted], name=f"{pfx}_q_reshape"),
    ]

    # K/V adapters: (B, 4, S, 128) -> (B, S, 512)
    init_kv = make_const_int64(f"{pfx}_kv_shape", [0, -1, KV_HIDDEN])
    k_adapted = f"{pfx}_k_in"
    k_nodes = [
        helper.make_node("Transpose", inputs=[fresh_k], outputs=[f"{pfx}_k_t"],
                         name=f"{pfx}_k_transpose", perm=[0, 2, 1, 3]),
        helper.make_node("Reshape", inputs=[f"{pfx}_k_t", f"{pfx}_kv_shape"],
                         outputs=[k_adapted], name=f"{pfx}_k_reshape"),
    ]
    v_adapted = f"{pfx}_v_in"
    v_nodes = [
        helper.make_node("Transpose", inputs=[fresh_v], outputs=[f"{pfx}_v_t"],
                         name=f"{pfx}_v_transpose", perm=[0, 2, 1, 3]),
        helper.make_node("Reshape", inputs=[f"{pfx}_v_t", f"{pfx}_kv_shape"],
                         outputs=[v_adapted], name=f"{pfx}_v_reshape"),
    ]

    gqa_out = f"{pfx}_out"
    scale = 1.0 / HEAD_DIM
    if pass_attention_bias:
        # attention_bias is GQA input slot 10. Fill slots 7..9 with empty.
        gqa_inputs = [
            q_adapted, k_adapted, v_adapted,
            past_k, past_v,
            "seqlens_k", "total_sequence_length",
            "", "",      # cos_cache, sin_cache (do_rotary=0)
            "",          # position_ids (not used; RoPE applied upstream)
            mask,        # attention_bias = where_2
        ]
    else:
        gqa_inputs = [
            q_adapted, k_adapted, v_adapted,
            past_k, past_v,
            "seqlens_k", "total_sequence_length",
        ]
    gqa_node = helper.make_node(
        "GroupQueryAttention",
        inputs=gqa_inputs,
        outputs=[gqa_out, present_k_name, present_v_name],
        name=f"{pfx}_node",
        domain="com.microsoft",
        num_heads=NUM_HEADS,
        kv_num_heads=NUM_KV_HEADS,
        scale=scale,
        do_rotary=0,
    )

    # Rewire output_tensor consumers (o_proj input) -> gqa_out.
    for c in consumers.get(output_tensor, []):
        for i, inp in enumerate(c.input):
            if inp == output_tensor:
                c.input[i] = gqa_out

    # Old Slice nodes that produced present_key_<L>/present_value_<L> are
    # removed; GQA's outputs claim those graph-output names. The upstream
    # K/V chain (Concat → Unsqueeze → Expand → Reshape → Transpose) becomes
    # dead once the original attention nodes are gone — ORT DCE handles it.
    new_nodes = q_nodes + k_nodes + v_nodes + [gqa_node]
    new_inits = [init_q, init_kv]
    to_remove_combined = to_remove + [k_slice, v_slice]
    return new_nodes, new_inits, to_remove_combined


def add_gqa_graph_inputs(graph: onnx.GraphProto, batch_dim: "int | str") -> None:
    """Append `seqlens_k` (int32, [B]) and `total_sequence_length`
    (int32, scalar) as new graph inputs so per-layer GQA nodes can
    reference them by name. batch_dim may be either an int (static
    batch) or a string symbol name (dynamic batch)."""
    existing = {i.name for i in graph.input}
    if "seqlens_k" not in existing:
        graph.input.append(helper.make_tensor_value_info(
            "seqlens_k", TensorProto.INT32, [batch_dim]))
    if "total_sequence_length" not in existing:
        graph.input.append(helper.make_tensor_value_info(
            "total_sequence_length", TensorProto.INT32, []))


def add_argmax_head(graph: onnx.GraphProto) -> None:
    """Append `Slice(logits, last_position) → ArgMax(V) → next_token` to
    the LM head, then replace `logits` in the graph outputs with
    `next_token` (shape [B, 1], int64).

    With left-padded prompts the real last token sits at S-1 for every
    row at prefill, and at position 0 at step (S=1). Either way "last
    position along axis 1" is what we want. Slicing first keeps the
    ArgMax cheap (over [B, 1, V] instead of [B, S, V]).

    Dropping `logits` from outputs lets ORT keep the logits tensor as
    an on-device intermediate (no GPU→CPU transfer per step). Greedy
    decoding only needs the argmax index, not the raw logits."""
    # Find and remove the `logits` graph output (preserve its dtype info).
    logits_vi = None
    for i, o in enumerate(graph.output):
        if o.name == "logits":
            logits_vi = o
            del graph.output[i]
            break
    if logits_vi is None:
        raise ValueError("`logits` not found in graph outputs")

    # Borrow batch dim spec from the (now-removed) logits output.
    batch_dim_proto = logits_vi.type.tensor_type.shape.dim[0]
    if batch_dim_proto.dim_value > 0:
        batch_dim: "int | str" = batch_dim_proto.dim_value
    else:
        batch_dim = batch_dim_proto.dim_param

    # Slice(logits, starts=[-1], ends=[INT_MAX], axes=[1]) → [B, 1, V].
    starts_init = numpy_helper.from_array(
        np.array([-1], dtype=np.int64), name="argmax_slice_starts")
    ends_init = numpy_helper.from_array(
        np.array([np.iinfo(np.int64).max], dtype=np.int64), name="argmax_slice_ends")
    axes_init = numpy_helper.from_array(
        np.array([1], dtype=np.int64), name="argmax_slice_axes")
    graph.initializer.extend([starts_init, ends_init, axes_init])

    slice_node = helper.make_node(
        "Slice",
        inputs=["logits", "argmax_slice_starts", "argmax_slice_ends", "argmax_slice_axes"],
        outputs=["logits_last"],
        name="argmax_slice",
    )
    argmax_node = helper.make_node(
        "ArgMax",
        inputs=["logits_last"],
        outputs=["next_token"],
        name="argmax_head",
        axis=2,
        keepdims=0,        # output shape [B, 1] (drop the V axis)
    )
    graph.node.extend([slice_node, argmax_node])

    # Add `next_token` as graph output.
    next_token_vi = helper.make_tensor_value_info(
        "next_token", TensorProto.INT64, [batch_dim, 1])
    graph.output.append(next_token_vi)


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
    print(f"  Mode: {args.mode}; rewriting layers: {target_layers}")

    if args.mode == "mha":
        rewrite_fn = lambda graph, sm, li, producer, consumers: rewrite_layer_mha(
            graph, sm, li, producer, consumers)
    else:
        rewrite_fn = lambda graph, sm, li, producer, consumers: rewrite_layer_gqa(
            graph, sm, li, producer, consumers,
            pass_attention_bias=pass_attention_bias)
    fused_op = "MHA" if args.mode == "mha" else "GQA"

    # GQA needs two extra graph inputs (seqlens_k, total_sequence_length)
    # that every layer's GQA node references. Infer batch size from any
    # past_key_<L> graph input (they're all the same B).
    pass_attention_bias = False
    if args.mode == "gqa":
        # batch dim: int if static (dim_value > 0), str symbol name if dynamic.
        pk_input = next(
            i for i in graph.input if i.name.startswith("past_key_")
        )
        pk_dim0 = pk_input.type.tensor_type.shape.dim[0]
        batch_dim = pk_dim0.dim_value if pk_dim0.dim_value > 0 else pk_dim0.dim_param
        # CUDA's GQA kernel doesn't yet support the attention_bias input
        # (only listed in the contrib spec; "attention_bias is not supported
        # in GroupQueryAttention cuda kernel" at runtime). So we never wire
        # it, even when shapes would match. Variable-length batching relies
        # on BatchSizer grouping similar-length segments and tolerating the
        # small pad contamination — the LLM is robust to PAD-token K at
        # the few trailing slots when realLen variation within a batch is
        # ~one or two tokens.
        pass_attention_bias = False
        add_gqa_graph_inputs(graph, batch_dim=batch_dim)
        bias_note = "with attention_bias=where_2" if pass_attention_bias else "(no attention_bias)"
        print(f"  Added GQA graph inputs: seqlens_k (int32, [{batch_dim!r}]) + "
              f"total_sequence_length (int32, scalar) {bias_note}")

    all_new_nodes: list[onnx.NodeProto] = []
    all_new_inits: list[onnx.TensorProto] = []
    all_remove: list[onnx.NodeProto] = []

    for li in target_layers:
        if li >= len(softmaxes):
            raise ValueError(f"Layer {li} out of range; graph has {len(softmaxes)} layers")
        sm = softmaxes[li]
        try:
            new_n, new_i, rem = rewrite_fn(graph, sm, li, producer, consumers)
        except ValueError as e:
            print(f"  layer {li}: SKIP — {e}")
            continue
        print(f"  layer {li}: replaced {len(rem)} nodes with {fused_op} + {len(new_n) - 1} adapter nodes")
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

    if args.add_argmax:
        add_argmax_head(graph)
        print(f"  Replaced `logits` output with `next_token` (int64 [B, 1]) "
              f"via Slice(last) + ArgMax(V).")

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
