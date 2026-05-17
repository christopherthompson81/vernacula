"""Merge three split cond-decoder sub-graphs into a single ONNX with an
embedded `Loop` op driving the CFM solve. Phase 2 follow-up to PR #50.

Input (produced by `export_chatterbox_to_onnx.py --split-cond-decoder`):
  - flow_encoder.onnx   speech_tokens, spk_emb, spk_feat
                          → mu, mel_mask, embedding, cond, z
  - cfm_estimator.onnx  x_in, mask_in, mu_in, t_in, spks_in, cond_in
                          → dphi_dt  (CFG-doubled batch)
  - mel2wav.onnx        mel → waveform

Output:
  conditional_decoder_loop.onnx

  Whose graph is:

    flow_encoder body
        ↓
    [CFG-double mu/mask/spks/cond once]
        ↓
    Loop(M=10, cond=True, body=cfm_estimator_body) starting from z
        - each iter: gather t/dt from precomputed t_span, CFG-double x,
          run cfm_estimator on the CFG-doubled inputs, split, CFG-combine,
          Euler step → new x
        ↓
    Slice mel: drop first PromptLen=500 frames (prompt prefix)
        ↓
    mel2wav body
        ↓
    waveform

Verification: run merged.onnx, compare against the 3-graph orchestration
(parity_split_pipeline in test_chatterbox_parity.py). Expect agreement
within ORT non-determinism floor.

Pure graph surgery — no PyTorch involvement.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import onnx
from onnx import GraphProto, NodeProto, TensorProto, helper, numpy_helper

# Model-level constants (match scripts/chatterbox_export/_common.py + listen_test.py).
N_TIMESTEPS = 10
MEL_BINS = 80
PROMPT_LEN = 500  # speaker_features.shape[1]
CFG_RATE = 0.7    # chatterbox.s3gen.flow.decoder.inference_cfg_rate


# ──────────────────────────────────────────────────────────────────────────
# Graph surgery helpers
# ──────────────────────────────────────────────────────────────────────────


def prefix_graph(graph: GraphProto, prefix: str,
                 protected: set[str]) -> tuple[GraphProto, dict[str, str]]:
    """Rename every node / tensor / initializer in `graph` by prepending
    `prefix`, EXCEPT names in `protected` (typically the graph's external
    input/output names, which we'll rewire externally).

    Returns the modified graph and a name-mapping dict so callers can look
    up where outputs ended up.

    Assumption (intentional, not enforced): the `protected` set is meant
    for the outer graph's contract, not nested subgraphs. Recursion into
    Loop/If bodies passes the same `protected` down, which is fine for
    the current inputs (flow_encoder, cfm_estimator, mel2wav have no
    inner subgraphs that name-collide with their own outer I/O), but
    would mis-handle a future graph that did. Revisit if that becomes
    relevant.
    """
    out = GraphProto()
    out.CopyFrom(graph)

    # Build the full set of names that need renaming: every initializer,
    # every node output, every value_info name. Inputs/outputs are
    # protected (rewired externally).
    rename: dict[str, str] = {}

    def maybe_rename(name: str) -> str:
        if not name or name in protected:
            return name
        if name not in rename:
            rename[name] = f"{prefix}{name}"
        return rename[name]

    # Initializers
    for init in out.initializer:
        init.name = maybe_rename(init.name)

    # value_info
    for vi in out.value_info:
        vi.name = maybe_rename(vi.name)

    # Nodes: rename node name + every input + every output. Also rename
    # subgraph attributes (e.g., Loop body, If branches) recursively.
    for node in out.node:
        if node.name:
            node.name = f"{prefix}{node.name}"
        for i, inp in enumerate(node.input):
            node.input[i] = maybe_rename(inp)
        for i, out_name in enumerate(node.output):
            node.output[i] = maybe_rename(out_name)
        # Subgraph attributes
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                sub, _ = prefix_graph(attr.g, prefix, protected)
                attr.g.CopyFrom(sub)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                new_graphs = []
                for g in attr.graphs:
                    nsub, _ = prefix_graph(g, prefix, protected)
                    new_graphs.append(nsub)
                # ClearField + extend
                del attr.graphs[:]
                attr.graphs.extend(new_graphs)

    return out, rename


def make_const_node(output_name: str, array: np.ndarray) -> NodeProto:
    """Create a Constant node that emits the given array as its output."""
    return helper.make_node(
        "Constant", inputs=[], outputs=[output_name],
        value=numpy_helper.from_array(array, name=f"{output_name}_value"),
    )


# ──────────────────────────────────────────────────────────────────────────
# Loop body construction
# ──────────────────────────────────────────────────────────────────────────


def build_loop_body(
    cfm_graph: GraphProto,
    *,
    # Outer-scope tensor names the body will reference (referenced by name
    # via outer-scope visibility — the body has no inputs for them):
    mu_in_pair: str,
    mask_in_pair: str,
    spks_in_pair: str,
    cond_in_pair: str,
    # t_span / dts are referenced by hardcoded literal names ("t_span_const",
    # "dts_const") that must match what merge() adds to outer_inits.
) -> tuple[GraphProto, list[TensorProto]]:
    """Build the Loop body subgraph. Body signature:
       inputs:  iter_num (int64 scalar), cond_in (bool scalar), x_carried (1, 80, T_mel)
       outputs: cond_out (bool scalar), x_new (1, 80, T_mel)
    The cfm_estimator graph is inlined with a "cfm__" prefix.

    Returns (body_graph, outer_initializers_to_hoist) — caller adds the
    second to the outer graph's initializer list. Hoisting matches how a
    native PyTorch ONNX Loop export emits weights (outer scope, not nested
    in the body). It does NOT by itself fix the cache round-trip bug
    (issue #56) — ORT still inserts body-scope constants after optimization;
    that bug is tracked separately.
    """
    # The cfm_estimator's six inputs become outputs of body-internal ops:
    # x_in     ← Concat([x_carried, x_carried], axis=0)
    # mask_in  ← outer-scope mask_in_pair
    # mu_in    ← outer-scope mu_in_pair
    # t_in     ← Concat([t_now, t_now], axis=0)  where t_now = Gather(t_span, iter_num)
    # spks_in  ← outer-scope spks_in_pair
    # cond_in  ← outer-scope cond_in_pair
    cfm_protected = {"x_in", "mask_in", "mu_in", "t_in", "spks_in", "cond_in", "dphi_dt"}
    cfm_inlined, _ = prefix_graph(cfm_graph, "cfm__", cfm_protected)

    # The body builds intermediate tensors and feeds them into the inlined
    # cfm graph via REWRITE of the inlined graph's input-tensor consumers:
    # rather than renaming the inlined inputs, we reroute them by inserting
    # Identity nodes at the boundary.
    #
    # Approach: keep cfm__ inputs as the contract (x_in, mask_in, ...) but
    # prefix them too so we can produce them as named outputs of preceding
    # ops in the body. We rename the protected names to "cfm__x_in" etc.
    boundary_rename = {
        n: f"cfm__{n}" for n in cfm_protected
    }
    for node in cfm_inlined.node:
        for i, inp in enumerate(node.input):
            if inp in boundary_rename:
                node.input[i] = boundary_rename[inp]
        for i, out in enumerate(node.output):
            if out in boundary_rename:
                node.output[i] = boundary_rename[out]
    # The cfm graph itself has its inputs/outputs listed; clear them
    # (we'll define the body's own inputs/outputs separately).
    del cfm_inlined.input[:]
    del cfm_inlined.output[:]

    # Pre-cfm setup nodes:
    pre_nodes = [
        # iter_num is int64 scalar. Gather wants axis 0; treat t_span as a
        # length-(N+1) tensor and dts as length-N.
        helper.make_node("Gather", ["t_span_const", "iter_num"], ["t_now"], axis=0),
        helper.make_node("Gather", ["dts_const", "iter_num"], ["dt_now"], axis=0),
    ]
    # Opset 13+ Unsqueeze: axes as a 1-D tensor input. Build via Constant + Unsqueeze.
    pre_nodes.extend(_make_unsqueeze_with_axes(
        "t_now", "t_now_1d", axes=[0], const_node_name="t_now_1d_axes"))
    # Compute dynamic 2*B from x_carried's leading dim so cfm__t_in is (2B,)
    # at B>1 (cfm_estimator wants one t per CFG-doubled row). At B=1 this
    # produces (2,) — same as the prior Concat-by-2 — preserving B=1 parity.
    pre_nodes.extend([
        helper.make_node("Shape", ["x_carried"], ["x_carried_shape"]),
        # x_b_dim is a length-1 1-D tensor = [B].
        make_const_node("two_b_starts", np.array([0], dtype=np.int64)),
        make_const_node("two_b_ends",   np.array([1], dtype=np.int64)),
        make_const_node("two_b_axes",   np.array([0], dtype=np.int64)),
        helper.make_node("Slice",
                         ["x_carried_shape", "two_b_starts", "two_b_ends", "two_b_axes"],
                         ["x_b_dim"]),
        make_const_node("two_const", np.array([2], dtype=np.int64)),
        # two_b is [2B] as a length-1 1-D tensor — Tile's repeats spec.
        helper.make_node("Mul", ["x_b_dim", "two_const"], ["two_b"]),
        # cfm__t_in = Tile(t_now_1d, [2B]) → shape (2B,).
        helper.make_node("Tile", ["t_now_1d", "two_b"], ["cfm__t_in"]),
        # x_in = Concat([x_carried, x_carried], axis=0) → [2B, 80, T_mel]
        helper.make_node("Concat", ["x_carried", "x_carried"], ["cfm__x_in"], axis=0),
        # Wire outer-scope inputs directly via Identity (keeps the graph
        # explicit; ORT will optimize away).
        helper.make_node("Identity", [mu_in_pair],   ["cfm__mu_in"]),
        helper.make_node("Identity", [mask_in_pair], ["cfm__mask_in"]),
        helper.make_node("Identity", [spks_in_pair], ["cfm__spks_in"]),
        helper.make_node("Identity", [cond_in_pair], ["cfm__cond_in"]),
    ])

    # Post-cfm CFG combine + Euler step:
    # dphi = cfm__dphi_dt, shape (2, 80, T_mel)
    # Split → cond_part [1, 80, T], uncond_part [1, 80, T]
    # combined = (1+cfg)*cond_part - cfg*uncond_part
    # x_new = x_carried + dt_now * combined
    #
    # CFG/Euler scalars are emitted as Constant NODES rather than
    # initializers. Avoids a name-uniqueness collision we hit while
    # debugging issue #56; Constant nodes don't go through the
    # initializer registry. This change alone does NOT fix issue #56.
    post_nodes = [
        helper.make_node("Split", ["cfm__dphi_dt"], ["cond_part", "uncond_part"],
                         axis=0, num_outputs=2),
        make_const_node("cfm_body__one_plus_cfg_const",
                        np.array(1.0 + CFG_RATE, dtype=np.float32)),
        make_const_node("cfm_body__cfg_const",
                        np.array(CFG_RATE, dtype=np.float32)),
        helper.make_node("Mul", ["cond_part", "cfm_body__one_plus_cfg_const"], ["cond_scaled"]),
        helper.make_node("Mul", ["uncond_part", "cfm_body__cfg_const"], ["uncond_scaled"]),
        helper.make_node("Sub", ["cond_scaled", "uncond_scaled"], ["combined"]),
        helper.make_node("Mul", ["combined", "dt_now"], ["step_delta"]),
        helper.make_node("Add", ["x_carried", "step_delta"], ["x_new"]),
        # cond_out = True scalar (we always continue; trip count drives termination).
        # Same Constant-node-not-initializer reasoning as above.
        make_const_node("cfm_body__true_const", np.array(True, dtype=np.bool_)),
        helper.make_node("Identity", ["cfm_body__true_const"], ["cond_out"]),
    ]

    # Stitch: pre_nodes + inlined cfm nodes + post_nodes.
    body_nodes = list(pre_nodes) + list(cfm_inlined.node) + list(post_nodes)

    # Empty body initializer list. cfm_estimator's own weights / attention
    # constants are added to the OUTER graph's initializer list (see merge()
    # below) and referenced from the body via name. This matches how a
    # native PyTorch ONNX Loop export would emit weights — at outer scope,
    # not nested in body. This does NOT by itself fix issue #56 (cache
    # round-trip) — ORT still inserts body-scope constants after
    # optimization. Tracked separately.
    body_inits: list[TensorProto] = []

    # Body inputs (signature): iter_num, cond_in, x_carried.
    # Body outputs: cond_out, x_new.
    # Leading dim is dynamic "batch_size" — Run 10 makes the whole graph
    # B>1-capable. cfm__t_in is dynamically sized to (2B,) via Tile in
    # pre_nodes; everything else (x_in, mu_in_pair, etc.) was already
    # constructed via axis-0 Concat that scales with B.
    body_inputs = [
        helper.make_tensor_value_info("iter_num", TensorProto.INT64, []),
        helper.make_tensor_value_info("cond_in", TensorProto.BOOL, []),
        helper.make_tensor_value_info("x_carried", TensorProto.FLOAT,
                                       ["batch_size", MEL_BINS, "T_mel"]),
    ]
    body_outputs = [
        helper.make_tensor_value_info("cond_out", TensorProto.BOOL, []),
        helper.make_tensor_value_info("x_new", TensorProto.FLOAT,
                                       ["batch_size", MEL_BINS, "T_mel"]),
    ]
    body = helper.make_graph(
        body_nodes, "cfm_loop_body",
        body_inputs, body_outputs,
        initializer=body_inits,
    )
    # Hoist cfm_estimator's weights/constants to outer scope (the body
    # references them by their cfm__-prefixed names; ONNX subgraphs see
    # outer-scope initializers).
    cfm_initializers_to_hoist = list(cfm_inlined.initializer)
    return body, cfm_initializers_to_hoist


def _make_unsqueeze_with_axes(input_name: str, output_name: str,
                              axes: list[int], const_node_name: str) -> list[NodeProto]:
    """Opset-18 Unsqueeze: axes is a tensor input. Returns 2 nodes
    (Constant for axes, then Unsqueeze)."""
    axes_arr = np.array(axes, dtype=np.int64)
    return [
        make_const_node(const_node_name, axes_arr),
        helper.make_node("Unsqueeze", [input_name, const_node_name], [output_name]),
    ]


# ──────────────────────────────────────────────────────────────────────────
# Outer graph construction
# ──────────────────────────────────────────────────────────────────────────


def merge(flow_enc_path: Path, cfm_est_path: Path, mel2wav_path: Path,
          out_path: Path, opset: int = 18) -> None:
    print("Loading sub-graphs ...")
    fe_model = onnx.load(str(flow_enc_path))
    ce_model = onnx.load(str(cfm_est_path))
    mw_model = onnx.load(str(mel2wav_path))

    # Inline flow_encoder. Its inputs become the OUTER graph's inputs
    # (same names). Its outputs become flow_enc__* intermediates that we
    # wire into the loop.
    fe_protected = {
        "speech_tokens", "speaker_embeddings", "speaker_features",
        "mu", "mel_mask", "embedding", "cond", "z",
    }
    fe_inlined, _ = prefix_graph(fe_model.graph, "flow_enc__", fe_protected)

    # Inline mel2wav similarly. mel2wav's input "mel" needs to receive our
    # trimmed feat; its output "waveform" is the outer graph's output.
    mw_protected = {"mel", "waveform"}
    mw_inlined, _ = prefix_graph(mw_model.graph, "mel2wav__", mw_protected)

    # Build precomputed t_span + dts as outer-graph initializers.
    t_linear = np.linspace(0.0, 1.0, N_TIMESTEPS + 1).astype(np.float32)
    t_span = (1.0 - np.cos(t_linear * 0.5 * math.pi)).astype(np.float32)
    dts = np.diff(t_span).astype(np.float32)

    outer_inits: list[TensorProto] = [
        numpy_helper.from_array(t_span, name="t_span_const"),
        numpy_helper.from_array(dts, name="dts_const"),
        numpy_helper.from_array(np.array(N_TIMESTEPS, dtype=np.int64),
                                name="trip_count_const"),
        numpy_helper.from_array(np.array(True, dtype=np.bool_),
                                name="loop_cond_init"),
    ]
    outer_inits.extend(fe_inlined.initializer)
    outer_inits.extend(mw_inlined.initializer)

    # CFG-doubling helpers — Concat([x, ZerosLike(x)], axis=0) for mu/spks/cond
    # (the "cond row 0, uncond row 1 = zeros" pattern), Concat([x, x], axis=0)
    # for mask (both rows = mask).
    pre_loop_nodes: list[NodeProto] = []
    pre_loop_nodes.extend(fe_inlined.node)

    # mu_in_pair = Concat([mu, ZerosLike(mu)], axis=0)
    pre_loop_nodes.extend([
        helper.make_node("ConstantOfShape", ["mu_shape"], ["mu_zeros"],
                         value=helper.make_tensor("v", TensorProto.FLOAT, [1], [0.0])),
        helper.make_node("Shape", ["mu"], ["mu_shape"]),
        helper.make_node("Concat", ["mu", "mu_zeros"], ["mu_in_pair"], axis=0),
    ])
    # spks_in_pair (embedding)
    pre_loop_nodes.extend([
        helper.make_node("Shape", ["embedding"], ["emb_shape"]),
        helper.make_node("ConstantOfShape", ["emb_shape"], ["emb_zeros"],
                         value=helper.make_tensor("v", TensorProto.FLOAT, [1], [0.0])),
        helper.make_node("Concat", ["embedding", "emb_zeros"], ["spks_in_pair"], axis=0),
    ])
    # cond_in_pair
    pre_loop_nodes.extend([
        helper.make_node("Shape", ["cond"], ["cond_shape"]),
        helper.make_node("ConstantOfShape", ["cond_shape"], ["cond_zeros"],
                         value=helper.make_tensor("v", TensorProto.FLOAT, [1], [0.0])),
        helper.make_node("Concat", ["cond", "cond_zeros"], ["cond_in_pair"], axis=0),
    ])
    # mask_in_pair = Concat([mel_mask, mel_mask], axis=0)
    pre_loop_nodes.append(
        helper.make_node("Concat", ["mel_mask", "mel_mask"], ["mask_in_pair"], axis=0),
    )

    # Build the Loop body subgraph.
    body, cfm_initializers_outer = build_loop_body(
        ce_model.graph,
        mu_in_pair="mu_in_pair",
        mask_in_pair="mask_in_pair",
        spks_in_pair="spks_in_pair",
        cond_in_pair="cond_in_pair",
    )
    outer_inits.extend(cfm_initializers_outer)

    # flow_encoder's `z` output stays at [1, MelBins, T_mel] regardless of
    # the input batch B — the export's pinned-rand_noise patch (Run 11 of
    # docs/chatterbox_investigation.md, for parity-test reproducibility)
    # makes the graph treat z as effectively constant. At B>1 the body
    # needs x_carried at [B, MelBins, T_mel], so we Expand z to mu's shape
    # before feeding it as the Loop's v_initial. At B=1 Expand([1,...], [1,...])
    # is a no-op pass-through, preserving prior parity.
    pre_loop_nodes.append(
        helper.make_node("Expand", ["z", "mu_shape"], ["z_b"]),
    )

    # Loop op: trip=10, cond=True, v_initial=[z_b], outputs x_final.
    loop_node = helper.make_node(
        "Loop",
        inputs=["trip_count_const", "loop_cond_init", "z_b"],
        outputs=["x_final"],
        body=body,
    )

    # Trim mel: Slice(x_final, starts=[PromptLen], ends=[INT64_MAX], axes=[2])
    # Opset 18 Slice: inputs (data, starts, ends, axes, steps) as tensors.
    trim_nodes = [
        make_const_node("trim_starts", np.array([PROMPT_LEN], dtype=np.int64)),
        make_const_node("trim_ends", np.array([np.iinfo(np.int64).max], dtype=np.int64)),
        make_const_node("trim_axes", np.array([2], dtype=np.int64)),
        helper.make_node("Slice", ["x_final", "trim_starts", "trim_ends", "trim_axes"],
                         ["mel"]),
    ]
    # mel2wav reads "mel" as input; its node outputs include "waveform" (protected).
    mw_nodes = list(mw_inlined.node)

    # Assemble all outer nodes in order: flow_enc → CFG-double prep → Loop → trim → mel2wav.
    all_nodes = list(pre_loop_nodes) + [loop_node] + list(trim_nodes) + list(mw_nodes)

    # Outer graph inputs/outputs.
    outer_inputs = [
        helper.make_tensor_value_info("speech_tokens", TensorProto.INT64,
                                       ["batch_size", "num_speech_tokens"]),
        helper.make_tensor_value_info("speaker_embeddings", TensorProto.FLOAT,
                                       ["batch_size", 192]),
        helper.make_tensor_value_info("speaker_features", TensorProto.FLOAT,
                                       ["batch_size", "prompt_len", 80]),
    ]
    outer_outputs = [
        helper.make_tensor_value_info("waveform", TensorProto.FLOAT,
                                       ["batch_size", "num_samples"]),
    ]
    outer_graph = helper.make_graph(
        all_nodes, "conditional_decoder_loop",
        outer_inputs, outer_outputs,
        initializer=outer_inits,
    )

    # Build the final model. Inherit opset_import from cfm_estimator (it has
    # the most ops; should be a superset of what flow_encoder and mel2wav use).
    model = helper.make_model(
        outer_graph,
        opset_imports=[helper.make_opsetid("", opset)],
        producer_name="vernacula.merge_cond_decoder_loop",
    )
    # Signal to Vocoder.cs that this graph accepts B>1 (Path B / Run 10).
    # Older bundles without this metadata get routed to the split-graph
    # fallback in SynthesizeBatch instead.
    meta = model.metadata_props.add()
    meta.key = "supports_batched"
    meta.value = "true"
    # Pin IR version to match the input sub-graphs (IR 8 ships from
    # torch.onnx.export on opset 18). onnx 1.18 otherwise defaults to IR
    # 13, which ORT < 1.24 can't load.
    model.ir_version = min(fe_model.ir_version, ce_model.ir_version, mw_model.ir_version)

    print(f"Saving {out_path} ...")
    onnx.save_model(
        model, str(out_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=out_path.name + "_data",
        size_threshold=1024 * 1024,
    )

    # Stats
    print(f"  outer graph: {len(outer_graph.node)} nodes, {len(outer_inits)} initializers")
    print(f"  loop body:   {len(body.node)} nodes, {len(body.initializer)} initializers")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--onnx-dir", type=Path, required=True,
                   help="Directory containing flow_encoder.onnx, "
                        "cfm_estimator.onnx, mel2wav.onnx (from "
                        "export_chatterbox_to_onnx.py --split-cond-decoder)")
    p.add_argument("--out", type=Path,
                   help="Output path (default: <onnx-dir>/conditional_decoder_loop.onnx)")
    p.add_argument("--opset", type=int, default=18)
    args = p.parse_args()

    out = args.out or (args.onnx_dir / "conditional_decoder_loop.onnx")
    merge(
        args.onnx_dir / "flow_encoder.onnx",
        args.onnx_dir / "cfm_estimator.onnx",
        args.onnx_dir / "mel2wav.onnx",
        out,
        opset=args.opset,
    )


if __name__ == "__main__":
    main()
