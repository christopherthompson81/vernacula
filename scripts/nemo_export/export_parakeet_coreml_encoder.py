#!/usr/bin/env python3
"""Export the Parakeet conformer encoder as a static-shape graph the CoreML
execution provider can compile, WITHOUT baking a constant length.

    python scripts/nemo_export/export_parakeet_coreml_encoder.py \
        --nemo ~/models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo \
        --output-dir ~/models/parakeet_coreml --frames 1000 --frames 2000

Why this is not the Sortformer recipe
-------------------------------------
docs/coreml_onnx_playbook.md reaches one CoreML partition by making every length
a graph constant, which folds the attention padding mask to an all-False constant
that Technique 3 then deletes. Sortformer can afford that: its lengths really are
constant, apart from the final chunk of a recording, which the runtime routes to
the stock graph instead.

Parakeet cannot. Segment lengths are arbitrary, so a constant length would be
wrong for essentially every segment, and there is no one-inference-per-file escape
hatch. Measured on real speech (see the investigation log), baking the length moves
the encoder output by 0.17-0.24 on valid frames and changes the transcript by
2-4% WER on 8-20 s segments, worse on short ones.

What this does instead: HOIST THE MASK
--------------------------------------
Masking is the only thing `length` is used for, and the encoder is padding-invariant
to 4e-7 as long as the mask is honest (verified at +10%/+50%/+100% padding). So the
graph takes the mask as an INPUT rather than deriving it from a length:

    audio_signal  [1, 128, F]        mel features, zero-padded to the bucket
    pad_keep      [1, F]             1.0 for a real mel frame, 0.0 for padding

That removes the `Range`/`Less`/`Expand`/`ConstantOfShape` chain that made every
downstream shape data-dependent, and it keeps the masking exact for any true length
<= the bucket.

`length` reaches masking at FOUR resolutions, not one, and missing any of them is a
silent 1e-2 error rather than a failure:

* `MaskedConvSequential` re-masks between every strided conv in the subsampler, at
  the mel rate and after each of the three strides;
* `_create_masks` builds the attention and conv-module masks at the encoder rate.

All four are exact stride-2 decimations of each other, so the graph derives them from
the single mel-rate input with constant-parameter `Slice`s (`Slice` with constant
starts/ends/steps is CoreML-supported). `check_mask_decimation` re-proves that
identity for every length in the bucket before each export, so a NeMo change to the
subsampler's arithmetic fails here rather than silently costing 1e-2. The
`masked_fill`s become `+bias` / `*keep` arithmetic (`Add`/`Mul`, also supported), so
no `Where` survives to fragment the graph.

The caller's obligations:

* zero-pad the mel features to the bucket and set `pad_keep` to match the true
  frame count. Passing all-ones for a short segment is the mistake this design
  exists to avoid.
* compute the true output length itself (`calc_encoded_length`, the same formula
  NeMo's `calc_length` uses) and ignore encoder output past it. The graph has no
  length output because it has no length input.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nemo", required=True, help="Path to the .nemo checkpoint.")
    p.add_argument("--output-dir", required=True,
                   help="Directory to write encoder-model.coreml-<frames>.onnx into.")
    p.add_argument("--frames", type=int, action="append", required=True, metavar="N",
                   help="Mel-frame bucket to export (100 frames = 1 s). Repeatable; "
                        "one graph is written per bucket.")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--weights-name", default="encoder-model.onnx.data",
                   help="Filename every bucket's external weights point at. Defaults to "
                        "the published encoder's sidecar, which these weights are "
                        "byte-identical to -- so the buckets add no weight download.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def calc_encoded_length(frames: int, *, sampling_num: int = 3, kernel: int = 3,
                        stride: int = 2, all_paddings: int = 2) -> int:
    """NeMo's `calc_length` for the dw_striding subsampler, in plain ints.

    The C# caller needs the identical formula, since the graph no longer reports
    an encoded length of its own.
    """
    length = frames
    for _ in range(sampling_num):
        length = math.floor((length + all_paddings - kernel) / stride) + 1
    return length


def check_mask_decimation(frames: int, *, sampling_num: int = 3, kernel: int = 3,
                          stride: int = 2, all_paddings: int = 2) -> None:
    """Prove `keep[0::2]` is the stage mask, for every true length in this bucket.

    The whole export rests on this: the subsampler's per-stage masks are
    `arange(T_k) < length_k`, and the graph instead decimates one input mask by 2.
    That is only the same tensor because `length_k = ceil(length_{k-1} / 2)` and
    `keep[2i]` is 1 exactly while `2i < length`. It is cheap to check outright, and a
    NeMo change to `calculate_conv_output_size` would otherwise cost a silent 1e-2
    that looks like a tolerance question.
    """
    for length in range(1, frames + 1):
        kept, current = length, length
        width = frames
        for _ in range(sampling_num):
            width = (width + all_paddings - kernel) // stride + 1
            # what the decimated mask keeps: indices 2i < kept, i.e. ceil(kept / 2)
            kept = -(-kept // 2)
            current = (current + all_paddings - kernel) // stride + 1
            if kept != current:
                raise SystemExit(
                    f"mask decimation does not match NeMo's arithmetic at bucket "
                    f"{frames}, true length {length}: decimated mask keeps {kept} "
                    f"frames, calc_length says {current}. The stride-2 slice in "
                    f"_patch_masking is no longer valid -- do not ship this export.")


def _patch_masking(torch: Any, encoder: Any) -> None:
    """Replace every length-derived mask in the encoder with a slice of one input mask.

    Patched onto the module INSTANCES, not the classes, so nothing else in the
    process is affected.

    Every substitution is exact, not an approximation:

    * `arange(T_k) < length_k` at subsampler stage k is exactly `keep[0::2]` applied k
      times to the mel-rate mask. `length_k = (length_{k-1} - 1) // 2 + 1 =
      ceil(length_{k-1} / 2)`, and `keep[2i]` is 1 exactly while `2i < length`, i.e.
      for `ceil(length / 2)` entries. Checked exhaustively over every length in
      several bucket sizes.
    * `apply_channel_mask` is already a multiply, so reusing the float keep is
      literally the same op.
    * `scores.masked_fill(mask, -INF_VAL)` sets masked scores to exactly -10000;
      adding `(keep - 1) * INF_VAL` makes them `score - 10000`. Scores are O(30),
      so both underflow to zero in the softmax against a max of the same order,
      and both are then multiplied to exactly zero by the second mask anyway.
    * `softmax(...).masked_fill(mask, 0.0)` is exactly `softmax(...) * keep`.
    """
    import types

    from nemo.collections.asr.parts.submodules.multi_head_attention import INF_VAL

    def masked_conv_forward(self, x, lengths):
        # NeMo re-masks around every layer and recomputes the mask after every
        # stride. Same schedule here, with the mask decimated instead of rebuilt.
        keep = encoder._exp_mel_keep
        x = x.unsqueeze(1)                                      # [B, 1, T, F]
        for layer in self:
            x = x * keep.reshape(1, 1, -1, 1)
            x = layer(x)
            if hasattr(layer, "stride") and layer.stride != (1, 1):
                keep = keep[:, 0::2]
        x = x * keep.reshape(1, 1, -1, 1)
        # The encoder-rate mask, for _create_masks. forward_internal always calls
        # pre_encode before it, so this is set by the time it is read.
        encoder._exp_enc_keep = keep
        # The real return value is the frame count, which this export discards -- the
        # caller computes it with calc_encoded_length instead.
        return x, lengths

    def forward_attention(self, value, scores, mask):
        # `mask` is a float keep tensor [B, T1, T2] here, not NeMo's bool mask.
        keep = mask.unsqueeze(1)                                  # [B, 1, T1, T2]
        scores = scores + (keep - 1.0) * INF_VAL
        attn = torch.softmax(scores, dim=-1) * keep
        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)
        x = x.transpose(1, 2).reshape(value.size(0), -1, self.h * self.d_k)
        return self.linear_out(x)

    def conv_forward(self, x, pad_mask=None, cache=None):
        # `pad_mask` is a float keep tensor [B, T] here. `cache` is unused: this
        # export has no streaming path.
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        if self.pointwise_activation == "glu_":
            x = torch.nn.functional.glu(x, dim=1)
        else:
            x = self.pointwise_activation(x)
        if pad_mask is not None:
            x = x * pad_mask.unsqueeze(1)
        x = self.depthwise_conv(x)
        if self.norm_type == "layer_norm":
            x = x.transpose(1, 2)
            x = self.batch_norm(x)
            x = x.transpose(1, 2)
        else:
            x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        return x.transpose(1, 2)

    def create_masks(self, **_kwargs):
        # Returning the hoisted mask here is what takes mask construction out of the
        # graph entirely. att_keep[i, j] = keep[i] * keep[j] -- a masked query row
        # keeps nothing, matching NeMo's symmetric `pad_mask & pad_mask.T`.
        keep = self._exp_enc_keep
        return keep, keep.unsqueeze(2) * keep.unsqueeze(1)

    encoder.pre_encode.conv.forward = types.MethodType(
        masked_conv_forward, encoder.pre_encode.conv)
    for layer in encoder.layers:
        layer.self_attn.forward_attention = types.MethodType(forward_attention, layer.self_attn)
        layer.conv.forward = types.MethodType(conv_forward, layer.conv)
    encoder._create_masks = types.MethodType(create_masks, encoder)


def build_wrapper(torch: Any, nn: Any, model: Any, frames: int) -> Any:
    encoder = model.encoder
    _patch_masking(torch, encoder)

    class ParakeetCoreMLEncoder(nn.Module):
        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, audio_signal: Any, pad_keep: Any) -> Any:
            self.inner._exp_mel_keep = pad_keep
            # `length` reaches nothing that matters any more: every masking site now
            # reads the hoisted mask, and the frame count it computes is discarded
            # with the second output.
            length = torch.full((audio_signal.size(0),), frames, dtype=torch.int64)
            encoded, _ = self.inner(audio_signal=audio_signal, length=length)
            return encoded

    return ParakeetCoreMLEncoder(encoder).eval()


def _all_tensors(graph: Any):
    """Every tensor that can carry external data: initializers AND the tensors sitting
    inside `Constant` node attributes, which the exporter spills too."""
    yield from graph.initializer
    for node in graph.node:
        for attr in node.attribute:
            if attr.HasField("t"):
                yield attr.t
            yield from attr.tensors


def consolidate_external_data(onnx_path: Path, sidecar: str) -> None:
    """Collapse the exporter's one-file-per-tensor external data into a single sidecar.

    The encoder's weights are 2.4 GB, well past the 2 GB protobuf ceiling, so
    torch.onnx.export spills them -- and it spills them as ~300 separate files
    named after the tensors, dumped straight into the output directory.
    """
    import onnx

    # Read the locations BEFORE the data: a full load resolves external_data into
    # raw_data and clears the field, so asking the loaded model where its tensors
    # came from silently returns nothing and leaves ~300 stale files behind.
    meta = onnx.load(str(onnx_path), load_external_data=False)
    stale = {kv.value for t in _all_tensors(meta.graph) for kv in t.external_data
             if kv.key == "location"}
    model = onnx.load(str(onnx_path))          # pulls the loose files into memory
    onnx.save_model(model, str(onnx_path), save_as_external_data=True,
                    all_tensors_to_one_file=True, location=sidecar, size_threshold=0)
    for loc in stale:
        p = onnx_path.parent / loc
        if p.exists() and p.name != sidecar:
            p.unlink()


def md5(path: Path) -> str:
    import hashlib

    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def repoint_external_data(onnx_path: Path, sidecar: str) -> None:
    """Point every external tensor at `sidecar` without touching the weights."""
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    for tensor in _all_tensors(model.graph):
        for kv in tensor.external_data:
            if kv.key == "location":
                kv.value = sidecar
    onnx.save_model(model, str(onnx_path))


def export_bucket(torch: Any, nn: Any, model: Any, frames: int, dst: Path, opset: int) -> dict:
    encoded = calc_encoded_length(frames)
    wrapper = build_wrapper(torch, nn, model, frames)

    audio_signal = torch.randn(1, model.encoder._feat_in, frames)
    pad_keep = torch.ones(1, frames)

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (audio_signal, pad_keep),
            str(dst),
            input_names=["audio_signal", "pad_keep"],
            output_names=["outputs"],
            dynamic_axes=None,          # fully static: Technique 1
            opset_version=opset,
            dynamo=False,
            do_constant_folding=True,
        )
    own_sidecar = dst.name + ".data"
    consolidate_external_data(dst, own_sidecar)

    return {
        "file": dst.name,
        "mel_frames": frames,
        "encoded_frames": encoded,
        "seconds": round(frames / 100.0, 2),
        "bytes": dst.stat().st_size,
        "weights_md5": md5(dst.parent / own_sidecar),
    }


def share_weights(out_dir: Path, exported: list, sidecar: str) -> bool:
    """Collapse the per-bucket weight sidecars into one, if they are byte-identical.

    Every bucket traces the same parameters in the same order, so the sidecars come
    out identical and shipping one per bucket would be N x 2.4 GB of the same bytes.
    Verified by digest rather than assumed -- a torch or NeMo change that reorders
    the trace would break the sharing silently otherwise.
    """
    digests = {b["weights_md5"] for b in exported}
    if len(digests) != 1:
        print("  ! the buckets' weight files differ; keeping one sidecar per bucket")
        return False
    keep = out_dir / (exported[0]["file"] + ".data")
    keep.replace(out_dir / sidecar)
    for bucket in exported[1:]:
        (out_dir / (bucket["file"] + ".data")).unlink()
    for bucket in exported:
        repoint_external_data(out_dir / bucket["file"], sidecar)
    return True


def main() -> None:
    args = parse_args()
    nemo_path = Path(args.nemo).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    if not nemo_path.exists():
        raise SystemExit(f".nemo file not found: {nemo_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    buckets = sorted(set(args.frames))
    for frames in buckets:
        if frames % 8 != 0:
            raise SystemExit(
                f"--frames {frames} is not a multiple of 8. The subsampler strides by 8, "
                "so a bucket that is not a multiple of 8 wastes frames and makes the "
                "caller's pad_keep arithmetic fiddlier for nothing.")

    import torch
    from torch import nn
    from nemo.collections.asr.models import ASRModel

    model = ASRModel.restore_from(str(nemo_path), map_location="cpu")
    model.freeze()
    model.eval()

    exported = []
    for frames in buckets:
        dst = out_dir / f"encoder-model.coreml-{frames}.onnx"
        if dst.exists() and not args.overwrite:
            raise SystemExit(f"{dst} exists. Re-run with --overwrite.")
        print(f"exporting {frames} mel frames ({frames / 100:.1f} s) -> {dst.name} ...")
        check_mask_decimation(frames)
        exported.append(export_bucket(torch, nn, model, frames, dst, args.opset))
        print(f"  {exported[-1]['bytes'] / 1e6:.0f} MB graph, {exported[-1]['encoded_frames']} encoded frames")

    shared = share_weights(out_dir, exported, args.weights_name)
    weights = out_dir / args.weights_name
    if shared:
        print(f"  all {len(exported)} buckets share {args.weights_name} "
              f"({weights.stat().st_size / 1e9:.2f} GB, md5 {exported[0]['weights_md5']})")
        print("  ^ compare that md5 against the published encoder-model.onnx.data: if it "
              "matches, these graphs need no new weight download")

    report = {
        "nemo_file": str(nemo_path),
        "opset": args.opset,
        "weights_file": args.weights_name if shared else "one per bucket",
        "weights_md5": exported[0]["weights_md5"] if shared else None,
        "inputs": {
            "audio_signal": "[1, 128, mel_frames] float32, zero-padded mel features",
            "pad_keep": "[1, mel_frames] float32, 1.0 real / 0.0 padding",
        },
        "outputs": {"outputs": "[1, 1024, encoded_frames] float32"},
        "caller_obligations": [
            "pad_keep must reflect the TRUE frame count; all-ones on a short segment "
            "reproduces the accuracy loss this export exists to avoid",
            "the graph has no encoded_lengths output -- compute it with calc_encoded_length "
            "and ignore output frames past it",
        ],
        "buckets": exported,
    }
    with (out_dir / "coreml-encoder-report.json").open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")
    print(f"wrote {out_dir / 'coreml-encoder-report.json'}")


if __name__ == "__main__":
    main()
