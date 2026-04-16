from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a NeMo streaming Sortformer diarization .nemo checkpoint into "
            "the ONNX contract used by this repository."
        )
    )
    parser.add_argument("--nemo", required=True, help="Path to diar_streaming_sortformer_4spk-v2.1.nemo")
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=17, help="Target ONNX opset")
    parser.add_argument(
        "--dynamo",
        choices=("auto", "true", "false"),
        default="false",
        help="Control whether torch.onnx.export uses the newer dynamo exporter path.",
    )
    parser.add_argument(
        "--optimize",
        choices=("auto", "true", "false"),
        default="auto",
        help="Control exporter graph optimization.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Enable torch.onnx.export verification.",
    )
    parser.add_argument(
        "--no-constant-folding",
        action="store_true",
        help="Disable constant folding during export.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Dummy batch size for export")
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=992,
        help="Dummy chunk length in mel frames. Vernacula's streaming path uses 992 mel frames per chunk.",
    )
    parser.add_argument(
        "--static-streaming-batch1",
        action="store_true",
        help=(
            "Export a batch-1 streaming-specialized graph with fixed input tensor shapes for "
            "chunk/spkcache/fifo. The graph still consumes logical lengths, but trims the fixed "
            "buffers before concatenation instead of using the generic concat_and_pad path."
        ),
    )
    parser.add_argument(
        "--dynamic-streaming-batch1",
        action="store_true",
        help=(
            "Export a batch-1 streaming-specialized graph that keeps dynamic time dimensions but "
            "uses direct batch-1 concatenation instead of the generic concat_and_pad path."
        ),
    )
    parser.add_argument("--fixed-spkcache-frames", type=int, default=188, help="Fixed speaker-cache frames for --static-streaming-batch1")
    parser.add_argument("--fixed-fifo-frames", type=int, default=124, help="Fixed FIFO frames for --static-streaming-batch1")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output file")
    return parser.parse_args()


def import_dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
        from torch import nn
        from nemo.collections.asr.models import SortformerEncLabelModel
    except Exception as exc:
        raise SystemExit(
            "Missing export dependencies. Use scripts/nemo_export/setup_nemo_export_env.ps1 first.\n"
            f"Original import error: {exc}"
        ) from exc
    return torch, nn, SortformerEncLabelModel


def resolve_export_bool(raw: str, *, default: bool) -> bool:
    if raw == "auto":
        return default
    return raw == "true"


def build_dynamic_shapes(torch: Any, *, dynamic_streaming_batch1: bool) -> dict[str, Any]:
    Dim = torch.export.Dim
    if dynamic_streaming_batch1:
        time_chunk_subsampled = Dim("time_chunk_subsampled", max=2048)
        return {
            # The benchmark/runtime feeds chunk in pre-subsampling mel frames, while the
            # encoder emits a sequence subsampled by 8. Keeping this affine constraint
            # symbolic avoids freezing chunk at the 124-frame export example length.
            "chunk": {1: 8 * time_chunk_subsampled},
            "chunk_lengths": None,
            "spkcache": {1: Dim("time_cache", max=4999)},
            "spkcache_lengths": None,
            "fifo": {1: Dim("time_fifo", max=4999)},
            "fifo_lengths": None,
        }
    batch_dim = Dim("batch")
    return {
        "chunk": {0: batch_dim, 1: Dim("time_chunk")},
        "chunk_lengths": {0: batch_dim},
        "spkcache": {0: batch_dim, 1: Dim("time_cache")},
        "spkcache_lengths": {0: batch_dim},
        "fifo": {0: batch_dim, 1: Dim("time_fifo")},
        "fifo_lengths": {0: batch_dim},
    }


def concat_and_pad_exportable(torch: Any, embs: list[Any], lengths: list[Any]) -> tuple[Any, Any]:
    device = embs[0].device
    dtype = embs[0].dtype
    batch_size = embs[0].shape[0]
    emb_dim = embs[0].shape[2]
    total_lengths = torch.sum(torch.stack(lengths), dim=0)
    max_total = torch.max(total_lengths)
    output = torch.zeros((batch_size, max_total, emb_dim), device=device, dtype=dtype)
    offsets = torch.zeros((batch_size,), device=device, dtype=torch.int64)

    for emb, length in zip(embs, lengths):
        t = emb.shape[1]
        pos = torch.arange(t, device=device, dtype=torch.int64).unsqueeze(0).expand(batch_size, t)
        valid = pos < length.unsqueeze(1)
        safe_target = torch.where(valid, offsets.unsqueeze(1) + pos, torch.zeros_like(pos))
        src = emb * valid.unsqueeze(-1).to(dtype)
        output = output.scatter_add(1, safe_target.unsqueeze(-1).expand(batch_size, t, emb_dim), src)
        offsets = offsets + length

    return output, total_lengths


def build_wrapper(
    torch: Any,
    nn: Any,
    model: Any,
    *,
    static_streaming_batch1: bool = False,
    dynamic_streaming_batch1: bool = False,
) -> Any:
    class SortformerExportWrapper(nn.Module):
        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, chunk: Any, chunk_lengths: Any, spkcache: Any, spkcache_lengths: Any, fifo: Any, fifo_lengths: Any) -> tuple[Any, Any, Any]:
            chunk_pre_encode_embs, chunk_pre_encode_lengths = self.inner.encoder.pre_encode(x=chunk, lengths=chunk_lengths)
            chunk_pre_encode_lengths = chunk_pre_encode_lengths.to(torch.int64)
            if static_streaming_batch1:
                spk_len = spkcache_lengths[0].to(torch.int64)
                fifo_len = fifo_lengths[0].to(torch.int64)
                chunk_len = chunk_pre_encode_lengths[0].to(torch.int64)

                spkcache_trimmed = spkcache[:, :spk_len, :]
                fifo_trimmed = fifo[:, :fifo_len, :]
                chunk_trimmed = chunk_pre_encode_embs[:, :chunk_len, :]

                spkcache_fifo_chunk_pre_encode_embs = torch.cat(
                    [spkcache_trimmed, fifo_trimmed, chunk_trimmed],
                    dim=1,
                )
                spkcache_fifo_chunk_pre_encode_lengths = chunk_pre_encode_lengths + spkcache_lengths + fifo_lengths
            elif dynamic_streaming_batch1:
                spkcache_fifo_chunk_pre_encode_embs = torch.cat(
                    [spkcache, fifo, chunk_pre_encode_embs],
                    dim=1,
                )
                spkcache_fifo_chunk_pre_encode_lengths = chunk_pre_encode_lengths + spkcache_lengths + fifo_lengths
            else:
                spkcache_fifo_chunk_pre_encode_embs, spkcache_fifo_chunk_pre_encode_lengths = concat_and_pad_exportable(
                    torch,
                    [spkcache, fifo, chunk_pre_encode_embs],
                    [spkcache_lengths, fifo_lengths, chunk_pre_encode_lengths],
                )
            spkcache_fifo_chunk_fc_encoder_embs, spkcache_fifo_chunk_fc_encoder_lengths = self.inner.frontend_encoder(
                processed_signal=spkcache_fifo_chunk_pre_encode_embs,
                processed_signal_length=spkcache_fifo_chunk_pre_encode_lengths,
                bypass_pre_encode=True,
            )
            spkcache_fifo_chunk_preds = self.inner.forward_infer(
                spkcache_fifo_chunk_fc_encoder_embs,
                spkcache_fifo_chunk_fc_encoder_lengths,
            )
            return spkcache_fifo_chunk_preds, chunk_pre_encode_embs, chunk_pre_encode_lengths

    return SortformerExportWrapper(model)


@dataclass
class ExportMetadata:
    nemo_file: str
    nemo_model_class: str
    device: str
    opset: int
    batch_size: int
    chunk_frames: int
    feature_dim: int
    emb_dim: int
    num_speakers: int
    notes: list[str]


def main() -> None:
    args = parse_args()
    nemo_path = Path(args.nemo).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    report_path = output_path.with_suffix(output_path.suffix + ".report.json")

    if not nemo_path.exists():
        raise SystemExit(f".nemo file not found: {nemo_path}")
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {output_path}. Re-run with --overwrite.")
    if args.static_streaming_batch1 and args.dynamic_streaming_batch1:
        raise SystemExit("Choose only one of --static-streaming-batch1 or --dynamic-streaming-batch1.")
    if (args.static_streaming_batch1 or args.dynamic_streaming_batch1) and args.batch_size != 1:
        raise SystemExit("The batch-1 streaming-specialized export modes currently require --batch-size 1.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch, nn, SortformerEncLabelModel = import_dependencies()
    model = SortformerEncLabelModel.restore_from(str(nemo_path), map_location="cpu")
    model.eval()

    feature_dim = int(model.cfg.encoder.feat_in)
    emb_dim = int(model.cfg.sortformer_modules.fc_d_model)
    num_speakers = int(model.cfg.sortformer_modules.num_spks)

    wrapper = build_wrapper(
        torch,
        nn,
        model,
        static_streaming_batch1=args.static_streaming_batch1,
        dynamic_streaming_batch1=args.dynamic_streaming_batch1,
    )
    wrapper.eval()

    batch = args.batch_size
    chunk = torch.randn(batch, args.chunk_frames, feature_dim)
    chunk_lengths = torch.full((batch,), args.chunk_frames, dtype=torch.int64)
    spkcache_frames = args.fixed_spkcache_frames if args.static_streaming_batch1 else 0
    fifo_frames = args.fixed_fifo_frames if args.static_streaming_batch1 else 0
    spkcache = torch.zeros(batch, spkcache_frames, emb_dim)
    spkcache_lengths = torch.zeros((batch,), dtype=torch.int64)
    fifo = torch.zeros(batch, fifo_frames, emb_dim)
    fifo_lengths = torch.zeros((batch,), dtype=torch.int64)
    dynamo = resolve_export_bool(args.dynamo, default=False)
    optimize = resolve_export_bool(args.optimize, default=dynamo)
    export_kwargs: dict[str, Any] = {
        "input_names": ["chunk", "chunk_lengths", "spkcache", "spkcache_lengths", "fifo", "fifo_lengths"],
        "output_names": ["spkcache_fifo_chunk_preds", "chunk_pre_encode_embs", "chunk_pre_encode_lengths"],
        "opset_version": args.opset,
        "dynamo": dynamo,
        "optimize": optimize,
        "verify": args.verify,
        "do_constant_folding": not args.no_constant_folding,
    }
    if not args.static_streaming_batch1:
        dynamic_axes = {
            "chunk": {1: "time_chunk"} if args.dynamic_streaming_batch1 else {0: "batch", 1: "time_chunk"},
            "spkcache": {1: "time_cache"} if args.dynamic_streaming_batch1 else {0: "batch", 1: "time_cache"},
            "fifo": {1: "time_fifo"} if args.dynamic_streaming_batch1 else {0: "batch", 1: "time_fifo"},
            "spkcache_fifo_chunk_preds": {1: "time_out"} if args.dynamic_streaming_batch1 else {0: "batch", 1: "time_out"},
            "chunk_pre_encode_embs": {1: "time_pre_encode"} if args.dynamic_streaming_batch1 else {0: "batch", 1: "time_pre_encode"},
        }
        if not args.dynamic_streaming_batch1:
            dynamic_axes["chunk_lengths"] = {0: "batch"}
            dynamic_axes["spkcache_lengths"] = {0: "batch"}
            dynamic_axes["fifo_lengths"] = {0: "batch"}
            dynamic_axes["chunk_pre_encode_lengths"] = {0: "batch"}
        if dynamo:
            export_kwargs["dynamic_shapes"] = build_dynamic_shapes(
                torch,
                dynamic_streaming_batch1=args.dynamic_streaming_batch1,
            )
        else:
            export_kwargs["dynamic_axes"] = dynamic_axes

    torch.onnx.export(
        wrapper,
        (chunk, chunk_lengths, spkcache, spkcache_lengths, fifo, fifo_lengths),
        str(output_path),
        **export_kwargs,
    )

    metadata = ExportMetadata(
        nemo_file=str(nemo_path),
        nemo_model_class=type(model).__name__,
        device="cpu",
        opset=args.opset,
        batch_size=batch,
        chunk_frames=args.chunk_frames,
        feature_dim=feature_dim,
        emb_dim=emb_dim,
        num_speakers=num_speakers,
        notes=[
            "Worked around NeMo's built-in streaming_input_examples(), which hardcodes an incompatible (120,80) chunk shape for this checkpoint.",
            "Used a custom ONNX-friendly concat_and_pad replacement to avoid the exporter failure in nemo.collections.asr.modules.sortformer_modules.SortformerModules.concat_and_pad.",
            f"Exporter settings: dynamo={dynamo}, optimize={optimize}, verify={args.verify}, constant_folding={not args.no_constant_folding}."
        ],
    )
    if args.static_streaming_batch1:
        metadata.notes.append(
            f"Exported a batch-1 static-streaming candidate with fixed shapes: chunk={args.chunk_frames}, "
            f"spkcache={args.fixed_spkcache_frames}, fifo={args.fixed_fifo_frames}."
        )
        metadata.notes.append(
            "The static candidate trims fixed-size cache/fifo buffers using the provided logical lengths before concatenation."
        )
    if args.dynamic_streaming_batch1:
        metadata.notes.append(
            "Exported a batch-1 dynamic-streaming candidate that uses direct concatenation for exact-length cache/fifo tensors."
        )

    with report_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(asdict(metadata), handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(output_path)
    print(report_path)


if __name__ == "__main__":
    main()
