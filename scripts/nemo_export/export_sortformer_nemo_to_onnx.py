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
    parser.add_argument("--batch-size", type=int, default=1, help="Dummy batch size for export")
    parser.add_argument("--chunk-frames", type=int, default=124, help="Dummy chunk length in mel frames")
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


def build_wrapper(torch: Any, nn: Any, model: Any) -> Any:
    class SortformerExportWrapper(nn.Module):
        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, chunk: Any, chunk_lengths: Any, spkcache: Any, spkcache_lengths: Any, fifo: Any, fifo_lengths: Any) -> tuple[Any, Any, Any]:
            chunk_pre_encode_embs, chunk_pre_encode_lengths = self.inner.encoder.pre_encode(x=chunk, lengths=chunk_lengths)
            chunk_pre_encode_lengths = chunk_pre_encode_lengths.to(torch.int64)
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

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch, nn, SortformerEncLabelModel = import_dependencies()
    model = SortformerEncLabelModel.restore_from(str(nemo_path), map_location="cpu")
    model.eval()

    feature_dim = int(model.cfg.encoder.feat_in)
    emb_dim = int(model.cfg.sortformer_modules.fc_d_model)
    num_speakers = int(model.cfg.sortformer_modules.num_spks)

    wrapper = build_wrapper(torch, nn, model)
    wrapper.eval()

    batch = args.batch_size
    chunk = torch.randn(batch, args.chunk_frames, feature_dim)
    chunk_lengths = torch.full((batch,), args.chunk_frames, dtype=torch.int64)
    spkcache = torch.zeros(batch, 0, emb_dim)
    spkcache_lengths = torch.zeros((batch,), dtype=torch.int64)
    fifo = torch.zeros(batch, 0, emb_dim)
    fifo_lengths = torch.zeros((batch,), dtype=torch.int64)

    torch.onnx.export(
        wrapper,
        (chunk, chunk_lengths, spkcache, spkcache_lengths, fifo, fifo_lengths),
        str(output_path),
        input_names=["chunk", "chunk_lengths", "spkcache", "spkcache_lengths", "fifo", "fifo_lengths"],
        output_names=["spkcache_fifo_chunk_preds", "chunk_pre_encode_embs", "chunk_pre_encode_lengths"],
        dynamic_axes={
            "chunk": {0: "batch", 1: "time_chunk"},
            "chunk_lengths": {0: "batch"},
            "spkcache": {0: "batch", 1: "time_cache"},
            "spkcache_lengths": {0: "batch"},
            "fifo": {0: "batch", 1: "time_fifo"},
            "fifo_lengths": {0: "batch"},
            "spkcache_fifo_chunk_preds": {0: "batch", 1: "time_out"},
            "chunk_pre_encode_embs": {0: "batch", 1: "time_pre_encode"},
            "chunk_pre_encode_lengths": {0: "batch"},
        },
        opset_version=args.opset,
        dynamo=False,
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
            "Used the legacy torch.onnx exporter path (dynamo=False), which succeeded where the newer exporter path failed."
        ],
    )

    with report_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(asdict(metadata), handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(output_path)
    print(report_path)


if __name__ == "__main__":
    main()
