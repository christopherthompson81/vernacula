#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from benchmark_sortformer_rtf import (
    CHUNK_LENGTH,
    EMBEDDING_DIMENSION,
    FIFO_LENGTH,
    NMELS,
    NUM_SPEAKERS,
    SAMPLE_RATE,
    SPEAKER_CACHE_LENGTH,
    SUBSAMPLING,
    NemoSortformerPipeline,
    OnnxSortformerPipeline,
    SortformerPipelineBase,
    load_audio,
    log_mel_spectrogram,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Sortformer streaming outputs chunk-by-chunk across two backends "
            "(for example legacy ONNX vs torch.export ONNX)."
        )
    )
    parser.add_argument("--audio", type=Path, required=True, help="16 kHz mono comparison audio.")
    parser.add_argument("--lhs-runtime", choices=("onnx", "nemo"), default="onnx")
    parser.add_argument("--lhs-model", type=Path, required=True)
    parser.add_argument("--lhs-label", default="lhs")
    parser.add_argument("--rhs-runtime", choices=("onnx", "nemo"), default="onnx")
    parser.add_argument("--rhs-model", type=Path, required=True)
    parser.add_argument("--rhs-label", default="rhs")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--max-audio-seconds", type=float, default=None)
    parser.add_argument("--max-chunks", type=int, default=None, help="Limit comparison to the first N chunks.")
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


@dataclass
class DiffSummary:
    max_abs: float
    mean_abs: float
    rmse: float
    compared_values: int


@dataclass
class ChunkComparison:
    chunk_index: int
    mel_frame_start: int
    mel_frame_end: int
    current_len: int
    lhs_cache_len_before: int
    rhs_cache_len_before: int
    lhs_fifo_len_before: int
    rhs_fifo_len_before: int
    lhs_raw_pred_shape: list[int]
    rhs_raw_pred_shape: list[int]
    lhs_raw_emb_shape: list[int]
    rhs_raw_emb_shape: list[int]
    lhs_chunk_pred_shape: list[int]
    rhs_chunk_pred_shape: list[int]
    raw_pred_diff: DiffSummary
    raw_emb_diff: DiffSummary
    chunk_pred_diff: DiffSummary
    chunk_input_diff: DiffSummary


@dataclass
class ComparisonSummary:
    audio_seconds: float
    total_chunks: int
    compared_chunks: int
    lhs_label: str
    rhs_label: str
    lhs_runtime: str
    rhs_runtime: str
    lhs_model: str
    rhs_model: str
    final_segment_count_lhs: int
    final_segment_count_rhs: int
    first_raw_pred_diff_chunk: int | None
    first_raw_emb_diff_chunk: int | None
    first_chunk_pred_diff_chunk: int | None
    max_raw_pred_diff: float
    max_raw_emb_diff: float
    max_chunk_pred_diff: float
    chunks: list[ChunkComparison]


def create_pipeline(runtime: str, model_path: Path, device: str, gpu_id: int) -> SortformerPipelineBase:
    if runtime == "onnx":
        return OnnxSortformerPipeline(model_path, device, gpu_id)
    return NemoSortformerPipeline(model_path, device, gpu_id)


def summarize_diff(lhs: np.ndarray, rhs: np.ndarray) -> DiffSummary:
    if lhs.size == 0 or rhs.size == 0:
        return DiffSummary(max_abs=0.0, mean_abs=0.0, rmse=0.0, compared_values=0)

    overlap_shape = tuple(min(a, b) for a, b in zip(lhs.shape, rhs.shape))
    lhs_overlap = lhs[tuple(slice(0, dim) for dim in overlap_shape)].astype(np.float64, copy=False)
    rhs_overlap = rhs[tuple(slice(0, dim) for dim in overlap_shape)].astype(np.float64, copy=False)
    diff = lhs_overlap - rhs_overlap
    compared = int(diff.size)
    if compared == 0:
        return DiffSummary(max_abs=0.0, mean_abs=0.0, rmse=0.0, compared_values=0)
    abs_diff = np.abs(diff)
    return DiffSummary(
        max_abs=float(np.max(abs_diff)),
        mean_abs=float(np.mean(abs_diff)),
        rmse=float(np.sqrt(np.mean(diff * diff))),
        compared_values=compared,
    )


def capture_process_chunk(
    pipeline: SortformerPipelineBase,
    idx: int,
    chunk_stride: int,
    total_frames: int,
    mel_spec: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    captured: dict[str, Any] = {
        "cache_len_before": int(pipeline._spkcache.shape[1]),
        "fifo_len_before": int(pipeline._fifo.shape[1]),
    }
    original_infer = pipeline.infer_chunk

    def wrapped_infer(chunk: np.ndarray, current_len: int) -> tuple[np.ndarray, np.ndarray]:
        captured["chunk_input"] = np.array(chunk, copy=True)
        captured["current_len"] = int(current_len)
        preds, embs = original_infer(chunk, current_len)
        captured["raw_preds"] = np.array(preds, copy=True)
        captured["raw_embs"] = np.array(embs, copy=True)
        return preds, embs

    pipeline.infer_chunk = wrapped_infer  # type: ignore[method-assign]
    try:
        chunk_preds, _ = pipeline.process_chunk(idx, chunk_stride, total_frames, mel_spec)
    finally:
        pipeline.infer_chunk = original_infer  # type: ignore[method-assign]

    captured["chunk_preds"] = np.array(chunk_preds, copy=True)
    captured["cache_len_after"] = int(pipeline._spkcache.shape[1])
    captured["fifo_len_after"] = int(pipeline._fifo.shape[1])
    return chunk_preds, captured


def compare_pipelines(
    lhs: SortformerPipelineBase,
    rhs: SortformerPipelineBase,
    mel_spec: np.ndarray,
    *,
    max_chunks: int | None,
) -> tuple[list[ChunkComparison], list[np.ndarray], list[np.ndarray]]:
    total_frames = mel_spec.shape[1]
    chunk_stride = CHUNK_LENGTH * SUBSAMPLING
    total_chunks = (total_frames + chunk_stride - 1) // chunk_stride if total_frames > 0 else 0
    if max_chunks is not None:
        total_chunks = min(total_chunks, max_chunks)

    lhs_all_preds: list[np.ndarray] = []
    rhs_all_preds: list[np.ndarray] = []
    comparisons: list[ChunkComparison] = []

    for idx in range(total_chunks):
        _, lhs_cap = capture_process_chunk(lhs, idx, chunk_stride, total_frames, mel_spec)
        _, rhs_cap = capture_process_chunk(rhs, idx, chunk_stride, total_frames, mel_spec)

        start = idx * chunk_stride
        end = min(start + chunk_stride, total_frames)
        comparisons.append(
            ChunkComparison(
                chunk_index=idx,
                mel_frame_start=int(start),
                mel_frame_end=int(end),
                current_len=int(lhs_cap["current_len"]),
                lhs_cache_len_before=int(lhs_cap["cache_len_before"]),
                rhs_cache_len_before=int(rhs_cap["cache_len_before"]),
                lhs_fifo_len_before=int(lhs_cap["fifo_len_before"]),
                rhs_fifo_len_before=int(rhs_cap["fifo_len_before"]),
                lhs_raw_pred_shape=list(lhs_cap["raw_preds"].shape),
                rhs_raw_pred_shape=list(rhs_cap["raw_preds"].shape),
                lhs_raw_emb_shape=list(lhs_cap["raw_embs"].shape),
                rhs_raw_emb_shape=list(rhs_cap["raw_embs"].shape),
                lhs_chunk_pred_shape=list(lhs_cap["chunk_preds"].shape),
                rhs_chunk_pred_shape=list(rhs_cap["chunk_preds"].shape),
                raw_pred_diff=summarize_diff(lhs_cap["raw_preds"], rhs_cap["raw_preds"]),
                raw_emb_diff=summarize_diff(lhs_cap["raw_embs"], rhs_cap["raw_embs"]),
                chunk_pred_diff=summarize_diff(lhs_cap["chunk_preds"], rhs_cap["chunk_preds"]),
                chunk_input_diff=summarize_diff(lhs_cap["chunk_input"], rhs_cap["chunk_input"]),
            )
        )
        lhs_all_preds.append(lhs_cap["chunk_preds"])
        rhs_all_preds.append(rhs_cap["chunk_preds"])

    return comparisons, lhs_all_preds, rhs_all_preds


def first_chunk_above(chunks: list[ChunkComparison], attr: str, threshold: float = 1e-4) -> int | None:
    for chunk in chunks:
        if getattr(chunk, attr).max_abs > threshold:
            return chunk.chunk_index
    return None


def max_diff(chunks: list[ChunkComparison], attr: str) -> float:
    if not chunks:
        return 0.0
    return max(getattr(chunk, attr).max_abs for chunk in chunks)


def main() -> None:
    args = parse_args()
    audio_path = args.audio.expanduser().resolve()
    lhs_model = args.lhs_model.expanduser().resolve()
    rhs_model = args.rhs_model.expanduser().resolve()
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")
    if not lhs_model.exists():
        raise SystemExit(f"Left-hand model not found: {lhs_model}")
    if not rhs_model.exists():
        raise SystemExit(f"Right-hand model not found: {rhs_model}")

    audio = load_audio(audio_path, args.max_audio_seconds)
    audio_seconds = len(audio) / SAMPLE_RATE
    mel_spec = log_mel_spectrogram(audio)

    lhs = create_pipeline(args.lhs_runtime, lhs_model, args.device, args.gpu_id)
    rhs = create_pipeline(args.rhs_runtime, rhs_model, args.device, args.gpu_id)
    lhs.reset_state()
    rhs.reset_state()

    chunks, lhs_preds, rhs_preds = compare_pipelines(lhs, rhs, mel_spec, max_chunks=args.max_chunks)
    lhs_segments = lhs.binarize_to_segments(lhs.filter_preds(lhs_preds))
    rhs_segments = rhs.binarize_to_segments(rhs.filter_preds(rhs_preds))

    summary = ComparisonSummary(
        audio_seconds=audio_seconds,
        total_chunks=(mel_spec.shape[1] + (CHUNK_LENGTH * SUBSAMPLING) - 1) // (CHUNK_LENGTH * SUBSAMPLING) if mel_spec.shape[1] > 0 else 0,
        compared_chunks=len(chunks),
        lhs_label=args.lhs_label,
        rhs_label=args.rhs_label,
        lhs_runtime=args.lhs_runtime,
        rhs_runtime=args.rhs_runtime,
        lhs_model=str(lhs_model),
        rhs_model=str(rhs_model),
        final_segment_count_lhs=len(lhs_segments),
        final_segment_count_rhs=len(rhs_segments),
        first_raw_pred_diff_chunk=first_chunk_above(chunks, "raw_pred_diff"),
        first_raw_emb_diff_chunk=first_chunk_above(chunks, "raw_emb_diff"),
        first_chunk_pred_diff_chunk=first_chunk_above(chunks, "chunk_pred_diff"),
        max_raw_pred_diff=max_diff(chunks, "raw_pred_diff"),
        max_raw_emb_diff=max_diff(chunks, "raw_emb_diff"),
        max_chunk_pred_diff=max_diff(chunks, "chunk_pred_diff"),
        chunks=chunks,
    )

    print(f"Audio: {audio_path} ({audio_seconds:.2f}s)")
    print(f"{args.lhs_label}: {args.lhs_runtime} -> {lhs_model}")
    print(f"{args.rhs_label}: {args.rhs_runtime} -> {rhs_model}")
    print(
        "First chunk above threshold:"
        f" raw_preds={summary.first_raw_pred_diff_chunk},"
        f" raw_embs={summary.first_raw_emb_diff_chunk},"
        f" chunk_preds={summary.first_chunk_pred_diff_chunk}"
    )
    print(
        "Max abs diff:"
        f" raw_preds={summary.max_raw_pred_diff:.6f},"
        f" raw_embs={summary.max_raw_emb_diff:.6f},"
        f" chunk_preds={summary.max_chunk_pred_diff:.6f}"
    )
    print(
        "Final segment counts:"
        f" {args.lhs_label}={summary.final_segment_count_lhs},"
        f" {args.rhs_label}={summary.final_segment_count_rhs}"
    )
    for chunk in chunks[: min(8, len(chunks))]:
        print(
            f"chunk {chunk.chunk_index:02d}"
            f" cache=({chunk.lhs_cache_len_before},{chunk.rhs_cache_len_before})"
            f" fifo=({chunk.lhs_fifo_len_before},{chunk.rhs_fifo_len_before})"
            f" raw_pred_max={chunk.raw_pred_diff.max_abs:.6f}"
            f" raw_emb_max={chunk.raw_emb_diff.max_abs:.6f}"
            f" chunk_pred_max={chunk.chunk_pred_diff.max_abs:.6f}"
            f" chunk_shapes={chunk.lhs_chunk_pred_shape}/{chunk.rhs_chunk_pred_shape}"
        )

    if args.json_out is not None:
        output_path = args.json_out.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(asdict(summary), handle, indent=2)
            handle.write("\n")
        print(f"JSON written to {output_path}")


if __name__ == "__main__":
    main()
