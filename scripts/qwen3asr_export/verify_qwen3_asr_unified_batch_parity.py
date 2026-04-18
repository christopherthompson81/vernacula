#!/usr/bin/env python3
"""
Verify mixed-length batched parity for the unified Qwen3-ASR decoder export.

This compares:
  - serial unified decoding (one segment at a time)
  - mixed-length unified batched decoding

The batched path uses per-row prompt padding, explicit attention masks, and
KV compaction between decode steps so shorter rows do not keep fake padding
positions alive in the cache.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from profile_qwen3_asr_pipeline import load_audio, make_session  # noqa: E402
from src.mel import log_mel_spectrogram  # noqa: E402
from src.prompt import EOS_TOKEN_IDS  # noqa: E402
from src.unified_batch import (  # noqa: E402
    DecoderConfig,
    build_prefill_inputs,
    build_step_inputs,
    compact_prefill_kv,
    compact_step_kv,
    gather_last_valid_logits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify unified mixed-length batch parity for Qwen3-ASR ONNX exports.")
    parser.add_argument("--onnx-dir", required=True, help="Directory containing encoder/decoder ONNX files")
    parser.add_argument("--audio", required=True, help="Audio file containing the test segments")
    parser.add_argument("--segments-json", required=True, help="JSON file with [{segId,start,end,...}, ...]")
    parser.add_argument(
        "--execution-provider",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Which ORT execution provider to use",
    )
    parser.add_argument("--max-tokens", type=int, default=128, help="Maximum decode tokens")
    parser.add_argument(
        "--show-tokens",
        action="store_true",
        help="Include token IDs in the output",
    )
    return parser.parse_args()


def get_providers(choice: str) -> list[str]:
    available = ort.get_available_providers()
    if choice == "cpu":
        return ["CPUExecutionProvider"]
    if choice == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError("CUDAExecutionProvider is not available in this environment")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def load_decoder_config(onnx_dir: str) -> DecoderConfig:
    config_path = Path(onnx_dir) / "config.json"
    config = json.loads(config_path.read_text())
    dec = config["decoder"]
    return DecoderConfig(
        hidden_size=dec["hidden_size"],
        num_layers=dec["num_layers"],
        num_kv_heads=dec["num_key_value_heads"],
        head_dim=dec["head_dim"],
    )


def load_embed_tokens(onnx_dir: str) -> np.ndarray:
    config_path = Path(onnx_dir) / "config.json"
    config = json.loads(config_path.read_text())
    vocab_size, hidden_size = config["embed_tokens_shape"]
    embed_path = Path(onnx_dir) / "embed_tokens.bin"
    return np.fromfile(embed_path, dtype=np.float32).reshape(vocab_size, hidden_size)


def extract_segment(audio: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    start = max(0, int(round(start_s * 16000.0)))
    end = max(start + 1, int(round(end_s * 16000.0)))
    return audio[start:end].copy()


def decode_batch(
    decoder: ort.InferenceSession,
    embed_table: np.ndarray,
    first_logits: np.ndarray,
    past_keys: np.ndarray,
    past_values: np.ndarray,
    past_lengths: np.ndarray,
    max_tokens: int,
) -> list[list[int]]:
    next_tokens = np.argmax(first_logits, axis=1).astype(np.int64)
    sequences = [[int(token)] for token in next_tokens.tolist()]
    done = np.isin(next_tokens, np.array(EOS_TOKEN_IDS, dtype=np.int64))

    for _ in range(max_tokens - 1):
        if np.all(done):
            break

        decoder_inputs = build_step_inputs(next_tokens, embed_table, past_keys, past_values, past_lengths)
        logits, present_keys, present_values = decoder.run(
            ["logits", "present_keys", "present_values"],
            decoder_inputs,
        )
        step_tokens = np.argmax(logits[:, 0, :], axis=1).astype(np.int64)
        active_mask = ~done

        for batch_index, token in enumerate(step_tokens.tolist()):
            if active_mask[batch_index]:
                sequences[batch_index].append(int(token))

        emitted_eos = np.isin(step_tokens, np.array(EOS_TOKEN_IDS, dtype=np.int64))
        past_keys, past_values, past_lengths = compact_step_kv(present_keys, present_values, past_lengths, active_mask)
        done = done | (active_mask & emitted_eos)

        for batch_index, token in enumerate(step_tokens.tolist()):
            if active_mask[batch_index]:
                next_tokens[batch_index] = int(token)

    return sequences


def run_serial_unified(
    encoder: ort.InferenceSession,
    decoder: ort.InferenceSession,
    audio: np.ndarray,
    segments: list[dict[str, object]],
    embed_table: np.ndarray,
    cfg: DecoderConfig,
    max_tokens: int,
) -> dict[int, list[int]]:
    results: dict[int, list[int]] = {}

    for segment in segments:
        segment_audio = extract_segment(audio, float(segment["start"]), float(segment["end"]))
        mel = log_mel_spectrogram(segment_audio).cpu().numpy()
        audio_features = encoder.run(["audio_features"], {"mel": mel})[0].astype(np.float32, copy=False)
        audio_lengths = np.array([audio_features.shape[1]], dtype=np.int64)

        decoder_inputs, seq_lengths = build_prefill_inputs(audio_features, audio_lengths, embed_table, cfg)
        logits, present_keys, present_values = decoder.run(
            ["logits", "present_keys", "present_values"],
            decoder_inputs,
        )
        first_logits = gather_last_valid_logits(logits, seq_lengths)
        compact_keys, compact_values = compact_prefill_kv(present_keys, present_values, seq_lengths)
        results[int(segment["segId"])] = decode_batch(
            decoder,
            embed_table,
            first_logits,
            compact_keys,
            compact_values,
            seq_lengths,
            max_tokens=max_tokens,
        )[0]

    return results


def run_batched_unified(
    encoder: ort.InferenceSession,
    decoder: ort.InferenceSession,
    audio: np.ndarray,
    segments: list[dict[str, object]],
    embed_table: np.ndarray,
    cfg: DecoderConfig,
    max_tokens: int,
) -> dict[int, list[int]]:
    mel_items: list[np.ndarray] = []
    mel_lengths: list[int] = []

    for segment in segments:
        segment_audio = extract_segment(audio, float(segment["start"]), float(segment["end"]))
        mel = log_mel_spectrogram(segment_audio).cpu().numpy().astype(np.float32)
        mel_items.append(mel[0])
        mel_lengths.append(int(mel.shape[2]))

    max_mel_len = max(mel_lengths)
    mel_batch = np.zeros((len(segments), 128, max_mel_len), dtype=np.float32)
    for batch_index, mel in enumerate(mel_items):
        mel_batch[batch_index, :, : mel.shape[1]] = mel

    audio_features, audio_lengths = encoder.run(
        ["audio_features", "audio_feature_lengths"],
        {
            "mel": mel_batch,
            "input_lengths": np.array(mel_lengths, dtype=np.int64),
        },
    )
    audio_features = audio_features.astype(np.float32, copy=False)
    audio_lengths = audio_lengths.astype(np.int64, copy=False)

    decoder_inputs, seq_lengths = build_prefill_inputs(audio_features, audio_lengths, embed_table, cfg)
    logits, present_keys, present_values = decoder.run(
        ["logits", "present_keys", "present_values"],
        decoder_inputs,
    )
    first_logits = gather_last_valid_logits(logits, seq_lengths)
    compact_keys, compact_values = compact_prefill_kv(present_keys, present_values, seq_lengths)
    sequences = decode_batch(
        decoder,
        embed_table,
        first_logits,
        compact_keys,
        compact_values,
        seq_lengths,
        max_tokens=max_tokens,
    )

    return {
        int(segment["segId"]): sequences[batch_index]
        for batch_index, segment in enumerate(segments)
    }


def decode_text(tokenizer: AutoTokenizer, tokens: list[int]) -> str:
    filtered = [token for token in tokens if token not in EOS_TOKEN_IDS]
    return tokenizer.decode(filtered, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def main() -> int:
    args = parse_args()
    providers = get_providers(args.execution_provider)
    cfg = load_decoder_config(args.onnx_dir)
    embed_table = load_embed_tokens(args.onnx_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.onnx_dir, trust_remote_code=False, fix_mistral_regex=True)

    segments = json.loads(Path(args.segments_json).read_text())
    if not segments:
        raise ValueError("segments-json must contain at least one segment")

    serial_encoder = make_session(str(Path(args.onnx_dir) / "encoder.onnx"), providers, enable_profiling=False)
    batched_encoder = make_session(str(Path(args.onnx_dir) / "encoder_batched.onnx"), providers, enable_profiling=False)
    decoder = make_session(str(Path(args.onnx_dir) / "decoder.onnx"), providers, enable_profiling=False)

    audio, _ = load_audio(args.audio)
    serial = run_serial_unified(serial_encoder, decoder, audio, segments, embed_table, cfg, args.max_tokens)
    batched = run_batched_unified(batched_encoder, decoder, audio, segments, embed_table, cfg, args.max_tokens)

    mismatch_count = 0
    print(f"Providers: {providers}")
    for segment in segments:
        seg_id = int(segment["segId"])
        serial_tokens = serial[seg_id]
        batched_tokens = batched[seg_id]
        serial_text = decode_text(tokenizer, serial_tokens)
        batched_text = decode_text(tokenizer, batched_tokens)
        same = serial_tokens == batched_tokens
        if not same:
            mismatch_count += 1

        row = {
            "segId": seg_id,
            "start": float(segment["start"]),
            "end": float(segment["end"]),
            "match": same,
            "serial_text": serial_text,
            "batched_text": batched_text,
        }
        if args.show_tokens or not same:
            row["serial_tokens"] = serial_tokens
            row["batched_tokens"] = batched_tokens
        print(json.dumps(row, ensure_ascii=True))

    if mismatch_count:
        print(f"Parity failed: {mismatch_count} segment(s) diverged.")
        return 1

    print("Parity passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
