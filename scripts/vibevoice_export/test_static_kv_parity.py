#!/usr/bin/env python3
"""
Quick parity check: run decoder_single_static.onnx and compare tokens against
decoder_single.onnx (the reference f32kv dynamic model).

Usage:
  python test_static_kv_parity.py \
      --static-dir  models/vibevoice_asr_static_bf16_f32kv \
      --dynamic-dir models/vibevoice_asr_single_bf16_f32kv \
      --audio       data/test_audio/en-US/en-US_sample_01.wav \
      --max-tokens  256
"""

from __future__ import annotations
import argparse
import ctypes
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


# ── helpers ──────────────────────────────────────────────────────────────────

def read_export_report(model_dir: Path) -> dict:
    return json.loads((model_dir / "export-report.json").read_text())


def make_session(onnx_path: Path, use_cuda: bool = True) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
    return ort.InferenceSession(str(onnx_path), opts, providers=providers)


def _ort_value_to_f32(val: ort.OrtValue) -> np.ndarray:
    """Convert an OrtValue (possibly bfloat16) to a float32 numpy array."""
    try:
        arr = val.numpy()
        return arr.astype(np.float32) if arr.dtype != np.float32 else arr
    except RuntimeError:
        # bfloat16 OrtValue: read raw bytes, reinterpret as uint16 → bfloat16 → float32
        import ml_dtypes
        raw = ctypes.string_at(val.data_ptr(), val.tensor_size_in_bytes())
        arr_u16 = np.frombuffer(raw, dtype=np.uint16).reshape(val.shape())
        return arr_u16.view(ml_dtypes.bfloat16).astype(np.float32)


def _decoder_fp_inputs(session: ort.InferenceSession) -> dict[str, np.dtype]:
    """Return {name: dtype} for non-float32 floating point inputs (bfloat16, float16)."""
    result = {}
    for inp in session.get_inputs():
        t = str(inp.type).lower()
        if "bfloat16" in t:
            import ml_dtypes
            result[inp.name] = ml_dtypes.bfloat16
        elif "float16" in t:
            result[inp.name] = np.float16
    return result


def _np_to_ort(name: str, arr: np.ndarray, fp_casts: dict[str, np.dtype]) -> ort.OrtValue:
    """Wrap a numpy array as an OrtValue; cast float32 → bfloat16/float16 when needed."""
    if name in fp_casts:
        target = fp_casts[name]
        import ml_dtypes
        if target == ml_dtypes.bfloat16:
            u16 = arr.astype(ml_dtypes.bfloat16).view(np.uint16)
            return ort.OrtValue.ortvalue_from_numpy_with_onnx_type(u16, 16)
        else:
            return ort.OrtValue.ortvalue_from_numpy(arr.astype(target))
    return ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(arr))


def run_decoder_session(session: ort.InferenceSession, feed: dict[str, np.ndarray],
                        fp_casts: dict[str, np.dtype], output_names: list[str]) -> list[np.ndarray]:
    """Run decoder session handling bfloat16/float16 I/O; returns all outputs as float32."""
    ort_feed = {name: _np_to_ort(name, arr, fp_casts) for name, arr in feed.items()}
    out_vals = session.run_with_ort_values(output_names, ort_feed)
    return [_ort_value_to_f32(v) for v in out_vals]


def run_audio_encoder(
    encoder: ort.InferenceSession,
    audio: np.ndarray,   # float32 [T]
    chunk_samples: int,
) -> np.ndarray:
    """Returns audio embeddings float32 [N, hidden]."""
    input_meta = {inp.name: inp for inp in encoder.get_inputs()}
    bf16_input = "bfloat16" in str(input_meta.get("input_values", object()).type).lower()

    all_embs = []
    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start: start + chunk_samples]
        # pad to chunk_samples
        padded = np.zeros(chunk_samples, dtype=np.float32)
        padded[:len(chunk)] = chunk
        mask = np.zeros(chunk_samples, dtype=bool)
        mask[:len(chunk)] = True

        if bf16_input:
            import ml_dtypes
            # Cast float32 → bfloat16, view as uint16 for ORT ONNX type 16
            padded_u16 = padded.astype(ml_dtypes.bfloat16).view(np.uint16)
            feed = {
                "input_values": ort.OrtValue.ortvalue_from_numpy_with_onnx_type(padded_u16[None], 16),
                "padding_mask": ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(mask[None])),
            }
            [emb_val] = encoder.run_with_ort_values(["audio_embeddings"], feed)
            emb = _ort_value_to_f32(emb_val)
        else:
            emb = encoder.run(None, {"input_values": padded[None], "padding_mask": mask[None]})[0]
        all_embs.append(emb)
    return np.concatenate(all_embs, axis=0)


# ── dynamic KV runner ─────────────────────────────────────────────────────────

def run_dynamic(
    decoder: ort.InferenceSession,
    audio_emb: np.ndarray,
    prefix_ids: np.ndarray,
    suffix_ids: np.ndarray,
    num_layers: int,
    kv_dtype: np.dtype,
    kv_heads: int,
    head_dim: int,
    max_tokens: int,
    eos_ids: set[int],
) -> list[int]:
    """Run decoder_single.onnx with growing KV cache."""
    bf16_ins = _decoder_fp_inputs(decoder)
    out_names = (["logits"]
                 + [f"present_key_{i}" for i in range(num_layers)]
                 + [f"present_value_{i}" for i in range(num_layers)])

    # Initial empty KV
    kv_shape = (1, kv_heads, 0, head_dim)
    past_kvs = [np.zeros(kv_shape, dtype=kv_dtype) for _ in range(num_layers * 2)]

    def run_once(pfx, audio_start, audio_count, sfx):
        nonlocal past_kvs
        feed = {
            "prefix_input_ids": pfx.astype(np.int64)[None],
            "audio_embeddings": (
                audio_emb[audio_start: audio_start + audio_count]
                if audio_count > 0
                else np.zeros((0, audio_emb.shape[1]), dtype=audio_emb.dtype)
            ),
            "suffix_input_ids": sfx.astype(np.int64)[None],
        }
        for i, kv in enumerate(past_kvs):
            name = f"past_key_{i//2}" if i % 2 == 0 else f"past_value_{i//2}"
            feed[name] = kv

        results = run_decoder_session(decoder, feed, bf16_ins, out_names)
        logits = results[0]      # float32 [1, seq, vocab]
        new_kvs = results[1:]

        # reorder: ORT returns present_key_0..27, present_value_0..27
        # match past_kvs order: k0,v0,k1,v1,...
        reordered = []
        for i in range(num_layers):
            reordered.append(new_kvs[i])
            reordered.append(new_kvs[i + num_layers])
        past_kvs = reordered

        seq_len = pfx.shape[0] + audio_count + sfx.shape[0]
        return int(np.argmax(logits[0, seq_len - 1, :]))

    # prefill
    n = len(audio_emb)
    chunk = 512
    num_chunks = (n + chunk - 1) // max(1, chunk)
    last_token = 0
    for ci in range(num_chunks):
        start = ci * chunk
        count = min(chunk, n - start)
        pfx = prefix_ids if ci == 0 else np.array([], dtype=np.int64)
        sfx = suffix_ids if ci == num_chunks - 1 else np.array([], dtype=np.int64)
        last_token = run_once(pfx, start, count, sfx)

    # decode
    tokens = []
    tok = last_token
    for _ in range(max_tokens):
        if tok in eos_ids:
            break
        tokens.append(tok)
        tok = run_once(np.array([tok], dtype=np.int64), 0, 0, np.array([], dtype=np.int64))

    return tokens


# ── static KV runner ──────────────────────────────────────────────────────────

def run_static(
    decoder: ort.InferenceSession,
    audio_emb: np.ndarray,
    prefix_ids: np.ndarray,
    suffix_ids: np.ndarray,
    num_layers: int,
    kv_dtype: np.dtype,
    kv_heads: int,
    head_dim: int,
    max_kv_tokens: int,
    max_tokens: int,
    eos_ids: set[int],
) -> list[int]:
    """Run decoder_single_static.onnx with pre-allocated static KV buffers."""
    bf16_ins = _decoder_fp_inputs(decoder)
    out_names = (["logits"]
                 + [f"present_key_{i}" for i in range(num_layers)]
                 + [f"present_value_{i}" for i in range(num_layers)])

    kv_shape = (1, kv_heads, max_kv_tokens, head_dim)
    past_kvs = [np.zeros(kv_shape, dtype=kv_dtype) for _ in range(num_layers * 2)]
    kv_pos = 0

    def run_once(pfx, audio_start, audio_count, sfx):
        nonlocal past_kvs, kv_pos
        feed = {
            "prefix_input_ids": pfx.astype(np.int64)[None],
            "audio_embeddings": (
                audio_emb[audio_start: audio_start + audio_count]
                if audio_count > 0
                else np.zeros((0, audio_emb.shape[1]), dtype=audio_emb.dtype)
            ),
            "suffix_input_ids": sfx.astype(np.int64)[None],
            "kv_pos": np.array(kv_pos, dtype=np.int64),
        }
        for i, kv in enumerate(past_kvs):
            name = f"past_key_{i//2}" if i % 2 == 0 else f"past_value_{i//2}"
            feed[name] = kv

        results = run_decoder_session(decoder, feed, bf16_ins, out_names)
        logits = results[0]  # float32 [1, seq, vocab]
        new_kvs = results[1:]

        # reorder present_key_0..27, present_value_0..27 → k0,v0,k1,v1,...
        reordered = []
        for i in range(num_layers):
            reordered.append(new_kvs[i])
            reordered.append(new_kvs[i + num_layers])
        past_kvs = reordered

        seq_len = pfx.shape[0] + audio_count + sfx.shape[0]
        token = int(np.argmax(logits[0, seq_len - 1, :]))
        kv_pos += seq_len
        return token

    # prefill
    n = len(audio_emb)
    chunk = 512
    num_chunks = (n + chunk - 1) // max(1, chunk)
    last_token = 0
    for ci in range(num_chunks):
        start = ci * chunk
        count = min(chunk, n - start)
        pfx = prefix_ids if ci == 0 else np.array([], dtype=np.int64)
        sfx = suffix_ids if ci == num_chunks - 1 else np.array([], dtype=np.int64)
        last_token = run_once(pfx, start, count, sfx)

    # decode
    tokens = []
    tok = last_token
    for _ in range(max_tokens):
        if tok in eos_ids:
            break
        tokens.append(tok)
        tok = run_once(np.array([tok], dtype=np.int64), 0, 0, np.array([], dtype=np.int64))

    return tokens


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--static-dir",  type=Path, required=True)
    ap.add_argument("--dynamic-dir", type=Path, required=True)
    ap.add_argument("--audio",       type=Path, required=True)
    ap.add_argument("--max-tokens",  type=int, default=256)
    ap.add_argument("--no-cuda",     action="store_true")
    args = ap.parse_args()

    use_cuda = not args.no_cuda

    # Load reports
    static_report  = read_export_report(args.static_dir)
    dynamic_report = read_export_report(args.dynamic_dir)

    num_layers   = static_report["num_layers"]
    kv_heads     = static_report["num_kv_heads"]
    head_dim     = static_report["head_dim"]
    max_kv_tokens = static_report["static_kv_max_tokens"]
    enc_chunk    = static_report["acoustic_tokenizer_chunk_size"]

    kv_dtype_s = np.float32 if static_report.get("f32_kv_cache") else np.float16
    kv_dtype_d = np.float32 if dynamic_report.get("f32_kv_cache") else np.float16

    prefix_ids = np.array(static_report["tokenizer"]["prefix_token_ids"], dtype=np.int64)
    # Build a short-audio suffix (just use the first sample's duration)
    audio_data, sr = sf.read(str(args.audio), dtype="float32")
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    # Resample to 24 kHz if needed
    if sr != 24000:
        import resampy
        audio_data = resampy.resample(audio_data, sr, 24000)
        sr = 24000
    dur = len(audio_data) / sr

    # Build suffix (before + duration digits + after)
    dur_str = f"{dur:.2f}"
    digit_map = {k: v for k, v in static_report["tokenizer"]["digit_char_to_token_id"].items()}
    sfx_before = np.array(static_report["tokenizer"]["suffix_before_duration_token_ids"], dtype=np.int64)
    sfx_after  = np.array(static_report["tokenizer"]["suffix_after_duration_token_ids"],  dtype=np.int64)
    dur_tokens = np.array([digit_map[c] for c in dur_str], dtype=np.int64)
    suffix_ids = np.concatenate([sfx_before, dur_tokens, sfx_after])

    eos_ids = {
        static_report["tokenizer"]["eos_token_id"],
        static_report["tokenizer"]["im_end_token_id"],
    }

    # Pad audio to stride multiple
    stride = 3200
    pad = (stride - len(audio_data) % stride) % stride
    audio_padded = np.concatenate([audio_data, np.zeros(pad, dtype=np.float32)])

    print(f"Audio: {dur:.1f}s  |  max_tokens={args.max_tokens}")

    # Load audio encoder (shared)
    print("Loading audio encoder ...")
    encoder = make_session(args.static_dir / "audio_encoder.onnx", use_cuda)
    audio_emb = run_audio_encoder(encoder, audio_padded, enc_chunk)
    print(f"Audio embeddings: {audio_emb.shape}")
    del encoder

    # Run dynamic (reference)
    print("\n=== Running DYNAMIC decoder (reference) ===")
    dec_dyn = make_session(args.dynamic_dir / "decoder_single.onnx", use_cuda)
    tokens_dyn = run_dynamic(
        dec_dyn, audio_emb.astype(np.float32),
        prefix_ids, suffix_ids, num_layers,
        kv_dtype_d, kv_heads, head_dim,
        args.max_tokens, eos_ids,
    )
    del dec_dyn
    print(f"  → {len(tokens_dyn)} tokens generated")

    # Run static
    print("\n=== Running STATIC decoder ===")
    dec_sta = make_session(args.static_dir / "decoder_single_static.onnx", use_cuda)
    tokens_sta = run_static(
        dec_sta, audio_emb.astype(np.float32),
        prefix_ids, suffix_ids, num_layers,
        kv_dtype_s, kv_heads, head_dim,
        max_kv_tokens, args.max_tokens, eos_ids,
    )
    del dec_sta
    print(f"  → {len(tokens_sta)} tokens generated")

    # Compare
    n = min(len(tokens_dyn), len(tokens_sta), args.max_tokens)
    first_div = None
    for i in range(n):
        if tokens_dyn[i] != tokens_sta[i]:
            first_div = i
            break

    print(f"\n=== Parity: dynamic vs static (first {n} tokens) ===")
    if first_div is None:
        print(f"  ✅  IDENTICAL for all {n} tokens compared")
    else:
        print(f"  ❌  First divergence at position {first_div}")
        print(f"       dynamic token: {tokens_dyn[first_div]}")
        print(f"       static  token: {tokens_sta[first_div]}")
        matches = sum(1 for a, b in zip(tokens_dyn[:n], tokens_sta[:n]) if a == b)
        print(f"       Agreement: {matches}/{n} = {100*matches/n:.1f}%")


if __name__ == "__main__":
    main()
