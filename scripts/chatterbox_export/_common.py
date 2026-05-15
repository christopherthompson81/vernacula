#!/usr/bin/env python3
"""Shared helpers for the Chatterbox ONNX export and parity scripts.

Mirrors the conventions in `scripts/vibevoice_export/_common.py`. Keep
helpers narrow — anything Chatterbox-specific (wrapper nn.Modules,
monkeypatches) stays in the export script itself or a dedicated
`_chatterbox_internals.py` once we decide how much of the Vlad-script
model code we need to vendor.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import ml_dtypes
except ImportError:  # pragma: no cover - optional until BF16 path is used
    ml_dtypes = None


# Chatterbox-wide constants extracted from the upstream Python package /
# Vlad's reference script. Keep these in sync with `chatterbox-tts`.
S3GEN_SR = 24000
S3_SR = 16_000

# Speech-token vocabulary.
#   - SPEECH_BASE_VOCAB_SIZE is the count of "real" speech tokens (ids
#     0..SPEECH_BASE_VOCAB_SIZE-1) emitted by the S3 tokenizer.
#   - The three special ids that follow it happen to share numeric
#     adjacency, but the names refer to different concepts: don't use
#     `SPEECH_BASE_VOCAB_SIZE` as a token id.
SPEECH_BASE_VOCAB_SIZE = 6561
START_SPEECH_TOKEN = 6561
STOP_SPEECH_TOKEN = 6562
EXAGGERATION_TOKEN = 6563

# Llama backbone dims (verified at runtime against
# chatterbox.t3.tfmr.config — 30 layers, 16 KV heads, 64 head_dim).
LLM_HIDDEN_SIZE = 1024
LLM_NUM_LAYERS = 30
LLM_NUM_KV_HEADS = 16
LLM_NUM_ATTN_HEADS = 16
LLM_HEAD_DIM = LLM_HIDDEN_SIZE // LLM_NUM_ATTN_HEADS  # 64

# Output projection dims (verified against chatterbox.t3.speech_head and
# .text_head Linear layers). The LM emits 8194 speech-vocab logits per
# step; only the lower SPEECH_BASE_VOCAB_SIZE + 3 are "named". The
# remaining 1630 are reserved/unused — likely a power-of-2-adjacent pad.
SPEECH_HEAD_OUTPUT_DIM = 8194
TEXT_HEAD_OUTPUT_DIM = 704


def fail(message: str):
    raise SystemExit(message)


def ensure_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        fail("CUDA was requested but torch.cuda.is_available() is False.")
    return requested


def resolve_dtype(torch: Any, name: str) -> Any:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def torch_dtype_name(dtype: Any) -> str:
    for name in ("bfloat16", "float32", "float16"):
        if str(dtype).endswith(name):
            return name
    return str(dtype)


def ensure_output_dir(path: Path, overwrite: bool, export_files: list[str]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if overwrite:
        return
    collisions = [name for name in export_files if (path / name).exists()]
    if collisions:
        fail(
            "Output directory already contains Chatterbox export targets. "
            "Re-run with --overwrite to replace them.\n"
            f"Existing files: {', '.join(collisions)}"
        )


def json_dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_export_report(model_dir: Path) -> dict[str, Any]:
    report_path = model_dir / "export-report.json"
    if not report_path.exists():
        fail(f"Missing export-report.json in {model_dir}")
    return load_json(report_path)


def kv_input_names(num_layers: int) -> list[str]:
    """LM input names for `past_key_values`, matching the HF-standard
    transformers exporter shape used by the published `onnx-community`
    `language_model.onnx`. The HF schema is
    `past_key_values.{layer}.{key|value}`, so we emit those literal names.
    """
    names: list[str] = []
    for idx in range(num_layers):
        names.append(f"past_key_values.{idx}.key")
        names.append(f"past_key_values.{idx}.value")
    return names


def kv_output_names(num_layers: int) -> list[str]:
    names: list[str] = []
    for idx in range(num_layers):
        names.append(f"present.{idx}.key")
        names.append(f"present.{idx}.value")
    return names


def load_audio_mono_24k(audio_path: Path) -> tuple[np.ndarray, int]:
    import librosa
    import soundfile as sf

    waveform, sr = sf.read(str(audio_path), always_2d=False)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    waveform = waveform.astype(np.float32, copy=False)
    if sr != S3GEN_SR:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=S3GEN_SR).astype(np.float32, copy=False)
        sr = S3GEN_SR
    return waveform, sr


def choose_onnx_providers(runtime: str) -> list[str]:
    runtime = runtime.lower()
    if runtime == "cpu":
        return ["CPUExecutionProvider"]
    if runtime == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if runtime == "tensorrt":
        return ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    fail(f"Unsupported runtime '{runtime}'.")


def read_ort_available_providers() -> list[str]:
    import onnxruntime as ort
    return list(ort.get_available_providers())


def nvidia_smi_query() -> dict[str, str] | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    except Exception:
        return None

    if not out:
        return None

    first = out.splitlines()[0]
    name, driver_version, memory_total, memory_used = [part.strip() for part in first.split(",")]
    return {
        "name": name,
        "driver_version": driver_version,
        "memory_total_mib": memory_total,
        "memory_used_mib": memory_used,
    }


def save_export_report(
    path: Path,
    *,
    repo_id: str,
    revision: str | None,
    device: str,
    dtype: str,
    opset_embed_tokens: int,
    opset_speech_encoder: int,
    opset_language_model: int,
    opset_conditional_decoder: int,
    lm_graph_mode: str,
    safe_dense_layer_patched: bool,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "repo_id": repo_id,
        "revision": revision,
        "device": device,
        "dtype": dtype,
        "opsets": {
            "embed_tokens": opset_embed_tokens,
            "speech_encoder": opset_speech_encoder,
            "language_model": opset_language_model,
            "conditional_decoder": opset_conditional_decoder,
        },
        "lm_graph_mode": lm_graph_mode,
        "safe_dense_layer_patched": safe_dense_layer_patched,
        "constants": {
            "s3gen_sr": S3GEN_SR,
            "start_speech_token": START_SPEECH_TOKEN,
            "stop_speech_token": STOP_SPEECH_TOKEN,
            "llm_num_layers": LLM_NUM_LAYERS,
            "llm_num_kv_heads": LLM_NUM_KV_HEADS,
            "llm_head_dim": LLM_HEAD_DIM,
        },
    }
    if extra:
        payload.update(extra)
    json_dump(path, payload)


def add_local_script_path() -> None:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
