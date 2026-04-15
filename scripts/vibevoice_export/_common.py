#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import ml_dtypes
except ImportError:  # pragma: no cover - optional dependency until BF16 ORT path is used
    ml_dtypes = None


def fail(message: str) -> "NoReturn":
    raise SystemExit(message)


def ensure_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


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
            "Output directory already contains VibeVoice export targets. "
            "Re-run with --overwrite to replace them.\n"
            f"Existing files: {', '.join(collisions)}"
        )


def json_dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def load_model_and_processor(
    repo_id: str,
    revision: str | None,
    device: str,
    dtype: Any,
    torch: Any,
) -> tuple[Any, Any]:
    from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration

    revision_suffix = f" @ {revision}" if revision else ""
    print(f"Loading processor from {repo_id}{revision_suffix} ...")
    processor = AutoProcessor.from_pretrained(repo_id, revision=revision, trust_remote_code=True)

    print(f"Loading model from {repo_id}{revision_suffix} onto {device} as {torch_dtype_name(dtype)} ...")
    model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
        repo_id,
        revision=revision,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Loaded {param_count:,} parameters ({param_count / 1e9:.2f}B)")
    return model, processor


def estimate_decoder_memory_gib(config: dict[str, Any]) -> dict[str, float]:
    text = config["text_config"]
    layers = int(text["num_hidden_layers"])
    hidden = int(text["hidden_size"])
    heads = int(text["num_attention_heads"])
    kv_heads = int(text["num_key_value_heads"])
    head_dim = hidden // heads
    intermediate = int(text["intermediate_size"])
    vocab = int(text["vocab_size"])

    embed = vocab * hidden
    attn_per_layer = hidden * hidden + hidden * (kv_heads * head_dim) * 2 + hidden * hidden
    mlp_per_layer = hidden * intermediate * 3
    norm_per_layer = hidden * 2
    total_params = embed + layers * (attn_per_layer + mlp_per_layer + norm_per_layer) + embed

    def gib(param_bytes: float) -> float:
        return float(param_bytes / (1024 ** 3))

    per_token_bf16 = 2 * layers * kv_heads * head_dim * 2

    return {
        "decoder_weight_gib_fp16_estimate": gib(total_params * 2),
        "decoder_weight_gib_int8_estimate": gib(total_params),
        "kv_cache_gib_16k_fp16_estimate": gib(per_token_bf16 * 16384),
        "kv_cache_gib_32k_fp16_estimate": gib(per_token_bf16 * 32768),
        "kv_cache_gib_64k_fp16_estimate": gib(per_token_bf16 * 65536),
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_export_report(model_dir: Path) -> dict[str, Any]:
    report_path = model_dir / "export-report.json"
    if not report_path.exists():
        fail(f"Missing export-report.json in {model_dir}")
    return load_json(report_path)


def onnx_elem_type_to_numpy(elem_type: Any) -> Any:
    if isinstance(elem_type, str):
        text = elem_type.lower()
        if "bfloat16" in text:
            if ml_dtypes is None:
                fail("bfloat16 ONNX feeds require `ml_dtypes`. Install the export requirements and try again.")
            return ml_dtypes.bfloat16
        if "float16" in text:
            return np.float16
        if "float" in text:
            return np.float32
        if "int64" in text:
            return np.int64
        if "bool" in text:
            return np.bool_
        fail(f"Unsupported ONNX Runtime type string '{elem_type}' in current tooling.")

    mapping = {
        1: np.float32,
        7: np.int64,
        9: np.bool_,
        10: np.float16,
    }
    if elem_type == 16:
        if ml_dtypes is None:
            fail("bfloat16 ONNX feeds require `ml_dtypes`. Install the export requirements and try again.")
        return ml_dtypes.bfloat16
    if elem_type not in mapping:
        fail(f"Unsupported ONNX elem_type {elem_type} in current tooling.")
    return mapping[elem_type]


def onnx_elem_type_name(elem_type: Any) -> str:
    if isinstance(elem_type, str):
        return elem_type.lower()

    mapping = {
        1: "float32",
        7: "int64",
        9: "bool",
        10: "float16",
        16: "bfloat16",
    }
    return mapping.get(elem_type, f"elem_type_{elem_type}")


def save_export_report(
    path: Path,
    *,
    repo_id: str,
    revision: str | None,
    device: str,
    dtype: str,
    opset: int,
    acoustic_tokenizer_chunk_size: int,
    deterministic_audio: bool,
    model_config: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "repo_id": repo_id,
        "revision": revision,
        "device": device,
        "dtype": dtype,
        "opset": opset,
        "deterministic_audio": deterministic_audio,
        "acoustic_tokenizer_chunk_size": acoustic_tokenizer_chunk_size,
        "memory_estimates": estimate_decoder_memory_gib(model_config),
        "text_config": model_config["text_config"],
        "audio_config": {
            "acoustic_tokenizer_encoder_config": model_config["acoustic_tokenizer_encoder_config"],
            "semantic_tokenizer_encoder_config": model_config["semantic_tokenizer_encoder_config"],
            "audio_token_id": model_config["audio_token_id"],
            "audio_bos_token_id": model_config["audio_bos_token_id"],
            "audio_eos_token_id": model_config["audio_eos_token_id"],
            "acoustic_tokenizer_chunk_size": model_config["acoustic_tokenizer_chunk_size"],
        },
    }
    if extra:
        payload.update(extra)
    json_dump(path, payload)


def flatten_past_key_values(past_key_values: Any) -> list[Any]:
    flat: list[Any] = []
    for layer in past_key_values:
        if len(layer) < 2:
            fail("Unexpected past_key_values layer shape; expected key/value pair.")
        flat.append(layer[0])
        flat.append(layer[1])
    return flat


def unflatten_past_key_values(values: list[Any]) -> tuple[tuple[Any, Any], ...]:
    if len(values) % 2 != 0:
        fail(f"Expected an even number of KV tensors, got {len(values)}.")
    return tuple((values[i], values[i + 1]) for i in range(0, len(values), 2))


def kv_input_names(num_layers: int) -> list[str]:
    names: list[str] = []
    for idx in range(num_layers):
        names.append(f"past_key_{idx}")
        names.append(f"past_value_{idx}")
    return names


def kv_output_names(num_layers: int) -> list[str]:
    names: list[str] = []
    for idx in range(num_layers):
        names.append(f"present_key_{idx}")
        names.append(f"present_value_{idx}")
    return names


def load_audio_mono_24k(audio_path: Path, target_sr: int = 24000) -> tuple[np.ndarray, int]:
    import librosa
    import soundfile as sf

    waveform, sr = sf.read(str(audio_path), always_2d=False)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    waveform = waveform.astype(np.float32, copy=False)
    if sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr).astype(np.float32, copy=False)
        sr = target_sr
    return waveform, sr


def truncate_audio_seconds(waveform: np.ndarray, sr: int, seconds: float | None) -> np.ndarray:
    if seconds is None:
        return waveform
    samples = max(1, int(round(seconds * sr)))
    return waveform[:samples]


def derive_legacy_audio_export_samples(
    *,
    export_report: dict[str, Any],
    sample_rate: int,
) -> int | None:
    exact = export_report.get("audio_export_input_samples")
    if exact:
        return int(exact)

    chunk_size = export_report.get("acoustic_tokenizer_chunk_size")
    dummy_audio_seconds = export_report.get("dummy_audio_seconds")
    if not chunk_size or not dummy_audio_seconds:
        return None

    chunk_size = int(chunk_size)
    requested = int(round(float(dummy_audio_seconds) * sample_rate))
    if requested <= 0:
        return None
    return ((requested + chunk_size - 1) // chunk_size) * chunk_size


def pad_or_trim_waveform(waveform: np.ndarray, target_samples: int) -> np.ndarray:
    if waveform.shape[0] == target_samples:
        return waveform.astype(np.float32, copy=False)
    if waveform.shape[0] > target_samples:
        return waveform[:target_samples].astype(np.float32, copy=False)
    padded = np.zeros((target_samples,), dtype=np.float32)
    padded[: waveform.shape[0]] = waveform.astype(np.float32, copy=False)
    return padded


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


@dataclass
class TimerResult:
    wall_seconds: float
    iterations: int


class Timer:
    def __init__(self) -> None:
        self._start = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._start


def add_local_script_path() -> None:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
