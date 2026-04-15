#!/usr/bin/env python3
"""
Export microsoft/VibeVoice-ASR-HF into ONNX graphs:

  audio_encoder.onnx
      input_values [1, num_samples] float
      padding_mask [1, num_samples] bool
      -> audio_embeddings [num_audio_tokens, hidden]

  decoder_prefill.onnx
      input_ids [1, prompt_len] int64
      attention_mask [1, prompt_len] int64
      audio_embeddings [num_audio_tokens, hidden] float
      -> logits [1, prompt_len, vocab]
      -> present_key_*/present_value_*

  decoder_step.onnx
      input_ids [1, 1] int64
      attention_mask [1, total_len] int64
      past_key_*/past_value_*
      -> logits [1, 1, vocab]
      -> present_key_*/present_value_*

  decoder_single.onnx
      prefix_input_ids [1, prefix_len] int64
      audio_embeddings [num_audio_tokens, hidden] float
      suffix_input_ids [1, suffix_len] int64
      past_key_0 .. past_key_27   [1, kv_heads, cache_len, head_dim]
      past_value_0 .. past_value_27 [1, kv_heads, cache_len, head_dim]
      -> logits [1, seq_len, vocab]
      -> present_key_0..27, present_value_0..27 [1, kv_heads, total_len, head_dim]

The initial export target is batch size 1 with dynamic prompt, audio-token, and cache lengths.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import sys
from pathlib import Path
from typing import Any

import onnx
from onnx.external_data_helper import load_external_data_for_model
import numpy as np
import torch
from transformers.cache_utils import DynamicCache

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _common import (
    ensure_output_dir,
    flatten_past_key_values,
    kv_input_names,
    kv_output_names,
    load_model_and_processor,
    resolve_device,
    resolve_dtype,
    save_export_report,
    torch_dtype_name,
)


EXPORT_FILES = [
    "audio_encoder.onnx",
    "decoder_prefill.onnx",
    "decoder_step.onnx",
    "decoder_single.onnx",
    "decoder_single_static.onnx",
    "config.json",
    "generation_config.json",
    "processor_config.json",
    "export-report.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export VibeVoice-ASR-HF to ONNX graphs.")
    parser.add_argument("--model-repo", default="microsoft/VibeVoice-ASR-HF")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float16")
    parser.add_argument(
        "--decoder-exporter",
        choices=("auto", "legacy", "torch-export"),
        default="auto",
        help="Decoder export backend. 'auto' keeps the audio export on the legacy path and prefers torch.export for decoder graphs.",
    )
    parser.add_argument(
        "--audio-exporter",
        choices=("auto", "legacy", "torch-export"),
        default="auto",
        help="Audio export backend. 'auto' keeps the current legacy path unless a specific backend is requested.",
    )
    parser.add_argument(
        "--torch-export-debug-artifacts",
        action="store_true",
        help="When using --decoder-exporter torch-export, emit the exporter report and exported-program artifacts.",
    )
    parser.add_argument("--dummy-audio-seconds", type=float, default=15.0)
    parser.add_argument("--dummy-prompt", default="Transcribe the meeting audio with speaker labels when available.")
    parser.add_argument("--acoustic-tokenizer-chunk-size", type=int, default=None)
    parser.add_argument(
        "--deterministic-audio",
        action="store_true",
        help="Disable stochastic acoustic-latent noise in the exported audio path.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-audio-encoder", action="store_true")
    parser.add_argument("--skip-prefill", action="store_true")
    parser.add_argument("--skip-step", action="store_true")
    parser.add_argument(
        "--decoder-graph-mode",
        choices=("split", "single", "both", "static-single"),
        default="single",
        help=(
            "Which decoder graph set to export: "
            "'split' exports decoder_prefill.onnx + decoder_step.onnx, "
            "'single' exports only decoder_single.onnx (dynamic KV cache, default), "
            "'both' exports all decoder graphs, "
            "'static-single' exports only decoder_single_static.onnx (pre-allocated KV buffers, "
            "eliminates O(n) Concat per step via ScatterElements)."
        ),
    )
    parser.add_argument(
        "--export-single-decoder",
        action="store_true",
        help=(
            "Deprecated alias for --decoder-graph-mode single. "
            "Exports only decoder_single.onnx, a unified decoder graph that handles prefill and step via one packed KV-cache input."
        ),
    )
    parser.add_argument(
        "--static-kv-max-tokens",
        type=int,
        default=6144,
        help=(
            "Pre-allocated KV buffer length for --decoder-graph-mode static-single. "
            "Must be >= the maximum number of tokens (prefill + decode) that will be generated. "
            "VRAM cost: 2 × num_layers × num_kv_heads × max_tokens × head_dim × dtype_bytes "
            "(e.g. 56 × [1,4,6144,128] float32 ≈ 706 MB). Default: 6144 (~10-min audio at ~10 tok/s)."
        ),
    )
    parser.add_argument(
        "--f32-kv-cache",
        action="store_true",
        help=(
            "Export decoder_single.onnx with float32 KV cache and float32 attention dot products. "
            "Patches Qwen2Attention during export so Q, K, and V are upcast to float32 before the "
            "Q*K^T and V*attn matmuls, and K/V are stored as float32 in the cache. "
            "Adds explicit Cast nodes to the ONNX graph so ORT executes float32 attention at inference. "
            "VRAM impact: ~2x KV cache size (typically +50-300 MB). "
            "Addresses BF16 matmul non-determinism that causes token-level drift between ORT and PyTorch."
        ),
    )
    parser.add_argument(
        "--f32-lm-head",
        action="store_true",
        help=(
            "Export decoder_single.onnx with a float32 lm_head projection. "
            "Patches lm_head during export so hidden states and weights are cast to float32 before "
            "the final vocab projection, giving float32-precision logits. "
            "Eliminates BF16 matmul accumulation differences in the lm_head that cause ORT and PyTorch "
            "to disagree at token positions where multiple logits round to the same BF16 value. "
            "VRAM impact: lm_head weight (~1.1 GB) is cast to float32 at inference time. "
            "Can be combined with --f32-kv-cache."
        ),
    )
    args = parser.parse_args()
    if args.export_single_decoder:
        args.decoder_graph_mode = "single"
    return args


def infer_export_chunk_size(
    *,
    requested_chunk_size: int | None,
    default_chunk_size: int,
    sampling_rate: int,
    dummy_audio_seconds: float,
    processor: Any,
) -> int:
    if requested_chunk_size is not None:
        return int(requested_chunk_size)

    pad_multiple = getattr(processor.feature_extractor, "pad_to_multiple_of", None) or 3200
    requested_samples = max(pad_multiple, int(round(dummy_audio_seconds * sampling_rate)))
    rounded = ((requested_samples + pad_multiple - 1) // pad_multiple) * pad_multiple

    # Use the smaller of:
    # - the model's full production chunk size
    # - the test export's requested dummy duration rounded to the tokenizer step
    return max(pad_multiple, min(default_chunk_size, rounded))


def round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def resolve_decoder_exporter(choice: str) -> str:
    if choice == "auto":
        return "torch-export"
    return choice


def resolve_audio_exporter(choice: str) -> str:
    if choice == "auto":
        return "legacy"
    return choice


def resolve_audio_export_dtype(dtype: Any, torch: Any, exporter: str) -> tuple[Any, str | None]:
    if dtype == torch.bfloat16 and exporter == "legacy":
        return (
            torch.bfloat16,
            "audio_encoder.onnx keeps bf16 inputs/outputs when --dtype bfloat16 is requested, but the Conv-heavy tokenizer towers are promoted to float32 during export because ONNX Runtime rejects bf16 Conv nodes in the current audio graph.",
        )
    return dtype, None


class AudioEncoderWrapper(torch.nn.Module):
    def __init__(self, model: Any, deterministic_audio: bool, encoder_compute_dtype: Any, projector_dtype: Any):
        super().__init__()
        self.model = model
        self.deterministic_audio = deterministic_audio
        self.encoder_compute_dtype = encoder_compute_dtype
        self.projector_dtype = projector_dtype

    def forward(self, input_values, padding_mask):
        # Export the tokenizer towers as one full-waveform pass. The chunked
        # streaming path traces `torch.split(...)` into a fixed ONNX `Split`
        # layout, which makes the current graph only look dynamic in metadata.
        # A single pass preserves a truly dynamic sample axis for offline export.
        audio = input_values.to(self.encoder_compute_dtype).unsqueeze(1)
        acoustic_latents = self.model.acoustic_tokenizer_encoder(
            audio,
            padding_cache=None,
            use_cache=False,
        ).latents
        semantic_latents = self.model.semantic_tokenizer_encoder(
            audio,
            padding_cache=None,
            use_cache=False,
        ).latents

        if not self.deterministic_audio:
            noise_std = self.model.config.acoustic_tokenizer_encoder_config.vae_std * torch.randn(
                acoustic_latents.shape[0], device=acoustic_latents.device, dtype=acoustic_latents.dtype
            )
            acoustic_latents = acoustic_latents + noise_std[:, None, None] * torch.randn_like(acoustic_latents)

        combined_features = self.model.multi_modal_projector(
            acoustic_latents.to(self.projector_dtype),
            semantic_latents.to(self.projector_dtype),
        )
        if padding_mask is not None:
            num_audio_tokens = torch.ceil(
                padding_mask.sum(dim=-1) / self.model.config.acoustic_tokenizer_encoder_config.hop_length
            ).to(torch.int64)
            token_mask = torch.arange(num_audio_tokens.max(), device=combined_features.device) < num_audio_tokens[
                :, None
            ].to(combined_features.device)
            combined_features = combined_features[token_mask]
        return combined_features


class DecoderPrefillWrapper(torch.nn.Module):
    def __init__(self, model: Any):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, audio_embeddings):
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        inputs_embeds = replace_audio_token_embeddings(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            audio_embeddings=audio_embeddings,
            audio_token_id=int(self.model.config.audio_token_id),
        )
        full_attention_mask = build_full_attention_mask(
            attention_mask=attention_mask,
            query_length=inputs_embeds.shape[1],
            kv_length=inputs_embeds.shape[1],
            past_length=0,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask={"full_attention": full_attention_mask},
            use_cache=True,
        )
        return (outputs.logits, *flatten_past_key_values(outputs.past_key_values))


class DecoderStepWrapper(torch.nn.Module):
    def __init__(self, model: Any):
        super().__init__()
        self.model = model
        self.num_layers = int(model.config.text_config.num_hidden_layers)

    def forward(self, input_ids, attention_mask, *past_key_values):
        past_length = past_key_values[0].shape[2] if past_key_values else 0
        full_attention_mask = build_full_attention_mask(
            attention_mask=attention_mask,
            query_length=input_ids.shape[1],
            kv_length=past_length + input_ids.shape[1],
            past_length=past_length,
            dtype=self.model.get_input_embeddings().weight.dtype,
            device=input_ids.device,
        )
        cache = DynamicCache(
            ddp_cache_data=tuple(
                (past_key_values[i], past_key_values[i + 1]) for i in range(0, len(past_key_values), 2)
            ),
            config=self.model.config,
        )
        outputs = self.model.language_model(
            input_ids=input_ids,
            attention_mask={"full_attention": full_attention_mask},
            past_key_values=cache,
            use_cache=True,
        )
        return (outputs.logits, *flatten_past_key_values(outputs.past_key_values))


class DecoderSingleWrapper(torch.nn.Module):
    def __init__(self, model: Any):
        super().__init__()
        self.model = model
        self.num_layers = int(model.config.text_config.num_hidden_layers)

    def forward(self, prefix_input_ids, audio_embeddings, suffix_input_ids, *past_key_values):
        prefix_embeds = self.model.get_input_embeddings()(prefix_input_ids)
        suffix_embeds = self.model.get_input_embeddings()(suffix_input_ids)
        inputs_embeds = torch.cat(
            (prefix_embeds, audio_embeddings.to(prefix_embeds.device).unsqueeze(0), suffix_embeds),
            dim=1,
        )

        past_length = past_key_values[0].shape[2] if past_key_values else 0
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0) + past_length
        full_attention_mask = build_full_attention_mask(
            attention_mask=None,
            query_length=inputs_embeds.shape[1],
            kv_length=past_length + inputs_embeds.shape[1],
            past_length=past_length,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )

        cache = DynamicCache(
            ddp_cache_data=tuple(
                (past_key_values[i], past_key_values[i + 1]) for i in range(0, len(past_key_values), 2)
            ),
            config=self.model.config,
        )
        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask={"full_attention": full_attention_mask},
            past_key_values=cache,
            use_cache=True,
        )
        return (outputs.logits, *flatten_past_key_values(outputs.past_key_values))


class StaticKVCache:
    """
    Drop-in for DynamicCache that writes K/V into pre-allocated static buffers via
    torch.scatter (ONNX ScatterElements) instead of torch.cat (ONNX Concat).

    Performance rationale: Concat is O(past_len + new_len) because it allocates and
    copies the entire growing tensor each step. Scatter is O(new_len) — it writes only
    the new slice into a fixed-size buffer. For a 600-second recording (~4720 decode steps
    after prefill of ~1200 tokens), each Concat copies an average of ~252 MB of KV data
    per step; scatter copies ~112 KB.

    Interface (matching DynamicCache.update):
        update(key_states, value_states, layer_idx) → (full_key_buffer, full_val_buffer)
    The returned full buffers are passed to attention; positions beyond the current fill
    position are masked out by the causal attention mask.
    """

    def __init__(
        self,
        key_buffers: list[Any],
        val_buffers: list[Any],
        kv_pos: Any,  # int64 scalar tensor (may be symbolic during torch.export)
    ) -> None:
        self._keys = list(key_buffers)  # [num_layers] × [1, kv_heads, max_tokens, head_dim]
        self._vals = list(val_buffers)
        self._pos  = kv_pos

    def update(self, key_states: Any, value_states: Any, layer_idx: int) -> tuple[Any, Any]:
        # key_states: [1, kv_heads, seq_len, head_dim]
        seq_len  = key_states.shape[2]
        kv_heads = key_states.shape[1]
        head_dim = key_states.shape[3]
        device   = key_states.device

        # Build scatter index: positions [kv_pos, kv_pos+1, ..., kv_pos+seq_len-1]
        # broadcast to [1, kv_heads, seq_len, head_dim] — matches key_states shape.
        pos = torch.arange(seq_len, device=device, dtype=torch.int64) + self._pos
        idx = pos.view(1, 1, seq_len, 1).expand(1, kv_heads, seq_len, head_dim)

        # Scatter new K/V into the pre-allocated buffers.
        # torch.scatter → ONNX ScatterElements: only writes the seq_len new positions.
        kb = torch.scatter(self._keys[layer_idx], 2, idx, key_states)
        vb = torch.scatter(self._vals[layer_idx], 2, idx, value_states)
        self._keys[layer_idx] = kb
        self._vals[layer_idx] = vb
        # Return the full updated buffer for attention computation.
        # Positions beyond kv_pos+seq_len contain stale or zero data but are masked
        # to -inf by the causal attention mask built from kv_pos, so they are inert.
        return kb, vb

    def get_seq_length(self, layer_idx: int = 0) -> Any:
        # Return the current fill position so callers that query cache length see
        # kv_pos rather than max_tokens.
        return self._pos


class DecoderSingleStaticWrapper(torch.nn.Module):
    """
    decoder_single_static.onnx: unified prefill+decode with pre-allocated KV buffers.

    Interface:
        inputs:
            prefix_input_ids    [1, prefix_len]
            audio_embeddings    [num_audio_tokens, hidden_size]
            suffix_input_ids    [1, suffix_len]
            kv_pos              [] (int64 scalar: current fill position in KV buffers)
            past_key_0..27      [1, kv_heads, max_tokens, head_dim]  float32
            past_value_0..27    [1, kv_heads, max_tokens, head_dim]  float32
        outputs:
            logits              [1, seq_len, vocab_size]
            present_key_0..27   [1, kv_heads, max_tokens, head_dim]  (SAME shape — no growth)
            present_value_0..27 [1, kv_heads, max_tokens, head_dim]

    kv_pos tells the model where to scatter new K/V; it does NOT increment kv_pos —
    the C# runtime advances kv_pos by seq_len after each call.

    Output shape is FIXED (max_tokens). This allows ORT's BFC arena to reuse the same
    CUDA memory blocks each step without growing allocations, and opens the door to
    binding outputs once rather than re-registering them every step.
    """

    def __init__(self, model: Any) -> None:
        super().__init__()
        self.model = model
        self.num_layers = int(model.config.text_config.num_hidden_layers)

    def forward(
        self,
        prefix_input_ids: Any,
        audio_embeddings: Any,
        suffix_input_ids: Any,
        kv_pos: Any,        # int64 scalar tensor
        *kv_buffers: Any,   # 2 × num_layers tensors: key_0, val_0, key_1, val_1, ...
    ) -> tuple[Any, ...]:
        prefix_embeds = self.model.get_input_embeddings()(prefix_input_ids)
        suffix_embeds = self.model.get_input_embeddings()(suffix_input_ids)
        inputs_embeds = torch.cat(
            (prefix_embeds, audio_embeddings.to(prefix_embeds.device).unsqueeze(0), suffix_embeds),
            dim=1,
        )

        seq_len    = inputs_embeds.shape[1]
        max_tokens = kv_buffers[0].shape[2]  # pre-allocated buffer length

        # Position IDs for the new tokens start at kv_pos.
        position_ids = (
            torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0) + kv_pos
        )

        # Causal attention mask over the full pre-allocated buffer.
        # query positions: kv_pos .. kv_pos+seq_len-1
        # kv positions:    0 .. max_tokens-1
        # Positions beyond kv_pos+seq_len-1 are masked to -inf by the causal logic,
        # so they do not contribute to attention regardless of buffer content.
        full_attention_mask = build_full_attention_mask(
            attention_mask=None,
            query_length=seq_len,
            kv_length=max_tokens,
            past_length=kv_pos,        # tensor is valid here — arithmetic broadcasts
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )

        key_buffers = [kv_buffers[i * 2]     for i in range(self.num_layers)]
        val_buffers = [kv_buffers[i * 2 + 1] for i in range(self.num_layers)]
        cache = StaticKVCache(key_buffers, val_buffers, kv_pos)

        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask={"full_attention": full_attention_mask},
            past_key_values=cache,
            use_cache=True,
        )

        # Flatten updated K/V buffers to match kv_output_names ordering.
        updated_kv: list[Any] = []
        for i in range(self.num_layers):
            updated_kv.append(cache._keys[i])
            updated_kv.append(cache._vals[i])

        return (outputs.logits, *updated_kv)


def copy_metadata_files(processor: Any, model: Any, output_dir: Path) -> None:
    model.config.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)


def build_tokenizer_extras(
    processor: Any,
    dummy_samples: int,
    dummy_prompt: str,
) -> dict[str, Any]:
    """
    Pre-compute prefix/suffix token arrays and digit→token_id map for C# inference.

    The C# VibeVoiceAsr class reads these from export-report.json so it does not
    need a full BPE tokenizer at runtime — only a trivial character lookup for the
    audio-duration float formatted as "%.2f".

    Returned dict is stored under the "tokenizer" key in export-report.json.
    """
    sample_rate = int(processor.feature_extractor.sampling_rate)

    prompt_inputs = processor.apply_transcription_request(
        [np.zeros(dummy_samples, dtype=np.float32)],
        prompt=[dummy_prompt],
    )
    ids: list[int] = prompt_inputs["input_ids"][0].tolist()

    audio_bos_id: int = processor.tokenizer.convert_tokens_to_ids("<|object_ref_start|>")
    audio_eos_id: int = processor.tokenizer.convert_tokens_to_ids("<|object_ref_end|>")
    eos_token_id: int = int(processor.tokenizer.eos_token_id)
    im_end_token_id: int = int(processor.tokenizer.convert_tokens_to_ids("<|im_end|>"))

    bos_pos = ids.index(audio_bos_id)
    eos_pos = ids.index(audio_eos_id)

    prefix_token_ids: list[int] = ids[: bos_pos + 1]   # includes audio_bos_token
    suffix_all: list[int] = ids[eos_pos:]               # includes audio_eos_token

    # Build digit-char→token_id map by encoding each character individually.
    # Qwen2 tokenises decimal digit characters as single tokens; verify this.
    digit_char_to_token_id: dict[str, int] = {}
    for ch in "0123456789.":
        tok_ids = processor.tokenizer.encode(ch, add_special_tokens=False)
        if len(tok_ids) != 1:
            raise ValueError(
                f"Digit character '{ch}' tokenises to {len(tok_ids)} tokens; "
                "expected exactly 1. The C# inference path cannot handle this."
            )
        digit_char_to_token_id[ch] = tok_ids[0]

    # Find the duration string in the suffix by matching the tokenised digits.
    duration_sec = dummy_samples / sample_rate
    duration_str = f"{duration_sec:.2f}"
    duration_token_ids = [digit_char_to_token_id[c] for c in duration_str]

    split_pos: int | None = None
    for i in range(len(suffix_all) - len(duration_token_ids) + 1):
        if suffix_all[i : i + len(duration_token_ids)] == duration_token_ids:
            split_pos = i
            break
    if split_pos is None:
        raise ValueError(
            f"Duration token sequence {duration_token_ids} not found in suffix {suffix_all}."
        )

    suffix_before_duration = suffix_all[:split_pos]
    suffix_after_duration = suffix_all[split_pos + len(duration_token_ids):]

    return {
        "prefix_token_ids": prefix_token_ids,
        "suffix_before_duration_token_ids": suffix_before_duration,
        "suffix_after_duration_token_ids": suffix_after_duration,
        "digit_char_to_token_id": digit_char_to_token_id,
        "eos_token_id": eos_token_id,
        "im_end_token_id": im_end_token_id,
    }


def build_cache_from_packed_past(model: Any, packed_past_key_values: torch.Tensor) -> DynamicCache:
    cache = DynamicCache(config=model.config)
    past_length = packed_past_key_values.shape[4]

    for layer_idx, layer_cache in enumerate(cache.layers):
        layer_keys = packed_past_key_values[layer_idx, 0].contiguous()
        layer_values = packed_past_key_values[layer_idx, 1].contiguous()
        layer_cache.keys = layer_keys
        layer_cache.values = layer_values
        layer_cache.is_initialized = True
        layer_cache.dtype = layer_keys.dtype
        layer_cache.device = layer_keys.device

        if hasattr(layer_cache, "cumulative_length"):
            cumulative_length = getattr(layer_cache, "cumulative_length")
            if torch.is_tensor(cumulative_length):
                layer_cache.cumulative_length = torch.full_like(cumulative_length, past_length)
            else:
                layer_cache.cumulative_length = past_length
        if hasattr(layer_cache, "cumulative_length_int"):
            layer_cache.cumulative_length_int = int(past_length)

    return cache


def replace_audio_token_embeddings(
    *,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    audio_embeddings: torch.Tensor,
    audio_token_id: int,
) -> torch.Tensor:
    audio_positions = torch.nonzero(input_ids[0] == audio_token_id, as_tuple=False).squeeze(-1)
    updated = inputs_embeds[0].index_copy(0, audio_positions, audio_embeddings.to(inputs_embeds.device))
    return updated.unsqueeze(0)


def attention_mask_fill_value(dtype: torch.dtype) -> float:
    if dtype in (torch.float16, torch.bfloat16):
        return -1.0e4
    return float(torch.finfo(dtype).min)


def build_full_attention_mask(
    *,
    attention_mask: torch.Tensor | None,
    query_length: int,
    kv_length: int,
    past_length: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    batch_size = 1
    min_value = attention_mask_fill_value(dtype)

    q_positions = torch.arange(query_length, device=device).view(1, 1, query_length, 1) + past_length
    kv_positions = torch.arange(kv_length, device=device).view(1, 1, 1, kv_length)
    causal = kv_positions <= q_positions

    if attention_mask is not None:
        kv_keep = attention_mask[:, -kv_length:].to(torch.bool).view(batch_size, 1, 1, kv_length)
        causal = causal & kv_keep

    zeros = torch.zeros((batch_size, 1, query_length, kv_length), dtype=dtype, device=device)
    return torch.where(causal, zeros, torch.full_like(zeros, min_value))


def force_eager_attention(model: Any) -> None:
    for cfg in [getattr(model, "config", None), getattr(getattr(model, "language_model", None), "config", None)]:
        if cfg is None:
            continue
        setattr(cfg, "_attn_implementation", "eager")
        setattr(cfg, "_attn_implementation_internal", "eager")

    language_model = getattr(model, "language_model", None)
    if language_model is not None and hasattr(language_model, "model") and hasattr(language_model.model, "config"):
        setattr(language_model.model.config, "_attn_implementation", "eager")
        setattr(language_model.model.config, "_attn_implementation_internal", "eager")


@contextlib.contextmanager
def f32_kv_cache_context():
    """Patch Qwen2Attention to store the KV cache in float32 and compute attention in float32.

    Addresses accumulated BF16 numerical drift across decode steps:
    - New K/V tensors (computed in BF16 via linear projections) are upcast to float32
      before being concatenated into the cache.
    - Q is upcast to float32 for the Q*K^T dot product.
    - The attention mask is cast to float32 to match.
    - All attention dot products (Q*K^T, V*attn) are float32; softmax is already float32
      in Qwen2 eager mode.
    - The attention output is cast back to BF16 before the output projection so the
      residual stream and MLP remain BF16.

    VRAM impact: the KV cache doubles in dtype width (~50-300 MB for 1K-5K token sequences).
    All linear projections and MLP operations remain BF16.

    The exported ONNX graph will contain explicit Cast nodes for all these transitions so
    ORT executes the same float32 attention path at inference time without any runtime patching.
    """
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2Attention,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    original_forward = Qwen2Attention.forward

    def _f32_kv_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
        compute_dtype = hidden_states.dtype
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Upcast new K/V to float32 before cache storage.
        # This ensures the DynamicLayer torch.cat sees matching float32 tensors
        # (past cache is float32 because packed_past_key_values is float32).
        key_states = key_states.to(torch.float32)
        value_states = value_states.to(torch.float32)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        # Upcast Q and attention mask to float32 for the dot products.
        query_states = query_states.to(torch.float32)
        f32_mask = attention_mask.to(torch.float32) if attention_mask is not None else None

        # eager_attention_forward: Q*K^T, softmax, V*attn — all float32.
        # The softmax line does `.to(query.dtype)` = `.to(float32)` which is a no-op.
        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            f32_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        # Cast back to BF16 before the output projection so the residual stream stays BF16.
        attn_output = attn_output.to(compute_dtype)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    Qwen2Attention.forward = _f32_kv_forward
    try:
        yield
    finally:
        Qwen2Attention.forward = original_forward


@contextlib.contextmanager
def f32_lm_head_context(model: Any):
    """Patch the lm_head to cast hidden states and weights to float32 before the projection.

    The lm_head is a large BF16 linear layer (hidden_size -> vocab_size).  BF16 matmul
    accumulation for this projection can collapse distinct logit values into the same BF16
    representable, creating false ties that different CUDA runtimes (PyTorch vs ORT) resolve
    differently.  Casting to float32 before the matmul gives float32-precision logits so the
    true winner is unambiguous and both runtimes agree.

    Weights stay stored in BF16; the Cast to float32 happens at compute time.  VRAM impact at
    inference is the weight cast (~1.1 GB temporarily during the lm_head step).  All other
    layers remain BF16.

    When used during torch.export, the traced ONNX graph contains explicit Cast nodes before
    the lm_head MatMul so ORT executes float32 logits at inference without runtime patching.
    """
    import torch
    import torch.nn.functional as F

    lm_head = model.language_model.lm_head
    original_forward = lm_head.forward

    def _f32_forward(hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(
            hidden_states.to(torch.float32),
            lm_head.weight.to(torch.float32),
            lm_head.bias.to(torch.float32) if lm_head.bias is not None else None,
        )

    lm_head.forward = _f32_forward
    try:
        yield
    finally:
        lm_head.forward = original_forward


def export_audio_encoder(
    *,
    model: Any,
    output_dir: Path,
    opset: int,
    dtype: Any,
    device: str,
    dummy_samples: int,
    deterministic_audio: bool,
    exporter: str,
    torch_export_debug_artifacts: bool,
) -> None:
    # Drop the decoder before tracing the audio path so the 9B LM weights do not
    # stay resident during audio export.
    if hasattr(model, "language_model"):
        del model.language_model
        gc.collect()

    model.to(device=device, dtype=dtype)
    encoder_compute_dtype = dtype
    projector_dtype = dtype
    if dtype == torch.bfloat16 and exporter == "legacy":
        # Keep the external audio graph in bf16, but promote the Conv-heavy tokenizer
        # towers to fp32 so ORT sees legal Conv dtypes without forcing the whole audio
        # graph or projector into fp32/fp16.
        encoder_compute_dtype = torch.float32
        model.acoustic_tokenizer_encoder.to(dtype=encoder_compute_dtype)
        model.semantic_tokenizer_encoder.to(dtype=encoder_compute_dtype)
        model.multi_modal_projector.to(dtype=projector_dtype)

    dummy_audio = torch.zeros((1, dummy_samples), dtype=dtype, device=device)
    dummy_padding_mask = torch.ones((1, dummy_samples), dtype=torch.bool, device=device)
    audio_wrapper = AudioEncoderWrapper(
        model,
        deterministic_audio=deterministic_audio,
        encoder_compute_dtype=encoder_compute_dtype,
        projector_dtype=projector_dtype,
    ).eval()

    print("Exporting audio_encoder.onnx ...")
    export_onnx_graph(
        model=audio_wrapper,
        args=(dummy_audio, dummy_padding_mask),
        output_path=output_dir / "audio_encoder.onnx",
        input_names=["input_values", "padding_mask"],
        output_names=["audio_embeddings"],
        opset=opset,
        dynamic_axes={
            "input_values": {1: "num_samples"},
            "padding_mask": {1: "num_samples"},
            "audio_embeddings": {0: "num_audio_tokens"},
        },
        exporter=exporter,
        dynamic_shapes=(
            {1: torch.export.Dim("num_samples")},
            {1: torch.export.Dim("num_samples")},
        )
        if exporter == "torch-export" and getattr(torch.export, "Dim", None) is not None
        else None,
        debug_artifacts=torch_export_debug_artifacts,
    )


def build_torch_export_dynamic_shapes(
    *,
    mode: str,
    num_layers: int = 0,
) -> tuple[Any, ...] | None:
    dim_ctor = getattr(torch.export, "Dim", None)
    if dim_ctor is None:
        return None

    if mode == "prefill":
        prompt_len = dim_ctor("prompt_len")
        num_audio_tokens = dim_ctor("num_audio_tokens")
        return (
            {1: prompt_len},
            {1: prompt_len},
            {0: num_audio_tokens},
        )

    if mode == "step":
        # The decoder step wrapper currently exposes KV cache as varargs.
        # torch.export shape validation treats that signature differently from
        # the legacy tracer, and the simple per-tensor dynamic spec used for
        # prefill does not line up with the captured input structure here.
        #
        # Until we redesign step export around a non-vararg cache interface,
        # keep the torch.export step graph on the fixed cache length used at
        # export time. This still lets us compare graph structure and memory
        # behavior without blocking on the dynamic-shapes API mismatch.
        return None

    if mode == "single":
        # decoder_single uses *past_key_values varargs (56 tensors).
        # torch.export sees the function signature as 4 positional args:
        #   (prefix_input_ids, audio_embeddings, suffix_input_ids, *past_key_values)
        # dynamic_shapes must match this 4-element pytree structure exactly, with
        # the varargs represented as a tuple of per-tensor shape dicts.
        cache_len = dim_ctor("cache_len")
        return (
            {1: dim_ctor("prefix_len")},   # prefix_input_ids
            {0: dim_ctor("num_audio_tokens")},  # audio_embeddings
            {1: dim_ctor("suffix_len")},   # suffix_input_ids
            tuple({2: cache_len} for _ in range(num_layers * 2)),  # *past_key_values
        )

    if mode == "static_single":
        # decoder_single_static uses *kv_buffers varargs (56 tensors) plus a kv_pos scalar.
        # torch.export sees the function signature as 5 positional args:
        #   (prefix_input_ids, audio_embeddings, suffix_input_ids, kv_pos, *kv_buffers)
        # kv_pos is a scalar (0-d tensor) — empty dict means no dynamic dims.
        # max_tokens is the pre-allocated buffer size; it can vary between export runs.
        max_tokens = dim_ctor("max_tokens")
        return (
            {1: dim_ctor("prefix_len")},        # prefix_input_ids
            {0: dim_ctor("num_audio_tokens")},  # audio_embeddings
            {1: dim_ctor("suffix_len")},        # suffix_input_ids
            {},                                  # kv_pos (scalar — no dynamic dims)
            tuple({2: max_tokens} for _ in range(num_layers * 2)),  # *kv_buffers
        )

    return None


def aggregate_onnx_external_data(onnx_path: Path) -> None:
    """Merge scattered per-tensor external-data files into one canonical <name>.data file.

    ``torch.onnx.export(..., external_data=True)`` writes one file per weight tensor
    (e.g. ``model.lm_head.weight``, ``model.layers.0.mlp.up_proj.weight``, …).
    ORT requires these files to exist *in the same directory as the .onnx file*, so a
    single-file aggregation is far more portable and easier to manage.

    Strategy (avoids loading the full model into CPU RAM twice):
      1. Load the proto *without* tensor bytes to discover which scattered files exist.
      2. If already aggregated (only one data file, already named correctly), return early.
      3. Load all tensor bytes into the proto (unavoidable spike; matches what's already
         on disk), re-save with ``all_tensors_to_one_file=True``, then delete the old files.
    """
    target_data_name = onnx_path.name + ".data"
    target_data_path = onnx_path.parent / target_data_name

    # Inspect proto metadata to find referenced external-data files.
    proto = onnx.load(str(onnx_path), load_external_data=False)
    scattered: set[Path] = set()
    for init in proto.graph.initializer:
        if init.data_location == onnx.TensorProto.EXTERNAL:
            for entry in init.external_data:
                if entry.key == "location":
                    scattered.add(onnx_path.parent / entry.value)

    if not scattered:
        return  # No external data at all — nothing to do.

    if scattered == {target_data_path}:
        return  # Already a single correctly-named data file.

    print(f"  Aggregating {len(scattered)} external-data file(s) → {target_data_name} …", flush=True)

    # Pull all tensor bytes into proto memory, then overwrite with aggregated layout.
    load_external_data_for_model(proto, str(onnx_path.parent))
    onnx.save_model(
        proto,
        str(onnx_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=target_data_name,
        size_threshold=0,
        convert_attribute=False,
    )

    # Remove the old scattered files (the new aggregated file has a different name).
    for f in scattered:
        if f != target_data_path and f.exists():
            f.unlink()


def export_onnx_graph(
    *,
    model: torch.nn.Module,
    args: tuple[Any, ...],
    output_path: Path,
    input_names: list[str],
    output_names: list[str],
    opset: int,
    dynamic_axes: dict[str, dict[int, str]] | None,
    exporter: str,
    dynamic_shapes: tuple[Any, ...] | None = None,
    debug_artifacts: bool = False,
) -> None:
    export_kwargs: dict[str, Any] = {
        "f": str(output_path),
        "input_names": input_names,
        "output_names": output_names,
        "opset_version": opset,
        "external_data": True,
    }

    if dynamic_axes is not None and exporter == "legacy":
        export_kwargs["dynamic_axes"] = dynamic_axes

    if exporter == "legacy":
        export_kwargs["dynamo"] = False
        export_kwargs["do_constant_folding"] = False
    elif exporter == "torch-export":
        export_kwargs["dynamo"] = True
        export_kwargs["optimize"] = False
        export_kwargs["verify"] = False
        export_kwargs["report"] = debug_artifacts
        export_kwargs["dump_exported_program"] = debug_artifacts
        if debug_artifacts:
            artifacts_dir = output_path.parent / "export_artifacts" / output_path.stem
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            export_kwargs["artifacts_dir"] = str(artifacts_dir)
        if dynamic_shapes is not None:
            export_kwargs["dynamic_shapes"] = dynamic_shapes
    else:
        raise ValueError(f"Unsupported exporter: {exporter}")

    torch.onnx.export(model, args=args, **export_kwargs)
    aggregate_onnx_external_data(output_path)


def export_decoder_graphs(
    *,
    model: Any,
    processor: Any,
    output_dir: Path,
    opset: int,
    dtype: Any,
    device: str,
    dummy_samples: int,
    dummy_prompt: str,
    chunk_size: int,
    exporter: str,
    export_prefill: bool,
    export_step: bool,
    export_single: bool,
    export_static_single: bool,
    torch_export_debug_artifacts: bool,
    f32_kv_cache: bool = False,
    f32_lm_head: bool = False,
    static_kv_max_tokens: int = 6144,
) -> dict[str, int]:
    # Drop the audio towers before tracing the decoder path so only the LM stays resident.
    for attr in ("acoustic_tokenizer_encoder", "semantic_tokenizer_encoder", "multi_modal_projector"):
        if hasattr(model, attr):
            delattr(model, attr)
    gc.collect()

    # Qwen2's SDPA mask path currently trips over ONNX tracing here. Force the
    # eager attention implementation for export stability.
    force_eager_attention(model)

    prompt_inputs = processor.apply_transcription_request(
        [np.zeros(dummy_samples, dtype=np.float32)],
        prompt=[dummy_prompt],
    )
    prompt_input_ids = prompt_inputs["input_ids"].to(device)
    prompt_attention_mask = prompt_inputs["attention_mask"].to(device)

    hidden_size = int(model.config.text_config.hidden_size)
    num_layers = int(model.config.text_config.num_hidden_layers)
    num_kv_heads = int(model.config.text_config.num_key_value_heads)
    num_heads = int(model.config.text_config.num_attention_heads)
    head_dim = hidden_size // num_heads
    prompt_len = int(prompt_input_ids.shape[1])
    num_audio_tokens = int((prompt_input_ids == int(model.config.audio_token_id)).sum().item())
    if num_audio_tokens <= 0:
        raise SystemExit("Expected the dummy prompt to contain at least one audio placeholder token.")
    audio_positions = torch.nonzero(prompt_input_ids[0] == int(model.config.audio_token_id), as_tuple=False).squeeze(-1)
    prompt_prefix_input_ids = prompt_input_ids[:, : audio_positions[0]]
    prompt_suffix_input_ids = prompt_input_ids[:, audio_positions[-1] + 1 :]

    dummy_audio_embeddings = torch.zeros((num_audio_tokens, hidden_size), dtype=dtype, device=device)

    if export_prefill:
        prefill_wrapper = DecoderPrefillWrapper(model).eval()
        print("Exporting decoder_prefill.onnx ...")
        export_onnx_graph(
            model=prefill_wrapper,
            args=(prompt_input_ids, prompt_attention_mask, dummy_audio_embeddings),
            output_path=output_dir / "decoder_prefill.onnx",
            input_names=["input_ids", "attention_mask", "audio_embeddings"],
            output_names=["logits", *kv_output_names(num_layers)],
            opset=opset,
            dynamic_axes={
                "input_ids": {1: "prompt_len"},
                "attention_mask": {1: "prompt_len"},
                "audio_embeddings": {0: "num_audio_tokens"},
                "logits": {1: "prompt_len"},
                **{name: {2: "cache_len"} for name in kv_output_names(num_layers)},
            },
            exporter=exporter,
            dynamic_shapes=build_torch_export_dynamic_shapes(mode="prefill"),
            debug_artifacts=torch_export_debug_artifacts,
        )
        del prefill_wrapper
        gc.collect()

    if export_step:
        step_wrapper = DecoderStepWrapper(model).eval()
        step_input_ids = torch.ones((1, 1), dtype=torch.long, device=device) * int(model.config.audio_eos_token_id)
        step_attention_mask = torch.ones((1, prompt_len + 1), dtype=torch.long, device=device)
        dummy_kv = []
        for _ in range(num_layers):
            kv_shape = (1, num_kv_heads, prompt_len, head_dim)
            dummy_kv.append(torch.zeros(kv_shape, dtype=dtype, device=device))
            dummy_kv.append(torch.zeros(kv_shape, dtype=dtype, device=device))

        print("Exporting decoder_step.onnx ...")
        export_onnx_graph(
            model=step_wrapper,
            args=(step_input_ids, step_attention_mask, *dummy_kv),
            output_path=output_dir / "decoder_step.onnx",
            input_names=["input_ids", "attention_mask", *kv_input_names(num_layers)],
            output_names=["logits", *kv_output_names(num_layers)],
            opset=opset,
            dynamic_axes={
                "attention_mask": {1: "total_seq_len"},
                "logits": {1: "step_len"},
                **{name: {2: "cache_len"} for name in kv_input_names(num_layers)},
                **{name: {2: "cache_len_out"} for name in kv_output_names(num_layers)},
            },
            exporter=exporter,
            dynamic_shapes=build_torch_export_dynamic_shapes(mode="step", num_layers=num_layers),
            debug_artifacts=torch_export_debug_artifacts,
        )
        del step_wrapper
        del dummy_kv
        gc.collect()

    if export_single:
        single_wrapper = DecoderSingleWrapper(model).eval()
        # Use float32 for the KV cache when --f32-kv-cache is requested; activations stay BF16.
        kv_cache_dtype = torch.float32 if f32_kv_cache else dtype
        # 56 separate empty KV tensors: [1, num_kv_heads, 0, head_dim] per layer × 2 (K+V).
        dummy_kv_single: list[Any] = []
        for _ in range(num_layers):
            kv_shape = (1, num_kv_heads, 0, head_dim)
            dummy_kv_single.append(torch.zeros(kv_shape, dtype=kv_cache_dtype, device=device))
            dummy_kv_single.append(torch.zeros(kv_shape, dtype=kv_cache_dtype, device=device))
        single_exporter = exporter

        print(f"Exporting decoder_single.onnx (kv_cache_dtype={kv_cache_dtype}, f32_lm_head={f32_lm_head}) ...")
        kv_ctx = f32_kv_cache_context() if f32_kv_cache else contextlib.nullcontext()
        lm_ctx = f32_lm_head_context(model) if f32_lm_head else contextlib.nullcontext()
        with kv_ctx, lm_ctx:
            export_onnx_graph(
                model=single_wrapper,
                args=(prompt_prefix_input_ids, dummy_audio_embeddings, prompt_suffix_input_ids, *dummy_kv_single),
                output_path=output_dir / "decoder_single.onnx",
                input_names=["prefix_input_ids", "audio_embeddings", "suffix_input_ids", *kv_input_names(num_layers)],
                output_names=["logits", *kv_output_names(num_layers)],
                opset=opset,
                dynamic_axes={
                    "prefix_input_ids": {1: "prefix_len"},
                    "audio_embeddings": {0: "num_audio_tokens"},
                    "suffix_input_ids": {1: "suffix_len"},
                    **{name: {2: "cache_len"} for name in kv_input_names(num_layers)},
                    "logits": {1: "seq_len"},
                    **{name: {2: "cache_len_out"} for name in kv_output_names(num_layers)},
                },
                exporter=single_exporter,
                dynamic_shapes=build_torch_export_dynamic_shapes(mode="single", num_layers=num_layers) if single_exporter == "torch-export" else None,
                debug_artifacts=torch_export_debug_artifacts,
            )
        del single_wrapper
        del dummy_kv_single
        gc.collect()

    if export_static_single:
        static_wrapper = DecoderSingleStaticWrapper(model).eval()
        kv_cache_dtype = torch.float32 if f32_kv_cache else dtype
        kv_pos_dummy = torch.tensor(0, dtype=torch.int64, device=device)
        # Pre-allocated dummy KV buffers: [1, num_kv_heads, max_tokens, head_dim]
        dummy_kv_static: list[Any] = []
        for _ in range(num_layers):
            kv_shape = (1, num_kv_heads, static_kv_max_tokens, head_dim)
            dummy_kv_static.append(torch.zeros(kv_shape, dtype=kv_cache_dtype, device=device))
            dummy_kv_static.append(torch.zeros(kv_shape, dtype=kv_cache_dtype, device=device))

        print(
            f"Exporting decoder_single_static.onnx "
            f"(kv_cache_dtype={kv_cache_dtype}, max_tokens={static_kv_max_tokens}, "
            f"f32_lm_head={f32_lm_head}) ..."
        )
        kv_ctx = f32_kv_cache_context() if f32_kv_cache else contextlib.nullcontext()
        lm_ctx = f32_lm_head_context(model) if f32_lm_head else contextlib.nullcontext()
        with kv_ctx, lm_ctx:
            export_onnx_graph(
                model=static_wrapper,
                args=(
                    prompt_prefix_input_ids,
                    dummy_audio_embeddings,
                    prompt_suffix_input_ids,
                    kv_pos_dummy,
                    *dummy_kv_static,
                ),
                output_path=output_dir / "decoder_single_static.onnx",
                input_names=[
                    "prefix_input_ids", "audio_embeddings", "suffix_input_ids", "kv_pos",
                    *kv_input_names(num_layers),
                ],
                output_names=["logits", *kv_output_names(num_layers)],
                opset=opset,
                dynamic_axes={
                    "prefix_input_ids": {1: "prefix_len"},
                    "audio_embeddings": {0: "num_audio_tokens"},
                    "suffix_input_ids": {1: "suffix_len"},
                    "logits": {1: "seq_len"},
                    **{name: {2: "max_tokens"} for name in kv_input_names(num_layers)},
                    **{name: {2: "max_tokens"} for name in kv_output_names(num_layers)},
                },
                exporter=exporter,
                dynamic_shapes=(
                    build_torch_export_dynamic_shapes(mode="static_single", num_layers=num_layers)
                    if exporter == "torch-export"
                    else None
                ),
                debug_artifacts=torch_export_debug_artifacts,
            )
        del static_wrapper
        del dummy_kv_static
        gc.collect()

    return {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "prompt_len": prompt_len,
        "num_audio_tokens": num_audio_tokens,
    }


def main() -> None:
    args = parse_args()

    device = resolve_device(torch, args.device)
    dtype = resolve_dtype(torch, args.dtype)
    decoder_exporter = resolve_decoder_exporter(args.decoder_exporter)
    audio_exporter = resolve_audio_exporter(args.audio_exporter)
    export_split_decoders       = args.decoder_graph_mode in ("split", "both")
    export_single_decoder       = args.decoder_graph_mode in ("single", "both")
    export_static_single_decoder = args.decoder_graph_mode in ("static-single",)
    ensure_output_dir(args.output_dir, args.overwrite, EXPORT_FILES)

    model_config_dict: dict[str, Any] | None = None
    processor: Any | None = None
    report_dims: dict[str, int] = {}
    audio_export_dtype, audio_export_note = resolve_audio_export_dtype(dtype, torch, audio_exporter)
    if audio_export_note:
        print(f"[export] {audio_export_note}")

    if not args.skip_audio_encoder:
        audio_model, processor = load_model_and_processor(args.model_repo, args.revision, device, dtype, torch)
        copy_metadata_files(processor, audio_model, args.output_dir)
        model_config_dict = audio_model.config.to_dict()

        chunk_size = infer_export_chunk_size(
            requested_chunk_size=args.acoustic_tokenizer_chunk_size,
            default_chunk_size=int(audio_model.config.acoustic_tokenizer_chunk_size),
            sampling_rate=int(processor.feature_extractor.sampling_rate),
            dummy_audio_seconds=args.dummy_audio_seconds,
            processor=processor,
        )

        dummy_samples = round_up_to_multiple(
            int(round(args.dummy_audio_seconds * processor.feature_extractor.sampling_rate)),
            chunk_size,
        )
        export_audio_encoder(
            model=audio_model,
            output_dir=args.output_dir,
            opset=args.opset,
            dtype=audio_export_dtype,
            device=device,
            dummy_samples=dummy_samples,
            deterministic_audio=args.deterministic_audio,
            exporter=audio_exporter,
            torch_export_debug_artifacts=args.torch_export_debug_artifacts,
        )
        del audio_model
        gc.collect()
    else:
        # Import the model class first so AutoProcessor can find the custom processing class.
        # VibeVoiceAsrForConditionalGeneration is present in the vibevoice-asr transformers build
        # but may not be a top-level export in all environments.  Fall back to trust_remote_code.
        try:
            from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration as _VVA  # noqa: F401
        except ImportError:
            from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(args.model_repo, revision=args.revision, trust_remote_code=True)
        chunk_size = infer_export_chunk_size(
            requested_chunk_size=args.acoustic_tokenizer_chunk_size,
            default_chunk_size=1440000,
            sampling_rate=int(processor.feature_extractor.sampling_rate),
            dummy_audio_seconds=args.dummy_audio_seconds,
            processor=processor,
        )
        dummy_samples = round_up_to_multiple(
            int(round(args.dummy_audio_seconds * processor.feature_extractor.sampling_rate)),
            chunk_size,
        )

    if not (
        (not export_split_decoders or (args.skip_prefill and args.skip_step))
        and not export_single_decoder
        and not export_static_single_decoder
    ):
        decoder_model, decoder_processor = load_model_and_processor(args.model_repo, args.revision, device, dtype, torch)
        if model_config_dict is None:
            copy_metadata_files(decoder_processor, decoder_model, args.output_dir)
            model_config_dict = decoder_model.config.to_dict()
        report_dims = export_decoder_graphs(
            model=decoder_model,
            processor=decoder_processor,
            output_dir=args.output_dir,
            opset=args.opset,
            dtype=dtype,
            device=device,
            dummy_samples=dummy_samples,
            dummy_prompt=args.dummy_prompt,
            chunk_size=chunk_size,
            exporter=decoder_exporter,
            export_prefill=export_split_decoders and not args.skip_prefill,
            export_step=export_split_decoders and not args.skip_step,
            export_single=export_single_decoder,
            export_static_single=export_static_single_decoder,
            torch_export_debug_artifacts=args.torch_export_debug_artifacts,
            f32_kv_cache=args.f32_kv_cache,
            f32_lm_head=args.f32_lm_head,
            static_kv_max_tokens=args.static_kv_max_tokens,
        )
        del decoder_model
        gc.collect()

    if model_config_dict is None:
        raise SystemExit("Nothing to export. Remove one of --skip-* flags and try again.")

    tokenizer_extras = build_tokenizer_extras(processor, dummy_samples, args.dummy_prompt)

    save_export_report(
        args.output_dir / "export-report.json",
        repo_id=args.model_repo,
        revision=args.revision,
        device=device,
        dtype=torch_dtype_name(dtype),
        opset=args.opset,
        acoustic_tokenizer_chunk_size=chunk_size,
        deterministic_audio=args.deterministic_audio,
        model_config=model_config_dict,
        extra={
            "dummy_audio_seconds": args.dummy_audio_seconds,
            "audio_export_input_samples": dummy_samples,
            "dummy_prompt": args.dummy_prompt,
            "audio_exporter": audio_exporter,
            "decoder_exporter": decoder_exporter,
            "decoder_graph_mode": args.decoder_graph_mode,
            "audio_encoder_dtype": torch_dtype_name(audio_export_dtype),
            "decoder_dtype": torch_dtype_name(dtype),
            "f32_kv_cache": args.f32_kv_cache,
            "f32_lm_head": args.f32_lm_head,
            "static_kv_cache": export_static_single_decoder,
            "static_kv_max_tokens": args.static_kv_max_tokens if export_static_single_decoder else 0,
            "mixed_precision_note": audio_export_note,
            "tokenizer": tokenizer_extras,
            **report_dims,
        },
    )

    print(f"Export completed in {args.output_dir}")


if __name__ == "__main__":
    main()
