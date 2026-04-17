"""
Helpers for mixed-length batching with the unified Qwen3-ASR decoder export.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .prompt import build_prompt_ids, get_audio_pad_range


class DecoderConfig(NamedTuple):
    hidden_size: int
    num_layers: int
    num_kv_heads: int
    head_dim: int


def build_empty_kv(batch_size: int, cfg: DecoderConfig) -> tuple[np.ndarray, np.ndarray]:
    empty = np.zeros((cfg.num_layers, batch_size, cfg.num_kv_heads, 0, cfg.head_dim), dtype=np.float32)
    return empty, empty.copy()


def build_prefill_attention_mask(seq_lengths: np.ndarray) -> np.ndarray:
    batch_size = int(seq_lengths.shape[0])
    max_seq_len = int(seq_lengths.max())
    mask = np.full((batch_size, 1, max_seq_len, max_seq_len), np.finfo(np.float32).min, dtype=np.float32)

    for batch_index, seq_len in enumerate(seq_lengths.astype(np.int64, copy=False)):
        valid_len = int(seq_len)
        for query_index in range(valid_len):
            mask[batch_index, 0, query_index, : query_index + 1] = 0.0

        if valid_len < max_seq_len:
            # Keep padded query rows numerically stable; their logits are ignored.
            mask[batch_index, 0, valid_len:, 0] = 0.0

    return mask


def build_prefill_inputs(
    audio_features: np.ndarray,
    audio_feature_lengths: np.ndarray,
    embed_table: np.ndarray,
    cfg: DecoderConfig,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    batch_size = int(audio_features.shape[0])
    hidden_size = int(cfg.hidden_size)

    prompt_embeds: list[np.ndarray] = []
    seq_lengths: list[int] = []

    for batch_index in range(batch_size):
        audio_len = int(audio_feature_lengths[batch_index])
        prompt_ids = build_prompt_ids(audio_len)
        prompt_embed = embed_table[prompt_ids].astype(np.float32, copy=True)
        audio_start, _ = get_audio_pad_range(prompt_ids)
        prompt_embed[audio_start : audio_start + audio_len] = audio_features[batch_index, :audio_len]
        prompt_embeds.append(prompt_embed)
        seq_lengths.append(prompt_embed.shape[0])

    seq_lengths_array = np.array(seq_lengths, dtype=np.int64)
    max_seq_len = int(seq_lengths_array.max())
    input_embeds = np.zeros((batch_size, max_seq_len, hidden_size), dtype=np.float32)
    position_ids = np.zeros((batch_size, max_seq_len), dtype=np.int64)

    for batch_index, prompt_embed in enumerate(prompt_embeds):
        seq_len = prompt_embed.shape[0]
        input_embeds[batch_index, :seq_len] = prompt_embed
        position_ids[batch_index, :seq_len] = np.arange(seq_len, dtype=np.int64)

    attention_mask = build_prefill_attention_mask(seq_lengths_array)
    past_keys, past_values = build_empty_kv(batch_size, cfg)

    return {
        "input_embeds": input_embeds,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "past_keys": past_keys,
        "past_values": past_values,
    }, seq_lengths_array


def gather_last_valid_logits(logits: np.ndarray, seq_lengths: np.ndarray) -> np.ndarray:
    batch_size = int(logits.shape[0])
    vocab_size = int(logits.shape[2])
    gathered = np.empty((batch_size, vocab_size), dtype=logits.dtype)

    for batch_index, seq_len in enumerate(seq_lengths.astype(np.int64, copy=False)):
        gathered[batch_index] = logits[batch_index, int(seq_len) - 1]

    return gathered


def compact_prefill_kv(
    present_keys: np.ndarray,
    present_values: np.ndarray,
    seq_lengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    max_seq_len = int(seq_lengths.max())
    compact_keys = np.zeros(
        (present_keys.shape[0], present_keys.shape[1], present_keys.shape[2], max_seq_len, present_keys.shape[4]),
        dtype=present_keys.dtype,
    )
    compact_values = np.zeros(
        (present_values.shape[0], present_values.shape[1], present_values.shape[2], max_seq_len, present_values.shape[4]),
        dtype=present_values.dtype,
    )

    for batch_index, seq_len in enumerate(seq_lengths.astype(np.int64, copy=False)):
        valid_len = int(seq_len)
        compact_keys[:, batch_index, :, :valid_len, :] = present_keys[:, batch_index, :, :valid_len, :]
        compact_values[:, batch_index, :, :valid_len, :] = present_values[:, batch_index, :, :valid_len, :]

    return compact_keys, compact_values


def build_step_attention_mask(past_lengths: np.ndarray) -> np.ndarray:
    batch_size = int(past_lengths.shape[0])
    max_past_len = int(past_lengths.max())
    total_len = max_past_len + 1
    mask = np.full((batch_size, 1, 1, total_len), np.finfo(np.float32).min, dtype=np.float32)

    for batch_index, past_len in enumerate(past_lengths.astype(np.int64, copy=False)):
        valid_past_len = int(past_len)
        mask[batch_index, 0, 0, :valid_past_len] = 0.0
        mask[batch_index, 0, 0, max_past_len] = 0.0

    return mask


def build_step_inputs(
    next_tokens: np.ndarray,
    embed_table: np.ndarray,
    past_keys: np.ndarray,
    past_values: np.ndarray,
    past_lengths: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "input_embeds": embed_table[next_tokens][:, np.newaxis, :].astype(np.float32, copy=False),
        "position_ids": past_lengths.astype(np.int64, copy=False)[:, np.newaxis],
        "attention_mask": build_step_attention_mask(past_lengths),
        "past_keys": past_keys,
        "past_values": past_values,
    }


def compact_step_kv(
    present_keys: np.ndarray,
    present_values: np.ndarray,
    past_lengths: np.ndarray,
    advance_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    append_index = int(present_keys.shape[3] - 1)
    next_lengths = past_lengths.astype(np.int64, copy=True)
    next_lengths[advance_mask] += 1
    max_next_len = int(next_lengths.max())

    compact_keys = np.zeros(
        (present_keys.shape[0], present_keys.shape[1], present_keys.shape[2], max_next_len, present_keys.shape[4]),
        dtype=present_keys.dtype,
    )
    compact_values = np.zeros(
        (present_values.shape[0], present_values.shape[1], present_values.shape[2], max_next_len, present_values.shape[4]),
        dtype=present_values.dtype,
    )

    for batch_index, past_len in enumerate(past_lengths.astype(np.int64, copy=False)):
        valid_past_len = int(past_len)
        if valid_past_len > 0:
            compact_keys[:, batch_index, :, :valid_past_len, :] = present_keys[:, batch_index, :, :valid_past_len, :]
            compact_values[:, batch_index, :, :valid_past_len, :] = present_values[:, batch_index, :, :valid_past_len, :]

        if bool(advance_mask[batch_index]):
            compact_keys[:, batch_index, :, valid_past_len, :] = present_keys[:, batch_index, :, append_index, :]
            compact_values[:, batch_index, :, valid_past_len, :] = present_values[:, batch_index, :, append_index, :]

    return compact_keys, compact_values, next_lengths
