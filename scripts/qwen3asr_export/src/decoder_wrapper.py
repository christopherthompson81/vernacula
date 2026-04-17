"""
Wrapper modules for Qwen3-ASR decoder export to ONNX.

The export uses split decoder graphs:
- `decoder_init.onnx` for prompt prefill
- `decoder_step.onnx` for autoregressive decoding with explicit KV cache I/O
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(query, key, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    query_embed = (query * cos) + (_rotate_half(query) * sin)
    key_embed = (key * cos) + (_rotate_half(key) * sin)
    return query_embed, key_embed


def _repeat_kv(hidden_states, n_rep):
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


def _attention(query, key, value, mask, scaling, num_kv_groups):
    key = _repeat_kv(key, num_kv_groups)
    value = _repeat_kv(value, num_kv_groups)
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if mask is not None:
        attn_weights = attn_weights + mask[:, :, :, : key.shape[-2]]
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value)
    return attn_output.transpose(1, 2).contiguous()


def _scatter_static_kv(
    *,
    buffer: torch.Tensor,
    update: torch.Tensor,
    kv_pos: torch.Tensor,
) -> torch.Tensor:
    seq_len = update.shape[2]
    kv_heads = update.shape[1]
    head_dim = update.shape[3]
    positions = torch.arange(seq_len, device=update.device, dtype=torch.int64) + kv_pos
    indices = positions.view(1, 1, seq_len, 1).expand(1, kv_heads, seq_len, head_dim)
    return torch.scatter(buffer, 2, indices, update)


def _decoder_layer_forward(layer, hidden_states, cos, sin, mask, past_key, past_value, num_kv_groups):
    attn = layer.self_attn

    residual = hidden_states
    normed = layer.input_layernorm(hidden_states)

    input_shape = normed.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)

    query_states = attn.q_norm(attn.q_proj(normed).view(hidden_shape)).transpose(1, 2)
    key_states = attn.k_norm(attn.k_proj(normed).view(hidden_shape)).transpose(1, 2)
    value_states = attn.v_proj(normed).view(hidden_shape).transpose(1, 2)

    query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key is not None:
        key_states = torch.cat([past_key, key_states], dim=2)
        value_states = torch.cat([past_value, value_states], dim=2)

    attn_output = _attention(query_states, key_states, value_states, mask, attn.scaling, num_kv_groups)
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = attn.o_proj(attn_output)

    hidden_states = residual + attn_output

    residual = hidden_states
    normed = layer.post_attention_layernorm(hidden_states)
    hidden_states = residual + layer.mlp(normed)

    return hidden_states, key_states, value_states


def _decoder_layer_forward_static(
    layer,
    hidden_states,
    cos,
    sin,
    key_buffer,
    value_buffer,
    kv_pos,
    num_kv_groups,
):
    attn = layer.self_attn

    residual = hidden_states
    normed = layer.input_layernorm(hidden_states)

    input_shape = normed.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)

    query_states = attn.q_norm(attn.q_proj(normed).view(hidden_shape)).transpose(1, 2)
    key_states = attn.k_norm(attn.k_proj(normed).view(hidden_shape)).transpose(1, 2)
    value_states = attn.v_proj(normed).view(hidden_shape).transpose(1, 2)

    query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

    updated_keys = _scatter_static_kv(buffer=key_buffer, update=key_states, kv_pos=kv_pos)
    updated_values = _scatter_static_kv(buffer=value_buffer, update=value_states, kv_pos=kv_pos)

    active_length = kv_pos + key_states.shape[2]
    active_positions = torch.arange(active_length, device=updated_keys.device, dtype=torch.int64)
    active_keys = torch.index_select(updated_keys, 2, active_positions)
    active_values = torch.index_select(updated_values, 2, active_positions)

    attn_output = _attention(query_states, active_keys, active_values, None, attn.scaling, num_kv_groups)
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = attn.o_proj(attn_output)

    hidden_states = residual + attn_output

    residual = hidden_states
    normed = layer.post_attention_layernorm(hidden_states)
    hidden_states = residual + layer.mlp(normed)

    return hidden_states, updated_keys, updated_values


class DecoderInitWrapper(nn.Module):
    def __init__(self, text_model, lm_head, text_config):
        super().__init__()
        self.embed_tokens = text_model.embed_tokens
        self.layers = text_model.layers
        self.norm = text_model.norm
        self.rotary_emb = text_model.rotary_emb
        self.lm_head = lm_head
        self.num_kv_groups = text_config.num_attention_heads // text_config.num_key_value_heads

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        audio_features: torch.Tensor,
        audio_offset: torch.Tensor,
    ):
        input_embeds = self.embed_tokens(input_ids)

        audio_len = audio_features.shape[1]
        indices = torch.arange(audio_len, device=input_ids.device) + audio_offset[0]
        indices = indices.unsqueeze(0).unsqueeze(-1).expand_as(audio_features)
        input_embeds = input_embeds.scatter(1, indices, audio_features)

        _, seq_len = input_embeds.shape[:2]

        pos_3d = position_ids.unsqueeze(0).expand(3, -1, -1)
        cos, sin = self.rotary_emb(input_embeds, pos_3d)

        causal_mask = torch.full(
            (seq_len, seq_len),
            torch.finfo(input_embeds.dtype).min,
            device=input_embeds.device,
            dtype=input_embeds.dtype,
        )
        causal_mask = torch.triu(causal_mask, diagonal=1).unsqueeze(0).unsqueeze(0)

        hidden_states = input_embeds
        all_keys = []
        all_values = []

        for layer in self.layers:
            hidden_states, key_states, value_states = _decoder_layer_forward(
                layer,
                hidden_states,
                cos,
                sin,
                causal_mask,
                past_key=None,
                past_value=None,
                num_kv_groups=self.num_kv_groups,
            )
            all_keys.append(key_states)
            all_values.append(value_states)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        present_keys = torch.stack(all_keys, dim=0)
        present_values = torch.stack(all_values, dim=0)
        return logits, present_keys, present_values


class DecoderInitBatchedWrapper(nn.Module):
    def __init__(self, text_model, lm_head, text_config):
        super().__init__()
        self.embed_tokens = text_model.embed_tokens
        self.layers = text_model.layers
        self.norm = text_model.norm
        self.rotary_emb = text_model.rotary_emb
        self.lm_head = lm_head
        self.num_kv_groups = text_config.num_attention_heads // text_config.num_key_value_heads

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        audio_features: torch.Tensor,
        audio_lengths: torch.Tensor,
        audio_offset: torch.Tensor,
    ):
        input_embeds = self.embed_tokens(input_ids)

        batch_size, audio_len, hidden_size = audio_features.shape
        start_index = audio_offset[0]
        audio_positions = torch.arange(audio_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        valid_mask = (audio_positions < audio_lengths.unsqueeze(1)).unsqueeze(-1)
        indices = audio_positions.unsqueeze(-1).expand(batch_size, audio_len, hidden_size) + start_index
        target_slice = torch.gather(input_embeds, 1, indices)
        scatter_values = torch.where(valid_mask, audio_features, target_slice)
        input_embeds = input_embeds.scatter(1, indices, scatter_values)

        _, seq_len = input_embeds.shape[:2]

        pos_3d = position_ids.unsqueeze(0).expand(3, -1, -1)
        cos, sin = self.rotary_emb(input_embeds, pos_3d)

        causal_mask = torch.full(
            (seq_len, seq_len),
            torch.finfo(input_embeds.dtype).min,
            device=input_embeds.device,
            dtype=input_embeds.dtype,
        )
        causal_mask = torch.triu(causal_mask, diagonal=1).unsqueeze(0).unsqueeze(0)

        hidden_states = input_embeds
        all_keys = []
        all_values = []

        for layer in self.layers:
            hidden_states, key_states, value_states = _decoder_layer_forward(
                layer,
                hidden_states,
                cos,
                sin,
                causal_mask,
                past_key=None,
                past_value=None,
                num_kv_groups=self.num_kv_groups,
            )
            all_keys.append(key_states)
            all_values.append(value_states)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        present_keys = torch.stack(all_keys, dim=0)
        present_values = torch.stack(all_values, dim=0)
        return logits, present_keys, present_values


class DecoderStepWrapper(nn.Module):
    def __init__(self, text_model, lm_head, text_config):
        super().__init__()
        self.layers = text_model.layers
        self.norm = text_model.norm
        self.rotary_emb = text_model.rotary_emb
        self.lm_head = lm_head
        self.num_kv_groups = text_config.num_attention_heads // text_config.num_key_value_heads

    def forward(
        self,
        input_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        past_keys: torch.Tensor,
        past_values: torch.Tensor,
    ):
        pos_3d = position_ids.unsqueeze(0).expand(3, -1, -1)
        cos, sin = self.rotary_emb(input_embeds, pos_3d)

        hidden_states = input_embeds
        all_keys = []
        all_values = []

        for index, layer in enumerate(self.layers):
            hidden_states, key_states, value_states = _decoder_layer_forward(
                layer,
                hidden_states,
                cos,
                sin,
                None,
                past_key=past_keys[index],
                past_value=past_values[index],
                num_kv_groups=self.num_kv_groups,
            )
            all_keys.append(key_states)
            all_values.append(value_states)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        present_keys = torch.stack(all_keys, dim=0)
        present_values = torch.stack(all_values, dim=0)
        return logits, present_keys, present_values


class DecoderStepStaticWrapper(nn.Module):
    def __init__(self, text_model, lm_head, text_config):
        super().__init__()
        self.layers = text_model.layers
        self.norm = text_model.norm
        self.rotary_emb = text_model.rotary_emb
        self.lm_head = lm_head
        self.num_kv_groups = text_config.num_attention_heads // text_config.num_key_value_heads

    def forward(
        self,
        input_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        kv_pos: torch.Tensor,
        past_keys: torch.Tensor,
        past_values: torch.Tensor,
    ):
        pos_3d = position_ids.unsqueeze(0).expand(3, -1, -1)
        cos, sin = self.rotary_emb(input_embeds, pos_3d)

        hidden_states = input_embeds
        all_keys = []
        all_values = []

        flat_kv_pos = kv_pos.reshape(())

        for index, layer in enumerate(self.layers):
            hidden_states, key_states, value_states = _decoder_layer_forward_static(
                layer,
                hidden_states,
                cos,
                sin,
                past_keys[index],
                past_values[index],
                flat_kv_pos,
                self.num_kv_groups,
            )
            all_keys.append(key_states)
            all_values.append(value_states)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        present_keys = torch.stack(all_keys, dim=0)
        present_values = torch.stack(all_values, dim=0)
        return logits, present_keys, present_values


def export_decoder_init(model, output_path: str, opset_version: int = 17, device: str = "cpu"):
    """Export the decoder prefill graph to ONNX."""
    text_config = model.config.thinker_config.text_config
    hidden_size = text_config.hidden_size

    text_model = model.thinker.model
    lm_head = model.thinker.lm_head
    wrapper = DecoderInitWrapper(text_model, lm_head, text_config).eval().to(device)

    seq_len = 100
    audio_len = 80
    audio_offset_value = 9
    dummy_ids = torch.zeros(1, seq_len, device=device, dtype=torch.long)
    dummy_pos = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    dummy_audio = torch.randn(1, audio_len, hidden_size, device=device, dtype=torch.float32)
    dummy_offset = torch.tensor([audio_offset_value], device=device, dtype=torch.long)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_ids, dummy_pos, dummy_audio, dummy_offset),
            output_path,
            input_names=["input_ids", "position_ids", "audio_features", "audio_offset"],
            output_names=["logits", "present_keys", "present_values"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq_len"},
                "position_ids": {0: "batch", 1: "seq_len"},
                "audio_features": {0: "batch", 1: "audio_len"},
                "logits": {0: "batch", 1: "seq_len"},
                "present_keys": {1: "batch", 3: "seq_len"},
                "present_values": {1: "batch", 3: "seq_len"},
            },
            opset_version=opset_version,
            do_constant_folding=True,
        )

    from .onnx_fixup import fix_reshape_allowzero

    fixed = fix_reshape_allowzero(output_path)
    print(f"Decoder init exported to {output_path} (fixed {fixed} Reshape allowzero attrs)")


def export_decoder_init_batched(model, output_path: str, opset_version: int = 18, device: str = "cpu"):
    """Export an experimental decoder prefill graph that accepts per-item audio lengths."""
    text_config = model.config.thinker_config.text_config
    hidden_size = text_config.hidden_size

    text_model = model.thinker.model
    lm_head = model.thinker.lm_head
    wrapper = DecoderInitBatchedWrapper(text_model, lm_head, text_config).eval().to(device)

    batch_size = 2
    seq_len = 120
    audio_len = 96
    audio_offset_value = 9
    dummy_ids = torch.zeros(batch_size, seq_len, device=device, dtype=torch.long)
    dummy_pos = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    dummy_audio = torch.randn(batch_size, audio_len, hidden_size, device=device, dtype=torch.float32)
    dummy_audio_lengths = torch.tensor([96, 61], device=device, dtype=torch.long)
    dummy_offset = torch.tensor([audio_offset_value], device=device, dtype=torch.long)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_ids, dummy_pos, dummy_audio, dummy_audio_lengths, dummy_offset),
            output_path,
            input_names=["input_ids", "position_ids", "audio_features", "audio_lengths", "audio_offset"],
            output_names=["logits", "present_keys", "present_values"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq_len"},
                "position_ids": {0: "batch", 1: "seq_len"},
                "audio_features": {0: "batch", 1: "audio_len"},
                "audio_lengths": {0: "batch"},
                "logits": {0: "batch", 1: "seq_len"},
                "present_keys": {1: "batch", 3: "seq_len"},
                "present_values": {1: "batch", 3: "seq_len"},
            },
            opset_version=opset_version,
            do_constant_folding=True,
        )

    from .onnx_fixup import fix_reshape_allowzero

    fixed = fix_reshape_allowzero(output_path)
    print(f"Batched decoder init exported to {output_path} (fixed {fixed} Reshape allowzero attrs)")


def export_decoder_step(model, output_path: str, opset_version: int = 17, device: str = "cpu"):
    """Export the autoregressive decoder step graph to ONNX."""
    text_config = model.config.thinker_config.text_config
    hidden_size = text_config.hidden_size
    num_layers = text_config.num_hidden_layers
    num_kv_heads = text_config.num_key_value_heads
    head_dim = text_config.head_dim

    text_model = model.thinker.model
    lm_head = model.thinker.lm_head
    wrapper = DecoderStepWrapper(text_model, lm_head, text_config).eval().to(device)

    past_seq_len = 100
    dummy_embeds = torch.randn(1, 1, hidden_size, device=device, dtype=torch.float32)
    dummy_pos = torch.tensor([[past_seq_len]], device=device, dtype=torch.long)
    dummy_past_keys = torch.randn(
        num_layers, 1, num_kv_heads, past_seq_len, head_dim, device=device, dtype=torch.float32
    )
    dummy_past_values = torch.randn(
        num_layers, 1, num_kv_heads, past_seq_len, head_dim, device=device, dtype=torch.float32
    )

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_embeds, dummy_pos, dummy_past_keys, dummy_past_values),
            output_path,
            input_names=["input_embeds", "position_ids", "past_keys", "past_values"],
            output_names=["logits", "present_keys", "present_values"],
            dynamic_axes={
                "input_embeds": {0: "batch"},
                "position_ids": {0: "batch"},
                "past_keys": {1: "batch", 3: "past_seq_len"},
                "past_values": {1: "batch", 3: "past_seq_len"},
                "logits": {0: "batch"},
                "present_keys": {1: "batch", 3: "total_seq_len"},
                "present_values": {1: "batch", 3: "total_seq_len"},
            },
            opset_version=opset_version,
            do_constant_folding=True,
        )

    from .onnx_fixup import fix_reshape_allowzero

    fixed = fix_reshape_allowzero(output_path)
    print(f"Decoder step exported to {output_path} (fixed {fixed} Reshape allowzero attrs)")


def export_decoder_step_static(
    model,
    output_path: str,
    *,
    static_kv_max_tokens: int,
    opset_version: int = 18,
    device: str = "cpu",
):
    """Export a decoder step graph with pre-allocated static KV buffers."""
    text_config = model.config.thinker_config.text_config
    hidden_size = text_config.hidden_size
    num_layers = text_config.num_hidden_layers
    num_kv_heads = text_config.num_key_value_heads
    head_dim = text_config.head_dim

    text_model = model.thinker.model
    lm_head = model.thinker.lm_head
    wrapper = DecoderStepStaticWrapper(text_model, lm_head, text_config).eval().to(device)

    dummy_embeds = torch.randn(1, 1, hidden_size, device=device, dtype=torch.float32)
    dummy_pos = torch.tensor([[100]], device=device, dtype=torch.long)
    dummy_kv_pos = torch.tensor(100, device=device, dtype=torch.long)
    dummy_past_keys = torch.zeros(
        num_layers, 1, num_kv_heads, static_kv_max_tokens, head_dim, device=device, dtype=torch.float32
    )
    dummy_past_values = torch.zeros(
        num_layers, 1, num_kv_heads, static_kv_max_tokens, head_dim, device=device, dtype=torch.float32
    )

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_embeds, dummy_pos, dummy_kv_pos, dummy_past_keys, dummy_past_values),
            output_path,
            input_names=["input_embeds", "position_ids", "kv_pos", "past_keys", "past_values"],
            output_names=["logits", "present_keys", "present_values"],
            opset_version=opset_version,
            do_constant_folding=True,
        )

    from .onnx_fixup import fix_reshape_allowzero

    fixed = fix_reshape_allowzero(output_path)
    print(f"Decoder step static exported to {output_path} (fixed {fixed} Reshape allowzero attrs)")
