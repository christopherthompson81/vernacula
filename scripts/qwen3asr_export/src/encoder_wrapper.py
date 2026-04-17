"""
Wrapper module for Qwen3-ASR audio encoder export to ONNX.

This reimplements the native encoder with trace-friendly tensor operations so
the exported graph avoids runtime-only structures that do not serialize well.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

CONV_WINDOW = 100
TOKENS_PER_WINDOW = 13
ATTN_WINDOW_SIZE = 104


def _conv_out_len(value):
    return (value + 1) // 2


def _get_feat_extract_output_lengths(input_lengths):
    leave = input_lengths % CONV_WINDOW
    value = _conv_out_len(leave)
    value = _conv_out_len(value)
    value = _conv_out_len(value)
    return value + (input_lengths // CONV_WINDOW) * TOKENS_PER_WINDOW


def _encoder_attention(query, key, value, mask, scaling):
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    attn_weights = attn_weights + mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    out = torch.matmul(attn_weights, value)
    return out.transpose(1, 2).reshape(query.shape[0], query.shape[2], -1)


def _encoder_layer_forward(layer, x, attn_mask, scaling, num_heads, head_dim):
    sa = layer.self_attn
    batch, seq, _ = x.shape

    residual = x
    normed = layer.self_attn_layer_norm(x)

    query = sa.q_proj(normed).view(batch, seq, num_heads, head_dim).transpose(1, 2)
    key = sa.k_proj(normed).view(batch, seq, num_heads, head_dim).transpose(1, 2)
    value = sa.v_proj(normed).view(batch, seq, num_heads, head_dim).transpose(1, 2)

    attn_out = _encoder_attention(query, key, value, attn_mask, scaling)
    attn_out = sa.out_proj(attn_out)
    x = residual + attn_out

    residual = x
    normed = layer.final_layer_norm(x)
    x = residual + layer.fc2(F.gelu(layer.fc1(normed)))

    return x


class EncoderWrapper(nn.Module):
    def __init__(self, audio_tower):
        super().__init__()
        self.conv2d1 = audio_tower.conv2d1
        self.conv2d2 = audio_tower.conv2d2
        self.conv2d3 = audio_tower.conv2d3
        self.conv_out = audio_tower.conv_out
        self.positional_embedding = audio_tower.positional_embedding
        self.layers = audio_tower.layers
        self.ln_post = audio_tower.ln_post
        self.proj1 = audio_tower.proj1
        self.proj2 = audio_tower.proj2
        self.act = audio_tower.act
        self.scaling = self.layers[0].self_attn.scaling

        self.d_model = audio_tower.config.d_model
        self.num_heads = audio_tower.config.encoder_attention_heads
        self.head_dim = self.d_model // self.num_heads
        self.output_dim = audio_tower.config.output_dim

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        time_steps = mel.shape[2]

        pad_amount = (CONV_WINDOW - time_steps % CONV_WINDOW) % CONV_WINDOW
        mel = F.pad(mel, (0, pad_amount))
        padded_time = mel.shape[2]
        num_conv_windows = padded_time // CONV_WINDOW

        if mel.shape[0] != 1:
            raise ValueError(f"Expected batch=1, got {mel.shape[0]}")

        x = mel.squeeze(0)
        x = x.reshape(128, num_conv_windows, CONV_WINDOW)
        x = x.permute(1, 0, 2).unsqueeze(1)

        x = F.gelu(self.conv2d1(x))
        x = F.gelu(self.conv2d2(x))
        x = F.gelu(self.conv2d3(x))

        batch, channels, freq_bins, tokens = x.shape
        x = x.permute(0, 3, 1, 2).reshape(batch, tokens, channels * freq_bins)
        x = self.conv_out(x)

        pos_embed = self.positional_embedding(tokens)
        x = x + pos_embed.unsqueeze(0)

        valid_count = _get_feat_extract_output_lengths(time_steps)
        flat = x.reshape(-1, self.d_model)[:valid_count]

        attn_pad = (ATTN_WINDOW_SIZE - valid_count % ATTN_WINDOW_SIZE) % ATTN_WINDOW_SIZE
        flat = F.pad(flat, (0, 0, 0, attn_pad))
        total_padded = valid_count + attn_pad
        num_attn_windows = total_padded // ATTN_WINDOW_SIZE
        x = flat.reshape(num_attn_windows, ATTN_WINDOW_SIZE, self.d_model)

        positions = torch.arange(total_padded, device=mel.device).reshape(num_attn_windows, ATTN_WINDOW_SIZE)
        pad_mask = (positions >= valid_count).to(mel.dtype) * torch.finfo(mel.dtype).min
        attn_mask = pad_mask.unsqueeze(1).unsqueeze(1)

        for layer in self.layers:
            x = _encoder_layer_forward(layer, x, attn_mask, self.scaling, self.num_heads, self.head_dim)

        x = x.reshape(-1, self.d_model)[:valid_count].unsqueeze(0)
        x = self.ln_post(x)
        x = self.act(self.proj1(x))
        x = self.proj2(x)
        return x


def export_encoder(model, output_path: str, opset_version: int = 17, device: str = "cpu"):
    """Export the audio encoder to ONNX."""
    audio_tower = model.thinker.audio_tower
    output_dim = audio_tower.config.output_dim
    wrapper = EncoderWrapper(audio_tower).eval().to(device)

    dummy_mel = torch.randn(1, 128, 997, device=device, dtype=torch.float32)

    with torch.no_grad():
        test_output = wrapper(dummy_mel)
        expected_tokens = _get_feat_extract_output_lengths(997)
        assert test_output.shape == (1, expected_tokens, output_dim)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_mel,),
            output_path,
            input_names=["mel"],
            output_names=["audio_features"],
            dynamic_axes={"mel": {2: "time"}, "audio_features": {1: "enc_time"}},
            opset_version=opset_version,
            do_constant_folding=True,
        )

    from .onnx_fixup import fix_reshape_allowzero

    fixed = fix_reshape_allowzero(output_path)
    print(f"Encoder exported to {output_path} (fixed {fixed} Reshape allowzero attrs)")
