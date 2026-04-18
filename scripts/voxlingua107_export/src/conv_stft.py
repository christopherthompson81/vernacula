"""Conv1D-based drop-in replacement for SpeechBrain's STFT module.

SpeechBrain's `speechbrain.processing.features.STFT` calls `torch.stft`,
which exports to the ONNX `STFT` op. `CUDAExecutionProvider` has no
kernel for that op, so it falls back to the host and dominates wall
time on a CUDA session (85.6% on a 30 s clip in Phase 1 profiling).

This module computes the same spectrogram with two `Conv1D` passes
(one with cos basis, one with sin basis) and emits a tensor in the
exact shape SpeechBrain's STFT produces:
    [batch, n_frames, n_freq, 2]  where the last dim is (real, imag).

Using only Conv1D / Pad / Stack / Transpose, all of which have
CUDAExecutionProvider kernels.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1DStft(nn.Module):
    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window: torch.Tensor,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: bool = True,
    ):
        super().__init__()
        if not onesided:
            raise NotImplementedError("only one-sided STFT is supported")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center
        self.pad_mode = pad_mode

        if window.numel() != win_length:
            raise ValueError(
                f"window has {window.numel()} samples but win_length={win_length}")

        # If win_length < n_fft, the window is centered in an n_fft-long frame
        # (matching torch.stft's behavior).
        if win_length < n_fft:
            pad_left = (n_fft - win_length) // 2
            pad_right = n_fft - win_length - pad_left
            full_window = F.pad(window, (pad_left, pad_right))
        else:
            full_window = window

        n_freq = n_fft // 2 + 1
        k = torch.arange(n_freq, dtype=torch.float32).unsqueeze(1)   # [n_freq, 1]
        n = torch.arange(n_fft,  dtype=torch.float32).unsqueeze(0)   # [1, n_fft]
        basis = (2.0 * math.pi / n_fft) * k * n                       # [n_freq, n_fft]
        cos_kernel = torch.cos(basis) * full_window.unsqueeze(0)     # [n_freq, n_fft]
        sin_kernel = -torch.sin(basis) * full_window.unsqueeze(0)    # forward DFT sign

        # Conv1D weight shape: [out_channels, in_channels, kernel_size]
        self.register_buffer("cos_kernel", cos_kernel.unsqueeze(1), persistent=False)
        self.register_buffer("sin_kernel", sin_kernel.unsqueeze(1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T] (SpeechBrain's STFT entry path for the LID pipeline).
        # Emulate torch.stft(center=True) by reflect-padding n_fft//2 on each
        # side before framing.
        if self.center:
            pad = self.n_fft // 2
            x = F.pad(x.unsqueeze(1), (pad, pad), mode=self.pad_mode).squeeze(1)

        x = x.unsqueeze(1)  # [B, 1, T_padded]
        real = F.conv1d(x, self.cos_kernel, stride=self.hop_length)  # [B, n_freq, T_frames]
        imag = F.conv1d(x, self.sin_kernel, stride=self.hop_length)

        # SpeechBrain returns [B, T_frames, n_freq, 2]; match that layout.
        stacked = torch.stack([real, imag], dim=-1)      # [B, n_freq, T_frames, 2]
        return stacked.transpose(1, 2).contiguous()      # [B, T_frames, n_freq, 2]


def replace_speechbrain_stft(speechbrain_stft: nn.Module) -> Conv1DStft:
    """Build a Conv1DStft matching SpeechBrain's configured STFT module."""
    return Conv1DStft(
        n_fft=speechbrain_stft.n_fft,
        hop_length=speechbrain_stft.hop_length,
        win_length=speechbrain_stft.win_length,
        window=speechbrain_stft.window.detach().clone(),
        center=speechbrain_stft.center,
        pad_mode=speechbrain_stft.pad_mode,
        onesided=speechbrain_stft.onesided,
    )
