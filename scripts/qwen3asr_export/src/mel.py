"""
Whisper-compatible mel spectrogram computation for Qwen3-ASR profiling.

This runs on the host side and mirrors the upstream validation helper.
"""

from __future__ import annotations

import numpy as np
import torch

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 128
FMIN = 0.0
FMAX = 8000.0


def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    import librosa

    return librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, norm="slaney")


def log_mel_spectrogram(
    audio: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
    fmin: float = FMIN,
    fmax: float = FMAX,
    device: str = "cpu",
) -> torch.Tensor:
    if sample_rate != 16000:
        raise ValueError(f"Expected 16kHz audio, got {sample_rate}Hz")

    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    mel_filters = _mel_filterbank(sample_rate, n_fft, n_mels, fmin, fmax)
    mel_filters_t = torch.from_numpy(mel_filters).float().to(device)

    window = torch.hann_window(n_fft).to(device)
    audio_tensor = torch.from_numpy(audio).float().to(device)

    stft = torch.stft(
        audio_tensor,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    )
    magnitudes = stft.abs() ** 2

    mel_spec = mel_filters_t @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    log_spec = log_spec[:, :-1]
    return log_spec.unsqueeze(0)
