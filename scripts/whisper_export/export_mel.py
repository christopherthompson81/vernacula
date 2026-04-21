#!/usr/bin/env python3
"""
Export a Whisper log-mel spectrogram as a small ONNX graph.

Motivation
----------
The onnx-community Whisper turbo bundle ships encoder + decoder graphs but
leaves mel feature extraction to the caller. In Vernacula's pipeline the
mel runs on CPU (MathNet FFT + Slaney filterbank + log clamp) and profiling
showed it's ~23 % of ASR time on a 600 s file. Moving it into an ONNX graph
lets ORT run it on the same EP as the encoder, eliminating the CPU bottleneck
and the CPU->GPU copy of raw audio.

Numerics
--------
Matches HuggingFace's WhisperFeatureExtractor (which is what onnx-community's
tokenizer/preprocessor configs reference). Constants:

  sample_rate = 16 000
  n_fft       = 400
  hop_length  = 160
  n_mels      = 128
  chunk_samples = 480 000 (30 s)

Pipeline:
  1. STFT: Hann window, center=True, pad_mode="reflect"
  2. Power spectrum: |X|^2
  3. Slaney mel filterbank (HTK=False)
  4. log10(max(power, 1e-10))
  5. Clamp to [max_log - 8, max_log]
  6. Normalise: (x + 4) / 4  ->  values end up ~[-1, 1]

Output shape: [batch, 128, 3000].

Usage
-----
    pip install -r requirements.txt
    python export_mel.py --out mel.onnx
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -- Slaney hz <-> mel (matches AudioUtils.HzToMelSlaney / MelToHzSlaney) ----

def _hz_to_mel_slaney(hz: np.ndarray) -> np.ndarray:
    f_min, f_sp = 0.0, 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0
    return np.where(
        hz >= min_log_hz,
        min_log_mel + np.log(hz / min_log_hz) / logstep,
        (hz - f_min) / f_sp,
    )


def _mel_to_hz_slaney(mel: np.ndarray) -> np.ndarray:
    f_min, f_sp = 0.0, 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0
    return np.where(
        mel >= min_log_mel,
        min_log_hz * np.exp(logstep * (mel - min_log_mel)),
        f_min + f_sp * mel,
    )


def _slaney_mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    n_freq = n_fft // 2 + 1
    fft_freq = np.linspace(0.0, sr / 2.0, n_freq, dtype=np.float64)

    mel_min = _hz_to_mel_slaney(np.array(0.0))
    mel_max = _hz_to_mel_slaney(np.array(sr / 2.0))
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = _mel_to_hz_slaney(mel_pts)

    fb = np.zeros((n_mels, n_freq), dtype=np.float64)
    for i in range(n_mels):
        lower = (fft_freq - hz_pts[i]) / (hz_pts[i + 1] - hz_pts[i])
        upper = (hz_pts[i + 2] - fft_freq) / (hz_pts[i + 2] - hz_pts[i + 1])
        fb[i] = np.maximum(0.0, np.minimum(lower, upper))
        # Slaney energy-normalisation so each band integrates to ~2.
        enorm = 2.0 / (hz_pts[i + 2] - hz_pts[i])
        fb[i] *= enorm
    return fb.astype(np.float32)


# -- Graph module -----------------------------------------------------------

class WhisperLogMel(nn.Module):
    SAMPLE_RATE = 16_000
    N_FFT = 400
    HOP_LENGTH = 160
    N_MELS = 128
    CHUNK_SAMPLES = 480_000   # 30 s
    CHUNK_FRAMES = CHUNK_SAMPLES // HOP_LENGTH   # 3000

    def __init__(self) -> None:
        super().__init__()
        # torch.stft exports to the ONNX STFT op, which CUDAExecutionProvider
        # has no kernel for; it falls back to host and triggers PCIe bounces,
        # dominating wall time. Measured on this graph: 148 ms/call CPU-EP,
        # 132 ms/call CUDA-EP-with-fallback, vs our hand-rolled CPU FFT at
        # ~58 ms/call. Same issue VoxLingua hit — see
        # scripts/voxlingua107_export/src/conv_stft.py for the prior art.
        #
        # Fix: build the STFT from two Conv1D ops (cos and sin kernels,
        # pre-windowed) so ORT sees only primitives it heavily optimises.
        window = torch.hann_window(self.N_FFT, periodic=True).float()
        n_freq = self.N_FFT // 2 + 1
        k = torch.arange(n_freq,     dtype=torch.float32).unsqueeze(1)      # freq-bin index  [n_freq, 1]
        n = torch.arange(self.N_FFT, dtype=torch.float32).unsqueeze(0)      # time index      [1, n_fft]
        basis = (2.0 * math.pi / self.N_FFT) * k * n                        # [n_freq, n_fft]
        # DFT: X[k] = sum_n x[n] * exp(-j * 2π * k * n / N)
        #   -> real kernel is cos(basis), imag kernel is -sin(basis).
        cos_kernel = (torch.cos(basis) * window.unsqueeze(0)).unsqueeze(1)  # [n_freq, 1, n_fft]
        sin_kernel = (-torch.sin(basis) * window.unsqueeze(0)).unsqueeze(1)
        self.register_buffer("cos_kernel", cos_kernel, persistent=False)
        self.register_buffer("sin_kernel", sin_kernel, persistent=False)

        fb = _slaney_mel_filterbank(self.SAMPLE_RATE, self.N_FFT, self.N_MELS)
        self.register_buffer("mel_filters", torch.from_numpy(fb))           # [N_MELS, n_freq]

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: [B, CHUNK_SAMPLES] -> [B, 1, CHUNK_SAMPLES]
        x = audio.unsqueeze(1)
        # torch.stft(center=True, pad_mode="reflect") reflection-pads by n_fft//2
        # on each side. Replicate manually.
        pad = self.N_FFT // 2
        x = F.pad(x, (pad, pad), mode="reflect")

        real = F.conv1d(x, self.cos_kernel, stride=self.HOP_LENGTH)         # [B, n_freq, frames+1]
        imag = F.conv1d(x, self.sin_kernel, stride=self.HOP_LENGTH)
        # Whisper drops the last frame so T_frames = T_samples / hop.
        real = real[..., :-1]
        imag = imag[..., :-1]

        power = real.pow(2) + imag.pow(2)                                   # [B, n_freq, N_FRAMES]

        mel = self.mel_filters @ power                                      # [B, N_MELS, N_FRAMES]
        log_mel = torch.log10(torch.clamp(mel, min=1e-10))

        max_log = log_mel.amax(dim=(-2, -1), keepdim=True)
        log_mel = torch.maximum(log_mel, max_log - 8.0)
        return (log_mel + 4.0) / 4.0


# -- Export + numerical sanity check ----------------------------------------

def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="mel.onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    model = WhisperLogMel().eval()

    # Fixed shape: [1, 480000]. Whisper encoders always consume exactly 30 s
    # per chunk anyway, so we don't need a dynamic time axis on the input.
    dummy = torch.zeros(1, WhisperLogMel.CHUNK_SAMPLES)

    with torch.no_grad():
        out = model(dummy)
    want = (1, WhisperLogMel.N_MELS, WhisperLogMel.CHUNK_FRAMES)
    assert out.shape == want, f"expected {want}, got {tuple(out.shape)}"
    print(f"[mel-export] reference shape OK: {tuple(out.shape)}")

    torch.onnx.export(
        model,
        dummy,
        args.out,
        input_names=["audio"],
        output_names=["mel"],
        # Batch is dynamic so one loaded graph serves B=1 ... B=8 without re-export.
        dynamic_axes={"audio": {0: "batch"}, "mel": {0: "batch"}},
        opset_version=args.opset,
    )

    # Newer torch.onnx exporters spill all initializers into a <out>.data
    # sidecar by default, even for small graphs where the weights easily fit
    # inside the protobuf 2 GiB limit. mel.onnx has a ~100 KB filterbank and
    # a 400-float Hann window total -- not worth the second file. Load it
    # back with the sidecar attached, then rewrite with weights inlined so
    # we ship a single self-contained asset.
    import onnx
    from onnx.external_data_helper import convert_model_from_external_data
    m = onnx.load(args.out)                             # auto-resolves the .data sidecar
    convert_model_from_external_data(m)                 # inline all initializers
    onnx.save(m, args.out)                              # overwrite, now self-contained
    sidecar = Path(args.out + ".data")
    if sidecar.exists():
        sidecar.unlink()

    size_kb = Path(args.out).stat().st_size / 1024.0
    print(f"[mel-export] wrote {args.out} ({size_kb:.1f} KB, opset {args.opset}, self-contained)")


if __name__ == "__main__":
    _main()
