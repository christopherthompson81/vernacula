#!/usr/bin/env python3
"""Minimal targeted patches that make upstream chatterbox modules ONNX-exportable.

Goal: drop the 600+ lines of vendored model code in `_chatterbox_internals.py`
(S3Tokenizer family + helpers) by patching only the specific upstream ops
that don't survive torch.onnx.export. Each patch is scoped via a
context manager so it does NOT mutate global module state.

Each patch lists:
  * The exact upstream symbol it overrides
  * The op it works around and why ONNX can't handle the original
  * A parity invariant: the patched code path is numerically identical to
    the unpatched version up to floating-point round-off, asserted by the
    parity test in test_chatterbox_parity.py.
"""
from __future__ import annotations

import types
from contextlib import contextmanager

import torch
import torch.nn.functional as F


# ── S3Tokenizer.log_mel_spectrogram ────────────────────────────────────
# Upstream calls `torch.stft(..., return_complex=True)` then does
# `stft.abs()**2`. ONNX symbolic doesn't support complex tensors as of
# opset 20 ("STFT does not currently support complex types"). The fix
# is to use `return_complex=False`, which returns a real tensor of shape
# (*, freq, time, 2) where the last dim is [real, imag], and compute
# magnitudes manually: `real**2 + imag**2`. Mathematically identical
# to `complex_stft.abs()**2`.

S3_HOP = 160  # matches chatterbox.models.s3tokenizer.s3tokenizer.S3_HOP


def _log_mel_spectrogram_real_stft(self, audio, padding=0):
    """Drop-in replacement that uses return_complex=False.

    Numerically equivalent to the upstream original; see module-level
    docstring for why ONNX requires this.
    """
    if not torch.is_tensor(audio):
        audio = torch.from_numpy(audio)

    audio = audio.to(self.device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))

    # Real-format STFT: shape (*, freq, time, 2), last dim = [real, imag]
    stft_real = torch.stft(
        audio, self.n_fft, S3_HOP,
        window=self.window.to(self.device),
        return_complex=False,
    )
    real = stft_real[..., 0]
    imag = stft_real[..., 1]
    # `stft.abs()**2` on the complex tensor equals `real**2 + imag**2`
    # on the real-format tensor. Identity, not approximation.
    magnitudes_full = real * real + imag * imag
    magnitudes = magnitudes_full[..., :-1]

    mel_spec = self._mel_filters.to(self.device) @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


# ── S3Tokenizer.forward ────────────────────────────────────────────────
# Upstream's forward iterates over `wavs` as a Python list, computing
# `log_mel_spectrogram` per element and concatenating with
# `s3tokenizer.utils.padding` (which uses `torch.nn.utils.rnn.pad_sequence`).
# ONNX doesn't export `aten::pad_sequence` (it operates on Python lists,
# not tensors). Replacement: accept a (B, N) tensor directly, batch
# through log_mel_spectrogram + quantize.
#
# Parity note: log_mel_spectrogram applies a global
# `torch.maximum(log_spec, log_spec.max() - 8.0)` normalization. Per-wav
# (upstream) this is computed within each utterance. Batched (ours)
# this reduces across the whole batch. For batch=1 the behaviors are
# identical — confirmed by the parity test. For batch>1 the behavior
# diverges slightly, but our export and runtime use batch=1
# (autoregressive TTS), so the divergence is moot for the current
# deployment.

def _forward_batched_for_export(self, wavs, max_len=None):
    """Drop-in replacement for S3Tokenizer.forward.

    Accepts a (B, N) tensor directly — no Python list iteration, no
    pad_sequence call. log_mel_spectrogram and quantize are both
    batch-aware, so this is a no-op transformation for batch=1.
    """
    if not torch.is_tensor(wavs):
        wavs = torch.from_numpy(wavs)
    if wavs.dim() == 1:
        wavs = wavs.unsqueeze(0)
    wavs = wavs.to(self.device)

    mel = self.log_mel_spectrogram(wavs)  # (B, n_mels, T)
    if max_len is not None:
        mel = mel[..., :max_len * 4]
    mel_lens = torch.full(
        (mel.shape[0],), mel.shape[-1],
        dtype=torch.int32, device=mel.device,
    )

    speech_tokens, speech_token_lens = self.quantize(mel, mel_lens)
    return speech_tokens.long(), speech_token_lens.long()


# ── Rotary embeddings: s3tokenizer.model_v2.{precompute_freqs_cis,
#    apply_rotary_emb} ────────────────────────────────────────────────
# Upstream represents rotary as `freqs_cis = torch.polar(ones, freqs)`
# (a complex64 tensor) and consumes it in apply_rotary_emb via
# `torch.view_as_real(freqs_cis)`. ONNX has no scalar type for
# ComplexFloat, so both ops fail to symbolic-convert.
#
# Replacement: store freqs as a real (T, D, 2) tensor with last-dim
# layout [cos, sin]. Mathematically identical — `polar(1, θ)` is
# `cos(θ) + i sin(θ)`, and `view_as_real(complex64)` returns the same
# (cos, sin) pair. apply_rotary_emb reads the (T, D, 2) tensor
# directly.
#
# The patch has two parts:
#   1. Replace the stored `self.freqs_cis` on each AudioEncoderV2-style
#      module (converts complex → real-format in-place).
#   2. Swap `s3tokenizer.model_v2.apply_rotary_emb` to a version that
#      reads the real-format tensor.

def _apply_rotary_emb_real(xq, xk, freqs_cis):
    """Real-tensor replacement for s3tokenizer.model_v2.apply_rotary_emb.

    freqs_cis must be a real (T, D, 2) tensor where last dim is [cos, sin]
    — produced by `_freqs_cis_to_real(...)` below.
    """
    cos = freqs_cis[..., 0]  # (T, D)
    sin = freqs_cis[..., 1]
    cos = cos.unsqueeze(0).unsqueeze(2).to(xq.dtype)
    sin = sin.unsqueeze(0).unsqueeze(2).to(xq.dtype)

    D = xq.shape[-1]
    half_l, half_r = xq[:, :, :, :D // 2], xq[:, :, :, D // 2:]
    xq_r = torch.cat((-half_r, half_l), dim=-1)

    D = xk.shape[-1]
    half_l, half_r = xk[:, :, :, :D // 2], xk[:, :, :, D // 2:]
    xk_r = torch.cat((-half_r, half_l), dim=-1)

    return xq * cos + xq_r * sin, xk * cos + xk_r * sin


def _freqs_cis_to_real(freqs_cis_complex):
    """Convert a complex64 freqs_cis tensor (T, D) to real (T, D, 2)
    with last-dim [cos, sin]. Done with `.real` / `.imag` which produce
    real tensors directly. Safe to run on a loaded model's buffer
    BEFORE the ONNX trace — the conversion is a one-time CPU op, not
    part of the traced graph.
    """
    real = freqs_cis_complex.real.contiguous()
    imag = freqs_cis_complex.imag.contiguous()
    return torch.stack([real, imag], dim=-1)


def _convert_freqs_cis_buffers(root_module):
    """Walk a model and convert any complex `freqs_cis` attribute on
    submodules to real-format (T, D, 2). Returns a dict of
    {module: original_complex_freqs_cis} for restoration.
    """
    saved = {}
    for m in root_module.modules():
        if hasattr(m, "freqs_cis") and torch.is_tensor(m.freqs_cis) and m.freqs_cis.is_complex():
            saved[m] = m.freqs_cis
            m.freqs_cis = _freqs_cis_to_real(m.freqs_cis)
    return saved


@contextmanager
def patched_rotary_for_export(root_module):
    """Patch rotary embeddings under the given root module for ONNX export.

    1. Walk root_module to find every submodule with a complex `freqs_cis`
       attribute; replace with real-format (T, D, 2).
    2. Monkey-patch s3tokenizer.model_v2.apply_rotary_emb to read the
       real format. Restore on exit.
    """
    import s3tokenizer.model_v2 as _m2
    original_apply = _m2.apply_rotary_emb
    saved_freqs = _convert_freqs_cis_buffers(root_module)
    _m2.apply_rotary_emb = _apply_rotary_emb_real
    try:
        yield root_module
    finally:
        _m2.apply_rotary_emb = original_apply
        for module, original in saved_freqs.items():
            module.freqs_cis = original


# ── chatterbox.models.s3gen.xvector.DenseLayer.forward ────────────────
# Upstream's DenseLayer.forward branches on `if len(x.shape) == 2` to
# handle both 2D and 3D inputs. The branch produces an ONNX `If` node
# whose output channel size is unknown to the symbolic checker, which
# then refuses BatchNorm1d ("ONNX export of batch_norm for unknown
# channel size"). Probe (probe_dense_shape.py) showed only one
# DenseLayer instance in the speech_encoder pipeline and it's always
# called with 2D input. Specialize the forward to the 2D branch and
# the symbolic check passes.
#
# Parity note: the if-branch only changes shape handling; mathematics
# is identical for inputs of any rank. Specializing to 2D drops the
# rank-3 path that's never taken in practice. Verified by parity:
# upstream eager 3D input would now fail, but no downstream caller
# uses 3D, so the specialization is safe at inference.

class _DenseLayerExportShim(torch.nn.Module):
    """Drop-in replacement for chatterbox DenseLayer used during ONNX
    export only. Same math, no opaque ops.

    Upstream's DenseLayer wraps Conv1d-with-kernel-1 plus a
    `if len(x.shape) == 2:` branch, then a Sequential containing
    BatchNorm1d (`affine=False`). Three nested ONNX-shape-inference
    failures cascade through:
      - The if-branch produces an `If` node hiding the channel dim
      - The squeeze(-1) is shape-conditional, producing another `If`
      - Even after replacing both with explicit Linear+Reshape, ONNX
        shape inference still can't propagate channel info to BatchNorm,
        which refuses with "unknown channel size".
    Workaround: inline the entire BatchNorm math as arithmetic ops.
    BatchNorm1d at inference (affine=False) is just
        y = (x - running_mean) / sqrt(running_var + eps)
    — pure ops with statically-shaped buffers, no symbolic gymnastics.
    Weights and BN running stats are copied byte-for-byte.
    """

    def __init__(self, upstream_dense):
        super().__init__()
        # Linear weights from the Conv1d-kernel-1
        weight = upstream_dense.linear.weight.detach()  # (C_out, C_in, 1)
        c_out, c_in = weight.shape[0], weight.shape[1]
        bias = upstream_dense.linear.bias.detach() if upstream_dense.linear.bias is not None else None
        self.linear = torch.nn.Linear(c_in, c_out, bias=bias is not None)
        with torch.no_grad():
            self.linear.weight.copy_(weight.squeeze(-1))
            if bias is not None:
                self.linear.bias.copy_(bias)

        # Inline BatchNorm1d: pull running stats from the upstream BN.
        # We assume affine=False (verified for chatterbox xvector dense).
        # If a different config appears, this needs to learn weight/bias.
        bn = upstream_dense.nonlinear.batchnorm
        assert not bn.affine, "DenseLayerExportShim assumes BatchNorm1d(affine=False)"
        self.register_buffer("bn_running_mean", bn.running_mean.detach().clone())
        self.register_buffer("bn_running_var", bn.running_var.detach().clone())
        self.bn_eps = float(bn.eps)

    def forward(self, x):
        y = self.linear(x)  # (B, C_out)
        # Inline BatchNorm: y = (y - mean) / sqrt(var + eps)
        y = (y - self.bn_running_mean) * torch.rsqrt(self.bn_running_var + self.bn_eps)
        return y


@contextmanager
def patched_dense_layer_for_export(speaker_encoder):
    """Swap every DenseLayer under speaker_encoder for the export shim.

    Restoration is by parent-attribute reassignment — captured at patch
    time so the original module returns to its slot on context exit.
    """
    from chatterbox.models.s3gen.xvector import DenseLayer
    swaps = []  # list of (parent, attr_name, original_module)
    for parent in speaker_encoder.modules():
        for name, child in parent.named_children():
            if isinstance(child, DenseLayer):
                shim = _DenseLayerExportShim(child).to(next(child.parameters()).device).eval()
                swaps.append((parent, name, child))
                setattr(parent, name, shim)
    try:
        yield speaker_encoder
    finally:
        for parent, name, original in swaps:
            setattr(parent, name, original)


@contextmanager
def patched_s3tokenizer_for_export(tokenizer):
    """Temporarily swap log_mel_spectrogram + forward for export-friendly versions.

    Usage:
        with patched_s3tokenizer_for_export(chatterbox_model.s3gen.tokenizer):
            torch.onnx.export(my_wrapper, ...)

    Originals are restored on context exit; trace and eager runs
    outside the `with` block see the unpatched methods.
    """
    orig_log_mel = tokenizer.log_mel_spectrogram
    orig_forward = tokenizer.forward
    tokenizer.log_mel_spectrogram = types.MethodType(_log_mel_spectrogram_real_stft, tokenizer)
    tokenizer.forward = types.MethodType(_forward_batched_for_export, tokenizer)
    try:
        yield tokenizer
    finally:
        tokenizer.log_mel_spectrogram = orig_log_mel
        tokenizer.forward = orig_forward
