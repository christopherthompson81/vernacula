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


def _audio_encoder_v2_forward_real_freqs(self, x, x_len):
    """Replacement for `s3tokenizer.model_v2.AudioEncoderV2.forward`.

    Upstream calls `torch.view_as_real(freqs_cis)` inline (line 344).
    Even with our converted real-format buffer, that op fails ONNX
    export ("view_as_real is only supported for complex tensors").
    The cos/sin computed from view_as_real are then **never used**
    downstream — they look like leftover refactoring debris. The
    actual rotary application happens inside MultiHeadAttention via
    `apply_rotary_emb(q, k, freqs_cis=freqs_cis)`, which we patch
    separately.

    This replacement is byte-for-byte identical to upstream EXCEPT
    that it omits the dead view_as_real / cos / sin block.
    """
    T = x.shape[-1]
    from s3tokenizer.model_v2 import make_non_pad_mask, mask_to_bias
    mask = make_non_pad_mask(x_len, T).unsqueeze(1)
    x = torch.nn.functional.gelu(self.conv1(x * mask))
    x_len = (x_len + 2 - 1 * (3 - 1) - 1) // self.stride + 1
    x_slen = (T + 2 - 1 * (3 - 1) - 1) // self.stride + 1
    mask = make_non_pad_mask(x_len, x_slen).unsqueeze(1)
    x = torch.nn.functional.gelu(self.conv2(x * mask))
    x_len = (x_len + 2 - 1 * (3 - 1) - 1) // 2 + 1
    x_slen = (x_slen + 2 - 1 * (3 - 1) - 1) // self.stride + 1
    mask = make_non_pad_mask(x_len, x_slen).unsqueeze(1)
    x = x.permute(0, 2, 1)  # (B, T // 2, n_state)
    freqs_cis = self.freqs_cis.to(x.device)  # already real-format (T, D, 2)
    mask_pad = mask.transpose(1, 2)
    mask = mask_to_bias(mask, x.dtype)

    for block in self.blocks:
        x = block(x, mask.unsqueeze(1), mask_pad, freqs_cis[:x.size(1)])

    return x, x_len


@contextmanager
def patched_rotary_for_export(root_module):
    """Patch rotary embeddings under the given root module for ONNX export.

    Three-part patch:

    1. Walk root_module to find every submodule with a complex
       `freqs_cis` attribute; replace with real-format (T, D, 2).
    2. Monkey-patch `s3tokenizer.model_v2.apply_rotary_emb` (the
       function called by MultiHeadAttention) to read the real format.
    3. Monkey-patch `AudioEncoderV2.forward` to skip its inline
       `torch.view_as_real(freqs_cis)` call — that's dead code anyway,
       the cos/sin it computes are never consumed.

    All three are restored on context exit.
    """
    import s3tokenizer.model_v2 as _m2
    from s3tokenizer.model_v2 import AudioEncoderV2

    original_apply = _m2.apply_rotary_emb
    saved_freqs = _convert_freqs_cis_buffers(root_module)

    saved_aev2_forward = {}
    for m in root_module.modules():
        if isinstance(m, AudioEncoderV2):
            saved_aev2_forward[m] = m.forward
            m.forward = types.MethodType(_audio_encoder_v2_forward_real_freqs, m)

    _m2.apply_rotary_emb = _apply_rotary_emb_real
    try:
        yield root_module
    finally:
        _m2.apply_rotary_emb = original_apply
        for module, original in saved_freqs.items():
            module.freqs_cis = original
        for module, original in saved_aev2_forward.items():
            module.forward = original


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


# ── Cond-decoder patches for chatterbox.s3gen.flow + mel2wav ─────────
# Upstream provides `flow.inference(...)` and `mel2wav.inference(...)`
# that together do what Vlad's vendored ConditionalDecoder
# reimplemented in ~360 LOC. To use them, we need to bypass three
# upstream ONNX-unfriendly patterns:
#
#   1. `@torch.inference_mode()` on `flow.inference`,
#      `CausalConditionalCFM.forward`, `mel2wav.inference`. Inference
#      mode poisons tensors with a flag that JIT trace's
#      save_for_backward chokes on. (Same fix as the FSQ codebook in
#      the S3Tokenizer chain — replace with @torch.no_grad.)
#
#   2. `mel2wav._stft` uses `torch.stft(return_complex=True)` then
#      `view_as_real`. Same issue as S3Tokenizer.log_mel_spectrogram.
#      Replacement uses `return_complex=False` and indexes [..., 0]/
#      [..., 1] for real/imag.
#
#   3. `mel2wav._istft` builds `torch.complex(real, img)` then calls
#      `torch.istft(...)`. ONNX has neither op. Replacement uses our
#      existing `ISTFT` class (in `_chatterbox_internals.py`) which
#      implements iSTFT via Conv1d + scatter_add window_sumsquare.


def _strip_inference_mode(method):
    """Wrap a method so its @torch.inference_mode decorator is skipped.

    Returns the raw function from the decorated method; callers should
    rebind via `types.MethodType`. The decorator chain wraps a _DecoratorContextManager
    around the original function, accessible via `__wrapped__`.
    """
    while hasattr(method, "__wrapped__"):
        method = method.__wrapped__
    return method


def _mel2wav_stft_real_format(self, x):
    """Replacement for `mel2wav._stft` — return_complex=False, no view_as_real."""
    spec = torch.stft(
        x,
        self.istft_params["n_fft"],
        self.istft_params["hop_len"],
        self.istft_params["n_fft"],
        window=self.stft_window.to(x.device),
        return_complex=False,
    )  # (B, F, TT, 2)
    return spec[..., 0], spec[..., 1]


def _make_mel2wav_istft_via_our_istft(istft_module):
    """Build a replacement for `mel2wav._istft` that uses our ISTFT
    (Conv1d + scatter_add window_sumsquare) instead of torch.istft +
    torch.complex.

    Closes over the istft module so the patched method has access to it.
    """

    def _mel2wav_istft_patched(self, magnitude, phase):
        magnitude = torch.clip(magnitude, max=1e2)
        real = magnitude * torch.cos(phase)
        img = magnitude * torch.sin(phase)
        # Stack real/imag along channel dim → (B, 2F, TT) — the format
        # our ISTFT class expects (recombine_magnitude_phase).
        spec = torch.cat([real, img], dim=1)
        return istft_module(spec)

    return _mel2wav_istft_patched


# ── SineGen noise/phase determinism (DIAGNOSTIC PROBE) ────────────────
# Upstream `SineGen.forward` has two random sources:
#   1. `phase_vec = Uniform(-pi, pi).sample(...)` — random initial phase
#      per call
#   2. `noise = noise_amp * torch.randn_like(sine_waves)` — random noise
#      mixed into the sine signal
# Both are stochastic per call in PyTorch eager. ONNX traces them as
# RandomNormal/RandomUniform ops which ORT runs with its own RNG —
# uncorrelated with PyTorch's. The result: eager and ONNX use different
# random samples even with same input → outputs diverge in the noise
# component, which the resblocks may amplify.
#
# This patch zeros both random sources for diagnostic purposes — if
# dec parity becomes good with this active, NSF random is the cause
# and the production fix is to precompute the noise as a buffer.

def _sinegen_forward_deterministic(self, f0):
    """Deterministic replacement for SineGen.forward — zero noise, zero phase."""
    import numpy as np
    import torch

    F_mat = torch.zeros(f0.size(0), self.harmonic_num + 1, f0.size(-1),
                        device=f0.device)
    for i in range(self.harmonic_num + 1):
        F_mat[:, i:i + 1, :] = f0 * (i + 1) / self.sampling_rate

    theta_mat = 2 * np.pi * (torch.cumsum(F_mat, dim=-1) % 1)
    # phase_vec ZEROED (was: u_dist.sample(...) — random per call)
    phase_vec = torch.zeros(
        f0.size(0), self.harmonic_num + 1, 1,
        device=f0.device, dtype=F_mat.dtype,
    )

    sine_waves = self.sine_amp * torch.sin(theta_mat + phase_vec)
    uv = self._f02uv(f0)
    # noise ZEROED (was: noise_amp * torch.randn_like(sine_waves))
    noise = torch.zeros_like(sine_waves)
    sine_waves = sine_waves * uv + noise
    return sine_waves, uv, noise


@contextmanager
def patched_sinegen_deterministic(mel2wav):
    """Make SineGen deterministic for diagnostic/parity work."""
    sine_gen = mel2wav.m_source.l_sin_gen
    original = sine_gen.forward
    sine_gen.forward = types.MethodType(_sinegen_forward_deterministic, sine_gen)
    try:
        yield sine_gen
    finally:
        sine_gen.forward = original


# ── flow.inference dynamic-shape rewrite ──────────────────────────────
# Upstream `flow.inference` does Python-int arithmetic on tensor
# shapes:
#     mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]
#     conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], ...)
#     ...
#     feat = feat[:, :, mel_len1:]
# These Python ints get baked into the ONNX trace as static constants.
# At runtime, inputs with different lengths fail with shape-mismatch
# errors (LeftShape vs RightShape on Mul). The export's `dynamic_axes`
# spec doesn't help because the shape values were captured pre-graph.
#
# Replacement uses tensor-shape-preserving construction: `h.new_zeros`
# with `h.shape[1]` (a SymInt during trace), and `narrow` slicing
# that keeps the start index symbolic.

# NOTE on dynamic-shape limits:
# The cond decoder uses `diffusers.models.attention_processor.Attention`
# (HuggingFace diffusers library) for its mid_blocks / up_blocks /
# down_blocks attention. We patched solve_euler + flow.inference's
# mask construction, which fixed several baked time-dim values, but
# additional shape-baking happens inside diffusers' Attention (and
# possibly elsewhere). Patching diffusers' attention has too large a
# surface area (multiple attention processors: SDPA, xformers,
# default; many code paths each).
#
# Pragmatic resolution: cond decoder ONNX is *trace-shape-only* for
# now — pad input to a fixed length at runtime (the C# orchestrator
# can pad speech_tokens to a max chunk length, decode, trim audio).
# A future fix could use `torch.onnx.dynamo_export` (PyTorch 2.x's
# newer ONNX path) which is designed for dynamic shapes — left as
# E5+ work.


def _solve_euler_dynamic(self, x, t_span, mu, mask, spks, cond):
    """Replacement for CausalConditionalCFM.solve_euler.

    Upstream allocates the CFG _in buffers via
    `torch.zeros([2, 80, x.size(2)], ...)` — `x.size(2)` becomes a
    Python int, baking the time dimension into the trace. Replace
    with `torch.cat([zeros_like(x), zeros_like(x)])` style
    construction that propagates symbolic shapes through ONNX.
    """
    t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
    t = t.unsqueeze(dim=0)

    sol = []

    # Build CFG-pair tensors via cat-of-zeros_like to keep last dim symbolic.
    x_in = torch.cat([torch.zeros_like(x), torch.zeros_like(x)], dim=0)
    mask_in = torch.cat([torch.zeros_like(mask), torch.zeros_like(mask)], dim=0)
    mu_in = torch.cat([torch.zeros_like(mu), torch.zeros_like(mu)], dim=0)
    t_in = torch.zeros(2, device=x.device, dtype=x.dtype)
    spks_in = torch.cat([torch.zeros_like(spks), torch.zeros_like(spks)], dim=0)
    cond_in = torch.cat([torch.zeros_like(cond), torch.zeros_like(cond)], dim=0)

    for step in range(1, len(t_span)):
        x_in[:] = x
        mask_in[:] = mask
        mu_in[0] = mu
        t_in[:] = t.unsqueeze(0)
        spks_in[0] = spks
        cond_in[0] = cond
        dphi_dt = self.forward_estimator(
            x_in, mask_in,
            mu_in, t_in,
            spks_in,
            cond_in,
        )
        dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
        dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt)
        x = x + dt * dphi_dt
        t = t + dt
        sol.append(x)
        if step < len(t_span) - 1:
            dt = t_span[step + 1] - t

    return sol[-1].float()


def _flow_inference_dynamic(self,
                            token, token_len,
                            prompt_token, prompt_token_len,
                            prompt_feat, prompt_feat_len,
                            embedding,
                            finalize):
    print(f"[patched _flow_inference_dynamic called, token shape={tuple(token.shape)}]")  # DEBUG
    """Drop-in replacement for chatterbox.s3gen.flow.MaskedDiffWithXvec.inference.

    Mathematically identical to upstream for the single-batch
    no-padding case (which is our entire deployment). Mask construction
    uses shape-derived `ones_like` to keep ONNX dynamic — upstream's
    `make_pad_mask(...)` does `lengths.max()` which collapses to a
    Python int and bakes the time dim.
    """
    import torch.nn.functional as F

    embedding = F.normalize(embedding, dim=1)
    embedding = self.spk_embed_affine_layer(embedding)

    token = torch.cat([prompt_token, token], dim=1)
    # Embed first so we have a tensor with the time dim symbolic.
    token = self.input_embedding(
        torch.clamp(token, min=0, max=self.input_embedding.num_embeddings - 1)
    )
    # All-True mask of shape (B, T, 1), derived from `token`'s shape via
    # ones_like on a sliced view. Avoids make_pad_mask's lengths.max()
    # int-collapse.
    mask = torch.ones_like(token[:, :, :1])
    token = token * mask

    # Encoder needs token_len. We pass the int-level length through but
    # use token's symbolic shape for downstream mask construction.
    h, h_lengths = self.encoder(token, prompt_token_len + (token_len.new_tensor([0]) + token.shape[1]))
    if finalize is False:
        h = h[:, :-self.pre_lookahead_len * self.token_mel_ratio]

    h = self.encoder_proj(h)
    prompt_len = prompt_feat.shape[1]

    # conds: zeros of shape (B, total_len, output_size) where
    # total_len = h.shape[1]. Constructed via a new_zeros that takes
    # h.shape[0:1] + Size((output_size,)) — symbolic through ONNX.
    conds_tail_len_tensor = h.shape[1] - prompt_len
    conds_head = prompt_feat  # (B, prompt_len, output_size)
    # tail zeros derived from h's shape, using symbolic subtraction
    # via narrow on h itself.
    conds_tail = torch.zeros_like(h.narrow(1, 0, conds_tail_len_tensor)[..., :self.output_size])
    conds = torch.cat([conds_head, conds_tail], dim=1).transpose(1, 2)

    # Mask for decoder: all-True (B, 1, total_len) derived from h.
    mask2 = torch.ones_like(h[:, :, :1]).transpose(1, 2)  # (B, 1, total_len)

    feat, _ = self.decoder(
        mu=h.transpose(1, 2).contiguous(),
        mask=mask2,
        spks=embedding,
        cond=conds,
        n_timesteps=10,
    )
    # Slice off the prompt portion via narrow to keep dim symbolic.
    feat = feat.narrow(2, prompt_len, feat.shape[2] - prompt_len)
    return feat.float(), None


@contextmanager
def patched_cond_decoder_for_export(s3gen, istft_module):
    """Patch upstream s3gen.flow + mel2wav for ONNX export.

    - Strips `@torch.inference_mode()` from flow.inference,
      flow.decoder.forward (CausalConditionalCFM), mel2wav.inference.
    - Patches mel2wav._stft and mel2wav._istft to avoid complex tensors,
      reusing the supplied `istft_module` (our scatter_add ISTFT).
    """
    flow = s3gen.flow
    mel2wav = s3gen.mel2wav

    # Save originals
    saved = {
        "flow.inference": flow.inference,
        "flow.decoder.forward": flow.decoder.forward,
        "flow.decoder.solve_euler": flow.decoder.solve_euler,
        "mel2wav.inference": mel2wav.inference,
        "mel2wav._stft": mel2wav._stft,
        "mel2wav._istft": mel2wav._istft,
    }

    # Strip inference_mode and swap flow.inference for the
    # dynamic-shape rewrite. Also patch solve_euler to use shape-
    # preserving zero allocation (zeros_like instead of torch.zeros
    # with .size() ints).
    flow.inference = types.MethodType(_flow_inference_dynamic, flow)
    flow.decoder.forward = types.MethodType(
        _strip_inference_mode(flow.decoder.forward.__func__), flow.decoder
    )
    flow.decoder.solve_euler = types.MethodType(_solve_euler_dynamic, flow.decoder)
    # NOTE: cond decoder estimator uses diffusers Attention internally,
    # which has its own shape-baking. Not patched here — see module
    # docstring above _solve_euler_dynamic for the deferred-work note.
    mel2wav.inference = types.MethodType(
        _strip_inference_mode(mel2wav.inference.__func__), mel2wav
    )

    # Swap STFT/iSTFT
    mel2wav._stft = types.MethodType(_mel2wav_stft_real_format, mel2wav)
    mel2wav._istft = types.MethodType(
        _make_mel2wav_istft_via_our_istft(istft_module), mel2wav
    )

    try:
        yield s3gen
    finally:
        flow.inference = saved["flow.inference"]
        flow.decoder.forward = saved["flow.decoder.forward"]
        flow.decoder.solve_euler = saved["flow.decoder.solve_euler"]
        mel2wav.inference = saved["mel2wav.inference"]
        mel2wav._stft = saved["mel2wav._stft"]
        mel2wav._istft = saved["mel2wav._istft"]


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
