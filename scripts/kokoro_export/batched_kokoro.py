#!/usr/bin/env python3
"""Variable-length BATCHED export of Kokoro-82M, at the model's own fidelity.

`KModel.forward_with_tokens` is batch=1: it sets input_lengths to the full token length (no
padding mask), squeezes pred_dur, and builds the length-regulator alignment with
repeat_interleave. This module rewrites that path to take a right-padded batch — and, crucially,
to make the padding *invisible*, so a batched item matches its solo render.

Why that needs care (docs/kokoro_onnx_investigation.md Runs 26-32):

  · The model is NOT deterministic. SourceModuleHnNSF draws torch.randn_like for the
    harmonic-plus-noise excitation every call, so two identical solo renders already differ by
    ~0.13 log-spec L1. That is the floor; "exact" means reaching it, not zero.

  · The padding error is a STEP FUNCTION, not a gradient: 2 frames of padding do as much damage
    as 160. Minimising padding (length bucketing) buys throughput, never fidelity.

  · Three separate leaks, all of which must be closed:
      1. AdaIN normalises over TIME, so padding frames pollute the per-item mean/variance
         -> mask the statistics.
      2. The bidirectional LSTMs read padding backwards into real tokens, which corrupts
         pred_dur itself -> pack them. `predictor.lstm` is the load-bearing one; without it
         durations shift with batch composition and word alignment breaks.
      3. AdaIN1d is `(1 + gamma) * norm(x) + beta`. Re-zeroing inside the InstanceNorm is undone
         by the `+ beta` OUTSIDE it, so padding carries beta into every downstream conv
         -> re-zero after the whole AdaIN1d.

  ⚠ Re-zero predictor convs but NOT the generator's. The predictor keeps the frame axis last
    throughout, so a last-dim mask is well defined; the generator mixes [B, T, 1] layouts and an
    iSTFT, where the same mask is wrong and makes the audio worse the more padding there is.

Frames->samples is the model constant 600 (investigation Run 15).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

FRAME = 600  # samples per duration unit at 24 kHz


class _MaskState:
    """Per-item real fraction of the padded frame axis, or None to disable masking.

    Held on the class rather than passed down because the masking has to reach inside
    stock kokoro modules (AdaIN, the decoder blocks) that we do not control the signature of.
    Set once per forward, from traced tensors, so it survives ONNX tracing.
    """
    frac = None


def _mask_of(x):
    """[B,1,T] 1.0 over each item's real span. The frame rate changes through the decoder
    (stride, upsampling), so the mask is rebuilt from the real FRACTION at whatever length
    the tensor has here."""
    if _MaskState.frac is None:
        return None
    T = x.shape[-1]
    real = torch.clamp((_MaskState.frac * T).round().long(), min=1)
    return (torch.arange(T, device=x.device)[None, :] < real[:, None]).to(x.dtype)[:, None, :]


class MaskedInstanceNorm1d(nn.Module):
    """Drop-in for nn.InstanceNorm1d(affine=True) that ignores padding in its statistics.
    Identical to the stock module when masking is disabled."""

    def __init__(self, orig: nn.InstanceNorm1d):
        super().__init__()
        self.eps = orig.eps
        self.weight = nn.Parameter(orig.weight.detach().clone())
        self.bias = nn.Parameter(orig.bias.detach().clone())

    def forward(self, x):
        m = _mask_of(x)
        if m is None:
            return nn.functional.instance_norm(x, weight=self.weight, bias=self.bias, eps=self.eps)
        n = m.sum(-1, keepdim=True).clamp(min=1)
        mean = (x * m).sum(-1, keepdim=True) / n
        var = (((x - mean) * m) ** 2).sum(-1, keepdim=True) / n
        y = (x - mean) / torch.sqrt(var + self.eps)
        return (y * self.weight[None, :, None] + self.bias[None, :, None]) * m


def _rezero(mod, inp, out):
    if not torch.is_tensor(out) or out.dim() != 3:
        return out
    m = _mask_of(out)
    return out if m is None else out * m


def install_masking(kmodel) -> tuple[int, int]:
    """Swap in masked norms and register the re-zero hooks. Returns (norms, hooks)."""
    norms = 0
    for mod in kmodel.modules():
        for name, child in list(mod.named_children()):
            if isinstance(child, nn.InstanceNorm1d):
                setattr(mod, name, MaskedInstanceNorm1d(child))
                norms += 1
    hooks = 0
    for m in kmodel.modules():
        if type(m).__name__ == "AdaIN1d":       # the `+ beta` leak
            m.register_forward_hook(_rezero)
            hooks += 1
    for m in kmodel.predictor.modules():        # predictor only — see module docstring
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            m.register_forward_hook(_rezero)
            hooks += 1
    return norms, hooks


class BatchedKokoroONNX(nn.Module):
    """Batched forward.

    Inputs:
        input_ids     : LongTensor  [B, T]   right-padded token ids
        ref_s         : FloatTensor [B, 256] per-item style vector
        speed         : FloatTensor [1]
        input_lengths : LongTensor  [B]      real token count per item

    Outputs:
        audio    : FloatTensor [B, max_frames * 600]  each item valid for its own
                   pred_dur.sum() * 600 samples; the remainder is padding, discard it.
        pred_dur : LongTensor  [B, T]  per-token frames, 0 on padded tokens. Identical to the
                   solo render, so word alignment is unaffected by batch composition.
    """

    def __init__(self, kmodel):
        super().__init__()
        self.kmodel = kmodel

    def forward(self, input_ids, ref_s, speed, input_lengths):
        km = self.kmodel
        B, T = input_ids.shape
        text_mask = torch.arange(T, device=input_ids.device)[None, :] + 1 > input_lengths[:, None]

        bert_dur = km.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = km.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = km.predictor.text_encoder(d_en, s, input_lengths, text_mask)

        # Packed: the backward direction would otherwise read padding into real tokens and
        # shift pred_dur with batch composition.
        packed = pack_padded_sequence(d, input_lengths.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = km.predictor.lstm(packed)
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=T)

        duration = torch.sigmoid(km.predictor.duration_proj(x)).sum(-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().masked_fill(text_mask, 0)

        frames = pred_dur.sum(1)
        max_frames = frames.max()
        _MaskState.frac = frames.float() / max_frames.float()

        # Length regulator, vectorized (repeat_interleave is single-item only).
        cum = pred_dur.cumsum(1)
        start = cum - pred_dur
        fi = torch.arange(max_frames, device=input_ids.device)
        aln = ((fi[None, None, :] >= start[:, :, None]) & (fi[None, None, :] < cum[:, :, None])).float()

        en = d.transpose(-1, -2) @ aln

        # F0Ntrain, with its frame-level LSTM packed by real frame count.
        pred = km.predictor
        packed_f = pack_padded_sequence(en.transpose(-1, -2), frames.cpu(),
                                        batch_first=True, enforce_sorted=False)
        xs, _ = pred.shared(packed_f)
        xs, _ = pad_packed_sequence(xs, batch_first=True, total_length=en.shape[-1])
        F0 = xs.transpose(-1, -2)
        for block in pred.F0:
            F0 = block(F0, s)
        F0 = pred.F0_proj(F0).squeeze(1)
        N = xs.transpose(-1, -2)
        for block in pred.N:
            N = block(N, s)
        N = pred.N_proj(N).squeeze(1)

        asr = km.text_encoder(input_ids, input_lengths, text_mask) @ aln
        audio = km.decoder(asr, F0, N, ref_s[:, :128])
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        return audio, pred_dur
