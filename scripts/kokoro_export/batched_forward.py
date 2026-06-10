#!/usr/bin/env python3
"""Experiment: batched Kokoro forward (model surgery) validated against sequential.

Kokoro's forward_with_tokens is batch=1: it sets input_lengths to the full token
length (no padding mask), squeezes pred_dur, and builds a single-item length-regulator
alignment via repeat_interleave. This rewrites the forward to handle a padded batch —
real per-item lengths → masks (the model already threads them), a vectorized duration
expansion, and no squeeze — then checks each item matches the sequential output.

Frames→samples is the model constant 600 (see investigation Run 15).
"""
import numpy as np
import torch
from kokoro import KModel, KPipeline

FRAME = 600  # samples per duration unit at 24 kHz


def capture(km, repo_id, text, voice):
    cap = {}
    orig = km.forward_with_tokens
    def spy(ids, ref_s, speed=1):
        cap.setdefault("ids", ids.detach().clone())
        cap.setdefault("ref_s", ref_s.detach().clone())
        return orig(ids, ref_s, speed)
    km.forward_with_tokens = spy
    pipe = KPipeline(lang_code=voice[0], repo_id=repo_id, model=km)
    for _ in pipe(text, voice=voice):
        break
    km.forward_with_tokens = orig
    return cap["ids"], cap["ref_s"]


@torch.no_grad()
def batched_forward(km, input_ids, ref_s, input_lengths, speed=1.0):
    """input_ids [B, T] (0-padded), ref_s [B, 256], input_lengths [B] real token counts."""
    B, T = input_ids.shape
    dev = km.device
    # text mask: True where padding (matches the model's convention).
    pos = torch.arange(T, device=dev).unsqueeze(0).expand(B, -1)
    text_mask = pos + 1 > input_lengths.unsqueeze(1)            # [B, T] True = pad
    bert_dur = km.bert(input_ids, attention_mask=(~text_mask).int())
    d_en = km.bert_encoder(bert_dur).transpose(-1, -2)
    s = ref_s[:, 128:]
    d = km.predictor.text_encoder(d_en, s, input_lengths, text_mask)
    # The bare prosody LSTM has no mask — pack so padding doesn't leak (esp. the backward
    # direction) into shorter items' real tokens.
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    d_packed = pack_padded_sequence(d, input_lengths.cpu(), batch_first=True, enforce_sorted=False)
    x_packed, _ = km.predictor.lstm(d_packed)
    x, _ = pad_packed_sequence(x_packed, batch_first=True, total_length=input_ids.shape[1])
    duration = km.predictor.duration_proj(x)
    duration = torch.sigmoid(duration).sum(axis=-1) / speed     # [B, T]
    pred_dur = torch.round(duration).clamp(min=1).long()        # [B, T] (no squeeze)
    pred_dur = pred_dur.masked_fill(text_mask, 0)               # padded tokens contribute no frames

    # Vectorized duration expansion → [B, T, max_frames] one-hot alignment.
    frames_per_item = pred_dur.sum(dim=1)                       # [B]
    max_frames = int(frames_per_item.max().item())
    cum = pred_dur.cumsum(dim=1)                                # [B, T] end positions
    start = cum - pred_dur                                      # [B, T] start positions
    fidx = torch.arange(max_frames, device=dev)                # [max_frames]
    aln = (fidx[None, None, :] >= start[:, :, None]) & (fidx[None, None, :] < cum[:, :, None])
    pred_aln_trg = aln.float()                                  # [B, T, max_frames]

    en = d.transpose(-1, -2) @ pred_aln_trg                     # [B, D, max_frames]
    F0_pred, N_pred = km.predictor.F0Ntrain(en, s)
    t_en = km.text_encoder(input_ids, input_lengths, text_mask)
    asr = t_en @ pred_aln_trg
    audio = km.decoder(asr, F0_pred, N_pred, ref_s[:, :128])    # [B, (1,) S]
    if audio.dim() == 3:
        audio = audio.squeeze(1)
    return audio, pred_dur, frames_per_item


def f0n_batched(km, en, s, frames_per_item):
    """ProsodyPredictor.F0Ntrain with the shared frame-LSTM packed by real frame length,
    so padding frames don't leak (backward) into shorter items."""
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    pred = km.predictor
    xt = en.transpose(-1, -2)                                   # [B, frames, D]
    packed = pack_padded_sequence(xt, frames_per_item.cpu(), batch_first=True, enforce_sorted=False)
    pred.shared.flatten_parameters()
    xs, _ = pred.shared(packed)
    xs, _ = pad_packed_sequence(xs, batch_first=True, total_length=en.shape[-1])  # [B, frames, D]
    F0 = xs.transpose(-1, -2)
    for block in pred.F0:
        F0 = block(F0, s)
    F0 = pred.F0_proj(F0)
    N = xs.transpose(-1, -2)
    for block in pred.N:
        N = block(N, s)
    N = pred.N_proj(N)
    return F0.squeeze(1), N.squeeze(1)


def logspec_l1(a, b):
    n = min(len(a), len(b))
    win = torch.hann_window(1024)
    def lm(x):
        return torch.log(torch.stft(torch.tensor(x[:n], dtype=torch.float32), 1024, 256,
                                    window=win, return_complex=True).abs() + 1e-5)
    return float((lm(a) - lm(b)).abs().mean())


def main():
    repo = "hexgrad/Kokoro-82M"
    km = KModel(repo_id=repo, disable_complex=True).eval()

    texts = [
        "Hello, this is a short test.",
        "In the quiet hours before dawn the old keeper climbed the spiral stair and lit the great lamp.",
    ]
    inputs = [capture(km, repo, t, "af_heart") for t in texts]

    # Sequential reference (the real forward).
    seq = []
    for ids, ref_s in inputs:
        a, pd = km.forward_with_tokens(ids, ref_s, 1.0)
        seq.append((a.squeeze().cpu().numpy(), pd.cpu().numpy()))
        print(f"[seq] ids={ids.shape[1]} pred_dur_sum={int(pd.sum())} audio={len(seq[-1][0])}")

    # Build a padded batch.
    lens = [ids.shape[1] for ids, _ in inputs]
    maxT = max(lens)
    B = len(inputs)
    ids_batch = torch.zeros((B, maxT), dtype=torch.long, device=km.device)
    for i, (ids, _) in enumerate(inputs):
        ids_batch[i, : ids.shape[1]] = ids[0]
    ref_batch = torch.cat([r for _, r in inputs], dim=0)
    len_batch = torch.tensor(lens, device=km.device)

    audio_b, pdur_b, frames_b = batched_forward(km, ids_batch, ref_batch, len_batch, 1.0)
    print(f"[batch] audio={tuple(audio_b.shape)} frames_per_item={frames_b.tolist()}")

    for i in range(B):
        n = int(frames_b[i].item()) * FRAME
        a_item = audio_b[i, :n].cpu().numpy()
        a_seq = seq[i][0]
        # pred_dur parity (the real lengths)
        pd_b = pdur_b[i, : lens[i]].cpu().numpy()
        pd_seq = seq[i][1]
        dur_match = np.array_equal(pd_b, pd_seq)
        L = logspec_l1(a_seq, a_item)
        print(f"  item {i}: len_seq={len(a_seq)} len_batch={n} pred_dur_match={dur_match} "
              f"logspecL1={L:.4f}")

    # ── Diagnostic: equal-length batch (item 0 duplicated, no padding) ──
    print("[diag] equal-length batch (item0 x2, no padding):")
    ids0, ref0 = inputs[0]
    ids_eq = ids0.repeat(2, 1)
    ref_eq = torch.cat([ref0, ref0], dim=0)
    len_eq = torch.tensor([lens[0], lens[0]], device=km.device)
    a_eq, pd_eq, fr_eq = batched_forward(km, ids_eq, ref_eq, len_eq, 1.0)
    for i in range(2):
        n = int(fr_eq[i].item()) * FRAME
        L = logspec_l1(seq[0][0], a_eq[i, :n].cpu().numpy())
        print(f"  eq item {i}: logspecL1 vs seq = {L:.4f}")


if __name__ == "__main__":
    main()
