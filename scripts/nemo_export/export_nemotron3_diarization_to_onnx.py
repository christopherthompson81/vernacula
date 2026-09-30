#!/usr/bin/env python3
"""Export nvidia/Nemotron-3-Diarization (.nemo) to the ONNX contract SortformerStreamer runs.

Nemotron-3-Diarization is a Streaming Sortformer successor (8 speakers, a 31-layer RoPE
transformer encoder, a learnable silence embedding, 10 ms output). The streaming loop around
the network is the same algorithm as diar_streaming_sortformer_4spk-v2.1's, so the graph keeps
that model's six inputs and three outputs and adds one:

    in   chunk              [1, T_mel, 128]     mel frames, exact length (NOT zero-padded)
    in   chunk_lengths      [1]  int64
    in   spkcache           [1, T_cache, 512]
    in   spkcache_lengths   [1]  int64
    in   fifo               [1, T_fifo, 512]
    in   fifo_lengths       [1]  int64
    out  spkcache_fifo_chunk_preds     [1, T, 8]     80 ms -- drives the cache/FIFO state
    out  chunk_pre_encode_embs         [1, T_ch, 512]
    out  chunk_pre_encode_lengths      [1]
    out  spkcache_fifo_chunk_preds_hr  [1, 8*T, 8]   10 ms -- what gets reported

The 80 ms output is NeMo's own `downsample_preds(hr, 8)`, which is exactly what
`forward_streaming_step` feeds `streaming_update`; the 10 ms output is what it reports.

Everything the runtime needs that is not a tensor -- the streaming schedule, the compression
constants, and the learnable silence embedding that replaces v2.1's running mean -- is written
to the model's metadata_props under `vernacula.diar.*`, so the artifact describes itself.

⚠ NEEDS NeMo MAIN, not a release. PyPI nemo-toolkit 3.0.0 cannot even build this checkpoint
(it requires a `transformer_encoder` the config does not have). See
docs/investigations/nemotron3_diarization_onnx_investigation.md for the venv recipe.

Usage:

    python scripts/nemo_export/export_nemotron3_diarization_to_onnx.py \\
        --nemo /path/to/Nemotron-3-Diarization.nemo \\
        --output /path/to/nemotron3_diarization/nemotron-3-diarization.onnx
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np

# The card's "very high latency (offline)" configuration, in 80 ms frames. This is the schedule
# the runtime uses; it is recorded in the metadata, not baked into the graph (all time axes stay
# dynamic).
OFFLINE_SCHEDULE = dict(spkcache_len=264, fifo_len=40, chunk_len=340,
                        chunk_right_context=40, spkcache_update_period=300)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nemo", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip-check", action="store_true",
                   help="Skip the ORT-vs-torch parity check after export.")
    return p.parse_args()


def load_model(nemo_path: Path):
    from nemo.collections.asr.models import SortformerEncLabelModel
    model = SortformerEncLabelModel.restore_from(str(nemo_path), map_location="cpu")
    model.eval()
    return model


def check_export_assumptions(model) -> None:
    """The export-mode encoder below re-implements NeMo's TransformerEncoder for exactly one
    configuration. Refuse anything else rather than export a graph that silently differs."""
    enc = model.encoder
    blk = enc.layers[0].attn
    want = {
        "encoder": (type(enc).__name__, "TransformerEncoder"),
        "pre_encode": (type(enc.pre_encode).__name__, "FeatureStacking"),
        "self_attention_model": (enc.self_attention_model, "rope"),
        "attn_mode": (enc.attn_mode, "full"),
        "causal_tail_len": (enc.causal_tail_len, 0),
        "xscale": (enc.xscale, None),
        "encoder.out_proj": (enc.out_proj, None),
        "qk_norm": (blk.qk_norm, False),
        "transformer_encoder": (model.transformer_encoder, None),
    }
    bad = {k: got for k, (got, exp) in want.items() if got != exp}
    if bad:
        raise SystemExit(f"Export-mode encoder does not cover this checkpoint: {bad}")


def build_wrapper(model):
    """Wrap the model in the streaming contract with an ONNX-traceable encoder.

    Two things in NeMo's encoder cannot be traced to ONNX, so they are re-expressed here:

    * attention is PyTorch FlexAttention driven by `create_block_mask` -- for this
      checkpoint's `attn_mode: full` with no causal tail, the mask_mod is padding-only
      (`kv_idx < length`), which is SDPA with a boolean key-padding mask;
    * FeatureStacking computes its pad and reshape from Python ints, which the tracer bakes
      at the example length. The runtime pads the mel chunk to a multiple of 8 itself (zero
      rows, exactly what FeatureStacking would add), so here a `-1` reshape suffices.

    `verify_against_nemo` checks this against NeMo's own FlexAttention forward.
    """
    import torch
    import torch.nn.functional as F
    from torch import nn

    check_export_assumptions(model)
    sub = model.encoder.subsampling_factor

    def attention(attn, x, key_keep):
        B, T, _ = x.shape
        H, D = attn.n_heads, attn.head_dim
        qkv = attn.w_qkv(x).view(B, T, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = attn.rope(q, k)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=key_keep)
        return attn.out_proj(out.transpose(1, 2).reshape(B, T, attn.d_model))

    def encode(enc, x, length):
        # forward_internal for rope / full / bypass_pre_encode=True, minus the block mask.
        x = enc.embed_norm(x)
        T = x.shape[1]
        key_keep = (torch.arange(T, device=x.device).unsqueeze(0) < length.unsqueeze(1))
        key_keep = key_keep[:, None, None, :]          # [B, 1, 1, T] -> broadcast over heads/queries
        for layer in enc.layers:
            x = x + attention(layer.attn, layer.norm1(x), key_keep)
            x = x + layer.ffn(layer.norm2(x))
        return enc.final_norm(x)                        # (B, T, D); NeMo transposes twice around here

    class Nemotron3DiarExport(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, chunk, chunk_lengths, spkcache, spkcache_lengths, fifo, fifo_lengths):
            m = self.inner
            sm = m.sortformer_modules
            b, _, c = chunk.shape
            chunk_embs = m.encoder.pre_encode.proj(chunk.reshape(b, -1, c * sub))
            chunk_emb_lengths = ((chunk_lengths + sub - 1) // sub).to(torch.int64)
            # Batch 1 with every buffer at its exact length, so a plain concat is NeMo's
            # concat_and_pad. Only the chunk can carry padding (<8 mel rows, from the multiple-
            # of-8 rule), and it is last, so the summed length masks it.
            embs = torch.cat([spkcache, fifo, chunk_embs], dim=1)
            lengths = spkcache_lengths + fifo_lengths + chunk_emb_lengths
            enc_out = encode(m.encoder, embs, lengths)
            enc_out = sm.encoder_proj(enc_out)
            preds_hr = m.forward_infer(enc_out, lengths)
            preds = sm.downsample_preds(preds_hr, m.upsample_factor)
            return preds, chunk_embs, chunk_emb_lengths, preds_hr

    return Nemotron3DiarExport(model).eval()


def reference_forward(model, chunk, chunk_lengths, spkcache, spkcache_lengths, fifo, fifo_lengths):
    """The same four outputs through NeMo's own modules (FlexAttention, FeatureStacking)."""
    import torch
    m = model
    chunk_embs, chunk_emb_lengths = m._call_pre_encode(chunk, chunk_lengths)
    embs = torch.cat([spkcache, fifo, chunk_embs], dim=1)
    lengths = spkcache_lengths + fifo_lengths + chunk_emb_lengths.to(torch.int64)
    enc, enc_lengths = m.frontend_encoder(processed_signal=embs, processed_signal_length=lengths,
                                          bypass_pre_encode=True)
    preds_hr = m.forward_infer(enc, enc_lengths)
    preds = m.sortformer_modules.downsample_preds(preds_hr, m.upsample_factor)
    return preds, chunk_embs, chunk_emb_lengths, preds_hr


def metadata(model) -> dict[str, str]:
    sm = model.sortformer_modules
    if not model.high_resolution or not sm.use_learnable_sil_emb:
        raise SystemExit("This exporter expects a high-resolution model with a learnable silence "
                         "embedding (Nemotron-3-Diarization). Use export_sortformer_nemo_to_onnx.py "
                         "for the v2.x Streaming Sortformer checkpoints.")
    sil = sm.learnable_sil_emb.detach().cpu().numpy().astype("<f4")
    md = {
        "vernacula.diar.contract": "sortformer-hr-1",
        "vernacula.diar.num_speakers": sm.n_spk,
        "vernacula.diar.emb_dim": sm.fc_d_model,
        "vernacula.diar.n_mels": model.encoder._feat_in,
        "vernacula.diar.subsampling": model.encoder.subsampling_factor,
        "vernacula.diar.upsample_factor": model.upsample_factor,
        "vernacula.diar.spkcache_sil_frames_per_spk": sm.spkcache_sil_frames_per_spk,
        "vernacula.diar.sil_threshold": sm.sil_threshold,
        "vernacula.diar.scores_boost_latest": sm.scores_boost_latest,
        "vernacula.diar.pred_score_threshold": sm.pred_score_threshold,
        "vernacula.diar.strong_boost_rate": sm.strong_boost_rate,
        "vernacula.diar.weak_boost_rate": sm.weak_boost_rate,
        "vernacula.diar.min_pos_scores_rate": sm.min_pos_scores_rate,
        "vernacula.diar.learnable_sil_emb_f32le_b64": base64.b64encode(sil.tobytes()).decode(),
    }
    md.update({f"vernacula.diar.{k}": v for k, v in OFFLINE_SCHEDULE.items()})
    return {k: str(v) for k, v in md.items()}


def example_inputs(torch, n_mels: int, emb_dim: int, chunk_frames: int, cache: int, fifo: int):
    g = torch.Generator().manual_seed(0)
    return (
        torch.randn(1, chunk_frames, n_mels, generator=g),
        torch.tensor([chunk_frames], dtype=torch.int64),
        torch.randn(1, cache, emb_dim, generator=g),
        torch.tensor([cache], dtype=torch.int64),
        torch.randn(1, fifo, emb_dim, generator=g),
        torch.tensor([fifo], dtype=torch.int64),
    )


def pad_to_multiple(torch, chunk, sub: int):
    """What the runtime does before every call: zero rows up to a multiple of `sub`."""
    t = chunk.shape[1]
    pad = (-t) % sub
    return torch.nn.functional.pad(chunk, (0, 0, 0, pad)) if pad else chunk


def parity_check(onnx_path: Path, model, wrapper, torch, n_mels: int, emb_dim: int) -> list[dict]:
    """Three-way, at shapes the runtime actually produces: NeMo's own modules on the UNPADDED
    chunk, the export wrapper and ORT on the padded one. Covers the first chunk (empty cache
    and FIFO) and short tails that are not a multiple of 8.

    Valid frames only: the reference has no padded tail rows, and the graph's padded rows are
    masked to zero, so comparing them would test nothing."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    names = ["chunk", "chunk_lengths", "spkcache", "spkcache_lengths", "fifo", "fifo_lengths"]
    sub = model.encoder.subsampling_factor
    cases = [
        ("first chunk", 380 * 8, 0, 0),
        ("steady", 380 * 8, 264, 40),
        ("tail, 1234 mel", 1234, 264, 40),
        ("tiny tail, 13 mel", 13, 264, 40),
    ]
    rows = []
    for label, t, c, f in cases:
        raw = example_inputs(torch, n_mels, emb_dim, t, c, f)
        padded = (pad_to_multiple(torch, raw[0], sub),) + raw[1:]
        with torch.inference_mode():
            ref = [x.numpy() for x in reference_forward(model, *raw)]
            wrp = [x.numpy() for x in wrapper(*padded)]
        ort_out = sess.run(None, dict(zip(names, [x.numpy() for x in padded])))
        n80 = c + f + (t + sub - 1) // sub
        valid = {"preds": n80, "embs": (t + sub - 1) // sub, "lengths": None, "preds_hr": n80 * sub}
        row = {"case": label}
        for i, name in enumerate(["preds", "embs", "lengths", "preds_hr"]):
            n = valid[name]
            r, w, o = (a if n is None else a[:, :n] for a in (ref[i], wrp[i], ort_out[i]))
            if not (r.shape == w.shape == o.shape):
                raise SystemExit(f"{label}: {name} shapes nemo {r.shape} wrapper {w.shape} ort {o.shape}")
            row[f"{name}:wrapper-nemo"] = float(np.abs(w.astype(np.float64) - r).max())
            row[f"{name}:ort-nemo"] = float(np.abs(o.astype(np.float64) - r).max())
        rows.append(row)
        print(f"  {label:18} " + "  ".join(f"{k}={v:.1e}" for k, v in row.items() if k != "case"))
    return rows


def main() -> None:
    args = parse_args()
    import torch
    import onnx

    out = args.output.expanduser().resolve()
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite.")
    out.parent.mkdir(parents=True, exist_ok=True)

    model = load_model(args.nemo.expanduser().resolve())
    md = metadata(model)
    wrapper = build_wrapper(model)
    n_mels, emb_dim = int(md["vernacula.diar.n_mels"]), int(md["vernacula.diar.emb_dim"])

    # Trace with a NON-empty cache and FIFO so no branch specialises on a zero-length axis.
    inputs = example_inputs(torch, n_mels, emb_dim, 380 * 8, 264, 40)
    torch.onnx.export(
        wrapper, inputs, str(out),
        input_names=["chunk", "chunk_lengths", "spkcache", "spkcache_lengths", "fifo", "fifo_lengths"],
        output_names=["spkcache_fifo_chunk_preds", "chunk_pre_encode_embs",
                      "chunk_pre_encode_lengths", "spkcache_fifo_chunk_preds_hr"],
        dynamic_axes={
            "chunk": {1: "time_chunk"},
            "spkcache": {1: "time_cache"},
            "fifo": {1: "time_fifo"},
            "spkcache_fifo_chunk_preds": {1: "time_out"},
            "chunk_pre_encode_embs": {1: "time_pre_encode"},
            "spkcache_fifo_chunk_preds_hr": {1: "time_out_hr"},
        },
        opset_version=args.opset,
        dynamo=False,
        do_constant_folding=True,
    )

    proto = onnx.load(str(out))
    del proto.metadata_props[:]
    for k, v in md.items():
        proto.metadata_props.add(key=k, value=v)
    onnx.save(proto, str(out))

    report = {"nemo_file": args.nemo.name, "opset": args.opset,
              "metadata": {k: v for k, v in md.items() if not k.endswith("_b64")}}
    if not args.skip_check:
        print("parity (max |ORT - torch|):")
        report["parity"] = parity_check(out, model, wrapper, torch, n_mels, emb_dim)
    report_path = out.with_suffix(out.suffix + ".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(out)
    print(report_path)


if __name__ == "__main__":
    main()
