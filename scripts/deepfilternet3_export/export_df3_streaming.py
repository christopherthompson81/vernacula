"""
Export DeepFilterNet3 in streaming/chunked mode.

The standard export (export.py) processes all T frames in a single ONNX call.
GRU layers must process T steps sequentially, so T=60,000 gives poor GPU utilisation.

This script exports three models (enc, erb_dec, df_dec) with explicit GRU hidden
state as inputs and outputs.  The C# inference loop then processes audio in chunks
of ~C frames (~1 s each), carrying the GRU state across chunks.

Architecture overview
---------------------
enc.onnx:
  inputs : feat_erb [1,1,T,32], feat_spec [1,2,T,96], h_enc [1,1,256]
  outputs: e0,e1,e2,e3,emb,c0,lsnr  (shapes as before)  + h_enc [1,1,256]

erb_dec.onnx:
  inputs : emb [1,T,512], e3,e2,e1,e0, h_erb [2,1,256]
  outputs: m [1,1,T,32] + h_erb [2,1,256]

df_dec.onnx:
  inputs : emb [1,T,512], c0 [1,64,T,96], h_df [2,1,256]
  outputs: coefs [1,T,96,10] + h_df [2,1,256]

Usage (from repo root):
    .venv-deepfilternet3/bin/python scripts/export_df3_streaming.py \\
        --out external/deepfilternet_onnx_streaming --opset 14

The output models replace the originals for chunked C# inference.
"""

import argparse
import os
from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from torch import Tensor

from df.enhance import ModelParams, df_features, init_df
from df.model import ModelParams  # noqa: ensure config is loaded
from libdf import DF


# ── Wrapper: Encoder with explicit GRU state ─────────────────────────────────

class EncStreaming(nn.Module):
    """
    Encoder wrapper that exposes the emb_gru hidden state.
    Replaces:   emb, _ = self.emb_gru(emb)
    With:       emb, h_out = self.emb_gru(emb, h_in)
    """
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(
        self,
        feat_erb: Tensor,   # [1, 1, T, 32]
        feat_spec: Tensor,  # [1, 2, T, 96]
        h_enc: Tensor,      # [1, 1, 256]  — GRU hidden state in
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        e = self.enc
        e0 = e.erb_conv0(feat_erb)
        e1 = e.erb_conv1(e0)
        e2 = e.erb_conv2(e1)
        e3 = e.erb_conv3(e2)
        c0 = e.df_conv0(feat_spec)
        c1 = e.df_conv1(c0)
        cemb = c1.permute(0, 2, 3, 1).flatten(2)
        cemb = e.df_fc_emb(cemb)
        emb = e3.permute(0, 2, 3, 1).flatten(2)
        emb = e.combine(emb, cemb)
        emb, h_enc_out = e.emb_gru(emb, h_enc)
        lsnr = e.lsnr_fc(emb) * e.lsnr_scale + e.lsnr_offset
        return e0, e1, e2, e3, emb, c0, lsnr, h_enc_out


# ── Wrapper: ErbDecoder with explicit GRU state ───────────────────────────────

class ErbDecStreaming(nn.Module):
    """
    ErbDecoder wrapper that exposes the emb_gru hidden state.
    """
    def __init__(self, erb_dec):
        super().__init__()
        self.erb_dec = erb_dec

    def forward(
        self,
        emb: Tensor,   # [1, T, 512]
        e3: Tensor,    # [1, 64, T, 8]
        e2: Tensor,    # [1, 64, T, 8]
        e1: Tensor,    # [1, 64, T, 16]
        e0: Tensor,    # [1, 64, T, 32]
        h_erb: Tensor, # [2, 1, 256]  — GRU hidden state in
    ) -> Tuple[Tensor, Tensor]:
        d = self.erb_dec
        b, _, t, f8 = e3.shape
        emb, h_erb_out = d.emb_gru(emb, h_erb)
        emb = emb.view(b, t, f8, -1).permute(0, 3, 1, 2)
        e3 = d.convt3(d.conv3p(e3) + emb)
        e2 = d.convt2(d.conv2p(e2) + e3)
        e1 = d.convt1(d.conv1p(e1) + e2)
        m = d.conv0_out(d.conv0p(e0) + e1)
        return m, h_erb_out


# ── Wrapper: DfDecoder with explicit GRU state ────────────────────────────────

class DfDecStreaming(nn.Module):
    """
    DfDecoder wrapper that exposes the df_gru hidden state.
    """
    def __init__(self, df_dec):
        super().__init__()
        self.df_dec = df_dec

    def forward(
        self,
        emb: Tensor,  # [1, T, 512]
        c0: Tensor,   # [1, 64, T, 96]
        h_df: Tensor, # [2, 1, 256]  — GRU hidden state in
    ) -> Tuple[Tensor, Tensor]:
        d = self.df_dec
        b, t, _ = emb.shape
        c, h_df_out = d.df_gru(emb, h_df)
        if d.df_skip is not None:
            c = c + d.df_skip(emb)
        c0 = d.df_convp(c0).permute(0, 2, 3, 1)
        c = d.df_out(c)
        c = c.view(b, t, d.df_bins, d.df_out_ch) + c0
        return c, h_df_out


# ── Export helpers ────────────────────────────────────────────────────────────

def _export_and_check(
    path: str,
    model: nn.Module,
    inputs: tuple,
    input_names: list,
    output_names: list,
    dynamic_axes: dict,
    opset: int,
) -> list:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with torch.no_grad():
        pt_outputs = model(*inputs)
    if not isinstance(pt_outputs, (list, tuple)):
        pt_outputs = (pt_outputs,)

    torch.onnx.export(
        model=deepcopy(model),
        f=path,
        args=inputs,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        keep_initializers_as_inputs=False,
    )
    print(f"  Exported → {path}")

    # Parity check
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    feed = {k: v.numpy() for k, v in zip(input_names, inputs)}
    ort_outputs = sess.run(None, feed)
    for name, pt_out, ort_out in zip(output_names, pt_outputs, ort_outputs):
        err = np.abs(pt_out.numpy() - ort_out).max()
        ok = "OK  " if err < 1e-4 else "WARN"
        print(f"    [{ok}] {name}: max_err={err:.2e}")

    return list(pt_outputs)


# ── Main export ───────────────────────────────────────────────────────────────

@torch.no_grad()
def export_streaming(out_dir: str, chunk_t: int = 100, opset: int = 14):
    print(f"Loading DeepFilterNet3 model…")
    model, df_state, _, epoch = init_df(None, log_level="NONE")
    model = deepcopy(model).to("cpu").eval()
    p = ModelParams()

    print(f"  Epoch: {epoch}, chunk_T: {chunk_t}")

    # Build dummy inputs for chunk_t frames
    audio = torch.randn((1, chunk_t * p.hop_size))
    spec, feat_erb, feat_spec = df_features(audio, df_state, p.nb_df, device="cpu")
    feat_spec = feat_spec.transpose(1, 4).squeeze(4)  # [1, 2, T, 96]

    # Initial zero hidden states
    enc = model.enc
    erb_dec = model.erb_dec
    df_dec = model.df_dec

    h_enc = torch.zeros(enc.emb_gru.gru.num_layers,     1, enc.emb_gru.gru.hidden_size)
    h_erb = torch.zeros(erb_dec.emb_gru.gru.num_layers, 1, erb_dec.emb_gru.gru.hidden_size)
    h_df  = torch.zeros(df_dec.df_gru.gru.num_layers,   1, df_dec.df_gru.gru.hidden_size)

    print(f"\n  h_enc shape: {h_enc.shape}")
    print(f"  h_erb shape: {h_erb.shape}")
    print(f"  h_df  shape: {h_df.shape}")

    enc_w    = EncStreaming(enc)
    erb_dec_w = ErbDecStreaming(erb_dec)
    df_dec_w  = DfDecStreaming(df_dec)

    T_dyn = {2: "T"}

    # ── Encoder ──────────────────────────────────────────────────────────────
    print("\n── enc_streaming.onnx ──")
    enc_path = os.path.join(out_dir, "enc.onnx")
    enc_outs = _export_and_check(
        enc_path, enc_w,
        inputs=(feat_erb, feat_spec, h_enc),
        input_names=["feat_erb", "feat_spec", "h_enc"],
        output_names=["e0", "e1", "e2", "e3", "emb", "c0", "lsnr", "h_enc_out"],
        dynamic_axes={
            "feat_erb": T_dyn, "feat_spec": T_dyn,
            "e0": T_dyn, "e1": T_dyn, "e2": T_dyn, "e3": T_dyn,
            "emb": {1: "T"}, "c0": T_dyn, "lsnr": {1: "T"},
        },
        opset=opset,
    )
    e0, e1, e2, e3, emb, c0, lsnr, h_enc_out = enc_outs

    # ── ErbDecoder ────────────────────────────────────────────────────────────
    print("\n── erb_dec_streaming.onnx ──")
    erb_dec_path = os.path.join(out_dir, "erb_dec.onnx")
    _export_and_check(
        erb_dec_path, erb_dec_w,
        inputs=(emb, e3, e2, e1, e0, h_erb),
        input_names=["emb", "e3", "e2", "e1", "e0", "h_erb"],
        output_names=["m", "h_erb_out"],
        dynamic_axes={
            "emb": {1: "T"},
            "e3": T_dyn, "e2": T_dyn, "e1": T_dyn, "e0": T_dyn,
            "m": T_dyn,
        },
        opset=opset,
    )

    # ── DfDecoder ─────────────────────────────────────────────────────────────
    print("\n── df_dec_streaming.onnx ──")
    df_dec_path = os.path.join(out_dir, "df_dec.onnx")
    _export_and_check(
        df_dec_path, df_dec_w,
        inputs=(emb, c0, h_df),
        input_names=["emb", "c0", "h_df"],
        output_names=["coefs", "h_df_out"],
        dynamic_axes={
            "emb": {1: "T"}, "c0": T_dyn,
            "coefs": {1: "T"},
        },
        opset=opset,
    )

    # Save metadata
    meta = {
        "h_enc_shape": list(h_enc.shape),
        "h_erb_shape": list(h_erb.shape),
        "h_df_shape":  list(h_df.shape),
        "chunk_t":     chunk_t,
        "export_epoch": int(epoch),
    }
    import json
    meta_path = os.path.join(out_dir, "streaming_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata → {meta_path}")
    print(json.dumps(meta, indent=2))
    print("\nDone.")


def main():
    ap = argparse.ArgumentParser(description="Export DeepFilterNet3 in streaming mode")
    ap.add_argument("--out",   default="external/deepfilternet_onnx_streaming",
                    help="Output directory (default: external/deepfilternet_onnx_streaming)")
    ap.add_argument("--opset", type=int, default=14, help="ONNX opset (default: 14)")
    ap.add_argument("--chunk-t", type=int, default=100,
                    help="Chunk size in frames for parity check (default: 100)")
    args = ap.parse_args()
    export_streaming(args.out, chunk_t=args.chunk_t, opset=args.opset)


if __name__ == "__main__":
    main()
