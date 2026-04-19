#!/usr/bin/env python3
"""
Export CohereLabs/cohere-transcribe-03-2026 to ONNX format.

Exports the Cohere ONNX package and a metadata file:

  encoder.onnx
      input_features  [batch, n_mels, time_frames]  float32
      -> encoder_hidden_states  [batch, enc_seq_len, d_model]  float32

  decoder_init.onnx / decoder_step.onnx   (default KV-cache decoder)
      First token uses decoder_init.onnx, then subsequent tokens use
      decoder_step.onnx with carried KV tensors for O(n) generation.

  decoder.onnx   (optional conventional decoder)
      decoder_input_ids      [batch, dec_seq_len]   int64
      encoder_hidden_states  [batch, enc_seq_len, d_model]  float32
      -> logits              [batch, dec_seq_len, vocab_size]  float32

  config.json
      Feature extractor parameters, special token IDs, vocabulary size, etc.
      Everything the C# inference layer needs to reconstruct mel features
      and run greedy/beam decoding.

Decoder strategy
----------------
KV-cache export is the default. The main exporter writes decoder_init.onnx
and decoder_step.onnx in the same pass as mel/encoder export, then patches
config.json with the KV metadata needed by downstream runtimes.

For compatibility or debugging, pass --conventional-decoder to export the
older decoder.onnx graph instead. That path performs full-sequence decode
at every step and is O(n²) in sequence length.

Usage
-----
    # Requires a HuggingFace account with access to the gated model.
    huggingface-cli login

    python public/scripts/cohere_export/export_cohere_transcribe_to_onnx.py \\
        --output-dir ./models/cohere_transcribe \\
        --opset 17

    # Export on GPU:
    python public/scripts/cohere_export/export_cohere_transcribe_to_onnx.py \\
        --output-dir ./models/cohere_transcribe \\
        --device cuda

Dependencies
------------
    pip install torch transformers accelerate onnx onnxruntime huggingface_hub
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export CohereLabs/cohere-transcribe-03-2026 to ONNX with KV-cache decoder by default."
        )
    )
    parser.add_argument(
        "--model-repo",
        default="CohereLabs/cohere-transcribe-03-2026",
        help="HuggingFace repo ID (default: CohereLabs/cohere-transcribe-03-2026).",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face revision or commit hash to pin remote code and weights.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory that will receive encoder/mel ONNX files, decoder export(s), config.json, and tokenizer assets.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help=(
            "ONNX opset version (default: 18).  The modern dynamo exporter "
            "auto-clamps to >= 18 because some ops (e.g. Pad) lack v17 "
            "adapters; the legacy --legacy-exporter path accepts 17."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for model loading and export (default: auto).",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16"),
        default="float32",
        help=(
            "Torch dtype for model weights (default: float32).  "
            "float16 halves the checkpoint size but some ONNX runtimes "
            "require fp32 inputs regardless."
        ),
    )
    parser.add_argument(
        "--dummy-seconds",
        type=float,
        default=5.0,
        help="Dummy audio length in seconds used to produce export dummy inputs (default: 5.0).",
    )
    parser.add_argument(
        "--skip-encoder",
        action="store_true",
        help="Skip encoder.onnx export (useful if it already exists).",
    )
    parser.add_argument(
        "--skip-decoder",
        action="store_true",
        help="Skip decoder export for the selected mode.",
    )
    parser.add_argument(
        "--conventional-decoder",
        action="store_true",
        help="Export conventional decoder.onnx instead of the default KV-cache decoder pair.",
    )
    parser.add_argument(
        "--skip-mel",
        action="store_true",
        help="Skip mel.onnx export.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the output directory.",
    )
    parser.add_argument(
        "--legacy-exporter",
        action="store_true",
        help=(
            "Use the legacy TorchScript-based ONNX exporter (dynamo=False). "
            "Default is the modern torch.export-based exporter (dynamo=True). "
            "Provided as an escape hatch for parity comparison or troubleshooting."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Device / dtype helpers
# ---------------------------------------------------------------------------

def resolve_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is False.")
    return requested


def resolve_dtype(torch: Any, name: str) -> Any:
    return torch.float16 if name == "float16" else torch.float32


NUM_LAYERS = 8
NUM_HEADS = 8
HEAD_DIM = 128
ENC_DIM = 1280


# ---------------------------------------------------------------------------
# Output directory guard
# ---------------------------------------------------------------------------

_EXPORT_FILES = (
    "encoder.onnx",
    "decoder.onnx",
    "decoder_init.onnx",
    "decoder_step.onnx",
    "mel.onnx",
    "config.json",
    "vocab.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "export-report.json",
)


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        collisions = [name for name in _EXPORT_FILES if (path / name).exists()]
        if collisions:
            raise SystemExit(
                "Output directory already contains export targets. "
                "Re-run with --overwrite to replace them.\n"
                f"Existing files: {', '.join(collisions)}"
            )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_processor(
    repo_id: str, revision: str | None, device: str, dtype: Any, torch: Any
) -> tuple[Any, Any]:
    """Load CohereAsrForConditionalGeneration and its processor."""
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

    revision_suffix = f" @ {revision}" if revision else ""
    print(f"Loading processor from {repo_id}{revision_suffix} …")
    processor = AutoProcessor.from_pretrained(
        repo_id,
        revision=revision,
        trust_remote_code=True,
    )

    print(f"Loading model from {repo_id}{revision_suffix} onto {device} as {dtype} …")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        repo_id,
        revision=revision,
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Loaded {param_count:,} parameters ({param_count / 1e9:.2f}B)")

    return model, processor


# ---------------------------------------------------------------------------
# Wrapper modules
# ---------------------------------------------------------------------------

def make_encoder_wrapper(torch: Any, model: Any, model_dtype: Any = None) -> Any:
    """
    Patched nn.Module wrapper that produces a batch-dynamic encoder.onnx.

        input_features [batch, n_mels, time_frames] -> encoder_hidden_states [batch, T_enc, encoder_d_model]

    The Cohere Conformer contains two data-dependent branches that get baked as
    constants during TorchScript tracing, locking the graph to the dummy batch size:

      1. RelPositionMultiHeadAttention.forward (line 310):
             if pos_emb.size(0) == 1 and batch_size > 1: pos_emb = pos_emb.expand(...)
         With B=1 dummy this traces as False and the expand is omitted.  At runtime
         with B>1 the un-expanded pos_emb [1, 2T-1, D] is passed to linear_pos which
         tries view(batch_size, -1, h, d_k) → wrong shape.
         Fix: monkey-patch forward to always expand pos_emb unconditionally.

      2. ConvSubsampling.forward (line 170):
             if self._needs_conv_split(x): ...
             self._check_input_shape(x)
         These helpers gate large-batch safety checks using Python conditionals. For
         practical batch sizes on modern hardware the split path is never needed, and
         tracing those checks bakes export-time constants into the graph.
         Fix: replace _needs_conv_split with a lambda that always returns False and
         _check_input_shape with a no-op during export.
         This is safe for batch sizes up to ~100 at typical segment lengths (see below).

      3. RelPositionalEncoding._materialize_pe (line 200):
             if hasattr(self, "pe") and self.pe.size(1) >= needed_size: ...
         The cached positional-encoding reuse path converts tensor-dependent shape checks
         into Python booleans during tracing.  Fix: rebuild the positional encoding
         deterministically for each traced call instead of branching on the cached buffer.

    Safety note for _needs_conv_split=False:
        projected = B * conv_channels * out_T * out_F
        conv_channels = 256 (from model config)
        For 30s audio at 8x downsampling: out_T ≈ 376, out_F ≈ 32
        projected(B=16) = 16 * 256 * 376 * 32 ≈ 49M  <<  2^31 = 2.1B   ✅
        projected(B=64) = 64 * 256 * 376 * 32 ≈ 196M  <<  2^31          ✅
    For all realistic inference batch sizes the guard is never triggered.

    encoder_decoder_proj is NOT applied — the encoder ONNX output is raw d_model=1280,
    not the projected decoder d_model (1024).
    """
    nn = torch.nn

    # --- Patch 1: RelPositionMultiHeadAttention.forward ---
    # Always expand pos_emb to batch_size, removing the data-dependent branch.
    def _patched_rel_pos_attn_forward(self_attn, x, pos_emb, mask=None):
        batch_size = x.size(0)
        q = self_attn.linear_q(x).view(batch_size, -1, self_attn.h, self_attn.d_k).transpose(1, 2)
        k = self_attn.linear_k(x).view(batch_size, -1, self_attn.h, self_attn.d_k).transpose(1, 2)
        v = self_attn.linear_v(x).view(batch_size, -1, self_attn.h, self_attn.d_k).transpose(1, 2)
        # Always expand — removes the `if pos_emb.size(0) == 1 and batch_size > 1` branch.
        pos_emb = pos_emb.expand(batch_size, -1, -1)
        p = self_attn.linear_pos(pos_emb).view(batch_size, -1, self_attn.h, self_attn.d_k).transpose(1, 2)
        q_with_u = q + self_attn.pos_bias_u.unsqueeze(0).unsqueeze(2)
        q_with_v = q + self_attn.pos_bias_v.unsqueeze(0).unsqueeze(2)
        matrix_ac = torch.matmul(q_with_u, k.transpose(-1, -2))
        matrix_bd = torch.matmul(q_with_v, p.transpose(-1, -2))
        matrix_bd = self_attn.rel_shift(matrix_bd)
        matrix_bd = matrix_bd[:, :, :, :matrix_ac.size(-1)]
        scores = (matrix_ac + matrix_bd) * self_attn.scaling
        if mask is not None:
            expanded_mask = mask.unsqueeze(1)
            # Use -1e4 instead of -1e9: fp16 max is ~65504, so -1e9 overflows.
            # -1e4 is sufficient to drive softmax to zero after exponentiation.
            fill_val = -1e4 if scores.dtype == torch.float16 else -1e9
            scores = scores.masked_fill(expanded_mask, fill_val)
        attn = torch.softmax(scores, dim=-1)
        if mask is not None:
            attn = attn.masked_fill(expanded_mask, 0.0)
        x = torch.matmul(self_attn.dropout(attn), v)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self_attn.h * self_attn.d_k)
        return self_attn.linear_out(x)

    import types
    encoder = model.encoder

    # Apply patch to every ConformerLayer's self_attn
    for layer in encoder.layers:
        layer.self_attn.forward = types.MethodType(
            lambda self, x, pos_emb, mask=None, _fn=_patched_rel_pos_attn_forward:
                _fn(self, x, pos_emb, mask),
            layer.self_attn,
        )

    # --- Patch 2: ConvSubsampling export guards ---
    # Safe for all practical batch sizes; see docstring above.
    if hasattr(encoder.pre_encode, "_needs_conv_split"):
        encoder.pre_encode._needs_conv_split = lambda x: False
    if hasattr(encoder.pre_encode, "_check_input_shape"):
        encoder.pre_encode._check_input_shape = lambda x: None

    # --- Patch 3: RelPositionalEncoding._materialize_pe ---
    # Rebuild the cached buffer unconditionally to avoid tracing Python boolean guards.
    if hasattr(encoder, "pos_enc") and hasattr(encoder.pos_enc, "_materialize_pe"):
        def _patched_materialize_pe(self_pos, length: int, device: Any, dtype: Any) -> None:
            positions = torch.arange(
                length - 1,
                -length,
                -1,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(1)
            pe = self_pos._create_pe(positions=positions, dtype=dtype)
            if hasattr(self_pos, "pe"):
                self_pos.pe = pe
            else:
                self_pos.register_buffer("pe", pe, persistent=False)

        encoder.pos_enc._materialize_pe = types.MethodType(_patched_materialize_pe, encoder.pos_enc)

    _dtype = model_dtype  # None or torch.float16

    class EncoderWrapper(nn.Module):
        def __init__(self, enc: nn.Module) -> None:
            super().__init__()
            self.encoder = enc

        def forward(self, input_features: Any, input_lengths: Any) -> Any:
            # input_lengths: int64 [B] — actual mel-frame count per batch item.
            # The ConformerEncoder uses this to compute a padding mask so that
            # self-attention does not attend to zero-padded positions beyond each
            # segment's true length.  This eliminates cross-contamination when
            # multiple segments of different lengths are batched together.
            #
            # When exporting fp16: cast float32 graph input → fp16 at the boundary,
            # run the Conformer in fp16, then cast hidden_states back to float32 so
            # C# can read the output without changes.  When _dtype is None/float32
            # these are no-ops and the graph stays fully float32.
            if _dtype is not None and _dtype != torch.float32:
                input_features = input_features.to(_dtype)
            hidden_states, _lengths = self.encoder(input_features, length=input_lengths)
            if _dtype is not None and _dtype != torch.float32:
                hidden_states = hidden_states.float()
            return hidden_states

    return EncoderWrapper(encoder)


def make_decoder_wrapper(torch: Any, model: Any) -> Any:
    """
    Wrapper that runs a full decoder forward pass (no KV cache) via the top-level model.

        decoder_input_ids      [batch, dec_seq_len]          int64
        encoder_hidden_states  [batch, enc_seq_len, d_model]  float32
        -> logits              [batch, dec_seq_len, vocab]    float32

    We call model.forward() with encoder_outputs=encoder_hidden_states (plain tensor is
    accepted), which lets the model build the causal mask and compute positions internally.
    Returning only logits keeps the output shape simple for ONNX export.
    """
    nn = torch.nn

    from transformers.modeling_outputs import BaseModelOutput

    class DecoderWrapper(nn.Module):
        def __init__(self, full_model: nn.Module) -> None:
            super().__init__()
            self.full_model = full_model

        def forward(self, decoder_input_ids: Any, encoder_hidden_states: Any) -> Any:
            # model.forward() accesses encoder_outputs.last_hidden_state unconditionally
            # at return time, so wrap in BaseModelOutput rather than passing a plain tensor.
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)
            out = self.full_model(
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
            )
            return out.logits

    return DecoderWrapper(model)


def make_kv_init_wrapper(torch: Any, model: Any, model_dtype: Any) -> Any:
    """Decoder-init wrapper: context tokens + encoder states -> logits + all KV."""
    F = torch.nn.functional
    _dtype = model_dtype

    class DecoderInitWrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = model.transf_decoder._embedding
            self.layers = model.transf_decoder._decoder.layers
            self.final_norm = model.transf_decoder._decoder.final_layer_norm
            self.proj = model.encoder_decoder_proj
            self.lm_head = model.log_softmax

        def forward(self, decoder_input_ids: Any, encoder_hidden_states: Any) -> tuple:
            if _dtype != torch.float32:
                encoder_hidden_states = encoder_hidden_states.to(_dtype)

            enc = self.proj(encoder_hidden_states) if self.proj is not None else encoder_hidden_states
            batch, seq = decoder_input_ids.shape
            positions = torch.arange(seq, device=decoder_input_ids.device).unsqueeze(0).expand(batch, -1)
            hidden = self.embedding(decoder_input_ids, positions)

            self_keys: list = []
            self_vals: list = []
            cross_keys: list = []
            cross_vals: list = []

            for layer in self.layers:
                residual = hidden
                h = layer.layer_norm_1(hidden)
                sa = layer.first_sub_layer
                q = sa._reshape(sa.query_net(h))
                sk = sa._reshape(sa.key_net(h))
                sv = sa._reshape(sa.value_net(h))
                ao = F.scaled_dot_product_attention(q, sk, sv, attn_mask=None, dropout_p=0.0, scale=sa.scale)
                ao = sa.out_projection(ao.transpose(1, 2).contiguous().view(batch, seq, sa.hidden_size))
                hidden = residual + ao
                self_keys.append(sk)
                self_vals.append(sv)

                residual = hidden
                h = layer.layer_norm_2(hidden)
                ca = layer.second_sub_layer
                cq = ca._reshape(ca.query_net(h))
                ck = ca._reshape(ca.key_net(enc))
                cv = ca._reshape(ca.value_net(enc))
                ao = F.scaled_dot_product_attention(cq, ck, cv, attn_mask=None, dropout_p=0.0, scale=ca.scale)
                ao = ca.out_projection(ao.transpose(1, 2).contiguous().view(batch, seq, ca.hidden_size))
                hidden = residual + ao
                cross_keys.append(ck)
                cross_vals.append(cv)

                residual = hidden
                h = layer.layer_norm_3(hidden)
                hidden = residual + layer.third_sub_layer(h)

            hidden = self.final_norm(hidden)
            logits = self.lm_head(hidden)
            if _dtype != torch.float32:
                logits = logits.float()

            return (logits, *self_keys, *self_vals, *cross_keys, *cross_vals)

    return DecoderInitWrapper()


def make_kv_step_wrapper(torch: Any, model: Any, model_dtype: Any) -> Any:
    """Decoder-step wrapper: next token + past KV + fixed cross KV -> next logits + updated self KV."""
    F = torch.nn.functional
    _dtype = model_dtype

    class DecoderStepWrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = model.transf_decoder._embedding
            self.layers = model.transf_decoder._decoder.layers
            self.final_norm = model.transf_decoder._decoder.final_layer_norm
            self.lm_head = model.log_softmax

        def forward(
            self,
            decoder_input_ids: Any,
            positions: Any,
            sk0: Any, sk1: Any, sk2: Any, sk3: Any, sk4: Any, sk5: Any, sk6: Any, sk7: Any,
            sv0: Any, sv1: Any, sv2: Any, sv3: Any, sv4: Any, sv5: Any, sv6: Any, sv7: Any,
            ck0: Any, ck1: Any, ck2: Any, ck3: Any, ck4: Any, ck5: Any, ck6: Any, ck7: Any,
            cv0: Any, cv1: Any, cv2: Any, cv3: Any, cv4: Any, cv5: Any, cv6: Any, cv7: Any,
        ) -> tuple:
            past_sk = [sk0, sk1, sk2, sk3, sk4, sk5, sk6, sk7]
            past_sv = [sv0, sv1, sv2, sv3, sv4, sv5, sv6, sv7]
            cross_k = [ck0, ck1, ck2, ck3, ck4, ck5, ck6, ck7]
            cross_v = [cv0, cv1, cv2, cv3, cv4, cv5, cv6, cv7]

            batch, seq = decoder_input_ids.shape
            hidden = self.embedding(decoder_input_ids, positions)

            new_sk: list = []
            new_sv: list = []

            for i, layer in enumerate(self.layers):
                residual = hidden
                h = layer.layer_norm_1(hidden)
                sa = layer.first_sub_layer
                q = sa._reshape(sa.query_net(h))
                sk_new = sa._reshape(sa.key_net(h))
                sv_new = sa._reshape(sa.value_net(h))
                k = torch.cat([past_sk[i], sk_new], dim=2)
                v = torch.cat([past_sv[i], sv_new], dim=2)
                ao = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, scale=sa.scale)
                ao = sa.out_projection(ao.transpose(1, 2).contiguous().view(batch, seq, sa.hidden_size))
                hidden = residual + ao
                new_sk.append(k)
                new_sv.append(v)

                residual = hidden
                h = layer.layer_norm_2(hidden)
                ca = layer.second_sub_layer
                cq = ca._reshape(ca.query_net(h))
                ao = F.scaled_dot_product_attention(cq, cross_k[i], cross_v[i], attn_mask=None, dropout_p=0.0, scale=ca.scale)
                ao = ca.out_projection(ao.transpose(1, 2).contiguous().view(batch, seq, ca.hidden_size))
                hidden = residual + ao

                residual = hidden
                h = layer.layer_norm_3(hidden)
                hidden = residual + layer.third_sub_layer(h)

            hidden = self.final_norm(hidden)
            logits = self.lm_head(hidden)
            if _dtype != torch.float32:
                logits = logits.float()

            return (logits, *new_sk, *new_sv)

    return DecoderStepWrapper()


# ---------------------------------------------------------------------------
# Dummy inputs
# ---------------------------------------------------------------------------

def make_encoder_dummy_inputs(
    torch: Any,
    processor: Any,
    device: str,
    dtype: Any,
    dummy_seconds: float,
) -> tuple:
    """Return (input_features, input_lengths) dummy tensors for the encoder.

    Uses B=2 where item 0 is full-length and item 1 is half-length (zero-padded).
    This forces the attention-mask branch to be traced into the ONNX graph so
    the exported model correctly handles batches with mixed-length segments.
    """
    import numpy as np

    sample_rate = processor.feature_extractor.sampling_rate
    num_samples = int(sample_rate * dummy_seconds)

    # Item 0: full-length audio.  Item 1: half-length (the rest will be padded zeros).
    audio_full = np.zeros(num_samples, dtype=np.float32)
    audio_half = np.zeros(num_samples // 2, dtype=np.float32)

    inputs_full = processor(
        audio_full,
        sampling_rate=sample_rate,
        return_tensors="pt",
        language="en",
    )
    inputs_half = processor(
        audio_half,
        sampling_rate=sample_rate,
        return_tensors="pt",
        language="en",
    )

    # Dummy input is always float32 — the wrapper casts internally for fp16 exports
    # so the ONNX graph I/O stays float32 and C# code requires no changes.
    feat_full = inputs_full["input_features"].to(device=device, dtype=torch.float32)  # [1, mels, T]
    feat_half = inputs_half["input_features"].to(device=device, dtype=torch.float32)  # [1, mels, T//2]

    T_full = feat_full.shape[2]
    T_half = feat_half.shape[2]

    # Pad feat_half to T_full along the time axis so we can stack into a batch.
    pad_len = T_full - T_half
    if pad_len > 0:
        feat_half = torch.nn.functional.pad(feat_half, (0, pad_len))

    # Stack into [2, mels, T_full]
    features = torch.cat([feat_full, feat_half], dim=0)
    # lengths: actual mel-frame count for each item (int64)
    lengths = torch.tensor([T_full, T_half], dtype=torch.int64, device=device)

    print(f"  Encoder dummy input shape: {tuple(features.shape)}  lengths: {lengths.tolist()}")
    return features, lengths


def make_decoder_dummy_inputs(
    torch: Any,
    model: Any,
    encoder_hidden_states: Any,
    device: str,
) -> tuple[Any, Any]:
    """Return (decoder_input_ids, encoder_hidden_states) for a minimal decode step."""
    # Token IDs live in generation_config for this model, not model.config.
    gen_cfg = getattr(model, "generation_config", None)
    bos_id = None
    for attr in ("decoder_start_token_id", "bos_token_id"):
        bos_id = getattr(gen_cfg, attr, None) if gen_cfg is not None else None
        if bos_id is not None:
            break
        bos_id = getattr(model.config, attr, None)
        if bos_id is not None:
            break
    if bos_id is None:
        raise RuntimeError(
            "Could not determine decoder start token ID from model.generation_config "
            "or model.config. Pass --bos-token-id explicitly."
        )

    # Shape: (batch=2, seq_len=2) — keep B and seq > 1 so torch.export does not
    # specialise either dim to a static 1.  Runtime accepts B=1, seq=1 once the
    # graph dims are symbolic.
    decoder_input_ids = torch.tensor([[bos_id, bos_id], [bos_id, bos_id]], dtype=torch.long, device=device)
    print(f"  Decoder dummy decoder_input_ids shape: {tuple(decoder_input_ids.shape)}")
    print(f"  Decoder dummy encoder_hidden_states shape: {tuple(encoder_hidden_states.shape)}")
    return decoder_input_ids, encoder_hidden_states


# ---------------------------------------------------------------------------
# ONNX external-data consolidation
# ---------------------------------------------------------------------------

def _consolidate_external_data(onnx_path: Path) -> None:
    """
    Normalise external-data layout so every export ends up as a single pair:
        <name>.onnx           — graph
        <name>.onnx.data      — consolidated weights

    The legacy TorchScript exporter scatters weights into many individual files
    alongside the .onnx, so a load+resave round-trip is required.  The modern
    dynamo exporter already writes a single consolidated <name>.onnx.data file,
    so we detect that case and only do stale-file cleanup — re-saving would
    require pointlessly loading 8GB of weights into memory.
    """
    import onnx

    data_file = onnx_path.name + ".data"
    data_path = onnx_path.parent / data_file

    proto = onnx.load(str(onnx_path), load_external_data=False)
    scattered: set[Path] = set()
    needs_consolidation = False
    for init in proto.graph.initializer:
        if init.data_location == onnx.TensorProto.EXTERNAL:
            for entry in init.external_data:
                if entry.key == "location":
                    candidate = (onnx_path.parent / entry.value).resolve()
                    if candidate != data_path.resolve():
                        scattered.add(candidate)
                        needs_consolidation = True

    if needs_consolidation:
        # Legacy path — weights are in many files; load and re-save consolidated.
        # Delete any pre-existing data file before writing: onnx.save_model opens
        # the data file in append mode, so without this every re-export would
        # grow the file by one model's worth of weights.
        if data_path.exists():
            data_path.unlink()

        print(f"  Consolidating external data → {data_file} …")
        model = onnx.load(str(onnx_path), load_external_data=True)
        onnx.save_model(
            model,
            str(onnx_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_file,
            size_threshold=1024,
        )
    else:
        print(f"  External data already consolidated in {data_file}; skipping resave.")

    # --- Sweep stale Constant_* files left by previous partial/failed exports.
    # Torch may prefix these with node paths such as `_encoder_pos_enc_Constant_*`,
    # so match any filename containing Constant.  Also sweep the now-superseded
    # scattered legacy weight files.
    keep = {onnx_path.resolve(), data_path.resolve()}
    deleted = 0
    candidates = set(scattered)
    for pattern in ("Constant_*", "*Constant*"):
        for stale in onnx_path.parent.glob(pattern):
            candidates.add(stale.resolve())
    for f in candidates:
        if f not in keep and f.exists() and f.is_file():
            f.unlink()
            deleted += 1
    if deleted:
        print(f"  Removed {deleted} scattered weight file(s).")

    total_mb = (onnx_path.stat().st_size + (data_path.stat().st_size if data_path.exists() else 0)) / 1024 / 1024
    print(f"  Saved {onnx_path.name} ({onnx_path.stat().st_size / 1024:.0f} KB graph) + "
          f"{data_file} ({data_path.stat().st_size / 1024 / 1024:.1f} MB weights)  "
          f"[total {total_mb:.1f} MB]")


# ---------------------------------------------------------------------------
# ONNX export helpers
# ---------------------------------------------------------------------------

def _run_torch_export(
    torch: Any,
    wrapper: Any,
    args_tuple: tuple,
    output_path: Path,
    *,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]],
    dynamic_shapes: Any,
    opset: int,
    legacy: bool,
    constant_folding: bool = True,
) -> None:
    """
    Unified entry point for both legacy and modern ONNX export paths.

    legacy=True  → TorchScript-based exporter (dynamo=False, dynamic_axes).
                   Emits a deprecation warning under torch >= 2.9.
    legacy=False → torch.export-based exporter (dynamo=True, dynamic_shapes).
                   Default for new exports; also writes external data side-by-side
                   so _consolidate_external_data still applies as a cleanup pass.

    constant_folding only takes effect on the legacy path.  The modern path uses
    its own optimisation passes (controlled by the `optimize` kwarg, default on)
    and the encoder's pos-encoding duplication issue does not arise there because
    the FX-based exporter shares buffers via initialisers rather than folding
    expand-of-broadcast into per-layer constants.
    """
    if legacy:
        torch.onnx.export(
            wrapper,
            args_tuple,
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=constant_folding,
            dynamo=False,
        )
        return

    # Modern (torch.export) path.  external_data=True is the default in torch>=2.10
    # and writes <name>.onnx + <name>.onnx.data automatically; pass explicitly so
    # behavior is stable across versions.
    #
    # opset_version is clamped to >= 18 because the dynamo exporter only ships
    # implementations for opset 18+; requesting 17 triggers an automatic downgrade
    # pass that fails on Pad (no v17 adapter).  Older runtime requirements are
    # already satisfied by ORT 1.20+ which speaks any opset >= 18.
    modern_opset = max(opset, 18)
    if modern_opset != opset:
        print(f"  Bumping opset {opset} → {modern_opset} for the dynamo exporter (no v17 adapter for some ops).")
    torch.onnx.export(
        wrapper,
        args_tuple,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_shapes=dynamic_shapes,
        opset_version=modern_opset,
        dynamo=True,
        external_data=True,
        optimize=True,
    )


def _make_dim(torch: Any, name: str, *, min: int | None = None, max: int | None = None) -> Any:
    kwargs: dict[str, Any] = {}
    if min is not None:
        kwargs["min"] = min
    if max is not None:
        kwargs["max"] = max
    return torch.export.Dim(name, **kwargs)


def _auto_dim(torch: Any) -> Any:
    """Return Dim.AUTO (torch infers bounds from the trace).

    Use for axes that only need to be dynamic and do not share a name across
    inputs.  Named Dims must be reserved for dimensions that are asserted to
    match between multiple inputs (e.g. `batch` shared across mel+encoder+KV,
    or `enc_seq_len` shared between encoder output and cross-KV).  Downstream
    derived symbols like the post-subsampling encoder time dim require either
    `Dim.AUTO` or an explicit min high enough that the derivation (e.g.
    `8*T - 3`) stays non-negative; the named-Dim `min=` kwarg does not always
    propagate through the internal symbol rename and has been observed to
    trigger `AssertionError: Expected derived min value ... to be >= 0`.
    """
    return torch.export.Dim.AUTO


def export_encoder(
    torch: Any,
    wrapper: Any,
    dummy_features: Any,
    dummy_lengths: Any,
    output_path: Path,
    opset: int,
    legacy: bool,
) -> None:
    print(f"\nExporting encoder to {output_path} …")

    with torch.no_grad():
        test_out = wrapper(dummy_features, dummy_lengths)
    print(f"  PyTorch encoder output shape: {tuple(test_out.shape)}")

    # Patched EncoderWrapper removes data-dependent branches in make_encoder_wrapper()
    # so both batch and time can be marked dynamic on either export path.
    #
    # Legacy-only note: do_constant_folding=False is required because the TorchScript
    # tracer materialises pos_emb.expand() into each of the 48 attention layers
    # separately (the B=1 dummy makes expand a no-op that looks foldable),
    # duplicating the PE tensor 48× and inflating the weight file by ~7 GB.
    # The modern (FX/dynamo) exporter shares the PE buffer as a single initialiser
    # and is unaffected.
    #
    # input_lengths [B] int64: actual mel-frame count per item.  The Conformer
    # encoder uses this to compute a padding mask so that self-attention does not
    # attend to zero-padded positions, preventing cross-contamination when items
    # of different lengths are batched together.
    batch = _make_dim(torch, "batch", min=1, max=65535)
    time_frames = _auto_dim(torch)  # 8x subsampling → derived dim 8*T-3 needs AUTO
    _run_torch_export(
        torch,
        wrapper,
        (dummy_features, dummy_lengths),
        output_path,
        input_names=["input_features", "input_lengths"],
        output_names=["encoder_hidden_states"],
        dynamic_axes={
            "input_features":        {0: "batch", 2: "time_frames"},
            "input_lengths":         {0: "batch"},
            "encoder_hidden_states": {0: "batch", 1: "enc_seq_len"},
        },
        dynamic_shapes=(
            {0: batch, 2: time_frames},
            {0: batch},
        ),
        opset=opset,
        legacy=legacy,
        constant_folding=False,
    )
    _consolidate_external_data(output_path)


def export_decoder(
    torch: Any,
    wrapper: Any,
    decoder_input_ids: Any,
    encoder_hidden_states: Any,
    output_path: Path,
    opset: int,
    legacy: bool,
) -> None:
    print(f"\nExporting decoder to {output_path} …")

    with torch.no_grad():
        test_out = wrapper(decoder_input_ids, encoder_hidden_states)
    print(f"  PyTorch decoder output shape: {tuple(test_out.shape)}")

    batch = _make_dim(torch, "batch", min=1, max=65535)
    dec = _auto_dim(torch)
    enc = _auto_dim(torch)
    _run_torch_export(
        torch,
        wrapper,
        (decoder_input_ids, encoder_hidden_states),
        output_path,
        input_names=["decoder_input_ids", "encoder_hidden_states"],
        output_names=["logits"],
        dynamic_axes={
            "decoder_input_ids": {0: "batch", 1: "dec_seq_len"},
            "encoder_hidden_states": {0: "batch", 1: "enc_seq_len"},
            "logits": {0: "batch", 1: "dec_seq_len"},
        },
        dynamic_shapes=(
            {0: batch, 1: dec},
            {0: batch, 1: enc},
        ),
        opset=opset,
        legacy=legacy,
    )
    _consolidate_external_data(output_path)


# ---------------------------------------------------------------------------
# ONNX validation
# ---------------------------------------------------------------------------

def validate_onnx(onnx: Any, path: Path, label: str) -> None:
    print(f"\nValidating {label} ONNX model …")
    # Pass the file path as a string so the checker resolves external data
    # relative to the file's directory.  Loading into memory first loses the
    # base-directory context and breaks serialization for large models.
    onnx.checker.check_model(str(path))
    print(f"  onnx.checker passed for {path.name}")


def _kv_names(prefix: str, n: int = NUM_LAYERS) -> list[str]:
    return [f"{prefix}_{i}" for i in range(n)]


def export_kv_init(
    torch: Any,
    wrapper: Any,
    output_dir: Path,
    opset: int,
    device: str,
    legacy: bool,
    enc_t: int = 80,
) -> None:
    out_path = output_dir / "decoder_init.onnx"
    print(f"\nExporting decoder_init to {out_path} …")

    # Dummy uses B=2, init_seq_len=2: torch.export specialises any symbolic dim
    # that is concretely 1 at trace time (broadcast guard, plus extra SDPA
    # decomposition guards on CUDA), but the exported graph still accepts B=1 /
    # seq_len=1 at runtime once the dim is symbolic.
    bos = torch.tensor([[13764, 13764], [13764, 13764]], dtype=torch.long, device=device)
    enc_h = torch.randn(2, enc_t, ENC_DIM, device=device, dtype=torch.float32)

    with torch.no_grad():
        out = wrapper(bos, enc_h)
    print(
        f"  PyTorch output shapes: logits={tuple(out[0].shape)}, "
        f"sk0={tuple(out[1].shape)}, ck0={tuple(out[1 + 2 * NUM_LAYERS].shape)}"
    )

    input_names = ["decoder_input_ids", "encoder_hidden_states"]
    output_names = (
        ["logits"]
        + _kv_names("self_key")
        + _kv_names("self_val")
        + _kv_names("cross_key")
        + _kv_names("cross_val")
    )

    dynamic_axes: dict[str, dict[int, str]] = {
        "decoder_input_ids": {0: "batch", 1: "init_seq_len"},
        "encoder_hidden_states": {0: "batch", 1: "enc_seq_len"},
        "logits": {0: "batch", 1: "init_seq_len"},
    }
    for name in _kv_names("self_key") + _kv_names("self_val"):
        dynamic_axes[name] = {0: "batch", 2: "init_seq_len"}
    for name in _kv_names("cross_key") + _kv_names("cross_val"):
        dynamic_axes[name] = {0: "batch", 2: "enc_seq_len"}

    batch = _make_dim(torch, "batch", min=1, max=65535)
    init_seq = _auto_dim(torch)
    enc = _auto_dim(torch)
    dynamic_shapes = (
        {0: batch, 1: init_seq},
        {0: batch, 1: enc},
    )

    _run_torch_export(
        torch,
        wrapper,
        (bos, enc_h),
        out_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamic_shapes=dynamic_shapes,
        opset=opset,
        legacy=legacy,
    )
    _consolidate_external_data(out_path)


def export_kv_step(
    torch: Any,
    wrapper: Any,
    output_dir: Path,
    opset: int,
    device: str,
    legacy: bool,
    past_len: int = 2,
    enc_t: int = 80,
) -> None:
    out_path = output_dir / "decoder_step.onnx"
    print(f"\nExporting decoder_step to {out_path} …")

    # past_len defaults to 2 (not 1) and B=2 (not 1) so torch.export does not
    # specialise these dims to static shapes.  The wrapper concatenates
    # past||new along axis 2 and the +1 step still works at runtime once the
    # dim is symbolic; same for runtime B=1.
    kv_dtype = next(wrapper.parameters()).dtype
    B = 2
    tok = torch.tensor([[42]] * B, dtype=torch.long, device=device)
    positions = torch.tensor([[past_len]] * B, dtype=torch.long, device=device)
    dummy_sk = [torch.randn(B, NUM_HEADS, past_len, HEAD_DIM, device=device, dtype=kv_dtype) for _ in range(NUM_LAYERS)]
    dummy_sv = [torch.randn(B, NUM_HEADS, past_len, HEAD_DIM, device=device, dtype=kv_dtype) for _ in range(NUM_LAYERS)]
    dummy_ck = [torch.randn(B, NUM_HEADS, enc_t, HEAD_DIM, device=device, dtype=kv_dtype) for _ in range(NUM_LAYERS)]
    dummy_cv = [torch.randn(B, NUM_HEADS, enc_t, HEAD_DIM, device=device, dtype=kv_dtype) for _ in range(NUM_LAYERS)]
    dummy_inputs = (tok, positions, *dummy_sk, *dummy_sv, *dummy_ck, *dummy_cv)

    with torch.no_grad():
        out = wrapper(*dummy_inputs)
    print(f"  PyTorch output shapes: logits={tuple(out[0].shape)}, sk0={tuple(out[1].shape)}")

    input_names = (
        ["decoder_input_ids", "positions"]
        + _kv_names("self_key")
        + _kv_names("self_val")
        + _kv_names("cross_key")
        + _kv_names("cross_val")
    )
    output_names = ["logits"] + _kv_names("new_self_key") + _kv_names("new_self_val")

    dynamic_axes: dict[str, dict[int, str]] = {
        "decoder_input_ids": {0: "batch"},
        "positions": {0: "batch"},
        "logits": {0: "batch"},
    }
    for name in _kv_names("self_key") + _kv_names("self_val"):
        dynamic_axes[name] = {0: "batch", 2: "kv_seq_len"}
    for name in _kv_names("new_self_key") + _kv_names("new_self_val"):
        dynamic_axes[name] = {0: "batch", 2: "kv_seq_len_plus_1"}
    for name in _kv_names("cross_key") + _kv_names("cross_val"):
        dynamic_axes[name] = {0: "batch", 2: "enc_seq_len"}

    batch = _make_dim(torch, "batch", min=1, max=65535)
    kv_seq = _make_dim(torch, "kv_seq_len", min=2)
    enc = _make_dim(torch, "enc_seq_len", min=1)
    self_kv_shape = {0: batch, 2: kv_seq}
    cross_kv_shape = {0: batch, 2: enc}
    dynamic_shapes = (
        {0: batch},  # decoder_input_ids — seq is fixed to 1 next-token
        {0: batch},  # positions
        *([self_kv_shape] * NUM_LAYERS),    # past self_key
        *([self_kv_shape] * NUM_LAYERS),    # past self_val
        *([cross_kv_shape] * NUM_LAYERS),   # cross_key
        *([cross_kv_shape] * NUM_LAYERS),   # cross_val
    )

    _run_torch_export(
        torch,
        wrapper,
        dummy_inputs,
        out_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamic_shapes=dynamic_shapes,
        opset=opset,
        legacy=legacy,
    )
    _consolidate_external_data(out_path)


def parity_check_encoder(
    ort: Any,
    torch: Any,
    encoder_wrapper: Any,
    dummy_features: Any,
    dummy_lengths: Any,
    onnx_path: Path,
) -> None:
    import numpy as np
    print("\nEncoder parity check (PyTorch vs ONNX Runtime) …")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # B=2 parity (dummy already has B=2 with one padded item)
    feat_np = dummy_features.float().cpu().numpy()
    lens_np = dummy_lengths.cpu().numpy()
    ort_out = session.run(None, {"input_features": feat_np, "input_lengths": lens_np})[0]
    with torch.no_grad():
        pt_out = encoder_wrapper(dummy_features, dummy_lengths).float().cpu().numpy()
    max_diff = abs(ort_out - pt_out).max()
    print(f"  B=2 (one padded)  max absolute diff: {max_diff:.6f}")
    if max_diff > 5e-3:
        print("  WARNING: parity difference is large — check dtype or op support.")
    else:
        print("  B=2  parity OK.")

    # B=4 batch shape test — verifies batch dimension is dynamic
    B, n_mels, n_frames = dummy_features.shape
    feat4_np = np.random.randn(4, n_mels, n_frames).astype(np.float32)
    lens4_np = np.array([n_frames, n_frames * 3 // 4, n_frames // 2, n_frames // 4],
                        dtype=np.int64)
    try:
        ort_out4 = session.run(None, {"input_features": feat4_np, "input_lengths": lens4_np})[0]
        print(f"  B=4  output shape: {ort_out4.shape}  ✅ batch-dynamic export confirmed")
    except Exception as e:
        print(f"  B=4  FAILED: {e}")
        print("  ❌ Encoder is not batch-dynamic — patch did not take effect.")


def parity_check_decoder(
    ort: Any,
    torch: Any,
    decoder_wrapper: Any,
    decoder_input_ids: Any,
    encoder_hidden_states: Any,
    onnx_path: Path,
) -> None:
    print("\nDecoder parity check (PyTorch vs ONNX Runtime) …")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ids_np = decoder_input_ids.cpu().numpy()
    enc_np = encoder_hidden_states.float().cpu().numpy()
    ort_out = session.run(None, {"decoder_input_ids": ids_np, "encoder_hidden_states": enc_np})[0]

    with torch.no_grad():
        pt_out = decoder_wrapper(decoder_input_ids, encoder_hidden_states).float().cpu().numpy()

    max_diff = abs(ort_out - pt_out).max()
    print(f"  Max absolute diff decoder: {max_diff:.6f}")
    if max_diff > 1e-3:
        print("  WARNING: decoder parity difference is large — check dtype or op support.")
    else:
        print("  Decoder parity OK.")


def parity_check_kv(torch: Any, ort: Any, model: Any, output_dir: Path, device: str) -> None:
    print("\nParity check (PyTorch reference vs ONNX KV-cache) …")
    from transformers.modeling_outputs import BaseModelOutput
    import numpy as np

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    init_sess = ort.InferenceSession(str(output_dir / "decoder_init.onnx"), providers=providers)
    step_sess = ort.InferenceSession(str(output_dir / "decoder_step.onnx"), providers=providers)

    bos_id = 13764
    tok1_id = 42
    model_dtype = next(model.parameters()).dtype
    dummy_enc = torch.randn(1, 80, ENC_DIM, device=device, dtype=model_dtype)

    with torch.no_grad():
        out0 = model(
            decoder_input_ids=torch.tensor([[bos_id]], device=device),
            encoder_outputs=BaseModelOutput(last_hidden_state=dummy_enc),
        )
        logits_ref0 = out0.logits[0, -1, :].float().cpu().numpy()

        out1 = model(
            decoder_input_ids=torch.tensor([[bos_id, tok1_id]], device=device),
            encoder_outputs=BaseModelOutput(last_hidden_state=dummy_enc),
        )
        logits_ref1 = out1.logits[0, -1, :].float().cpu().numpy()

    enc_np = dummy_enc.float().cpu().numpy()
    bos_np = np.array([[bos_id]], dtype=np.int64)

    init_out = init_sess.run(None, {
        "decoder_input_ids": bos_np,
        "encoder_hidden_states": enc_np,
    })
    logits_onnx0 = init_out[0][0, -1, :]
    kv_cache = init_out[1:]
    diff0 = abs(logits_ref0 - logits_onnx0).max()
    print(f"  Step 0 (init) max logit diff: {diff0:.2e}  top-ref={logits_ref0.argmax()}  top-onnx={logits_onnx0.argmax()}")
    if diff0 > 0.05:
        print("  WARNING: step 0 difference is large!")

    n = NUM_LAYERS
    sk = list(kv_cache[:n])
    sv = list(kv_cache[n:2 * n])
    ck = list(kv_cache[2 * n:3 * n])
    cv = list(kv_cache[3 * n:4 * n])

    step_inputs: dict[str, Any] = {
        "decoder_input_ids": np.array([[tok1_id]], dtype=np.int64),
        "positions": np.array([[1]], dtype=np.int64),
    }
    for i in range(n):
        step_inputs[f"self_key_{i}"] = sk[i]
        step_inputs[f"self_val_{i}"] = sv[i]
        step_inputs[f"cross_key_{i}"] = ck[i]
        step_inputs[f"cross_val_{i}"] = cv[i]

    step_out = step_sess.run(None, step_inputs)
    logits_onnx1 = step_out[0][0, -1, :]
    diff1 = abs(logits_ref1 - logits_onnx1).max()
    print(f"  Step 1 (step) max logit diff: {diff1:.2e}  top-ref={logits_ref1.argmax()}  top-onnx={logits_onnx1.argmax()}")
    if diff1 > 0.05:
        print("  WARNING: step 1 difference is large!")
    if diff0 < 0.05 and diff1 < 0.05:
        print("  Parity OK.")


# ---------------------------------------------------------------------------
# Mel preprocessor export (DFT conv1d — same approach as nemo128.onnx "dft" mode)
# ---------------------------------------------------------------------------

def make_mel_wrapper(torch: Any, processor: Any) -> Any:
    """
    Build a DFT-conv1d mel spectrogram wrapper from the feature extractor's _fb_config.

    Input / output contract:
        waveforms      [batch, samples]   float32   — raw 16 kHz PCM, dither removed
        waveforms_lens [batch]            int64     — valid sample counts
        -> features      [batch, n_mels, frames]  float32
        -> features_lens [batch]                  int64

    Uses conv1d with a precomputed windowed DFT basis matrix to avoid the ONNX STFT
    op (which diverges in ONNX Runtime on this toolchain).  The pipeline is:
        preemphasis → time-mask → center-pad → conv1d DFT → magnitude → power
        → mel filterbank → log → per-feature norm → frame-mask
    """
    import math
    nn = torch.nn

    fb_cfg = processor.feature_extractor._fb_config

    sample_rate = int(fb_cfg.get("sample_rate", 16000))
    win_length = int(fb_cfg.get("n_window_size", 400))
    hop_length = int(fb_cfg.get("n_window_stride", 160))
    n_fft_cfg = fb_cfg.get("n_fft")
    n_fft = int(n_fft_cfg) if n_fft_cfg else 2 ** math.ceil(math.log2(win_length))
    preemph = float(fb_cfg.get("preemph", 0.97))
    mag_power = float(fb_cfg.get("mag_power", 2.0))
    log_guard_t = str(fb_cfg.get("log_zero_guard_type", "add"))
    log_guard_v = float(fb_cfg.get("log_zero_guard_value", 2.0 ** -24))
    normalize = str(fb_cfg.get("normalize", "per_feature"))
    # pad_to: 0 means no temporal padding needed (Cohere default)
    pad_to = int(fb_cfg.get("pad_to", 0) or 0)
    pad_value = float(fb_cfg.get("pad_value", fb_cfg.get("padding_value", 0.0)))
    # exact_pad is not in Cohere's preprocessor_config — default False
    exact_pad = bool(fb_cfg.get("exact_pad", False))
    stft_pad_amount = (n_fft - hop_length) // 2 if exact_pad else None

    # Build windowed DFT basis.
    # The processor's AudioToMelSpectrogramPreprocessor stores its window as bfloat16
    # (see processing_cohere_asr.py line ~147: self.window = self.window.to(bfloat16)).
    # The STFT then casts it back to float32 at call time.  We must use the same
    # bfloat16→float32 round-trip here so the DFT basis matches the processor exactly;
    # using a native float32 hann window differs by up to 0.002 per sample, and that
    # error squares under mag_power=2 (power spectrum), causing ~4 unit divergence in
    # the log-mel output.
    #
    # We extract the window from the lazily-initialised _filterbank object.  If it
    # hasn't been initialised yet, force one forward pass to trigger it.
    fe = processor.feature_extractor
    fb_obj = getattr(fe, "_filterbank", None)
    if fb_obj is None:
        import numpy as _np
        _dummy = _np.zeros(win_length * 2, dtype=_np.float32)
        fe([_dummy], sampling_rate=sample_rate, return_tensors="pt")
        fb_obj = fe._filterbank

    # window: bfloat16 on the object → cast to float32 for DFT basis construction
    win = fb_obj.window.to(dtype=torch.float32).cpu()

    # Zero-pad window to n_fft (centre-pad, same as torch.stft internal convention)
    if win.shape[0] < n_fft:
        left = (n_fft - win.shape[0]) // 2
        right = n_fft - win.shape[0] - left
        win = torch.nn.functional.pad(win, [left, right])

    n_bins = n_fft // 2 + 1
    n_range = torch.arange(n_fft, dtype=torch.float64)
    k_range = torch.arange(n_bins, dtype=torch.float64)
    angles = 2.0 * math.pi * k_range.unsqueeze(1) * n_range.unsqueeze(0) / n_fft
    win64 = win.to(dtype=torch.float64)
    cos_basis = (torch.cos(angles) * win64.unsqueeze(0)).to(torch.float32)
    sin_basis = (torch.sin(angles) * win64.unsqueeze(0)).to(torch.float32)
    dft_matrix = torch.cat([cos_basis, sin_basis], dim=0).unsqueeze(1)  # [2*n_bins, 1, n_fft]

    # Mel filterbank: use the processor's fb buffer (also bfloat16→float32) so the
    # filterbank coefficients are bit-identical to what the processor uses.
    fb = fb_obj.fb.to(dtype=torch.float32).cpu()  # [1, nfilt, n_bins]

    class MelDFTWrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.preemph = preemph
            self.n_fft = n_fft
            self.hop_length = hop_length
            self.mag_power = mag_power
            self.exact_pad = exact_pad
            self.stft_pad_amount = stft_pad_amount
            self.log_guard_type = log_guard_t
            self.log_guard_value = log_guard_v
            self.normalize = normalize
            self.pad_to = pad_to
            self.pad_value        = pad_value
            self.register_buffer("dft_matrix", dft_matrix)
            self.register_buffer("fb",         fb)

        def _get_seq_len(self, waveforms_lens: Any) -> Any:
            pad_amount = (
                self.stft_pad_amount * 2
                if self.stft_pad_amount is not None
                else self.n_fft // 2 * 2
            )
            return torch.floor_divide(
                (waveforms_lens + pad_amount - self.n_fft), self.hop_length
            ).to(dtype=torch.long)

        def _normalize_per_feature(self, features: Any, seq_len: Any) -> Any:
            max_time   = features.shape[2]
            time_steps = torch.arange(max_time, device=features.device).unsqueeze(0).expand(features.shape[0], -1)
            valid      = time_steps < seq_len.unsqueeze(1)
            masked     = torch.where(valid.unsqueeze(1), features, 0.0)
            denom      = valid.sum(dim=1).unsqueeze(1).to(features.dtype)
            mean       = masked.sum(dim=2) / denom
            variance   = (
                torch.where(valid.unsqueeze(1), features - mean.unsqueeze(2), 0.0).pow(2).sum(dim=2)
                / (denom - 1.0)
            )
            std = torch.sqrt(variance).masked_fill(variance.isnan(), 0.0) + 1e-5
            return (features - mean.unsqueeze(2)) / std.unsqueeze(2)

        def forward(self, waveforms: Any, waveforms_lens: Any) -> tuple[Any, Any]:
            seq_len = torch.where(
                waveforms_lens == 0,
                torch.zeros_like(waveforms_lens),
                self._get_seq_len(waveforms_lens),
            )

            # exact_pad: pad before dither/preemphasis (matches processor order)
            if self.stft_pad_amount is not None:
                waveforms = torch.nn.functional.pad(
                    waveforms.unsqueeze(1), (self.stft_pad_amount, self.stft_pad_amount), "constant",
                ).squeeze(1)

            # Pre-emphasis then time mask (dither is zero — deterministic export)
            waveforms = torch.cat(
                (waveforms[:, :1], waveforms[:, 1:] - self.preemph * waveforms[:, :-1]),
                dim=1,
            )
            time_mask = (
                torch.arange(waveforms.shape[1], device=waveforms.device)
                .unsqueeze(0) < waveforms_lens.unsqueeze(1)
            )
            waveforms = waveforms.masked_fill(~time_mask, 0.0)

            # center pad for non-exact-pad path.
            # torch.stft(center=True, pad_mode="constant") zero-pads by n_fft//2 on
            # each side — the processor passes pad_mode="constant" explicitly.
            if self.stft_pad_amount is None:
                half = self.n_fft // 2
                waveforms = torch.nn.functional.pad(
                    waveforms.unsqueeze(1), (half, half), "constant",
                ).squeeze(1)

            # DFT via conv1d  →  [batch, 2*n_bins, frames]
            frames = torch.nn.functional.conv1d(
                waveforms.unsqueeze(1),
                self.dft_matrix.to(dtype=waveforms.dtype, device=waveforms.device),
                stride=self.hop_length,
                padding=0,
            )
            n_bins = self.n_fft // 2 + 1
            features = torch.sqrt(frames[:, :n_bins].pow(2) + frames[:, n_bins:].pow(2))

            if self.mag_power != 1.0:
                features = features.pow(self.mag_power)

            # Mel filterbank: [1, nfilt, n_bins] @ [batch, n_bins, frames] → [batch, nfilt, frames]
            features = torch.matmul(
                self.fb.to(dtype=features.dtype, device=features.device),
                features,
            )

            # Log compression
            if self.log_guard_type == "add":
                features = torch.log(features + self.log_guard_value)
            else:
                features = torch.log(torch.clamp(features, min=self.log_guard_value))

            # Per-feature normalization
            if self.normalize == "per_feature":
                features = self._normalize_per_feature(features, seq_len)

            # Frame mask
            max_len = features.size(-1)
            frame_mask = (
                torch.arange(max_len, device=features.device)
                .unsqueeze(0) >= seq_len.unsqueeze(1)
            )
            features = features.masked_fill(frame_mask.unsqueeze(1), self.pad_value)

            # Temporal padding to multiple of pad_to (disabled when pad_to=0)
            if self.pad_to > 0:
                pad_amt = features.size(-1) % self.pad_to
                if pad_amt != 0:
                    features = torch.nn.functional.pad(
                        features, (0, self.pad_to - pad_amt), value=self.pad_value,
                    )

            return features, seq_len

    return MelDFTWrapper()


def export_mel(
    torch: Any,
    wrapper: Any,
    processor: Any,
    output_path: Path,
    opset: int,
    dummy_seconds: float,
    legacy: bool,
) -> tuple[Any, Any]:
    """Export mel.onnx and return (dummy_waveforms, dummy_lens) for parity check."""
    sample_rate = int(processor.feature_extractor._fb_config.get("sample_rate", 16000))
    num_samples = max(int(sample_rate * dummy_seconds), sample_rate)

    # Use B=2 dummy with mixed lengths so torch.export traces a symbolic batch
    # dim instead of specialising to B=1.
    dummy_wave = torch.zeros((2, num_samples), dtype=torch.float32)
    dummy_lens = torch.tensor([num_samples, num_samples // 2], dtype=torch.int64)

    print(f"\nExporting mel preprocessor to {output_path} …")
    with torch.no_grad():
        test_feat, test_lens = wrapper(dummy_wave, dummy_lens)
    print(f"  PyTorch mel output shape: {tuple(test_feat.shape)}, lens: {test_lens.tolist()}")

    batch = _make_dim(torch, "batch", min=1, max=65535)
    # samples → frames is a hop_length-stride conv → also a derived dim.
    samples = _auto_dim(torch)
    _run_torch_export(
        torch,
        wrapper,
        (dummy_wave, dummy_lens),
        output_path,
        input_names=["waveforms", "waveforms_lens"],
        output_names=["features", "features_lens"],
        dynamic_axes={
            "waveforms":      {0: "batch", 1: "samples"},
            "waveforms_lens": {0: "batch"},
            "features":       {0: "batch", 2: "frames"},
            "features_lens":  {0: "batch"},
        },
        dynamic_shapes=(
            {0: batch, 1: samples},
            {0: batch},
        ),
        opset=opset,
        legacy=legacy,
    )
    print(f"  Saved {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    return dummy_wave, dummy_lens


def parity_check_mel(
    ort: Any,
    torch: Any,
    wrapper: Any,
    processor: Any,
    dummy_wave: Any,
    dummy_lens: Any,
    onnx_path: Path,
) -> None:
    """
    Compare DFT wrapper output against the Python feature extractor.

    Uses band-limited noise rather than the silent dummy input because per-feature
    normalization on a constant signal (all-zeros → all-same log values) is a
    degenerate case: variance=0, std=1e-5, output≈0, and any tiny numerical
    divergence between ORT and PyTorch appears amplified.  Real-ish noise exercises
    the full normalization path.
    """
    import numpy as np

    sample_rate = int(processor.feature_extractor._fb_config.get("sample_rate", 16000))
    duration_s  = max(dummy_wave.shape[-1] / sample_rate, 1.0)
    n_samples   = int(duration_s * sample_rate)

    rng      = np.random.default_rng(42)
    noise_np = rng.standard_normal(n_samples).astype(np.float32) * 0.1
    noise_t  = torch.from_numpy(noise_np).unsqueeze(0)          # [1, T]
    lens_t   = torch.tensor([n_samples], dtype=torch.int64)
    lens_np  = np.array([n_samples], dtype=np.int64)

    print("\nMel parity check (DFT wrapper vs Python processor) …")

    # ---- ORT vs PyTorch wrapper (should be near-zero) -------------------------
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_feat, ort_lens = session.run(None, {
        "waveforms":      noise_t.numpy(),
        "waveforms_lens": lens_np,
    })
    with torch.no_grad():
        pt_feat, pt_lens = wrapper(noise_t, lens_t)
    pt_feat_np = pt_feat.numpy()

    feat_diff  = abs(ort_feat - pt_feat_np).max()
    lens_match = bool((ort_lens == pt_lens.numpy()).all())
    print(f"  ORT vs PT wrapper — max abs diff: {feat_diff:.6f}  |  lens match: {lens_match}")
    if feat_diff > 5e-3:
        print("  WARNING: ONNX export diverges from PyTorch wrapper — check pipeline.")
    else:
        print("  ONNX export parity OK.")

    # ---- ORT vs Python processor (dither zeroed for determinism) ---------------
    fe     = processor.feature_extractor
    fb_obj = getattr(fe, "_filterbank", None)
    # Ensure filterbank is initialised before we patch dither
    if fb_obj is None:
        fe([noise_np], sampling_rate=sample_rate, return_tensors="pt")
        fb_obj = getattr(fe, "_filterbank", None)

    orig_dither = getattr(fb_obj, "dither", 0.0) if fb_obj is not None else 0.0
    if fb_obj is not None:
        fb_obj.dither = 0.0
    try:
        proc_out  = processor(noise_np, sampling_rate=sample_rate,
                              return_tensors="pt", language="en")
    finally:
        if fb_obj is not None:
            fb_obj.dither = orig_dither

    proc_feat   = proc_out["input_features"].numpy()
    frames_onnx = ort_feat.shape[2]
    frames_proc = proc_feat.shape[2]
    proc_diff   = abs(ort_feat - proc_feat[:, :, :frames_onnx]).max()
    print(f"  ORT vs Python proc — max abs diff: {proc_diff:.6f}")
    print(f"  ONNX frames: {frames_onnx}  |  Python frames: {frames_proc}")
    if proc_diff > 0.1:
        print("  WARNING: DFT pipeline diverges from Python processor — check preemphasis/padding/norm.")

@dataclass
class ExportConfig:
    model_repo: str
    # Feature extractor
    sampling_rate: int
    num_mel_bins: int
    n_fft: int | None
    hop_length: int | None
    win_length: int | None
    chunk_length: int | None          # max audio seconds (model hard limit)
    # Tokenizer / generation
    vocab_size: int
    bos_token_id: int | None
    eos_token_id: int | None
    pad_token_id: int | None
    decoder_start_token_id: int | None
    # Model shape
    d_model: int | None           # encoder d_model (raw output of encoder.onnx)
    encoder_layers: int | None
    decoder_layers: int | None
    decoder_hidden_size: int | None  # decoder d_model (after encoder_decoder_proj)
    # Export
    opset: int
    dtype: str


def _tok_id(model: Any, attr: str) -> Any:
    """Read a token ID from generation_config first, falling back to model.config."""
    gen_cfg = getattr(model, "generation_config", None)
    val = getattr(gen_cfg, attr, None) if gen_cfg is not None else None
    if val is None:
        val = getattr(model.config, attr, None)
    return val


def _nested_get(obj: Any, *keys: str, default: Any = None) -> Any:
    """Walk a chain of dict/attr lookups, returning default if any step is missing."""
    cur = obj
    for key in keys:
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            cur = getattr(cur, key, None)
        if cur is None:
            return default
    return cur


def extract_config(
    repo_id: str,
    model: Any,
    processor: Any,
    opset: int,
    dtype_name: str,
) -> ExportConfig:
    fe = processor.feature_extractor
    cfg = model.config

    # CohereAsrFeatureExtractor stores mel params in a private _fb_config dict
    # (passed later to a lazily-constructed filterbank).  Flat getattr misses them.
    fb = getattr(fe, "_fb_config", {}) or {}

    def _fe(attr: str, fb_key: str | None = None) -> Any:
        """Check fe directly, then _fb_config under attr or an alternate key."""
        val = getattr(fe, attr, None)
        if val is None:
            val = fb.get(fb_key or attr)
        return val

    # This model stores encoder/decoder shape params in nested config dicts.
    enc_cfg = getattr(cfg, "encoder", {}) or {}
    dec_cfg = _nested_get(cfg, "transf_decoder", "config_dict") or {}

    return ExportConfig(
        model_repo=repo_id,
        sampling_rate=_fe("sampling_rate") or 16000,
        num_mel_bins=_fe("feature_size") or _fe("num_mel_bins"),
        n_fft=_fe("n_fft"),
        hop_length=_fe("hop_length", "n_window_stride"),
        win_length=_fe("win_length", "n_window_size"),
        chunk_length=_fe("chunk_length"),
        vocab_size=(getattr(cfg, "vocab_size", None)
                    or _nested_get(cfg, "head", "num_classes")),
        bos_token_id=_tok_id(model, "bos_token_id"),
        eos_token_id=_tok_id(model, "eos_token_id"),
        pad_token_id=_tok_id(model, "pad_token_id"),
        decoder_start_token_id=_tok_id(model, "decoder_start_token_id"),
        d_model=enc_cfg.get("d_model"),
        encoder_layers=enc_cfg.get("n_layers"),
        decoder_layers=dec_cfg.get("num_layers"),
        decoder_hidden_size=dec_cfg.get("hidden_size"),
        opset=opset,
        dtype=dtype_name,
    )


def write_export_report(
    config: ExportConfig,
    output_dir: Path,
    conventional_decoder: bool,
    skip_decoder: bool,
    notes: list[str],
) -> None:
    if skip_decoder:
        files = {
            "mel": "mel.onnx",
            "encoder": "encoder.onnx",
            "config": "config.json",
        }
        decoder_strategy = "skipped"
        decoder_notes = "Decoder export was skipped for this run."
    elif conventional_decoder:
        files = {
            "mel": "mel.onnx",
            "encoder": "encoder.onnx",
            "decoder": "decoder.onnx",
            "config": "config.json",
        }
        decoder_strategy = "no_kv_cache"
        decoder_notes = (
            "decoder.onnx takes the full token sequence at each step (no KV cache). "
            "For greedy decoding: call encoder once, then call decoder with growing "
            "decoder_input_ids, taking argmax of logits[:, -1, :] at each step."
        )
    else:
        files = {
            "mel": "mel.onnx",
            "encoder": "encoder.onnx",
            "decoder_init": "decoder_init.onnx",
            "decoder_step": "decoder_step.onnx",
            "config": "config.json",
        }
        decoder_strategy = "kv_cache"
        decoder_notes = (
            "decoder_init.onnx produces logits plus self/cross-attention KV for the "
            "initial context, then decoder_step.onnx consumes one token at a time "
            "with carried KV tensors for O(n) generation."
        )

    report = {
        **asdict(config),
        "files": files,
        "decoder_strategy": decoder_strategy,
        "decoder_notes": decoder_notes,
        "notes": notes,
    }
    config_path = output_dir / "config.json"
    with config_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote config: {config_path}")


def update_config_with_kv_decoder(output_dir: Path) -> None:
    cfg_path = output_dir / "config.json"
    with cfg_path.open() as f:
        cfg = json.load(f)

    cfg["files"]["decoder_init"] = "decoder_init.onnx"
    cfg["files"]["decoder_step"] = "decoder_step.onnx"
    cfg["decoder_kv_cache"] = {
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "decoder_init_inputs": ["decoder_input_ids [1,1] int64", "encoder_hidden_states [1,T,1280] float32"],
        "decoder_init_outputs": (
            ["logits [1,1,vocab] float32"]
            + [f"self_key_{i} [1,8,1,128] float32" for i in range(NUM_LAYERS)]
            + [f"self_val_{i} [1,8,1,128] float32" for i in range(NUM_LAYERS)]
            + [f"cross_key_{i} [1,8,T',128] float32" for i in range(NUM_LAYERS)]
            + [f"cross_val_{i} [1,8,T',128] float32" for i in range(NUM_LAYERS)]
        ),
        "decoder_step_inputs": (
            ["decoder_input_ids [1,1] int64", "positions [1,1] int64"]
            + [f"self_key_{i} [1,8,past,128] float32" for i in range(NUM_LAYERS)]
            + [f"self_val_{i} [1,8,past,128] float32" for i in range(NUM_LAYERS)]
            + [f"cross_key_{i} [1,8,T',128] float32" for i in range(NUM_LAYERS)]
            + [f"cross_val_{i} [1,8,T',128] float32" for i in range(NUM_LAYERS)]
        ),
        "decoder_step_outputs": (
            ["logits [1,1,vocab] float32"]
            + [f"new_self_key_{i} [1,8,past+1,128] float32" for i in range(NUM_LAYERS)]
            + [f"new_self_val_{i} [1,8,past+1,128] float32" for i in range(NUM_LAYERS)]
        ),
        "notes": [
            "Run decoder_init once with the start token and encoder hidden states.",
            "decoder_init outputs: logits + 8 self_key + 8 self_val + 8 cross_key + 8 cross_val.",
            "Run decoder_step for each subsequent token; pass self KV from previous step output.",
            "cross_key/cross_val come from decoder_init and are fixed for the whole utterance.",
            "decoder_step self KV outputs are already concatenated (past||new) and can be passed directly to the next step.",
            "positions input to decoder_step = number of tokens decoded so far (1 after init, 2 after step 1, ...).",
        ],
    }

    with cfg_path.open("w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\nUpdated {cfg_path}")


def export_tokenizer_assets(
    repo_id: str,
    revision: str | None,
    processor: Any,
    output_dir: Path,
) -> None:
    """
    Make the exported model directory self-contained for downstream runtimes.

    We always write vocab.json because the C# loader expects an index-ordered array
    of token strings. We also copy the original tokenizer artifacts from the local
    Hugging Face snapshot cache for debugging, future compatibility, and parity with
    the source model layout.
    """
    vocab = processor.tokenizer.convert_ids_to_tokens(
        list(range(processor.tokenizer.vocab_size))
    )
    vocab_path = output_dir / "vocab.json"
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"Wrote vocab: {vocab_path}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("WARNING: huggingface_hub unavailable; skipped copying tokenizer artifacts from cache.")
        return

    snapshot_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=[
                "tokenizer.json",
                "tokenizer.model",
                "tokenizer_config.json",
                "special_tokens_map.json",
            ],
        )
    )

    copied_any = False
    for name in (
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        src = snapshot_dir / name
        if not src.exists():
            continue
        shutil.copy2(src, output_dir / name)
        copied_any = True
        print(f"Copied tokenizer asset: {name}")

    if not copied_any:
        print("WARNING: no tokenizer artifacts were found in the Hugging Face snapshot cache.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output_dir, args.overwrite)

    # ---- Lazy imports -------------------------------------------------------
    try:
        import torch
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Install with:\n"
            "  pip install torch transformers accelerate onnx onnxruntime huggingface_hub\n"
            f"Original error: {exc}"
        ) from exc

    device = resolve_device(torch, args.device)
    dtype = resolve_dtype(torch, args.dtype)
    print(f"Device: {device}  |  dtype: {args.dtype}  |  opset: {args.opset}")

    if args.conventional_decoder and args.dtype != "float32" and not args.skip_decoder:
        print(
            "\nSkipping conventional decoder export (dtype != float32; "
            "decoder.onnx wrapper requires float32 graph I/O)."
        )
        args.skip_decoder = True

    # ---- Load model ---------------------------------------------------------
    model, processor = load_model_and_processor(args.model_repo, args.revision, device, dtype, torch)

    # ---- Build wrappers -----------------------------------------------------
    encoder_wrapper = make_encoder_wrapper(torch, model, model_dtype=dtype).to(device)
    encoder_wrapper.eval()
    decoder_wrapper = None
    if args.conventional_decoder and not args.skip_decoder:
        decoder_wrapper = make_decoder_wrapper(torch, model).to(device)
        decoder_wrapper.eval()

    # ---- Dummy inputs -------------------------------------------------------
    dummy_features, dummy_lengths = make_encoder_dummy_inputs(
        torch, processor, device, dtype, args.dummy_seconds
    )

    with torch.no_grad():
        enc_hidden = encoder_wrapper(dummy_features, dummy_lengths)

    # Decoder dummy uses B=1 — take the first item from the B=2 encoder output.
    decoder_input_ids = None
    enc_hidden_for_dec = None
    if args.conventional_decoder and not args.skip_decoder:
        # Keep the full B=2 encoder hidden so the decoder dummy stays B=2 and
        # torch.export does not specialise the decoder batch dim to 1.
        decoder_input_ids, enc_hidden_for_dec = make_decoder_dummy_inputs(
            torch, model, enc_hidden, device
        )

    legacy = args.legacy_exporter
    exporter_label = "legacy TorchScript" if legacy else "torch.export (dynamo)"
    print(f"ONNX exporter: {exporter_label}")

    # ---- Export encoder -----------------------------------------------------
    encoder_path = args.output_dir / "encoder.onnx"
    if not args.skip_encoder:
        export_encoder(torch, encoder_wrapper, dummy_features, dummy_lengths, encoder_path, args.opset, legacy)
        validate_onnx(onnx, encoder_path, "encoder")
        parity_check_encoder(ort, torch, encoder_wrapper, dummy_features, dummy_lengths, encoder_path)
    else:
        print("\nSkipping encoder export (--skip-encoder).")

    # ---- Export mel preprocessor -------------------------------------------
    mel_path = args.output_dir / "mel.onnx"
    if not args.skip_mel:
        mel_wrapper = make_mel_wrapper(torch, processor)
        mel_wrapper.eval()
        dummy_wave, dummy_wave_lens = export_mel(
            torch, mel_wrapper, processor, mel_path, args.opset, args.dummy_seconds, legacy
        )
        validate_onnx(onnx, mel_path, "mel")
        parity_check_mel(ort, torch, mel_wrapper, processor, dummy_wave, dummy_wave_lens, mel_path)
    else:
        print("\nSkipping mel export (--skip-mel).")

    # ---- Export decoder -----------------------------------------------------
    if args.conventional_decoder:
        decoder_path = args.output_dir / "decoder.onnx"
        if not args.skip_decoder:
            export_decoder(
                torch,
                decoder_wrapper,
                decoder_input_ids,
                enc_hidden_for_dec,
                decoder_path,
                args.opset,
                legacy,
            )
            validate_onnx(onnx, decoder_path, "decoder")
            parity_check_decoder(
                ort, torch, decoder_wrapper, decoder_input_ids, enc_hidden_for_dec, decoder_path
            )
        else:
            print("\nSkipping conventional decoder export (--skip-decoder).")
    else:
        if not args.skip_decoder:
            kv_init_wrapper = make_kv_init_wrapper(torch, model, dtype).to(device)
            kv_step_wrapper = make_kv_step_wrapper(torch, model, dtype).to(device)
            kv_init_wrapper.eval()
            kv_step_wrapper.eval()

            export_kv_init(torch, kv_init_wrapper, args.output_dir, args.opset, device, legacy)
            export_kv_step(torch, kv_step_wrapper, args.output_dir, args.opset, device, legacy)
            validate_onnx(onnx, args.output_dir / "decoder_init.onnx", "decoder_init")
            validate_onnx(onnx, args.output_dir / "decoder_step.onnx", "decoder_step")
            parity_check_kv(torch, ort, model, args.output_dir, device)
        else:
            print("\nSkipping KV-cache decoder export (--skip-decoder).")

    # ---- Config / report ----------------------------------------------------
    export_tokenizer_assets(args.model_repo, args.revision, processor, args.output_dir)

    config = extract_config(args.model_repo, model, processor, args.opset, args.dtype)
    write_export_report(
        config,
        args.output_dir,
        args.conventional_decoder,
        args.skip_decoder,
        notes=[
            "Model requires trust_remote_code=True when loading via transformers.",
            "mel.onnx: waveforms [B,T] float32 + waveforms_lens [B] int64 → features [B,128,F] + features_lens [B].",
            "mel.onnx uses DFT conv1d (no STFT op); dither is disabled — add 1e-5*randn in training but not inference.",
            "encoder.onnx: input_features [B,128,F] float32 → encoder_hidden_states [B,F/8,1280] float32.",
            "Default decoder export is KV-cache based: decoder_init.onnx + decoder_step.onnx.",
            "Pass --conventional-decoder to export decoder.onnx instead.",
            "Pass --revision <commit-or-tag> to pin the Hugging Face remote code and model files.",
            "vocab.json is generated from the tokenizer ID mapping; tokenizer.json/model metadata are copied from the Hugging Face snapshot cache when available.",
            "The model is gated on HuggingFace; run `huggingface-cli login` before export.",
        ],
    )

    if not args.conventional_decoder and not args.skip_decoder:
        update_config_with_kv_decoder(args.output_dir)

    print(f"\nExport complete.  Models written to: {args.output_dir}")


if __name__ == "__main__":
    main()
