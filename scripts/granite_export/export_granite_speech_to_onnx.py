#!/usr/bin/env python3
"""
Export ibm-granite/granite-speech-4.1-2b to ONNX format.

Outputs (default)
-----------------
    encoder.onnx
        input_features [batch, time_stacked, 160] float32
        -> encoder_hidden [batch, time_stacked, 1024] float32

    projector.onnx
        encoder_hidden [batch, time_stacked, 1024] float32
        -> audio_embeds [batch, audio_len, 2048] float32

    decoder_init.onnx
        Prefill graph that fuses audio embeddings into the prompt via
        masked_scatter on the audio_token_id (100352).
        input_ids        [batch, prompt_len]              int64
        audio_embeds     [batch, audio_len_padded, 2048]  float32
        audio_mask       [batch, audio_len_padded]        bool
        attention_mask   [batch, prompt_len]              int64
        -> logits        [batch, prompt_len, 100353]      float32
        -> present_key/value_<L>  40 x 2 x [batch, 4, prompt_len, 128]  float32

    decoder_step.onnx
        Single-token step on top of an existing KV cache.
        input_id         [batch, 1]                       int64
        attention_mask   [batch, total_len]               int64
        cache_position   [batch, 1]                       int64
        past_key/value_<L>  40 x 2 x [batch, 4, past_len, 128]  float32
        -> logits        [batch, 1, 100353]               float32
        -> present_key/value_<L>  40 x 2 x [batch, 4, past_len+1, 128]  float32

    config.json, processor assets (tokenizer + audio extractor),
    export-report.json

The mel frontend (`GraniteSpeechFeatureExtractor`) is intentionally NOT
exported as ONNX in this first cut. It uses torchaudio's MelSpectrogram
plus a frame-stacking step; running the algorithm on the host (C# or
Python smoke harness) avoids the dynamo-export risk for FFT ops. See
docs/dev/granite_speech_investigation.md for the rationale.

Usage
-----
    python public/scripts/granite_export/export_granite_speech_to_onnx.py \\
        --output-dir ./models/granite_speech_4_1_2b \\
        --opset 18

    # Run only one piece while iterating:
    python public/scripts/granite_export/export_granite_speech_to_onnx.py \\
        --output-dir ./models/granite_speech_4_1_2b \\
        --skip-projector --skip-decoder
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_MODEL_REPO = "ibm-granite/granite-speech-4.1-2b"
DEFAULT_OPSET = 18

# Granite-4.0-1b-base architecture constants (mirrored from text_config).
# These are used for KV-tensor naming and dummy shape construction; the
# values come from the loaded config at runtime, but we hardcode the
# ones that drive output_names lists below.
NUM_DECODER_LAYERS = 40
NUM_KV_HEADS = 4
HEAD_DIM = 128
DECODER_HIDDEN = 2048
VOCAB_SIZE = 100353
AUDIO_TOKEN_ID = 100352

# Encoder / projector constants (mirrored from encoder_config / projector_config).
ENCODER_INPUT_DIM = 160          # 80 mels x 2 (frame-stacked)
ENCODER_HIDDEN = 1024
PROJECTOR_WINDOW = 15
PROJECTOR_DOWNSAMPLE = 5         # 5x temporal downsample
PROJECTOR_OUT_DIM = DECODER_HIDDEN

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export ibm-granite/granite-speech-4.1-2b to ONNX.",
    )
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-mel", action="store_true")
    parser.add_argument("--skip-encoder", action="store_true")
    parser.add_argument("--skip-projector", action="store_true")
    parser.add_argument("--skip-decoder", action="store_true")
    parser.add_argument(
        "--unified-decoder",
        action="store_true",
        help=(
            "Export a single decoder.onnx that handles both prefill and step "
            "via past_kv with variable past_len. Eliminates the duplicate LM "
            "weights (fp32: 7 GB saved on disk and on GPU). The init/step "
            "pair is skipped when this is set."
        ),
    )
    parser.add_argument(
        "--legacy-exporter",
        action="store_true",
        help="Use the legacy TorchScript ONNX exporter instead of dynamo.",
    )
    parser.add_argument(
        "--static-shapes-probe",
        action="store_true",
        help=(
            "Run 15 phase 1 (issue #41): export the unified decoder with "
            "all dims pinned to small fixed values (B=4, S=1, past_len=8, "
            "audio_len=8). Step-only graph; not production-usable. The probe "
            "confirms whether the dynamo exporter folds the shape-arithmetic "
            "backbone (Run 13's CPU island) when dims are constants. "
            "Implies --unified-decoder."
        ),
    )
    parser.add_argument(
        "--dummy-seconds",
        type=float,
        default=2.0,
        help="Dummy audio length used to build trace inputs (default 2.0).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Output directory guard
# ---------------------------------------------------------------------------

_EXPORT_FILES = (
    "mel.onnx",
    "encoder.onnx",
    "projector.onnx",
    "decoder.onnx",
    "decoder_init.onnx",
    "decoder_step.onnx",
    "config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
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
    from transformers import AutoProcessor, GraniteSpeechForConditionalGeneration

    rev_suffix = f" @ {revision}" if revision else ""
    print(f"Loading processor from {repo_id}{rev_suffix} ...")
    processor = AutoProcessor.from_pretrained(repo_id, revision=revision)

    # `attn_implementation="eager"` keeps the language-model attention out of
    # transformers' sdpa_attention_forward path, which contains a data-dependent
    # branch (`attention_mask.shape[-1] != q.shape[-1]`) that the dynamo
    # exporter cannot guard. Eager is mathematically equivalent and traces cleanly.
    print(f"Loading model from {repo_id}{rev_suffix} on {device} as {dtype} (attn_implementation=eager) ...")
    model = GraniteSpeechForConditionalGeneration.from_pretrained(
        repo_id,
        revision=revision,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded {n_params:,} parameters ({n_params / 1e9:.2f}B)")
    return model, processor


# ---------------------------------------------------------------------------
# Dummy inputs
# ---------------------------------------------------------------------------

def make_dummy_processor_inputs(
    torch: Any, processor: Any, dummy_seconds: float, device: str
) -> dict[str, Any]:
    """Run the processor twice on dummy audio (full + half length) and stack
    into a B=2 batch. The dynamo exporter specializes any symbolic dim that
    is concretely 1 at trace time, so dummy inputs must be B>=2 with different
    lengths to keep batch and time dims symbolic in the exported graph.
    """
    sr = processor.audio_processor.sampling_rate
    n_full = int(dummy_seconds * sr)
    n_half = n_full // 2

    import numpy as np
    audio_full = np.zeros(n_full, dtype=np.float32)
    audio_half = np.zeros(n_half, dtype=np.float32)

    prompt = processor.tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "<|audio|>transcribe the speech with proper punctuation and capitalization.",
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    in_full = processor(prompt, audio_full, return_tensors="pt")
    in_half = processor(prompt, audio_half, return_tensors="pt")

    # input_features: pad shorter [..., T_half, 160] up to [..., T_full, 160].
    feats_full = in_full["input_features"]
    feats_half = in_half["input_features"]
    t_full = feats_full.shape[1]
    t_half = feats_half.shape[1]
    if t_half < t_full:
        feats_half = torch.nn.functional.pad(feats_half, (0, 0, 0, t_full - t_half))
    input_features = torch.cat([feats_full, feats_half], dim=0)

    # input_features_mask: pad with False up to the longer length.
    mask_full = in_full["input_features_mask"]
    mask_half = in_half["input_features_mask"]
    a_full = mask_full.shape[1]
    a_half = mask_half.shape[1]
    a_max = max(a_full, a_half)
    if a_full < a_max:
        mask_full = torch.nn.functional.pad(mask_full, (0, a_max - a_full), value=False)
    if a_half < a_max:
        mask_half = torch.nn.functional.pad(mask_half, (0, a_max - a_half), value=False)
    input_features_mask = torch.cat([mask_full, mask_half], dim=0)

    # input_ids / attention_mask: Granite Speech inserts one <|audio|> per valid
    # projector frame, so the prompt token count differs between full/half items.
    # Pad shorter ids with a non-audio token (pad_token_id) and shorter mask with 0.
    ids_full = in_full["input_ids"]
    ids_half = in_half["input_ids"]
    am_full = in_full["attention_mask"]
    am_half = in_half["attention_mask"]
    s_full = ids_full.shape[1]
    s_half = ids_half.shape[1]
    s_max = max(s_full, s_half)

    pad_id = processor.tokenizer.pad_token_id or 0
    if s_full < s_max:
        ids_full = torch.nn.functional.pad(ids_full, (0, s_max - s_full), value=pad_id)
        am_full = torch.nn.functional.pad(am_full, (0, s_max - s_full), value=0)
    if s_half < s_max:
        ids_half = torch.nn.functional.pad(ids_half, (0, s_max - s_half), value=pad_id)
        am_half = torch.nn.functional.pad(am_half, (0, s_max - s_half), value=0)
    input_ids = torch.cat([ids_full, ids_half], dim=0)
    attention_mask = torch.cat([am_full, am_half], dim=0)

    return {
        "input_features": input_features.to(device),
        "input_features_mask": input_features_mask.to(device),
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
    }


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------

def _patch_encoder_attention_for_export(torch: Any, encoder: Any) -> None:
    """Replace each GraniteSpeechConformerAttention.forward with a 4D-friendly
    manual-math version.

    Why: the original code calls F.scaled_dot_product_attention with 5D
    [bsz, num_blocks, num_heads, ctx, head_dim] tensors. PyTorch handles 5D
    SDPA at runtime via the MATH backend, but the dynamo->ONNX converter
    in torch 2.11 only registers a 4D adapter for aten.scaled_dot_product_attention
    and fails with `only 4D query, key, and value are supported`.

    The MATH backend is mathematically equivalent to:
        softmax(QK^T * scale + attn_mask) @ V
    Inlining that here avoids the converter limitation, with no numerical
    change (verified by parity in the smoke test).
    """
    import math
    import types

    def _patched_forward(self, hidden_states, attention_dists):
        """Full-attention reformulation of Granite Speech block attention.

        The upstream code does block attention by reshaping
        `[B, T, inner]` into `[B, num_blocks, ctx, num_heads, head_dim]` and
        running attention per-block. `num_blocks = ceil(T/ctx)` is a Python
        int derived from a tensor shape, which the dynamo->ONNX exporter
        consistently bakes as a static value at trace time — every
        downstream reshape ends up specialising `num_blocks * ctx` to its
        trace-time product (e.g. `200`) and fails at runtime whenever the
        audio length yields a different block count.

        The Cohere encoder export hits a similar shape of problem with
        `_needs_conv_split` and works around it by monkey-patching the
        Python control-flow that bakes the constant. The analogous fix
        here is to remove the block-reshape entirely and run **full
        attention with a block-diagonal mask** — mathematically identical,
        traces with no num_blocks dim. Cost is O(T^2) attention work
        instead of O(T*ctx); typical Vernacula segments (10-30 s,
        T<=1500) fit comfortably.

        The Shaw rel-pos bias is computed without materialising a
        `[T, T, head_dim]` lookup tensor (which would hit ~5 GB at 60 s).
        Instead we precompute `Q @ rel_pos_emb.weight.T` of shape
        `[B, H, T, num_indices]` and gather the per-pair bias by indexing
        with the [T, T] distance matrix.
        """
        hidden_states = self.pre_norm(hidden_states)
        bsz, num_features, _ = hidden_states.shape

        ctx = self.context_size
        H = self.num_heads
        D = self.dim_head
        device = hidden_states.device

        # Pad to multiple of ctx (always — branch-free).
        remainder = num_features % ctx
        pad_amt = (ctx - remainder) % ctx
        hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, pad_amt))
        T = hidden_states.shape[1]                                   # SymInt

        # Q/K/V: [B, T, inner] -> [B, T, H, D] -> [B, H, T, D].
        Q = self.to_q(hidden_states).unflatten(-1, (H, D)).transpose(1, 2)
        kv = self.to_kv(hidden_states).chunk(2, dim=-1)
        K = kv[0].unflatten(-1, (H, D)).transpose(1, 2)
        V = kv[1].unflatten(-1, (H, D)).transpose(1, 2)

        # Standard scaled dot-product attention over the full padded T.
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale   # [B, H, T, T]

        # Shaw relative positional bias.
        # rel_pos_emb is an Embedding of size (2*max_pos_emb+1, D).
        # We want pos_bias[b, h, i, j] = sum_d Q[b, h, i, d] * rel_pos_emb[idx(i, j), d] * scale,
        # where idx(i, j) = clamp(i - j, -ctx, ctx) + max_pos_emb.
        # Materialising rel_pos_emb at every (i, j) as a [T, T, D] tensor
        # blows up memory at long T (~5 GB at 60 s). Instead, dot Q with
        # the embedding weight matrix once, then gather per-pair.
        rel_pos_w = self.rel_pos_emb.weight                          # [num_indices, D]
        q_dot_emb = torch.matmul(Q, rel_pos_w.T)                     # [B, H, T, num_indices]

        positions = torch.arange(T, device=device)
        rel_dist = positions.view(-1, 1) - positions.view(1, -1)     # [T, T]
        rel_idx = rel_dist.clamp(-ctx, ctx) + self.max_pos_emb       # [T, T] in [0, 2*max_pos_emb]
        # gather: q_dot_emb has shape [B, H, T, num_indices].
        # We want pos_bias[b, h, i, j] = q_dot_emb[b, h, i, rel_idx[i, j]].
        # Expand rel_idx to [B, H, T, T] for gather.
        rel_idx_exp = rel_idx.unsqueeze(0).unsqueeze(0).expand(bsz, H, T, T)
        pos_bias = q_dot_emb.gather(-1, rel_idx_exp) * self.scale    # [B, H, T, T]

        # Block mask: i, j attend only when in the same block.
        block_idx = positions // ctx                                 # [T]
        same_block = block_idx.unsqueeze(0) == block_idx.unsqueeze(1)  # [T, T] bool
        # Apply rel-pos bias only within the same block.
        pos_bias = pos_bias * same_block.to(pos_bias.dtype)

        # Padded positions in the last block: positions >= num_features.
        # Mask both rows and columns.
        in_pad = positions >= num_features                           # [T] bool
        pad_pair = in_pad.unsqueeze(0) | in_pad.unsqueeze(1)         # [T, T] bool

        # Combine masks. Cross-block and padding positions get -inf so
        # softmax assigns them zero probability.
        attn_bool_mask = (~same_block) | pad_pair                    # [T, T] bool
        mask_value = -torch.finfo(scores.dtype).max
        attn_additive = attn_bool_mask.to(scores.dtype) * mask_value  # [T, T]

        scores = scores + pos_bias + attn_additive.unsqueeze(0).unsqueeze(0)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)                                  # [B, H, T, D]

        out = out.transpose(1, 2).flatten(-2, -1)                    # [B, T, inner]
        out = self.to_out(out[:, :num_features, :])
        return self.dropout(out)

    patched = 0
    for module in encoder.modules():
        if type(module).__name__ == "GraniteSpeechConformerAttention":
            module.forward = types.MethodType(_patched_forward, module)
            patched += 1
    print(f"  Patched {patched} GraniteSpeechConformerAttention layers (5D SDPA -> manual math).")


def make_mel_wrapper(torch: Any, processor: Any) -> Any:
    """Mel + frame-stack frontend, wrapping torchaudio's MelSpectrogram +
    Granite's GraniteSpeechFeatureExtractor post-processing chain.

    Inputs:
      audio [B, samples] float32

    Outputs:
      input_features [B, T_stacked, 160] float32

    The C# runtime can drive this graph instead of re-implementing
    torchaudio's HTK mel filterbank + the log-clamp-stack chain. We keep
    the upstream feature extractor's exact arithmetic — log10 with a
    -8 dB floor relative to the per-clip max, then divide by 4 and add 1
    — so parity with `model(input_features=...)` holds.

    Note: the upstream `if logmel.shape[1] % 2 == 1: drop last frame` is
    data-dependent. We replace it with an unconditional even-frame slice
    using `T = (T // 2) * 2` so the trace doesn't branch on parity.
    """
    nn = torch.nn

    fe = processor.audio_processor
    melspec_module = fe.mel_filters  # torchaudio.transforms.MelSpectrogram

    class MelWrapper(nn.Module):
        def __init__(self, melspec: Any) -> None:
            super().__init__()
            self.melspec = melspec

        def forward(self, audio: Any) -> Any:
            # torchaudio MelSpectrogram: [B, samples] -> [B, n_mels, T_mel]
            mel = self.melspec(audio.float())
            logmel = mel.transpose(-1, -2).clamp(min=1e-10).log10()
            # max over (T, n_mels) per batch item
            mx = logmel.amax(dim=(-2, -1), keepdim=True)
            logmel = torch.maximum(logmel, mx - 8.0).div(4).add(1)
            # Drop last frame if odd. Branch-free reformulation:
            T = logmel.shape[1]
            T_even = (T // 2) * 2
            logmel = logmel[:, :T_even]
            # Frame-stack pairs: [B, T_even, n_mels] -> [B, T_even/2, 2*n_mels]
            n_mels = logmel.shape[-1]
            return logmel.reshape(logmel.shape[0], T_even // 2, 2 * n_mels)

    return MelWrapper(melspec_module)


def make_encoder_wrapper(torch: Any, model: Any) -> Any:
    """Wrap GraniteSpeechCTCEncoder for ONNX export.

    The encoder forward is single-input single-output, but the per-layer
    attention uses 5D SDPA which the dynamo->ONNX converter does not
    support; see _patch_encoder_attention_for_export.

    Encoder always runs at fp32 — ONNX Conv only added BF16 type support at
    opset 22, and even there the export hits dtype-binding errors at Conv
    nodes where input/weight don't agree. Encoder is ~25% of wall time vs
    decoder's 60%+, so keeping it fp32 sacrifices a modest fraction of the
    BF16 win to avoid a deep ONNX/Conv compatibility hunt.
    """
    nn = torch.nn
    _patch_encoder_attention_for_export(torch, model.encoder)

    class EncoderWrapper(nn.Module):
        def __init__(self, enc: Any) -> None:
            super().__init__()
            self.encoder = enc

        def forward(self, input_features: Any) -> Any:
            return self.encoder(input_features)

    return EncoderWrapper(model.encoder)


def make_projector_wrapper(torch: Any, model: Any) -> Any:
    """Projector: pads to multiple of window_size=15, runs Q-Former with 3 queries
    per 15-frame block, downsamples 5x, projects to LLM hidden dim 2048.
    Single-input single-output.

    Projector also stays fp32 — same Conv compatibility caveat as encoder
    applies (Q-Former blocks include conv-flavoured ops). The decoder is
    where BF16 yields the dominant win, so projector stays at native dtype.
    """
    nn = torch.nn

    class ProjectorWrapper(nn.Module):
        def __init__(self, proj: Any) -> None:
            super().__init__()
            self.projector = proj

        def forward(self, encoder_hidden: Any) -> Any:
            return self.projector(encoder_hidden)

    return ProjectorWrapper(model.projector)


def make_decoder_init_wrapper(torch: Any, model: Any) -> Any:
    """Prefill graph.

    Inputs:
      input_ids       [B, S]    int64
      audio_embeds    [B, A, 2048] float32 (projector output; padded along A)
      attention_mask  [B, S]    int64

    Returns:
      logits          [B, S, V] float32
      present_key_<L>, present_value_<L> for L in 0..39 (NUM_DECODER_LAYERS)

    Audio fusion is mathematically equivalent to
    `GraniteSpeechForConditionalGeneration.get_merged_audio_embeddings`,
    but uses a cumsum + torch.gather + torch.where pattern instead of the
    boolean-indexing + masked_scatter pattern. Background:

      * The reference uses `audio_embeds[input_features_mask]` (NonZero +
        GatherND in ONNX) followed by `masked_scatter` (ScatterND). The
        dynamo->ONNX conversion of this pair traced numerically WRONG:
        prefill logits diverged by ~14 vs the PyTorch reference even at
        the trace shape, while step parity was tight to 1e-5. The bug is
        likely in the masked_scatter -> ScatterND translator (PyTorch's
        masked_scatter takes a 1-D-flat source and walks it sequentially;
        ScatterND needs explicit per-element [b, s] indices, which the
        translator must derive from the bool mask).
      * cumsum + gather + where avoids both NonZero and ScatterND. It
        also drops the need for an explicit audio_mask input: the cumsum
        of `is_audio` gives the per-batch audio index at every position,
        and gather + where assemble the merged embeddings without ever
        addressing the padded slots in audio_embeds (no audio token in
        input_ids points to them).
    """
    nn = torch.nn

    class DecoderInitWrapper(nn.Module):
        def __init__(self, m: Any) -> None:
            super().__init__()
            self.language_model = m.language_model
            self.audio_token_id = m.config.audio_token_id

        def forward(
            self,
            input_ids: Any,
            audio_embeds: Any,
            attention_mask: Any,
        ) -> Any:
            is_audio = input_ids == self.audio_token_id
            llm_input_ids = torch.where(is_audio, torch.zeros_like(input_ids), input_ids)
            text_embeds = self.language_model.get_input_embeddings()(llm_input_ids)

            # Per-position audio index: cumulative count of audio tokens up
            # to and including position s, minus 1 (0-based). At non-audio
            # positions the index is unused (torch.where below selects text).
            audio_idx = is_audio.long().cumsum(dim=1) - 1
            audio_idx = audio_idx.clamp(min=0)
            # gather along dim=1 across the audio axis. audio_embeds: [B, A, D]
            # indices: [B, S, D] (broadcast over D)
            d = audio_embeds.shape[-1]
            idx_expanded = audio_idx.unsqueeze(-1).expand(-1, -1, d)
            gathered_audio = torch.gather(audio_embeds, 1, idx_expanded)  # [B, S, D]

            embeds = torch.where(is_audio.unsqueeze(-1), gathered_audio, text_embeds)

            out = self.language_model(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
            kv = out.past_key_values
            keys = [layer.keys for layer in kv.layers]
            values = [layer.values for layer in kv.layers]
            return (out.logits, *keys, *values)

    return DecoderInitWrapper(model)


def make_decoder_unified_wrapper(torch: Any, model: Any) -> Any:
    """One ONNX graph that handles BOTH prefill and step.

    Eliminates the duplicate LM-weight copy that the split init/step pair
    requires (each currently 7 GB at fp32 = 14 GB resident on GPU). With
    a single graph, only one 7 GB LM copy is loaded.

    The graph is structurally a step graph: it always takes `past_key/value_<L>`
    inputs and a `cache_position`. To run prefill, the caller passes
    zero-length past_kv tensors and `cache_position=[0, 1, ..., S-1]`; to
    run a step, the caller passes populated past_kv and
    `cache_position=[past_len]`. HF's `GraniteForCausalLM.forward` handles
    empty `DynamicCache` correctly — verified directly before writing this.

    The audio fuse runs unconditionally on every call. At step time,
    `input_ids` is just the next-token id which never matches
    `audio_token_id`; the cumsum-gather + torch.where pattern collapses to
    a no-op (text_embeds wins everywhere). To keep the gather valid we
    require `audio_embeds.shape[1] >= 1` even at step time — caller passes
    a dummy 1-row tensor.

    Inputs:
      input_ids       [B, S]       int64
      audio_embeds    [B, A, 2048] float32  (A >= 1; padded; zero-fill at step)
      attention_mask  [B, T]       int64    (T = past_len + S)
      cache_position  [S]          int64
      past_key_<L>    [B, 4, past_len, 128] float32  for L in 0..39
      past_value_<L>  [B, 4, past_len, 128] float32

    Outputs:
      logits          [B, S, 100353]
      present_key_<L>, present_value_<L>  [B, 4, T, 128]
    """
    nn = torch.nn
    from transformers.cache_utils import DynamicCache

    weight_dtype = next(model.language_model.parameters()).dtype

    class DecoderUnifiedWrapper(nn.Module):
        def __init__(self, m: Any) -> None:
            super().__init__()
            self.language_model = m.language_model
            self.audio_token_id = m.config.audio_token_id
            self._weight_dtype = weight_dtype

        def forward(
            self,
            input_ids: Any,
            audio_embeds: Any,
            attention_mask: Any,
            cache_position: Any,
            *past_kv: Any,
        ) -> Any:
            # audio_embeds arrives as fp32 (projector stays fp32; see
            # make_projector_wrapper). Cast to LM weight dtype at the
            # boundary so all internal matmuls stay in weight dtype.
            audio_embeds = audio_embeds.to(self._weight_dtype)

            # Audio merge — same cumsum-gather-where as the prefill-only
            # wrapper. Collapses to a no-op at step time because no audio
            # tokens appear in the next-token input.
            is_audio = input_ids == self.audio_token_id
            llm_input_ids = torch.where(is_audio, torch.zeros_like(input_ids), input_ids)
            text_embeds = self.language_model.get_input_embeddings()(llm_input_ids)
            audio_idx = is_audio.long().cumsum(dim=1) - 1
            audio_idx = audio_idx.clamp(min=0)
            d = audio_embeds.shape[-1]
            idx_expanded = audio_idx.unsqueeze(-1).expand(-1, -1, d)
            gathered_audio = torch.gather(audio_embeds, 1, idx_expanded)
            embeds = torch.where(is_audio.unsqueeze(-1), gathered_audio, text_embeds)

            # Build a DynamicCache from the past_kv inputs. At prefill the
            # caller supplies zero-length tensors; HF treats this as an
            # empty cache and runs prefill normally. past_kv stays in the
            # weight dtype across the chain — ORT chains them between
            # `Run` calls without ever materialising on the host, so a
            # BF16 KV is half the memory bandwidth and concat cost vs fp32.
            n = NUM_DECODER_LAYERS
            past_keys = list(past_kv[:n])
            past_values = list(past_kv[n:])
            cache = DynamicCache.from_legacy_cache(tuple(zip(past_keys, past_values)))

            out = self.language_model(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                past_key_values=cache,
                cache_position=cache_position,
                use_cache=True,
                return_dict=True,
            )
            new_keys = [layer.keys for layer in out.past_key_values.layers]
            new_values = [layer.values for layer in out.past_key_values.layers]
            # Logits → fp32 boundary so C# argmax stays unchanged. KV stays
            # in weight dtype (uncast).
            return (out.logits.to(torch.float32), *new_keys, *new_values)

    return DecoderUnifiedWrapper(model)


def make_decoder_step_wrapper(torch: Any, model: Any) -> Any:
    """Single-token step.

    Inputs:
      input_id        [B, 1]     int64
      attention_mask  [B, T]     int64  (T = past_len + 1)
      cache_position  [1]        int64  (single value = past_len; HF convention is 1D)
      past_key_<L>, past_value_<L>  for L in 0..39, each [B, 4, past_len, 128]

    Returns:
      logits           [B, 1, V]
      present_key_<L>, present_value_<L>  each [B, 4, past_len + 1, 128]
    """
    nn = torch.nn
    from transformers.cache_utils import DynamicCache

    class DecoderStepWrapper(nn.Module):
        def __init__(self, m: Any) -> None:
            super().__init__()
            self.language_model = m.language_model

        def forward(
            self,
            input_id: Any,
            attention_mask: Any,
            cache_position: Any,
            *past_kv: Any,
        ) -> Any:
            n = NUM_DECODER_LAYERS
            past_keys = list(past_kv[:n])
            past_values = list(past_kv[n:])
            cache = DynamicCache.from_legacy_cache(
                tuple(zip(past_keys, past_values))
            )
            out = self.language_model(
                input_ids=input_id,
                attention_mask=attention_mask,
                past_key_values=cache,
                cache_position=cache_position,
                use_cache=True,
                return_dict=True,
            )
            new_keys = [layer.keys for layer in out.past_key_values.layers]
            new_values = [layer.values for layer in out.past_key_values.layers]
            return (out.logits, *new_keys, *new_values)

    return DecoderStepWrapper(model)


# ---------------------------------------------------------------------------
# Export helpers (mirrors cohere_export's style)
# ---------------------------------------------------------------------------

# 2 GiB protobuf wire-format cap minus headroom.
_INGRAPH_MAX_BYTES = (2 << 30) - (100 << 20)


def _wrapper_weight_bytes(wrapper: Any) -> int:
    total = 0
    for p in wrapper.parameters():
        total += p.numel() * p.element_size()
    for b in wrapper.buffers():
        total += b.numel() * b.element_size()
    return total


def _make_dim(torch: Any, name: str, *, min: int | None = None, max: int | None = None) -> Any:
    kwargs: dict[str, Any] = {}
    if min is not None:
        kwargs["min"] = min
    if max is not None:
        kwargs["max"] = max
    return torch.export.Dim(name, **kwargs)


def _auto_dim(torch: Any) -> Any:
    return torch.export.Dim.AUTO


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
) -> None:
    if legacy:
        torch.onnx.export(
            wrapper,
            args_tuple,
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            dynamo=False,
        )
        return

    modern_opset = max(opset, 18)
    if modern_opset != opset:
        print(f"  Bumping opset {opset} -> {modern_opset} for the dynamo exporter.")

    weight_bytes = _wrapper_weight_bytes(wrapper)
    use_external = weight_bytes > _INGRAPH_MAX_BYTES
    print(
        f"  Weights ~ {weight_bytes / 1024 / 1024:.1f} MB -> "
        f"{'external sidecar' if use_external else 'in-graph (no .data file)'}"
    )

    torch.onnx.export(
        wrapper,
        args_tuple,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_shapes=dynamic_shapes,
        opset_version=modern_opset,
        dynamo=True,
        external_data=use_external,
        optimize=True,
    )


def _consolidate_external_data(onnx_path: Path) -> None:
    """Sweep stale Constant_* files left by the dynamo exporter and
    report the final layout."""
    import onnx

    data_file = onnx_path.name + ".data"
    data_path = onnx_path.parent / data_file

    proto = onnx.load(str(onnx_path), load_external_data=False)
    has_external = any(
        init.data_location == onnx.TensorProto.EXTERNAL
        for init in proto.graph.initializer
    )
    if not has_external and data_path.exists():
        print(f"  Removing stale {data_file} (weights are now in-graph).")
        data_path.unlink()

    keep = {onnx_path.resolve(), data_path.resolve()}
    deleted = 0
    for pattern in ("Constant_*", "*Constant*"):
        for stale in onnx_path.parent.glob(pattern):
            stale_resolved = stale.resolve()
            if stale_resolved not in keep and stale.is_file():
                stale.unlink()
                deleted += 1
    if deleted:
        print(f"  Removed {deleted} scattered weight file(s).")

    if data_path.exists():
        total_mb = (onnx_path.stat().st_size + data_path.stat().st_size) / 1024 / 1024
        print(
            f"  Saved {onnx_path.name} "
            f"({onnx_path.stat().st_size / 1024:.0f} KB graph) + "
            f"{data_file} ({data_path.stat().st_size / 1024 / 1024:.1f} MB weights) "
            f"[total {total_mb:.1f} MB]"
        )
    else:
        print(
            f"  Saved {onnx_path.name} "
            f"({onnx_path.stat().st_size / 1024 / 1024:.1f} MB, weights in-graph)"
        )


# ---------------------------------------------------------------------------
# Per-module export entrypoints
# ---------------------------------------------------------------------------

def export_mel(
    torch: Any,
    wrapper: Any,
    dummy_audio: Any,
    output_path: Path,
    opset: int,
    legacy: bool,
) -> None:
    print(f"\nExporting mel to {output_path} ...")

    with torch.no_grad():
        out = wrapper(dummy_audio)
    print(f"  PyTorch mel output shape: {tuple(out.shape)}")

    batch = _make_dim(torch, "batch", min=1, max=65535)
    samples = _auto_dim(torch)
    _run_torch_export(
        torch,
        wrapper,
        (dummy_audio,),
        output_path,
        input_names=["audio"],
        output_names=["input_features"],
        dynamic_axes={
            "audio": {0: "batch", 1: "samples"},
            "input_features": {0: "batch", 1: "T_stacked"},
        },
        dynamic_shapes=({0: batch, 1: samples},),
        opset=opset,
        legacy=legacy,
    )
    _consolidate_external_data(output_path)


def export_encoder(
    torch: Any,
    wrapper: Any,
    dummy_features: Any,
    output_path: Path,
    opset: int,
    legacy: bool,
) -> None:
    print(f"\nExporting encoder to {output_path} ...")

    with torch.no_grad():
        out = wrapper(dummy_features)
    print(f"  PyTorch encoder output shape: {tuple(out.shape)}")

    batch = _make_dim(torch, "batch", min=1, max=65535)
    time_dim = _auto_dim(torch)
    _run_torch_export(
        torch,
        wrapper,
        (dummy_features,),
        output_path,
        input_names=["input_features"],
        output_names=["encoder_hidden"],
        dynamic_axes={
            "input_features": {0: "batch", 1: "time_stacked"},
            "encoder_hidden": {0: "batch", 1: "time_stacked"},
        },
        dynamic_shapes=({0: batch, 1: time_dim},),
        opset=opset,
        legacy=legacy,
    )
    _consolidate_external_data(output_path)


def export_projector(
    torch: Any,
    wrapper: Any,
    dummy_encoder_hidden: Any,
    output_path: Path,
    opset: int,
    legacy: bool,
) -> None:
    print(f"\nExporting projector to {output_path} ...")

    with torch.no_grad():
        out = wrapper(dummy_encoder_hidden)
    print(f"  PyTorch projector output shape: {tuple(out.shape)}")

    batch = _make_dim(torch, "batch", min=1, max=65535)
    time_dim = _auto_dim(torch)
    _run_torch_export(
        torch,
        wrapper,
        (dummy_encoder_hidden,),
        output_path,
        input_names=["encoder_hidden"],
        output_names=["audio_embeds"],
        dynamic_axes={
            "encoder_hidden": {0: "batch", 1: "time_stacked"},
            "audio_embeds":   {0: "batch", 1: "audio_len"},
        },
        dynamic_shapes=({0: batch, 1: time_dim},),
        opset=opset,
        legacy=legacy,
    )
    _consolidate_external_data(output_path)


def _kv_names(prefix: str) -> list[str]:
    return [f"{prefix}_{i}" for i in range(NUM_DECODER_LAYERS)]


def export_decoder_init(
    torch: Any,
    wrapper: Any,
    inputs: dict[str, Any],
    output_path: Path,
    opset: int,
    legacy: bool,
) -> None:
    print(f"\nExporting decoder_init to {output_path} ...")

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    audio_embeds = inputs["audio_embeds"]

    with torch.no_grad():
        out = wrapper(input_ids, audio_embeds, attention_mask)
    logits = out[0]
    print(f"  PyTorch logits shape: {tuple(logits.shape)}, "
          f"present_key_0 shape: {tuple(out[1].shape)}")

    output_names = (
        ["logits"]
        + _kv_names("present_key")
        + _kv_names("present_value")
    )

    batch = _make_dim(torch, "batch", min=1, max=65535)
    prompt_len = _auto_dim(torch)
    audio_len = _auto_dim(torch)

    dynamic_axes: dict[str, dict[int, str]] = {
        "input_ids":      {0: "batch", 1: "prompt_len"},
        "audio_embeds":   {0: "batch", 1: "audio_len"},
        "attention_mask": {0: "batch", 1: "prompt_len"},
        "logits":         {0: "batch", 1: "prompt_len"},
    }
    for name in _kv_names("present_key") + _kv_names("present_value"):
        dynamic_axes[name] = {0: "batch", 2: "prompt_len"}

    _run_torch_export(
        torch,
        wrapper,
        (input_ids, audio_embeds, attention_mask),
        output_path,
        input_names=["input_ids", "audio_embeds", "attention_mask"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamic_shapes=(
            {0: batch, 1: prompt_len},
            {0: batch, 1: audio_len},
            {0: batch, 1: prompt_len},
        ),
        opset=opset,
        legacy=legacy,
    )
    _consolidate_external_data(output_path)


def export_decoder_unified(
    torch: Any,
    wrapper: Any,
    inputs: dict[str, Any],
    output_path: Path,
    opset: int,
    legacy: bool,
    static_shapes_probe: bool = False,
) -> None:
    """Export the unified prefill+step graph.

    The trace dummy uses S=2 (mid-prompt) and past_len=2 (mid-cache) so
    BOTH dims stay symbolic. Specialising either to a fixed value would
    break one of the two runtime modes.

    With ``static_shapes_probe=True`` (Run 15 phase 1 / issue #41), all
    five dims (batch, seq, past_len, total_len, audio_len) are pinned to
    integer literals so the dynamo exporter can constant-fold the
    per-step shape-arithmetic backbone identified in Run 13. The result
    is a step-only graph, not production-usable; phase 1 just confirms
    whether the constant folding works at all.
    """
    print(f"\nExporting decoder_unified to {output_path}"
          f"{' (static-shapes probe)' if static_shapes_probe else ''} ...")

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    audio_embeds = inputs["audio_embeds"]
    cache_position = inputs["cache_position"]
    keys = inputs["past_keys"]
    values = inputs["past_values"]

    with torch.no_grad():
        out = wrapper(input_ids, audio_embeds, attention_mask, cache_position, *keys, *values)
    print(f"  PyTorch logits shape: {tuple(out[0].shape)}, "
          f"present_key_0 shape: {tuple(out[1].shape)}")

    output_names = (
        ["logits"]
        + _kv_names("present_key")
        + _kv_names("present_value")
    )
    input_names = (
        ["input_ids", "audio_embeds", "attention_mask", "cache_position"]
        + _kv_names("past_key")
        + _kv_names("past_value")
    )

    if static_shapes_probe:
        # Phase-1 probe: pin every dim to its trace-dummy value. The
        # dummy was built at B=input_ids.shape[0], S=input_ids.shape[1],
        # past=past_keys[0].shape[2], audio_len=audio_embeds.shape[1].
        # Empty dynamic_axes / dynamic_shapes => the dynamo exporter
        # treats the trace-dummy shape as the static shape.
        dynamic_axes: dict[str, dict[int, str]] = {}
        dyn_shapes = ({}, {}, {}, {}) + (tuple({} for _ in range(2 * NUM_DECODER_LAYERS)),)
    else:
        batch = _make_dim(torch, "batch", min=1, max=65535)
        seq = _auto_dim(torch)         # input_ids[1] / cache_position[0] / new positions
        past = _auto_dim(torch)        # past_kv[2]
        total = _auto_dim(torch)       # attention_mask[1] / present_kv[2]
        audio_len = _auto_dim(torch)   # audio_embeds[1]

        dynamic_axes = {
            "input_ids":      {0: "batch", 1: "seq"},
            "audio_embeds":   {0: "batch", 1: "audio_len"},
            "attention_mask": {0: "batch", 1: "total_len"},
            # cache_position is 1-D; its single axis is "seq".
            "cache_position": {0: "seq"},
            "logits":         {0: "batch", 1: "seq"},
        }
        for name in _kv_names("past_key") + _kv_names("past_value"):
            dynamic_axes[name] = {0: "batch", 2: "past_len"}
        for name in _kv_names("present_key") + _kv_names("present_value"):
            dynamic_axes[name] = {0: "batch", 2: "total_len"}

        past_kv_shapes = tuple({0: batch, 2: past} for _ in range(2 * NUM_DECODER_LAYERS))
        dyn_shapes = (
            {0: batch, 1: seq},          # input_ids
            {0: batch, 1: audio_len},    # audio_embeds
            {0: batch, 1: total},        # attention_mask
            {0: seq},                    # cache_position
            past_kv_shapes,              # *past_kv
        )

    _run_torch_export(
        torch,
        wrapper,
        (input_ids, audio_embeds, attention_mask, cache_position, *keys, *values),
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamic_shapes=dyn_shapes,
        opset=opset,
        legacy=legacy,
    )
    _consolidate_external_data(output_path)


def export_decoder_step(
    torch: Any,
    wrapper: Any,
    init_inputs: dict[str, Any],
    init_outputs: tuple,
    output_path: Path,
    opset: int,
    legacy: bool,
) -> None:
    print(f"\nExporting decoder_step to {output_path} ...")

    # Build dummy step inputs from the prefill outputs.
    batch_size = init_inputs["input_ids"].shape[0]
    past_len = init_inputs["input_ids"].shape[1]
    device = init_inputs["input_ids"].device

    input_id = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
    # HF convention: cache_position is 1D of length seq_len (= 1 for a step).
    cache_position = torch.tensor([past_len], dtype=torch.long, device=device)
    attention_mask = torch.ones((batch_size, past_len + 1), dtype=torch.long, device=device)

    keys = init_outputs[1 : 1 + NUM_DECODER_LAYERS]
    values = init_outputs[1 + NUM_DECODER_LAYERS : 1 + 2 * NUM_DECODER_LAYERS]

    with torch.no_grad():
        out = wrapper(input_id, attention_mask, cache_position, *keys, *values)
    print(f"  PyTorch step logits shape: {tuple(out[0].shape)}")

    input_names = (
        ["input_id", "attention_mask", "cache_position"]
        + _kv_names("past_key")
        + _kv_names("past_value")
    )
    output_names = (
        ["logits"]
        + _kv_names("present_key")
        + _kv_names("present_value")
    )

    batch = _make_dim(torch, "batch", min=1, max=65535)
    past = _auto_dim(torch)
    total = _auto_dim(torch)

    dynamic_axes: dict[str, dict[int, str]] = {
        "input_id": {0: "batch"},
        "attention_mask": {0: "batch", 1: "total_len"},
        # cache_position is 1D [seq_len]; the step graph fixes seq_len=1
        # so this axis is static and we omit it from dynamic_axes.
        "logits": {0: "batch"},
    }
    for name in _kv_names("past_key") + _kv_names("past_value"):
        dynamic_axes[name] = {0: "batch", 2: "past_len"}
    for name in _kv_names("present_key") + _kv_names("present_value"):
        dynamic_axes[name] = {0: "batch", 2: "total_len"}

    # The wrapper's signature is (input_id, attention_mask, cache_position, *past_kv),
    # so torch.export sees 4 top-level args (the varargs collapses into one).
    # Mirror that with the dynamic_shapes structure: 3 dicts + a tuple of 80 dicts.
    past_kv_shapes = tuple({0: batch, 2: past} for _ in range(2 * NUM_DECODER_LAYERS))
    dyn_shapes = (
        {0: batch},          # input_id
        {0: batch, 1: total},  # attention_mask
        {},                  # cache_position: [1], static
        past_kv_shapes,      # *past_kv as a single nested tuple
    )

    _run_torch_export(
        torch,
        wrapper,
        (input_id, attention_mask, cache_position, *keys, *values),
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamic_shapes=dyn_shapes,
        opset=opset,
        legacy=legacy,
    )
    _consolidate_external_data(output_path)


# ---------------------------------------------------------------------------
# Tokenizer / processor asset copy
# ---------------------------------------------------------------------------

def copy_processor_assets(processor: Any, output_dir: Path) -> list[str]:
    """Save the processor's tokenizer + audio extractor assets next to the
    ONNX files so a Vernacula runtime has everything it needs locally."""
    print("\nSaving processor assets ...")
    processor.save_pretrained(str(output_dir))
    saved = sorted(p.name for p in output_dir.iterdir() if p.is_file())
    print(f"  Saved {len(saved)} files: {', '.join(saved)}")
    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    try:
        import torch
    except ImportError:
        print(
            "torch is not installed. Install requirements first:\n"
            "  pip install -r public/scripts/granite_export/requirements.txt",
            file=sys.stderr,
        )
        return 2

    ensure_output_dir(args.output_dir, args.overwrite)

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    model, processor = load_model_and_processor(
        args.model_repo, args.revision, args.device, dtype, torch
    )

    # Mixed-precision policy: when --dtype is BF16 (or fp16), only the
    # language-model decoder runs in the reduced precision. Encoder and
    # projector are cast back to fp32 because ONNX Conv has historically
    # patchy reduced-precision support (BF16 added at opset 22 but with
    # type-binding errors on the Conformer's depthwise convs at trace),
    # and they account for ~25% of inference time vs the decoder's 60%+.
    if dtype != torch.float32:
        print(f"  Casting encoder + projector back to fp32 (mixed-precision policy)")
        model.encoder = model.encoder.to(torch.float32)
        model.projector = model.projector.to(torch.float32)

    inputs = make_dummy_processor_inputs(
        torch, processor, args.dummy_seconds, args.device
    )
    print(
        "  Dummy processor outputs: "
        + ", ".join(f"{k}={tuple(v.shape)}" for k, v in inputs.items() if hasattr(v, "shape"))
    )
    input_features = inputs["input_features"]
    input_features_mask = inputs["input_features_mask"]
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    mel_wrapper = make_mel_wrapper(torch, processor)
    encoder_wrapper = make_encoder_wrapper(torch, model)
    projector_wrapper = make_projector_wrapper(torch, model)

    report: dict[str, Any] = {
        "model_repo": args.model_repo,
        "revision": args.revision,
        "device": args.device,
        "dtype": args.dtype,
        "opset": args.opset,
        "exporter": "legacy" if args.legacy_exporter else "dynamo",
        "stages": {},
    }

    # -------- Mel --------
    if not args.skip_mel:
        # B=2 mixed-length dummy audio: full and half. Using `torch.zeros`
        # yields a degenerate logmel max (-inf-ish), so we use a small
        # nonzero signal to keep the trace numerics stable. The graph
        # itself is not value-dependent past the trace.
        sr = processor.audio_processor.sampling_rate
        n_full = int(args.dummy_seconds * sr)
        n_half = n_full // 2
        dummy_full = torch.full((n_full,), 1e-3, dtype=torch.float32, device=args.device)
        dummy_half = torch.full((n_half,), 1e-3, dtype=torch.float32, device=args.device)
        dummy_half_padded = torch.nn.functional.pad(dummy_half, (0, n_full - n_half))
        dummy_audio = torch.stack([dummy_full, dummy_half_padded], dim=0)
        t0 = time.time()
        export_mel(
            torch,
            mel_wrapper,
            dummy_audio,
            args.output_dir / "mel.onnx",
            args.opset,
            args.legacy_exporter,
        )
        report["stages"]["mel"] = {"seconds": round(time.time() - t0, 2)}

    # -------- Encoder --------
    enc_out: Any = None
    if not args.skip_encoder:
        t0 = time.time()
        export_encoder(
            torch,
            encoder_wrapper,
            input_features,
            args.output_dir / "encoder.onnx",
            args.opset,
            args.legacy_exporter,
        )
        report["stages"]["encoder"] = {"seconds": round(time.time() - t0, 2)}
    with torch.no_grad():
        enc_out = encoder_wrapper(input_features)

    # -------- Projector --------
    proj_out: Any = None
    if not args.skip_projector:
        t0 = time.time()
        export_projector(
            torch,
            projector_wrapper,
            enc_out,
            args.output_dir / "projector.onnx",
            args.opset,
            args.legacy_exporter,
        )
        report["stages"]["projector"] = {"seconds": round(time.time() - t0, 2)}
    with torch.no_grad():
        proj_out = projector_wrapper(enc_out)

    # -------- Decoder (unified or split) --------
    # --static-shapes-probe implies --unified-decoder.
    use_unified = args.unified_decoder or args.static_shapes_probe
    if use_unified and not args.skip_decoder:
        unified_wrapper = make_decoder_unified_wrapper(torch, model)
        # Trace dummy. Without --static-shapes-probe: B=input_ids.shape[0],
        # S=2 (mid-prompt), past=2 (mid-cache); both non-zero so dynamo doesn't
        # specialise either to a fixed value.
        # With --static-shapes-probe (Run 15 phase 1): B=4, S=1 (step-only),
        # past=8, audio_len=8. Small fixed values; the export pins every dim
        # so the constant folder can collapse the shape-arithmetic backbone.
        if args.static_shapes_probe:
            bsz = 4
            seq_dummy = 1
            past_dummy = 8
            audio_dummy_len = 8
        else:
            bsz = input_ids.shape[0]
            seq_dummy = 2
            past_dummy = 2
            audio_dummy_len = None  # use proj_out as-is below
        # Take the first `seq_dummy` columns of input_ids/attention_mask, then
        # extend attention_mask to length seq_dummy + past_dummy so the
        # cache+seq alignment matches. In probe mode, build fresh dummies at
        # the pinned dims rather than slicing proj_out (whose batch / audio_len
        # came from the real audio).
        if args.static_shapes_probe:
            audio_hidden = proj_out.shape[-1]
            u_input_ids = torch.zeros((bsz, seq_dummy), dtype=input_ids.dtype, device=input_ids.device)
            u_audio_embeds = torch.zeros(
                (bsz, audio_dummy_len, audio_hidden), dtype=proj_out.dtype, device=proj_out.device
            )
            u_attention_mask = torch.ones(
                (bsz, seq_dummy + past_dummy), dtype=attention_mask.dtype, device=attention_mask.device
            )
        else:
            u_input_ids = input_ids[:, :seq_dummy]
            u_audio_embeds = proj_out
            u_attention_mask = torch.ones(
                (bsz, seq_dummy + past_dummy), dtype=attention_mask.dtype, device=attention_mask.device
            )
        u_cache_position = torch.arange(
            past_dummy, past_dummy + seq_dummy, dtype=torch.long, device=input_ids.device
        )
        # Past-KV must match decoder weight dtype — the LM's K/V projection
        # outputs land in `dtype`, and the in-graph concat across the past_len
        # axis fails if the supplied past tensor doesn't match.
        u_past_keys = [
            torch.zeros((bsz, NUM_KV_HEADS, past_dummy, HEAD_DIM), dtype=dtype, device=input_ids.device)
            for _ in range(NUM_DECODER_LAYERS)
        ]
        u_past_values = [
            torch.zeros((bsz, NUM_KV_HEADS, past_dummy, HEAD_DIM), dtype=dtype, device=input_ids.device)
            for _ in range(NUM_DECODER_LAYERS)
        ]
        unified_inputs = {
            "input_ids": u_input_ids,
            "audio_embeds": u_audio_embeds,
            "attention_mask": u_attention_mask,
            "cache_position": u_cache_position,
            "past_keys": u_past_keys,
            "past_values": u_past_values,
        }
        t0 = time.time()
        export_decoder_unified(
            torch,
            unified_wrapper,
            unified_inputs,
            args.output_dir / "decoder.onnx",
            args.opset,
            args.legacy_exporter,
            static_shapes_probe=args.static_shapes_probe,
        )
        report["stages"]["decoder_unified"] = {"seconds": round(time.time() - t0, 2)}
        if args.static_shapes_probe:
            report["stages"]["decoder_unified"]["static_shapes_probe"] = True
            report["stages"]["decoder_unified"]["pinned_dims"] = {
                "batch": bsz, "seq": seq_dummy, "past_len": past_dummy,
                "audio_len": audio_dummy_len,
            }
    elif not args.skip_decoder:
        decoder_init_wrapper = make_decoder_init_wrapper(torch, model)
        decoder_step_wrapper = make_decoder_step_wrapper(torch, model)

        # The prefill graph expects projector output + the same
        # input_features_mask the merger uses to gate audio frames.
        decoder_init_inputs = {
            "input_ids": input_ids,
            "audio_embeds": proj_out,
            "attention_mask": attention_mask,
        }
        t0 = time.time()
        export_decoder_init(
            torch,
            decoder_init_wrapper,
            decoder_init_inputs,
            args.output_dir / "decoder_init.onnx",
            args.opset,
            args.legacy_exporter,
        )
        report["stages"]["decoder_init"] = {"seconds": round(time.time() - t0, 2)}

        with torch.no_grad():
            init_out = decoder_init_wrapper(
                input_ids, proj_out, attention_mask
            )

        t0 = time.time()
        export_decoder_step(
            torch,
            decoder_step_wrapper,
            decoder_init_inputs,
            init_out,
            args.output_dir / "decoder_step.onnx",
            args.opset,
            args.legacy_exporter,
        )
        report["stages"]["decoder_step"] = {"seconds": round(time.time() - t0, 2)}

    # -------- Tokenizer / processor assets --------
    saved = copy_processor_assets(processor, args.output_dir)
    report["processor_assets"] = saved

    # -------- Export report --------
    report_path = args.output_dir / "export-report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {report_path}")
    print("Export complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
