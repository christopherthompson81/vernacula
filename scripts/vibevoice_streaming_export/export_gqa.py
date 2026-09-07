#!/usr/bin/env python
"""Export the streaming decoder with com.microsoft.GroupQueryAttention and a shared
(pre-allocated, in-place) KV cache.

Why: the dynamic cache Concats the whole past into a new tensor every step (O(cache) copy,
14% of node time in the sibling port's profile) and grows without bound; the static-KV
export bounds memory but pays attention over the whole buffer every step (Run 8: 2x slower).
GQA does both right — attention costs only the filled length, the cache is updated in place
in a buffer allocated once — and, because the shapes are now fixed, it is the precondition
for CUDA graph capture against the 56% dispatch overhead.

Cache dtype is float16 by design: the float32 cache falls off the flash kernel and is 60x
slower (Run 11 addendum), while float16 GQA is *more* accurate than the BF16 build we
shipped.

Graph:
  inputs   prefix_input_ids [1,P] int64, audio_embeddings [N,hidden] fp16,
           suffix_input_ids [1,S] int64, seqlens_k [1] int32 (= total_seq - 1),
           total_sequence_length [1] int32, past_key/value_i [1,KH,MAX,HD] fp16
  outputs  logits [1,P+N+S,vocab], present_key/value_i (the same buffers, updated)
The caller binds past_key_i and present_key_i to the SAME device tensor.
"""
from __future__ import annotations

import argparse, gc, json, shutil, sys, time
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "vibevoice_export"))
from export_vibevoice_asr_to_onnx import export_onnx_graph  # noqa: E402
from export_vibevoice_streaming_to_onnx import (  # noqa: E402
    METADATA_FILES, load, tokenizer_extras,
)


@torch.library.custom_op("vernacula::gqa", mutates_args=())
def _gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
         past_k: torch.Tensor, past_v: torch.Tensor,
         seqlens_k: torch.Tensor, total_seq: torch.Tensor,
         num_heads: int, kv_num_heads: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Placeholder for com.microsoft::GroupQueryAttention.

    Never computes anything real: the export only needs an op in the traced graph with the
    right output shapes and dtypes, which the symbolic below rewrites into the ONNX node.
    (torch.autograd.Function no longer traces on 2.11, hence a registered custom op.)
    """
    # Clones, not the inputs themselves: a custom op may not return a tensor that aliases
    # one of its inputs. In the real ONNX node present_* IS past_* (that is the whole point
    # of the shared buffer); the aliasing happens at bind time in the runtime, not here.
    return torch.zeros_like(q), past_k.clone(), past_v.clone()


@_gqa.register_fake
def _(q, k, v, past_k, past_v, seqlens_k, total_seq, num_heads, kv_num_heads):
    return torch.empty_like(q), torch.empty_like(past_k), torch.empty_like(past_v)


def _gqa_symbolic(g, q, k, v, past_k, past_v, seqlens_k, total_seq, num_heads, kv_num_heads):
    # The head counts reach the symbolic as traced Constant values, not Python ints, and the
    # *_i attribute setters need real ints.
    from torch.onnx._internal.torchscript_exporter import symbolic_helper as sh
    return g.op("com.microsoft::GroupQueryAttention", q, k, v, past_k, past_v,
                seqlens_k, total_seq,
                num_heads_i=sh._parse_arg(num_heads, "i"),
                kv_num_heads_i=sh._parse_arg(kv_num_heads, "i"), outputs=3)


class GqaState:
    """Per-call scratch the patched attention reads: the buffers and the two length inputs."""
    buffers: list = []          # [k0, v0, k1, v1, ...]
    seqlens_k = None
    total_seq = None
    present: list = []          # collected [k0, v0, ...] outputs, in layer order


def gqa_attention_forward(self, hidden_states, position_embeddings, attention_mask,
                          past_key_values=None, cache_position=None, **kwargs):
    """Replacement for Qwen2Attention.forward.

    RoPE is applied here (do_rotary=0 on the node), exactly as the eager path does, so the
    only thing that changes is who computes softmax(QK^T)V and where the cache lives. The
    causal mask is GQA's own business: it derives it from seqlens_k, so `attention_mask` is
    deliberately unused.
    """
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    b, s, _ = hidden_states.shape
    q = self.q_proj(hidden_states).view(b, s, -1, self.head_dim).transpose(1, 2)
    k = self.k_proj(hidden_states).view(b, s, -1, self.head_dim).transpose(1, 2)
    v = self.v_proj(hidden_states).view(b, s, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    # GQA wants BSND packed as [batch, seq, heads * head_dim].
    q = q.transpose(1, 2).reshape(b, s, -1)
    k = k.transpose(1, 2).reshape(b, s, -1)
    v = v.transpose(1, 2).reshape(b, s, -1)

    i = self.layer_idx
    out, pk, pv = torch.ops.vernacula.gqa(
        q, k, v, GqaState.buffers[2 * i], GqaState.buffers[2 * i + 1],
        GqaState.seqlens_k, GqaState.total_seq,
        self.config.num_attention_heads, self.config.num_key_value_heads)
    GqaState.present.append(pk)
    GqaState.present.append(pv)
    return self.o_proj(out), None


class DecoderGqa(nn.Module):
    def __init__(self, inner, f32_lm_head=False):
        super().__init__()
        self.lm = inner.model.language_model
        self.lm_head = inner.lm_head
        self.f32_lm_head = f32_lm_head

    def forward(self, prefix_input_ids, audio_embeddings, suffix_input_ids,
                seqlens_k, total_seq, *kv_buffers):
        emb = self.lm.embed_tokens
        x = torch.cat((emb(prefix_input_ids),
                       audio_embeddings.to(emb.weight.dtype).unsqueeze(0),
                       emb(suffix_input_ids)), dim=1)
        q = x.shape[1]
        # Positions start where the cache currently ends: total_seq counts this call's tokens.
        past = total_seq.to(torch.int64).reshape(()) - q
        position_ids = torch.arange(q, device=x.device).unsqueeze(0) + past

        GqaState.buffers = list(kv_buffers)
        GqaState.seqlens_k = seqlens_k
        GqaState.total_seq = total_seq
        GqaState.present = []
        # Hand the model a mask dict so it does NOT call create_causal_mask: that helper
        # builds the mask with torch.vmap, which the TorchScript exporter cannot trace. The
        # value is never read — the patched attention masks via GQA's seqlens_k instead.
        dummy_mask = torch.zeros((1, 1, 1, 1), dtype=x.dtype, device=x.device)
        out = self.lm(inputs_embeds=x, position_ids=position_ids,
                      attention_mask={"full_attention": dummy_mask},
                      past_key_values=None, use_cache=False)
        h = out.last_hidden_state
        logits = (torch.nn.functional.linear(h.to(torch.float32), self.lm_head.weight.to(torch.float32))
                  if self.f32_lm_head else self.lm_head(h))
        present = GqaState.present
        GqaState.buffers, GqaState.present = [], []
        return (logits, *present)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-tokens", type=int, default=16384, help="KV buffer ceiling")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--f32-lm-head", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{out} is not empty; pass --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    mp = Path(args.model_path)
    pc = json.loads((mp / "preprocessor_config.json").read_text())

    t0 = time.time()
    model, proc = load(args.model_path, torch.float16, args.device, "eager")
    print(f"loaded in {time.time() - t0:.1f}s")
    dc = model.config.decoder_config
    L, KH = dc.num_hidden_layers, dc.num_key_value_heads
    HD, hidden = dc.hidden_size // dc.num_attention_heads, dc.hidden_size
    extras = tokenizer_extras(proc.tokenizer, pc)
    for f in METADATA_FILES:
        if (mp / f).exists():
            shutil.copy(mp / f, out / f)

    torch.onnx.register_custom_op_symbolic("vernacula::gqa", _gqa_symbolic, args.opset)
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
    original = Qwen2Attention.forward
    Qwen2Attention.forward = gqa_attention_forward
    try:
        dec = DecoderGqa(model, args.f32_lm_head).eval()
        pfx = torch.tensor([extras["prompt_token_ids"][:5]], dtype=torch.long, device=args.device)
        aud = torch.zeros((4, hidden), dtype=torch.float16, device=args.device)
        sfx = torch.tensor([[extras["speech_end_id"]]], dtype=torch.long, device=args.device)
        seq = 10
        seqlens = torch.tensor([seq - 1], dtype=torch.int32, device=args.device)
        total = torch.tensor([seq], dtype=torch.int32, device=args.device)
        kv = [torch.zeros((1, KH, args.max_tokens, HD), dtype=torch.float16, device=args.device)
              for _ in range(2 * L)]
        names_in = [n for i in range(L) for n in (f"past_key_{i}", f"past_value_{i}")]
        names_out = [n for i in range(L) for n in (f"present_key_{i}", f"present_value_{i}")]
        print(f"exporting decoder_gqa.onnx (max_tokens {args.max_tokens}, fp16 cache) ...")
        export_onnx_graph(
            model=dec, args=(pfx, aud, sfx, seqlens, total, *kv),
            output_path=out / "decoder_gqa.onnx",
            input_names=["prefix_input_ids", "audio_embeddings", "suffix_input_ids",
                         "seqlens_k", "total_sequence_length", *names_in],
            output_names=["logits", *names_out], opset=args.opset,
            dynamic_axes={"prefix_input_ids": {1: "prefix_len"},
                          "audio_embeddings": {0: "num_audio_tokens"},
                          "suffix_input_ids": {1: "suffix_len"},
                          "logits": {1: "seq_len"}},
            exporter="legacy")
    finally:
        Qwen2Attention.forward = original
    del dec, kv
    gc.collect(); torch.cuda.empty_cache()

    report = {
        "model": "vibevoice_asr_streaming", "repo_id": mp.name, "dtype": "float16",
        "opset": args.opset, "device": args.device, "deterministic_audio": True,
        "attention": "GroupQueryAttention", "kv_cache": "shared-buffer fp16",
        "f32_kv_cache": False, "f32_lm_head": bool(args.f32_lm_head),
        "static_kv_cache": True, "static_kv_max_tokens": args.max_tokens,
        "audio_embeddings_dtype": "float32",
        "num_layers": L, "num_kv_heads": KH, "head_dim": HD, "hidden_size": hidden,
        "vocab_size": dc.vocab_size,
        "acoustic_tokenizer_chunk_size": extras["window_samples"],
        "streaming": {k: extras[k] for k in ("sample_rate", "samples_per_frame", "chunk_frames",
                                             "lookahead_frames", "window_samples", "hop_samples")},
        "tokenizer": {k: extras[k] for k in ("speech_start_id", "speech_end_id", "text_chunk_end_id",
                                             "eos_token_id", "prompt_token_ids",
                                             "prompt_hotwords_head_token_ids",
                                             "prompt_tail_token_ids", "strip_tokens")},
    }
    (out / "export-report.json").write_text(json.dumps(report, indent=1))
    print("wrote", out / "export-report.json")


if __name__ == "__main__":
    main()
